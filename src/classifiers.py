"""
Model Zoo for EE5130 MNIST Supervised Classification.

Provides uniform interface for all classical classifiers:
- predict(image28x28) -> {digit, probabilities, confidence, inference_time_ms, method_name, family}
- fit(X_train, y_train)
- predict_proba(X)
- predict_batch(X)
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.utils import MODELS_DIR


class MNISTBaseClassifier(ABC):
    """Abstract base class ensuring a uniform inference and serialization protocol."""

    def __init__(self, name: str, family: str = "Supervised"):
        self.name = name
        self.family = family
        self.is_fitted = False
        self.test_accuracy: Optional[float] = None
        self.val_accuracy: Optional[float] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MNISTBaseClassifier":
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict single image of shape (28, 28) or (784,).
        Returns standardized dictionary.
        """
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(1, 784)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        start = time.perf_counter()
        probs = self.predict_proba(arr)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pred_digit = int(np.argmax(probs))
        confidence = float(probs[pred_digit])

        return {
            "digit": pred_digit,
            "probabilities": [float(p) for p in probs],
            "confidence": confidence,
            "inference_time_ms": elapsed_ms,
            "method_name": self.name,
            "family": self.family,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        if filepath is None:
            filename = f"{self.name.lower().replace(' ', '_').replace('+', '_')}.joblib"
            filepath = MODELS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "MNISTBaseClassifier":
        return joblib.load(filepath)


class SklearnClassifierWrapper(MNISTBaseClassifier):
    """Wraps scikit-learn estimators to expose the uniform API."""

    def __init__(self, estimator: Any, name: str, family: str = "Supervised Classical"):
        super().__init__(name=name, family=family)
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnClassifierWrapper":
        X_f = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)
        self.estimator.fit(X_f, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, f"{self.name} is not fitted."
        X_f = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_f)
        elif hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X_f)
            # Softmax transformation for decision function logits
            exp = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        else:
            preds = self.estimator.predict(X_f)
            onehot = np.zeros((len(preds), 10), dtype=np.float32)
            onehot[np.arange(len(preds)), preds] = 1.0
            return onehot


class LinearRegressionClassifier(MNISTBaseClassifier):
    """
    Linear Regression used as a 10-class classifier.
    Fits least-squares Ridge regression on one-hot targets, takes argmax,
    and applies calibrated softmax to produce smooth probability distributions.
    """

    def __init__(self, alpha: float = 1.0, name: str = "Linear Regression (Ridge)"):
        super().__init__(name=name, family="Supervised Classical")
        self.alpha = alpha
        self.weights: Optional[np.ndarray] = None  # (785, 10) including bias
        self.temperature: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionClassifier":
        X_f = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)
        N = X_f.shape[0]
        # Add bias column
        X_bias = np.hstack([np.ones((N, 1), dtype=np.float32), X_f])
        
        # One-hot encode targets to {-1, +1} or {0, 1}
        Y_onehot = np.eye(10, dtype=np.float32)[y]

        # Solve Ridge normal equation: W = (X^T X + alpha * I)^(-1) X^T Y
        D = X_bias.shape[1]
        I = np.eye(D, dtype=np.float32)
        I[0, 0] = 0.0  # Do not regularize bias
        XtX = X_bias.T @ X_bias + self.alpha * I
        XtY = X_bias.T @ Y_onehot
        self.weights = np.linalg.solve(XtX, XtY)
        self.is_fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "LinearRegressionClassifier is not fitted."
        X_f = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)
        N = X_f.shape[0]
        X_bias = np.hstack([np.ones((N, 1), dtype=np.float32), X_f])
        return X_bias @ self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X) / self.temperature
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


def build_model_zoo(seed: int = 42) -> Dict[str, MNISTBaseClassifier]:
    """Factory creating all course diagram supervised classifiers."""
    models: Dict[str, MNISTBaseClassifier] = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = SklearnClassifierWrapper(
        LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=seed),
        name="Logistic Regression",
        family="Supervised Classical",
    )

    # 2. Linear Regression (Least-Squares Classifier)
    models["Linear Regression"] = LinearRegressionClassifier(
        alpha=10.0,
        name="Linear Regression",
    )

    # 3. Support Vector Machine (Linear with calibration)
    linear_svc = LinearSVC(C=0.1, dual="auto", random_state=seed, max_iter=2000)
    models["Linear SVM"] = SklearnClassifierWrapper(
        CalibratedClassifierCV(linear_svc, cv=3),
        name="Linear SVM",
        family="Supervised Classical",
    )

    # 4. Support Vector Machine (RBF kernel)
    rbf_svc = SVC(C=5.0, kernel="rbf", gamma="scale", random_state=seed)
    models["RBF SVM"] = SklearnClassifierWrapper(
        CalibratedClassifierCV(rbf_svc, cv=3),
        name="RBF SVM",
        family="Supervised Classical",
    )

    # 5. Random Forest
    models["Random Forest"] = SklearnClassifierWrapper(
        RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed),
        name="Random Forest",
        family="Supervised Classical",
    )

    # 6. k-Nearest Neighbours
    models["k-NN (k=3)"] = SklearnClassifierWrapper(
        KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1),
        name="k-NN (k=3)",
        family="Supervised Classical",
    )

    # 7. Decision Tree
    models["Decision Tree"] = SklearnClassifierWrapper(
        DecisionTreeClassifier(max_depth=15, random_state=seed),
        name="Decision Tree",
        family="Supervised Classical",
    )

    # 8. Naive Bayes
    models["Naive Bayes"] = SklearnClassifierWrapper(
        GaussianNB(var_smoothing=1e-2),
        name="Naive Bayes",
        family="Supervised Classical",
    )

    # 9. Linear Discriminant Analysis
    models["LDA"] = SklearnClassifierWrapper(
        LinearDiscriminantAnalysis(solver="svd"),
        name="LDA",
        family="Supervised Classical",
    )

    # 10. Gradient Boosting
    models["HistGradientBoosting"] = SklearnClassifierWrapper(
        HistGradientBoostingClassifier(max_iter=100, random_state=seed),
        name="HistGradientBoosting",
        family="Supervised Classical",
    )

    # 11-15. Dimension-Reduction Pipelines
    models["PCA + SVM"] = SklearnClassifierWrapper(
        Pipeline([
            ("pca", PCA(n_components=50, random_state=seed)),
            ("svm", CalibratedClassifierCV(SVC(C=5.0, kernel="rbf", gamma="scale", random_state=seed), cv=3)),
        ]),
        name="PCA + SVM",
        family="PCA/SVD Pipeline",
    )

    models["PCA + kNN"] = SklearnClassifierWrapper(
        Pipeline([
            ("pca", PCA(n_components=50, random_state=seed)),
            ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)),
        ]),
        name="PCA + kNN",
        family="PCA/SVD Pipeline",
    )

    models["PCA + Logistic Regression"] = SklearnClassifierWrapper(
        Pipeline([
            ("pca", PCA(n_components=50, random_state=seed)),
            ("logreg", LogisticRegression(C=1.0, max_iter=200, random_state=seed)),
        ]),
        name="PCA + Logistic Regression",
        family="PCA/SVD Pipeline",
    )

    models["SVD + Logistic Regression"] = SklearnClassifierWrapper(
        Pipeline([
            ("svd", TruncatedSVD(n_components=50, random_state=seed)),
            ("logreg", LogisticRegression(C=1.0, max_iter=200, random_state=seed)),
        ]),
        name="SVD + Logistic Regression",
        family="PCA/SVD Pipeline",
    )

    models["SVD + Random Forest"] = SklearnClassifierWrapper(
        Pipeline([
            ("svd", TruncatedSVD(n_components=50, random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=seed)),
        ]),
        name="SVD + Random Forest",
        family="PCA/SVD Pipeline",
    )

    return models
