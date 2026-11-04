"""
Ensemble and Model Combination Framework for EE5130 MNIST Project.

Implements:
1. Hard Majority Voting
2. Soft Voting (Unweighted Probability Averaging)
3. Weighted Soft Voting (Weights tuned on validation accuracy / log-loss)
4. Stacking Meta-Classifier (Logistic Regression trained on validation predictions)
5. Full Ablation Evaluator
6. Tri-fold 'Most Precise Answer' Resolver
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from src.classifiers import MNISTBaseClassifier
from src.utils import MODELS_DIR


class AllModelEnsemble(MNISTBaseClassifier):
    """
    Ensemble container holding all trained project models.
    Supports dynamic inclusion/exclusion of models and live switching of combination methods.
    """

    def __init__(
        self,
        models: Dict[str, MNISTBaseClassifier],
        weights: Optional[Dict[str, float]] = None,
        combination_method: str = "weighted",
    ):
        super().__init__(name=f"Ensemble ({combination_method.capitalize()})", family="Ensemble")
        self.models = models
        self.combination_method = combination_method
        self.weights = weights or {m: 1.0 for m in models}
        self.stacker: Optional[LogisticRegression] = None
        self.best_model_overall_name: str = "CNN (Deep Learning)"
        self.is_fitted = True

    def fit_stacker(self, X_val: np.ndarray, y_val: np.ndarray) -> "AllModelEnsemble":
        """Train stacking meta-classifier strictly on validation data predictions."""
        print("Training Stacking Meta-Classifier on validation set predictions...")
        val_meta_features = []
        for name, m in self.models.items():
            probs = m.predict_proba(X_val)
            val_meta_features.append(probs)

        # Shape: (N_val, num_models * 10)
        X_meta = np.hstack(val_meta_features)
        self.stacker = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        self.stacker.fit(X_meta, y_val)
        print("Stacking Meta-Classifier trained successfully.")
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        active_models: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        if active_models is None:
            active_models = list(self.models.keys())
        if method is None:
            method = self.combination_method

        N = len(X)
        selected_models = [self.models[m] for m in active_models if m in self.models]
        if not selected_models:
            return np.ones((N, 10), dtype=np.float32) / 10.0

        if method == "stacking" and self.stacker is not None and len(active_models) == len(self.models):
            meta_features = [self.models[m].predict_proba(X) for m in self.models]
            X_meta = np.hstack(meta_features)
            return self.stacker.predict_proba(X_meta)

        # Soft or Weighted Soft
        combined_probs = np.zeros((N, 10), dtype=np.float32)
        total_weight = 0.0

        for m_name in active_models:
            if m_name in self.models:
                m = self.models[m_name]
                p = m.predict_proba(X)
                w = self.weights.get(m_name, 1.0) if method == "weighted" else 1.0
                combined_probs += w * p
                total_weight += w

        if total_weight > 0:
            combined_probs /= total_weight
        else:
            combined_probs = np.ones((N, 10), dtype=np.float32) / 10.0

        return combined_probs

    def predict_batch(
        self,
        X: np.ndarray,
        active_models: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        if method == "hard":
            if active_models is None:
                active_models = list(self.models.keys())
            votes = np.zeros((len(X), len(active_models)), dtype=int)
            for idx, m_name in enumerate(active_models):
                if m_name in self.models:
                    votes[:, idx] = self.models[m_name].predict_batch(X)
            # Majority vote
            out = np.zeros(len(X), dtype=int)
            for i in range(len(X)):
                out[i] = int(np.bincount(votes[i], minlength=10).argmax())
            return out
        else:
            probas = self.predict_proba(X, active_models=active_models, method=method)
            return np.argmax(probas, axis=1)

    def predict(
        self,
        image: np.ndarray,
        active_models: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive live exhibition inference method.
        Returns the triple answers: Final Answer, Most Confident for input, Best Overall,
        plus the per-model breakdown table.
        """
        if active_models is None:
            active_models = list(self.models.keys())
        if method is None:
            method = self.combination_method

        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(1, 784)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        start = time.perf_counter()
        breakdown = []
        max_conf = -1.0
        most_confident_model = None

        for m_name in active_models:
            if m_name in self.models:
                m = self.models[m_name]
                pred_dict = m.predict(arr)
                breakdown.append(pred_dict)
                if pred_dict["confidence"] > max_conf:
                    max_conf = pred_dict["confidence"]
                    most_confident_model = {
                        "name": m_name,
                        "digit": pred_dict["digit"],
                        "confidence": pred_dict["confidence"],
                    }

        # Ensemble prediction
        probs = self.predict_proba(arr, active_models=active_models, method=method)[0]
        final_digit = int(np.argmax(probs))
        confidence = float(probs[final_digit])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Mark agreement with final answer in breakdown
        agree_count = 0
        for b in breakdown:
            agrees = (b["digit"] == final_digit)
            b["agrees_with_final"] = agrees
            if agrees:
                agree_count += 1

        return {
            "digit": final_digit,
            "final_answer": final_digit,
            "probabilities": [float(p) for p in probs],
            "confidence": confidence,
            "inference_time_ms": elapsed_ms,
            "method_name": f"ENSEMBLE ({method.upper()})",
            "family": "Ensemble",
            "most_confident_method": most_confident_model,
            "best_method_overall": self.best_model_overall_name,
            "agreement_summary": f"{agree_count} of {len(breakdown)} models agree on digit {final_digit}",
            "model_breakdown": breakdown,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AllModelEnsemble":
        return self

