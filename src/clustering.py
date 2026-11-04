"""
Unsupervised Clustering and Topology Analysis Suite for EE5130 MNIST Project.

Algorithms:
1. K-Means
2. Mini-Batch K-Means
3. Gaussian Mixture Models (GMM with 4 covariance types)
4. Agglomerative Hierarchical Clustering (Ward & Average linkage)
5. DBSCAN (Density-based with noise detection and k-distance estimation)
6. Spectral Clustering (Normalized Cuts on graph Laplacian)
7. Unsupervised Digit Voter (post-hoc cluster -> majority digit mapping)
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
import joblib

from src.classifiers import MNISTBaseClassifier
from src.features import FeaturePipeline
from src.utils import MODELS_DIR


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    sample_size_silhouette: int = 5000,
) -> Dict[str, Any]:
    """Compute internal and external clustering validation metrics."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    noise_count = int(np.sum(labels == -1))
    noise_ratio = float(noise_count / len(labels))

    metrics: Dict[str, Any] = {
        "n_clusters": int(n_clusters),
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "ari": None,
        "nmi": None,
        "homogeneity": None,
        "completeness": None,
        "v_measure": None,
        "purity": None,
    }

    # Internal metrics require at least 2 non-noise clusters
    valid_mask = labels >= 0
    if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        # Subsample for silhouette if dataset is large to maintain speed
        if len(X_valid) > sample_size_silhouette:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(len(X_valid), size=sample_size_silhouette, replace=False)
            metrics["silhouette"] = float(silhouette_score(X_valid[sub_idx], labels_valid[sub_idx]))
        else:
            metrics["silhouette"] = float(silhouette_score(X_valid, labels_valid))

        metrics["davies_bouldin"] = float(davies_bouldin_score(X_valid, labels_valid))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_valid, labels_valid))

    # External metrics (post-hoc evaluation with ground truth labels)
    if y_true is not None:
        y_valid = y_true[valid_mask]
        labels_valid = labels[valid_mask]
        metrics["ari"] = float(adjusted_rand_score(y_valid, labels_valid))
        metrics["nmi"] = float(normalized_mutual_info_score(y_valid, labels_valid))
        h, c, v = homogeneity_completeness_v_measure(y_valid, labels_valid)
        metrics["homogeneity"] = float(h)
        metrics["completeness"] = float(c)
        metrics["v_measure"] = float(v)

        # Cluster Purity
        total_correct = 0
        for k in unique_labels[unique_labels >= 0]:
            cluster_y = y_valid[labels_valid == k]
            if len(cluster_y) > 0:
                counts = np.bincount(cluster_y)
                total_correct += np.max(counts)
        metrics["purity"] = float(total_correct / len(y_valid))

    return metrics


class ClusterDigitVoter(MNISTBaseClassifier):
    """
    Unsupervised K-Means / GMM digit classifier.
    Fits clustering without labels; then maps each cluster to the majority
    training digit post-hoc. Assigns probability based on cluster purity and distance.
    """

    def __init__(self, clusterer: Any, name: str = "K-Means Digit Voter", feature_pipeline: Optional[FeaturePipeline] = None):
        super().__init__(name=name, family="Unsupervised Clustering")
        self.clusterer = clusterer
        self.feature_pipeline = feature_pipeline
        self.cluster_to_digit: Dict[int, int] = {}
        self.cluster_purities: Dict[int, float] = {}
        self.centroids: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ClusterDigitVoter":
        # 1. Transform features to PCA or normalized
        if self.feature_pipeline is not None:
            X_feat = self.feature_pipeline.transform_pca(X)
        else:
            X_feat = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)

        # 2. Fit unsupervised clusterer WITHOUT labels
        print(f"Fitting {self.name} clustering without labels...")
        self.clusterer.fit(X_feat)

        if hasattr(self.clusterer, "cluster_centers_"):
            self.centroids = self.clusterer.cluster_centers_
        elif hasattr(self.clusterer, "means_"):
            self.centroids = self.clusterer.means_

        cluster_labels = self.clusterer.predict(X_feat) if hasattr(self.clusterer, "predict") else self.clusterer.labels_

        # 3. Post-hoc map each cluster to majority training label
        if y is not None:
            for k in range(len(self.centroids) if self.centroids is not None else 10):
                mask = (cluster_labels == k)
                if np.sum(mask) > 0:
                    y_sub = y[mask]
                    majority_digit = int(np.bincount(y_sub, minlength=10).argmax())
                    purity = float(np.max(np.bincount(y_sub)) / len(y_sub))
                    self.cluster_to_digit[k] = majority_digit
                    self.cluster_purities[k] = purity
                else:
                    self.cluster_to_digit[k] = k % 10
                    self.cluster_purities[k] = 0.1

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, f"{self.name} is not fitted."
        if self.feature_pipeline is not None:
            X_feat = self.feature_pipeline.transform_pca(X)
        else:
            X_feat = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)

        N = len(X_feat)
        probas = np.zeros((N, 10), dtype=np.float32)

        if self.centroids is not None:
            # Distance-based softmax over centroids
            # dist shape: (N, K)
            dists = np.linalg.norm(X_feat[:, None, :] - self.centroids[None, :, :], axis=2)
            # Softmin over distances
            neg_dists = -dists
            weights = np.exp(neg_dists - np.max(neg_dists, axis=1, keepdims=True))
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            for k in range(weights.shape[1]):
                target_digit = self.cluster_to_digit.get(k, k % 10)
                purity = self.cluster_purities.get(k, 0.5)
                probas[:, target_digit] += weights[:, k] * purity

            # Normalize to valid probability distribution
            row_sums = probas.sum(axis=1, keepdims=True) + 1e-9
            probas = probas / row_sums
        else:
            clusters = self.clusterer.predict(X_feat)
            for i, c in enumerate(clusters):
                d = self.cluster_to_digit.get(c, 0)
                p = self.cluster_purities.get(c, 0.5)
                probas[i, d] = p
                rem = (1.0 - p) / 9.0
                probas[i] += rem
                probas[i, d] = p
        return probas

