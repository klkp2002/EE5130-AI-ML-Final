"""
Feature extraction and dimensionality reduction pipelines for EE5130 MNIST Project.

Features:
- Pixel Normalization to [0, 1]
- Pixel Standardization (Z-score)
- PCA (Principal Component Analysis)
- SVD (Truncated SVD)
- HOG (Histogram of Oriented Gradients)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils import MODELS_DIR


class FeaturePipeline:
    """
    Unified feature pipeline that fits all transformers strictly on training data
    and provides transform methods for validation, test, and inference.
    """
    def __init__(self, pca_components: int = 50, svd_components: int = 50):
        self.pca_components = pca_components
        self.svd_components = svd_components
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=pca_components, random_state=42)
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.is_fitted = False

    @staticmethod
    def normalize_pixels(X: np.ndarray) -> np.ndarray:
        """Scale uint8 pixel values from [0, 255] to float32 [0.0, 1.0]."""
        if X.dtype == np.uint8 or X.max() > 1.0:
            return (X.astype(np.float32) / 255.0)
        return X.astype(np.float32)

    def fit(self, X_train: np.ndarray) -> "FeaturePipeline":
        """Fit scaler, PCA, and SVD strictly on training data."""
        X_norm = self.normalize_pixels(X_train)
        print("Fitting StandardScaler on training data...")
        self.scaler.fit(X_norm)
        
        print(f"Fitting PCA ({self.pca_components} components) on training data...")
        self.pca.fit(X_norm)
        
        print(f"Fitting TruncatedSVD ({self.svd_components} components) on training data...")
        self.svd.fit(X_norm)
        
        self.is_fitted = True
        return self

    def transform_normalized(self, X: np.ndarray) -> np.ndarray:
        return self.normalize_pixels(X)

    def transform_standardized(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "FeaturePipeline must be fitted before transform."
        X_norm = self.normalize_pixels(X)
        return self.scaler.transform(X_norm)

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "FeaturePipeline must be fitted before transform."
        X_norm = self.normalize_pixels(X)
        return self.pca.transform(X_norm)

    def transform_svd(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "FeaturePipeline must be fitted before transform."
        X_norm = self.normalize_pixels(X)
        return self.svd.transform(X_norm)

    def save(self, filepath: Optional[Path] = None) -> Path:
        if filepath is None:
            filepath = MODELS_DIR / "feature_pipeline.joblib"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"FeaturePipeline saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "FeaturePipeline":
        if filepath is None:
            filepath = MODELS_DIR / "feature_pipeline.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Saved FeaturePipeline not found at {filepath}")
        return joblib.load(filepath)


def extract_hog_features(images: np.ndarray, cell_size: int = 7, n_bins: int = 8) -> np.ndarray:
    """
    Fast vectorized pure-numpy HOG (Histogram of Oriented Gradients) descriptor for 28x28 MNIST images.
    No heavy external skimage dependency required.
    
    For 28x28 image with 7x7 cells -> 4x4 = 16 cells.
    With 8 orientation bins -> 16 * 8 = 128-dimensional feature vector.
    """
    if images.ndim == 1:
        images = images.reshape(1, -1)
    
    N = images.shape[0]
    imgs_2d = images.reshape(N, 28, 28).astype(np.float32) / 255.0
    
    # Compute Sobel-like gradients along x and y
    gx = np.zeros_like(imgs_2d)
    gy = np.zeros_like(imgs_2d)
    gx[:, :, 1:-1] = imgs_2d[:, :, 2:] - imgs_2d[:, :, :-2]
    gy[:, 1:-1, :] = imgs_2d[:, 2:, :] - imgs_2d[:, :-2, :]

    magnitude = np.hypot(gx, gy)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180.0) # 0 to 180 degrees

    bin_width = 180.0 / n_bins
    bin_idx = (orientation / bin_width).astype(int) % n_bins

    n_cells_y = 28 // cell_size
    n_cells_x = 28 // cell_size
    n_features = n_cells_y * n_cells_x * n_bins
    hog_feats = np.zeros((N, n_features), dtype=np.float32)

    feat_offset = 0
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start = cy * cell_size
            y_end = y_start + cell_size
            x_start = cx * cell_size
            x_end = x_start + cell_size

            cell_mag = magnitude[:, y_start:y_end, x_start:x_end]
            cell_bin = bin_idx[:, y_start:y_end, x_start:x_end]

            for b in range(n_bins):
                mask = (cell_bin == b)
                hog_feats[:, feat_offset + b] = np.sum(cell_mag * mask, axis=(1, 2))

            feat_offset += n_bins

    # L2-Hys normalization per sample
    norms = np.linalg.norm(hog_feats, axis=1, keepdims=True) + 1e-6
    hog_feats = hog_feats / norms
    hog_feats = np.clip(hog_feats, 0, 0.2)
    hog_feats = hog_feats / (np.linalg.norm(hog_feats, axis=1, keepdims=True) + 1e-6)

    return hog_feats

