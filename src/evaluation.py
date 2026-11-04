"""
Evaluation metrics and reporting tools for EE5130 MNIST Project.

Computes:
- Multiclass Accuracy, Precision, Recall, F1 (Macro, Weighted, Micro)
- Per-digit Precision, Recall, F1-score, and Support
- Confusion Matrix (counts and normalized)
- One-vs-Rest ROC curves and ROC-AUC
- Model inference latency (per sample and batch)
- Serialized model size on disk
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    inference_time_ms: float = 0.0,
    model_name: str = "Classifier",
    family: str = "Supervised",
    model_size_bytes: int = 0,
) -> Dict[str, Any]:
    """
    Compute comprehensive multiclass evaluation dictionary.
    Includes macro, weighted, micro averages, per-digit stats, and ROC-AUC.
    """
    acc = float(accuracy_score(y_true, y_pred))

    # Average metrics
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    r_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # Per-digit metrics
    p_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    r_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    support_per_class = np.bincount(y_true, minlength=10).tolist()

    # Confusion matrix
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm_counts.astype(np.float64) / (cm_counts.sum(axis=1, keepdims=True) + 1e-9)

    # ROC-AUC if probabilities provided
    roc_auc_macro = None
    roc_auc_weighted = None
    per_class_auc = {}
    if y_proba is not None and y_proba.shape[1] == 10:
        try:
            # One-hot encode true labels
            y_onehot = np.eye(10)[y_true]
            roc_auc_macro = float(roc_auc_score(y_onehot, y_proba, multi_class="ovr", average="macro"))
            roc_auc_weighted = float(roc_auc_score(y_onehot, y_proba, multi_class="ovr", average="weighted"))
            for d in range(10):
                fpr, tpr, _ = roc_curve(y_onehot[:, d], y_proba[:, d])
                per_class_auc[str(d)] = float(auc(fpr, tpr))
        except Exception as e:
            print(f"Warning: ROC-AUC calculation failed for {model_name}: {e}")

    results = {
        "model_name": model_name,
        "family": family,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "per_class_precision": p_per_class,
        "per_class_recall": r_per_class,
        "per_class_f1": f1_per_class,
        "per_class_support": support_per_class,
        "per_class_auc": per_class_auc,
        "confusion_matrix_counts": cm_counts.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "inference_time_ms_per_sample": float(inference_time_ms),
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": float(model_size_bytes / (1024 * 1024)),
    }
    return results


def find_most_confused_pairs(cm: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """Identify the most frequently confused true -> pred digit pairs."""
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    
    indices = np.unravel_index(np.argsort(cm_off_diag.ravel())[::-1], cm_off_diag.shape)
    pairs = []
    for i in range(min(top_k, len(indices[0]))):
        t = int(indices[0][i])
        p = int(indices[1][i])
        count = int(cm_off_diag[t, p])
        if count > 0:
            pairs.append({"true_digit": t, "predicted_digit": p, "count": count})
    return pairs

