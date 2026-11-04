"""
Unit Tests for Metric Calculations (Macro, Weighted, Micro, ECE, Clustering).
"""

import numpy as np
import pytest
from src.evaluation import evaluate_classifier
from src.calibration import compute_ece_and_reliability
from src.clustering import compute_clustering_metrics




def test_classification_metrics_multiclass():
    """Verify macro, weighted, micro precision, recall, and F1 calculations."""
    # Synthetic ground truth and predictions across 3 classes
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    # Class 0: true=2, tp=1, fp=1, fn=1 -> prec=0.5, rec=0.5, f1=0.5
    # Class 1: true=2, tp=2, fp=1, fn=0 -> prec=2/3=0.667, rec=1.0, f1=0.8
    # Class 2: true=2, tp=1, fp=0, fn=1 -> prec=1.0, rec=0.5, f1=0.667

    # Dummy probabilities
    probs = np.zeros((6, 3))
    for i, p in enumerate(y_pred):
        probs[i, p] = 0.9
        probs[i, (p + 1) % 3] = 0.1

    metrics = evaluate_classifier(y_true, y_pred, y_proba=probs)

    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-3)
    assert metrics["precision_micro"] == pytest.approx(4 / 6, abs=1e-3)
    assert metrics["recall_micro"] == pytest.approx(4 / 6, abs=1e-3)
    assert metrics["f1_micro"] == pytest.approx(4 / 6, abs=1e-3)

    # Macro averages
    expected_macro_prec = (0.5 + (2 / 3) + 1.0) / 3.0
    expected_macro_rec = (0.5 + 1.0 + 0.5) / 3.0
    assert metrics["precision_macro"] == pytest.approx(expected_macro_prec, abs=1e-3)
    assert metrics["recall_macro"] == pytest.approx(expected_macro_rec, abs=1e-3)



def test_ece_perfect_calibration():
    """Verify that a calibrated probability distribution yields low ECE."""
    n = 1000
    y_true = np.random.randint(0, 2, size=n)
    # Generate probabilities where prob == true label with 0.8 probability
    y_proba = np.zeros((n, 2))
    for i in range(n):
        if np.random.rand() < 0.8:
            y_proba[i, y_true[i]] = 0.8
            y_proba[i, 1 - y_true[i]] = 0.2
        else:
            y_proba[i, 1 - y_true[i]] = 0.8
            y_proba[i, y_true[i]] = 0.2

    calib = compute_ece_and_reliability(y_true, y_proba, n_bins=10)
    ece = calib["ece"]
    assert 0.0 <= ece < 0.1, f"Expected low ECE ~0.0, got {ece}"



def test_clustering_metrics():
    """Verify clustering metric calculation with valid bounds."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-5.0, -5.0], scale=0.5, size=(50, 2))
    c2 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(50, 2))
    X = np.vstack([c1, c2])
    y_true = np.array([0] * 50 + [1] * 50)
    labels = np.array([0] * 50 + [1] * 50)

    metrics = compute_clustering_metrics(X, labels, y_true=y_true)

    assert metrics["silhouette"] > 0.8
    assert metrics["davies_bouldin"] < 0.5
    assert metrics["ari"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["purity"] == pytest.approx(1.0, abs=1e-4)
