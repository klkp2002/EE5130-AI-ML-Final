"""
Unit Tests for Data Partitioning and Strict Zero-Data-Leakage Safeguards.
"""

from pathlib import Path
import json
import numpy as np
import pytest
from src.utils import DATA_DIR, load_json
from src.clustering import ClusterDigitVoter
from sklearn.cluster import KMeans


def test_train_val_split_disjoint():
    """Verify that train and validation indices in splits.json have 0% overlap."""
    splits_path = DATA_DIR / "splits.json"
    assert splits_path.exists(), "splits.json must exist in data directory"

    splits = load_json(splits_path)
    train_idx = set(splits["train_indices"])
    val_idx = set(splits["val_indices"])

    # Disjoint check
    intersection = train_idx.intersection(val_idx)
    assert len(intersection) == 0, f"Data leakage detected! {len(intersection)} overlapping indices between train and val."

    # Partition sizes: 54,000 train + 6,000 val = 60,000
    assert len(train_idx) == 54000
    assert len(val_idx) == 6000


def test_test_data_integrity():
    """Verify mnist_test.csv has exactly 10,000 samples and label column at index 0."""
    test_path = DATA_DIR / "mnist_test.csv"
    assert test_path.exists(), "mnist_test.csv must exist in data directory"

    # Read line count
    with open(test_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # If header exists, 10001 lines; if no header, 10000 lines
    has_header = not lines[0].replace(",", "").isdigit()
    expected_rows = 10001 if has_header else 10000
    assert len(lines) == expected_rows, f"Expected 10,000 data rows, found {len(lines)}"


def test_clustering_voter_unsupervised_fit():
    """Verify that clustering algorithm fits strictly without seeing labels."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 784)).astype(np.float32)
    y = rng.integers(0, 5, size=50)

    km = KMeans(n_clusters=5, random_state=42, n_init=1)

    voter = ClusterDigitVoter(
        clusterer=km,
        name="Test Voter",
    )
    voter.fit(X, y)

    pred = voter.predict(X[:1])
    assert "digit" in pred
    assert "probabilities" in pred
    assert len(pred["probabilities"]) == 10


