"""
Data acquisition, verification, and preprocessing module for EE5130 MNIST Project.

Responsibilities:
1. Verify `mnist_test.csv` (10,000 rows, 785 columns, correct classes and pixel bounds).
2. Download official 60,000 MNIST training set if not present and cache as `mnist_train.csv`.
3. Verify downloaded test images against `mnist_test.csv`.
4. Partition training data into stratified 54,000 train and 6,000 validation sets.
5. Provide data loaders for quick mode (subsets) and full mode.
"""

import gzip
import os
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from src.utils import PROJECT_ROOT, DATA_DIR, JSON_DIR, save_json, load_json

# URLs for official MNIST dataset mirrors
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url_list, filename: str, target_path: Path) -> None:
    """Download a file trying multiple mirrors with timeout."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for base_url in url_list:
        full_url = base_url + filename
        try:
            print(f"Downloading {filename} from {base_url}...")
            response = requests.get(full_url, headers=headers, timeout=20)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(target_path, "wb") as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filename} ({target_path.stat().st_size} bytes)")
                return
        except Exception as e:
            print(f"Failed to fetch {full_url}: {e}")

    raise RuntimeError(
        f"Could not download {filename} from any mirror. "
        f"Please place 'mnist_train.csv' manually into '{DATA_DIR}'."
    )


def read_idx_images(gz_path: Path) -> np.ndarray:
    """Parse MNIST idx3 gzip file into uint8 numpy array of shape (N, 784)."""
    with gzip.open(gz_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} in {gz_path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols)


def read_idx_labels(gz_path: Path) -> np.ndarray:
    """Parse MNIST idx1 gzip file into uint8 numpy array of shape (N,)."""
    with gzip.open(gz_path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} in {gz_path}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def verify_test_csv(test_csv_path: Path) -> pd.DataFrame:
    """Verify format, dimensions, labels, and pixel ranges of test CSV."""
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    print(f"Verifying {test_csv_path}...")
    df = pd.read_csv(test_csv_path)

    # Check shape
    assert df.shape == (10000, 785), f"Unexpected test shape: {df.shape}, expected (10000, 785)"
    
    # Check label column
    label_col = df.columns[0]
    labels = df[label_col].values
    assert set(np.unique(labels)).issubset(set(range(10))), f"Invalid labels found: {np.unique(labels)}"
    
    # Check pixel value range
    pixels = df.iloc[:, 1:].values
    assert pixels.min() >= 0 and pixels.max() <= 255, f"Pixel values out of [0, 255]: min={pixels.min()}, max={pixels.max()}"

    print(f"Verified test CSV: 10,000 samples, 784 features, classes: {np.bincount(labels)}")
    return df


def acquire_mnist_train_set(data_dir: Path) -> Path:
    """Download raw MNIST and convert to mnist_train.csv if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv_path = data_dir / "mnist_train.csv"

    if train_csv_path.exists() and train_csv_path.stat().st_size > 10_000_000:
        print(f"Cached training data exists at {train_csv_path}")
        return train_csv_path

    # Download raw gzip archives
    download_dir = data_dir / "raw"
    download_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in FILES.items():
        target = download_dir / filename
        download_file(MNIST_MIRRORS, filename, target)

    # Read train images and labels
    train_x = read_idx_images(download_dir / FILES["train_img"])
    train_y = read_idx_labels(download_dir / FILES["train_lbl"])
    assert train_x.shape == (60000, 784), f"Unexpected train_x shape: {train_x.shape}"
    assert train_y.shape == (60000,), f"Unexpected train_y shape: {train_y.shape}"

    # Read test images and labels for verification
    test_x = read_idx_images(download_dir / FILES["test_img"])
    test_y = read_idx_labels(download_dir / FILES["test_lbl"])
    assert test_x.shape == (10000, 784), f"Unexpected test_x shape: {test_x.shape}"
    assert test_y.shape == (10000,), f"Unexpected test_y shape: {test_y.shape}"

    # Verify against mnist_test.csv
    test_csv_path = data_dir / "mnist_test.csv"
    if test_csv_path.exists():
        df_test = pd.read_csv(test_csv_path)
        csv_y = df_test.iloc[:, 0].values.astype(np.uint8)
        csv_x = df_test.iloc[:, 1:].values.astype(np.uint8)
        
        lbl_diff = np.sum(test_y != csv_y)
        pix_diff = np.sum(test_x != csv_x)
        print(f"Verification against mnist_test.csv: Label differences = {lbl_diff}, Pixel differences = {pix_diff}")
        assert lbl_diff == 0, "Downloaded test labels do not match mnist_test.csv!"
        assert pix_diff == 0, "Downloaded test pixels do not match mnist_test.csv!"
        print("Downloaded MNIST test images match mnist_test.csv with 100% precision.")

    # Save mnist_train.csv with header matching mnist_test.csv
    print(f"Saving 60,000 training samples to {train_csv_path}...")
    columns = ["label"] + [f"{r}x{c}" for r in range(1, 29) for c in range(1, 29)]
    train_df = pd.DataFrame(np.hstack([train_y[:, None], train_x]), columns=columns)
    train_df.to_csv(train_csv_path, index=False)
    print(f"Saved {train_csv_path} ({train_csv_path.stat().st_size} bytes)")
    return train_csv_path


def create_train_val_split(train_csv_path: Path, val_size: int = 6000, seed: int = 42) -> Dict[str, list]:
    """Create a stratified 54,000 train / 6,000 validation split and persist indices."""
    split_file = train_csv_path.parent / "splits.json"
    if split_file.exists():
        return load_json(split_file)

    df = pd.read_csv(train_csv_path, usecols=[0])
    labels = df.iloc[:, 0].values

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for digit in range(10):
        digit_idx = np.where(labels == digit)[0]
        rng.shuffle(digit_idx)
        val_count = val_size // 10
        val_indices.extend(digit_idx[:val_count].tolist())
        train_indices.extend(digit_idx[val_count:].tolist())

    train_indices.sort()
    val_indices.sort()

    splits = {
        "seed": seed,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    save_json(splits, split_file)
    print(f"Created stratified split: {len(train_indices)} train, {len(val_indices)} val. Saved to {split_file}")
    return splits


def load_mnist(
    quick: bool = False,
    quick_train_samples: int = 5000,
    quick_val_samples: int = 1000,
    quick_test_samples: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST train, val, and test partitions.
    Returns: (X_train, y_train, X_val, y_val, X_test, y_test)
    All X arrays are uint8 in range [0, 255] and shape (N, 784).
    All y arrays are uint8 in range [0, 9] and shape (N,).
    """
    train_csv = DATA_DIR / "mnist_train.csv"
    test_csv = DATA_DIR / "mnist_test.csv"

    if not train_csv.exists():
        acquire_mnist_train_set(DATA_DIR)

    splits = create_train_val_split(train_csv, val_size=6000, seed=seed)

    print("Loading datasets from disk...")
    df_train_full = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train_full = df_train_full.iloc[:, 1:].values.astype(np.uint8)
    y_train_full = df_train_full.iloc[:, 0].values.astype(np.uint8)

    X_test = df_test.iloc[:, 1:].values.astype(np.uint8)
    y_test = df_test.iloc[:, 0].values.astype(np.uint8)

    train_idx = np.array(splits["train_indices"])
    val_idx = np.array(splits["val_indices"])

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    if quick:
        print(f"Extracting stratified quick subset: {quick_train_samples} train, {quick_val_samples} val, {quick_test_samples} test")
        rng = np.random.default_rng(seed)

        def get_stratified_sub(X, y, n_samples):
            sub_idx = []
            per_class = n_samples // 10
            for d in range(10):
                c_idx = np.where(y == d)[0]
                sub_idx.extend(rng.choice(c_idx, size=per_class, replace=False))
            sub_idx = np.array(sub_idx)
            rng.shuffle(sub_idx)
            return X[sub_idx], y[sub_idx]

        X_train, y_train = get_stratified_sub(X_train, y_train, quick_train_samples)
        X_val, y_val = get_stratified_sub(X_val, y_val, quick_val_samples)
        X_test, y_test = get_stratified_sub(X_test, y_test, quick_test_samples)

    print(f"Loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    acquire_mnist_train_set(DATA_DIR)
    verify_test_csv(DATA_DIR / "mnist_test.csv")
    create_train_val_split(DATA_DIR / "mnist_train.csv")
    print("Data acquisition & verification complete!")

