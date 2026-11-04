"""
Script to analyze PCA and TruncatedSVD dimensional reduction pipelines on MNIST.
Supports --quick and --full modes.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.features import FeaturePipeline, extract_hog_features
from src.plots import plot_scree_variance
from src.utils import TABLES_DIR, timer, set_seed


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST Dimension Reduction & Feature Extraction")
    parser.add_argument("--quick", action="store_true", help="Run with stratified subset for quick verification")
    args = parser.parse_args()

    set_seed(42)
    print("=" * 70)
    print(f"RUNNING DIMENSIONALITY REDUCTION EXPERIMENT (Mode: {'QUICK' if args.quick else 'FULL'})")
    print("=" * 70)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)
    print(f"Data loaded in {t_load['elapsed_sec']:.2f}s")

    X_train_norm = FeaturePipeline.normalize_pixels(X_train)
    components_sweep = [10, 20, 30, 50, 100]
    records = []

    print("\n--- Sweeping PCA components on Training Set (Fit on Train Only) ---")
    pca_100 = PCA(n_components=100, random_state=42)
    pca_100.fit(X_train_norm)
    cum_var_pca = np.cumsum(pca_100.explained_variance_ratio_)

    for k in components_sweep:
        records.append({
            "Method": "PCA",
            "Components": k,
            "Individual_Variance_Last": float(pca_100.explained_variance_ratio_[k - 1]),
            "Cumulative_Explained_Variance": float(cum_var_pca[k - 1]),
        })
        print(f"PCA k={k:3d} -> Cumulative Explained Variance: {cum_var_pca[k - 1] * 100:.2f}%")

    plot_scree_variance(pca_100.explained_variance_ratio_[:50], method_name="PCA")

    print("\n--- Sweeping TruncatedSVD components on Training Set (Fit on Train Only) ---")
    svd_100 = TruncatedSVD(n_components=100, random_state=42)
    svd_100.fit(X_train_norm)
    cum_var_svd = np.cumsum(svd_100.explained_variance_ratio_)

    for k in components_sweep:
        records.append({
            "Method": "TruncatedSVD",
            "Components": k,
            "Individual_Variance_Last": float(svd_100.explained_variance_ratio_[k - 1]),
            "Cumulative_Explained_Variance": float(cum_var_svd[k - 1]),
        })
        print(f"SVD k={k:3d} -> Cumulative Explained Variance: {cum_var_svd[k - 1] * 100:.2f}%")

    plot_scree_variance(svd_100.explained_variance_ratio_[:50], method_name="TruncatedSVD")

    # Save summary table
    df_var = pd.DataFrame(records)
    csv_path = TABLES_DIR / "dimred_variance.csv"
    df_var.to_csv(csv_path, index=False)
    print(f"\nSaved variance table to {csv_path}")

    # Fit and save production feature pipeline (k=50)
    print("\nFitting and saving standard production FeaturePipeline (k=50)...")
    pipeline = FeaturePipeline(pca_components=50, svd_components=50)
    pipeline.fit(X_train)
    pipeline.save()

    # Test HOG feature extraction
    print("\nVerifying HOG feature extraction...")
    hog_sample = extract_hog_features(X_train[:100])
    print(f"Extracted HOG features shape: {hog_sample.shape} (values in [{hog_sample.min():.2f}, {hog_sample.max():.2f}])")

    print("\nDimensionality reduction pipeline experiments successfully completed!")


if __name__ == "__main__":
    main()

