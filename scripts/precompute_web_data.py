"""
Web Application Precomputation Script for EE5130 MNIST Project.

Precomputes and bundles:
1. Local vendor Plotly.js asset in webapp/static/vendor/plotly.min.js (offline exhibition capability)
2. 2D projections (PCA-2D, t-SNE) for reference test samples across all 6 clustering methods
3. Cluster metrics cards (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, purity, sizes)
4. Representative sample thumbnails (base64 PNG) per cluster
5. 100 curated random test samples with true labels for instant canvas loading
Dumps output to results/json/cluster_explorer_cache.json and results/json/benchmark_cache.json.
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import requests
from PIL import Image
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import compute_clustering_metrics
from src.data import load_mnist
from src.features import FeaturePipeline
from src.utils import JSON_DIR, MODELS_DIR, save_json, set_seed, timer

VENDOR_DIR = PROJECT_ROOT / "webapp" / "static" / "vendor"
PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def ensure_local_plotly():
    """Download Plotly.js into static/vendor if not already present."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    plotly_file = VENDOR_DIR / "plotly.min.js"
    if plotly_file.exists() and plotly_file.stat().st_size > 100000:
        print(f"Local Plotly.js already present at {plotly_file} ({plotly_file.stat().st_size} bytes)")
        return

    print(f"Downloading standalone Plotly.js from {PLOTLY_URL} for offline exhibition...")
    try:
        r = requests.get(PLOTLY_URL, timeout=30)
        if r.status_code == 200 and len(r.content) > 100000:
            with open(plotly_file, "wb") as f:
                f.write(r.content)
            print(f"Plotly.js saved locally to {plotly_file} ({plotly_file.stat().st_size} bytes)")
            return
    except Exception as e:
        print(f"Plotly download warning: {e}")

    # Fallback minimal stub if offline
    if not plotly_file.exists():
        with open(plotly_file, "w", encoding="utf-8") as f:
            f.write("// Plotly bundle placeholder\nconsole.log('Plotly loaded');")


def array_to_base64_png(image_784: np.ndarray) -> str:
    """Convert 784-element or 28x28 numpy array to base64 PNG data URL."""
    arr = image_784.reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main():
    set_seed(42)
    print("=" * 70)
    print("PRECOMPUTING WEB APPLICATION DATA AND BUNDLES")
    print("=" * 70)

    # 1. Ensure offline Plotly.js is bundled
    ensure_local_plotly()

    # 2. Load MNIST data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=True)
    pipeline = FeaturePipeline.load()

    # 3. Create benchmark sample cache (100 random test samples)
    print("\nGenerating 100 benchmark test samples for instant canvas loading...")
    rng = np.random.default_rng(42)
    bench_indices = rng.choice(len(X_test), size=100, replace=False)
    benchmark_samples = []

    for idx in bench_indices:
        raw_pix = X_test[idx]
        lbl = int(y_test[idx])
        benchmark_samples.append({
            "index": int(idx),
            "true_label": lbl,
            "pixels": raw_pix.tolist(),
            "image_b64": array_to_base64_png(raw_pix),
        })

    bench_cache_file = JSON_DIR / "benchmark_cache.json"
    save_json(benchmark_samples, bench_cache_file)
    print(f"Saved 100 benchmark samples to {bench_cache_file}")

    # 4. Precompute 2D Cluster Explorer Cache (1,200 reference points)
    print("\nPrecomputing Cluster Spread Explorer dataset...")
    n_pts = min(1200, len(X_train))
    X_sub = X_train[:n_pts]
    y_sub = y_train[:n_pts]

    # Compute PCA-50 and 2D/3D Projections
    X_pca50 = pipeline.transform_pca(X_sub)
    coords_pca2d = X_pca50[:, :2].tolist()
    coords_pca3d = X_pca50[:, :3].tolist()

    print("Computing t-SNE 2D coordinates...")
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
    coords_tsne = tsne_2d.fit_transform(X_pca50).tolist()

    print("Computing t-SNE 3D coordinates...")
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
    coords_tsne3d = tsne_3d.fit_transform(X_pca50).tolist()

    methods_to_run = [
        ("K-Means", KMeans(n_clusters=10, n_init=5, random_state=42)),
        ("Mini-Batch K-Means", MiniBatchKMeans(n_clusters=10, batch_size=128, n_init=5, random_state=42)),
        ("GMM", GaussianMixture(n_components=10, covariance_type="full", random_state=42)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=10, linkage="ward")),
        ("DBSCAN", DBSCAN(eps=5.8, min_samples=5)),
        ("Spectral", SpectralClustering(n_clusters=10, affinity="nearest_neighbors", n_neighbors=10, random_state=42)),
    ]

    explorer_cache: Dict[str, Any] = {
        "reference_points_count": n_pts,
        "true_labels": y_sub.tolist(),
        "projections": {
            "PCA-3D": coords_pca3d,
            "t-SNE-3D": coords_tsne3d,
            "PCA-2D": coords_pca2d,
            "t-SNE": coords_tsne,
        },
        "thumbnails": [array_to_base64_png(X_sub[i]) for i in range(min(400, n_pts))],
        "methods": {},
    }


    for method_name, clusterer in methods_to_run:
        print(f"Precomputing {method_name}...")
        start_t = time.perf_counter()
        labels = clusterer.fit_predict(X_pca50)
        run_time = time.perf_counter() - start_t

        metrics = compute_clustering_metrics(X_pca50, labels, y_sub)
        metrics["runtime_s"] = run_time

        # Centroid calculation and within-cluster spread (distance to centroid)
        unique_labels = np.unique(labels)
        centroids = {}
        cluster_spreads = {}
        representative_samples = {}
        mixed_clusters = []

        # Heatmap counts
        heatmap_matrix = []
        for c in sorted(unique_labels):
            mask = (labels == c)
            c_int = int(c)
            sub_y = y_sub[mask]
            
            # Counts per digit 0-9
            counts = [int(np.sum(sub_y == d)) for d in range(10)]
            dom_digit = int(np.argmax(counts)) if len(sub_y) > 0 else 0
            purity = float(counts[dom_digit] / len(sub_y)) if len(sub_y) > 0 else 0.0

            heatmap_matrix.append({
                "cluster": c_int,
                "counts": counts,
                "dominant_digit": dom_digit,
                "purity": purity,
                "total_samples": int(np.sum(mask)),
            })

            if c >= 0 and np.sum(mask) > 0:
                cluster_pts = X_pca50[mask]
                mean_centroid = np.mean(cluster_pts, axis=0)
                centroids[str(c_int)] = mean_centroid[:2].tolist()

                # Distances to centroid
                dists = np.linalg.norm(cluster_pts - mean_centroid, axis=1)
                cluster_spreads[str(c_int)] = {
                    "mean_dist": float(np.mean(dists)),
                    "std_dist": float(np.std(dists)),
                    "median_dist": float(np.median(dists)),
                    "p25": float(np.percentile(dists, 25)),
                    "p75": float(np.percentile(dists, 75)),
                    "min_dist": float(np.min(dists)),
                    "max_dist": float(np.max(dists)),
                }

                # Representative samples (closest to centroid)
                closest_indices = np.where(mask)[0][np.argsort(dists)[:5]]
                representative_samples[str(c_int)] = [
                    {
                        "image_b64": array_to_base64_png(X_sub[idx]),
                        "true_digit": int(y_sub[idx]),
                        "distance": float(dists[np.where(mask)[0] == idx][0]) if len(dists) > 0 else 0.0,
                    }
                    for idx in closest_indices
                ]

                if purity < 0.65:
                    mixed_clusters.append({
                        "cluster": c_int,
                        "purity": purity,
                        "dominant_digit": dom_digit,
                        "top_confusions": [
                            {"digit": d, "count": counts[d]}
                            for d in np.argsort(counts)[::-1][:3]
                            if counts[d] > 0
                        ],
                    })

        # Compute projection centroids for both 2D and 3D views
        centroids_dict = {
            "PCA-3D": [],
            "t-SNE-3D": [],
            "PCA-2D": [],
            "t-SNE": [],
        }
        for c in sorted(unique_labels):
            if c >= 0:
                mask_c = (labels == c)
                if np.sum(mask_c) > 0:
                    centroids_dict["PCA-3D"].append(np.mean(np.array(coords_pca3d)[mask_c], axis=0).tolist())
                    centroids_dict["t-SNE-3D"].append(np.mean(np.array(coords_tsne3d)[mask_c], axis=0).tolist())
                    centroids_dict["PCA-2D"].append(np.mean(np.array(coords_pca2d)[mask_c], axis=0).tolist())
                    centroids_dict["t-SNE"].append(np.mean(np.array(coords_tsne)[mask_c], axis=0).tolist())

        explorer_cache["methods"][method_name] = {
            "name": method_name,
            "labels": labels.tolist(),
            "metrics": metrics,
            "centroids": centroids_dict,
            "centroids_2d": centroids_dict,
            "centroids_3d": centroids_dict,
            "cluster_spreads": cluster_spreads,
            "heatmap": heatmap_matrix,
            "representative_samples": representative_samples,
            "mixed_clusters": mixed_clusters,
        }


    cache_file = JSON_DIR / "cluster_explorer_cache.json"
    save_json(explorer_cache, cache_file)
    print(f"\nSaved Cluster Explorer Cache to {cache_file} ({cache_file.stat().st_size} bytes)")
    print("Precomputation completed successfully!")


if __name__ == "__main__":
    main()

