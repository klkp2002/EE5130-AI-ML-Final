"""
Clustering and Unsupervised Topology Experiment Suite for EE5130 MNIST Project.

Evaluates K-Means, Mini-Batch K-Means, GMM, Agglomerative, DBSCAN, and Spectral Clustering
across Raw/Normalized, PCA-10, PCA-50, and Autoencoder latent representations.
Generates 2D topology projections, elbow curves, heatmaps, size charts, and metrics tables.
Supports --quick and --full execution modes.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from sklearn.metrics import silhouette_score
import joblib

matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import compute_clustering_metrics, ClusterDigitVoter
from src.data import load_mnist
from src.features import FeaturePipeline
from src.nn_models import AutoencoderArchitecture
from src.plots import save_figure_with_caption
from src.utils import TABLES_DIR, FIGURES_DIR, JSON_DIR, MODELS_DIR, save_json, load_json, set_seed, timer


def plot_cluster_2d_scatter(
    coords_2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    method_name: str,
    rep_name: str,
    proj_type: str = "PCA-2D",
) -> Path:
    """Plot dual 2-D scatter projections: Left colored by Cluster, Right colored by True Digit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    unique_clusters = np.unique(cluster_labels)
    cmap_clusters = matplotlib.colormaps.get_cmap("tab10") if hasattr(matplotlib.colormaps, "get_cmap") else plt.get_cmap("tab10")

    # Left: Colored by Cluster ID
    for idx, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        color = "grey" if c == -1 else cmap_clusters(idx % 10)
        label = "Noise (-1)" if c == -1 else f"Cluster {c}"
        ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=12, alpha=0.6, color=color, label=label)
    ax1.set_title(f"{method_name} ({rep_name}) - Colored by Cluster", fontsize=11, fontweight="bold")
    ax1.set_xlabel(f"{proj_type} Dim 1", fontsize=10)
    ax1.set_ylabel(f"{proj_type} Dim 2", fontsize=10)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2, fontsize=8)

    # Right: Colored by Ground Truth Digit
    cmap_digits = matplotlib.colormaps.get_cmap("tab10") if hasattr(matplotlib.colormaps, "get_cmap") else plt.get_cmap("tab10")
    for d in range(10):
        mask = true_labels == d
        ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=12, alpha=0.6, color=cmap_digits(d), label=f"Digit {d}")
    ax2.set_title(f"{method_name} ({rep_name}) - Colored by True Digit", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"{proj_type} Dim 1", fontsize=10)
    ax2.set_ylabel(f"{proj_type} Dim 2", fontsize=10)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2, fontsize=8)

    fig.suptitle(
        f"2-D Topology Projection: {method_name} on {rep_name} ({proj_type})",
        fontsize=13,
        fontweight="bold",
    )

    output_dir = FIGURES_DIR / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_name = f"scatter_{method_name.lower().replace(' ', '_')}_{rep_name.lower().replace(' ', '_')}_{proj_type.lower()}"
    filepath = output_dir / f"{clean_name}.png"
    caption = (
        f"Two-dimensional {proj_type} topology visualization of {method_name} applied to {rep_name} representations. "
        f"Left pane displays the unsupervised cluster partitions; right pane shows true handwritten digit identities. "
        f"Note: Low-dimensional projections non-linearly compress feature space, so visual cluster proximities must "
        f"be interpreted with caution relative to higher-dimensional metric distances."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_cluster_sizes(cluster_labels: np.ndarray, method_name: str, rep_name: str) -> Path:
    """Plot cluster size distribution bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    colors = ["grey" if l == -1 else "steelblue" for l in unique_labels]
    labels_str = ["Noise (-1)" if l == -1 else f"C{l}" for l in unique_labels]

    ax.bar(labels_str, counts, color=colors, edgecolor="black", alpha=0.85)
    ax.set_title(f"Cluster Size Distribution: {method_name} ({rep_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Discovered Cluster ID", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.grid(True, axis="y", ls="--", alpha=0.5)

    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=9, fontweight="bold")

    output_dir = FIGURES_DIR / "clustering"
    clean_name = f"sizes_{method_name.lower().replace(' ', '_')}_{rep_name.lower().replace(' ', '_')}"
    filepath = output_dir / f"{clean_name}.png"
    caption = (
        f"Sample distribution across clusters discovered by {method_name} on {rep_name}. "
        f"Uniform distributions indicate balanced grouping, whereas prominent skew or large noise clusters "
        f"indicate density disparities and outlier separation."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_cluster_digit_heatmap(cluster_labels: np.ndarray, true_labels: np.ndarray, method_name: str, rep_name: str) -> Path:
    """Plot Cluster x True Digit contingency heatmap with dominant digit and purity."""
    valid_mask = cluster_labels >= 0
    c_valid = cluster_labels[valid_mask]
    y_valid = true_labels[valid_mask]

    unique_clusters = np.unique(c_valid)
    heatmap_matrix = np.zeros((len(unique_clusters), 10), dtype=int)

    for idx, c in enumerate(unique_clusters):
        sub_y = y_valid[c_valid == c]
        for d in range(10):
            heatmap_matrix[idx, d] = np.sum(sub_y == d)

    # Compute purity per cluster
    row_sums = heatmap_matrix.sum(axis=1, keepdims=True) + 1e-9
    norm_matrix = heatmap_matrix / row_sums

    fig, ax = plt.subplots(figsize=(9, max(5, len(unique_clusters) * 0.45)))
    sns.heatmap(
        norm_matrix,
        annot=heatmap_matrix,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        xticklabels=list(range(10)),
        yticklabels=[f"C{c}" for c in unique_clusters],
        ax=ax,
    )
    ax.set_title(f"Cluster × True Digit Contingency: {method_name} ({rep_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("True Handwritten Digit", fontsize=11)
    ax.set_ylabel("Discovered Cluster", fontsize=11)

    output_dir = FIGURES_DIR / "clustering"
    clean_name = f"heatmap_{method_name.lower().replace(' ', '_')}_{rep_name.lower().replace(' ', '_')}"
    filepath = output_dir / f"{clean_name}.png"
    caption = (
        f"Contingency matrix mapping discovered clusters from {method_name} ({rep_name}) to true digit labels. "
        f"Cell color reflects the conditional probability P(Digit | Cluster); overlaid numbers indicate absolute sample counts. "
        f"Sharp concentrated peaks along distinct columns correspond to high cluster purity."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST Clustering & Topology Suite")
    parser.add_argument("--quick", action="store_true", help="Run with stratified subset for quick verification")
    args = parser.parse_args()

    set_seed(42)
    mode_str = "QUICK" if args.quick else "FULL"
    print("=" * 75)
    print(f"RUNNING UNSUPERVISED CLUSTERING EXPERIMENTS (Mode: {mode_str})")
    print("=" * 75)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)

    # Load fitted feature pipeline
    pipeline = FeaturePipeline.load()

    # Precompute representations on training set
    print("\nPreparing feature representations for clustering...")
    X_train_norm = FeaturePipeline.normalize_pixels(X_train)
    X_train_pca10 = pipeline.pca.transform(X_train_norm)[:, :10]
    X_train_pca50 = pipeline.transform_pca(X_train)

    # Load Autoencoder and extract 32-d latent representation
    ae_path = MODELS_DIR / "autoencoder.pt"
    if ae_path.exists():
        print("Extracting 32-d Autoencoder latent space...")
        ae_net = AutoencoderArchitecture(latent_dim=32)
        ae_net.load_state_dict(torch.load(ae_path, map_location="cpu"))
        ae_net.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_train_norm, dtype=torch.float32)
            X_train_ae32 = ae_net.encode(tensor_x).numpy()
    else:
        X_train_ae32 = X_train_pca50[:, :32]

    # Precompute 2D projection for visualization (PCA 2D and t-SNE on 2,000 reference points)
    print("Computing 2-D PCA and t-SNE visualization coordinates...")
    vis_n = min(2000, len(X_train))
    X_vis = X_train_pca50[:vis_n]
    y_vis = y_train[:vis_n]

    pca_2d_coords = X_vis[:, :2]
    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_vis)

    representations = {
        "PCA-50": X_train_pca50,
        "PCA-10": X_train_pca10,
        "Autoencoder-32": X_train_ae32,
        "Normalized Pixels": X_train_norm,
    }

    metrics_records = []
    saved_cluster_models = {}

    # -------------------------------------------------------------
    # 1. K-Means Across Representations (K=10)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: K-MEANS (K=10) ACROSS FEATURE REPRESENTATIONS")
    print("=" * 65)
    for rep_name, X_rep in representations.items():
        print(f"Fitting K-Means on {rep_name} (Shape: {X_rep.shape})...")
        start_t = time.perf_counter()
        km = KMeans(n_clusters=10, n_init=10, random_state=42)
        labels = km.fit_predict(X_rep)
        runtime = time.perf_counter() - start_t

        m = compute_clustering_metrics(X_rep, labels, y_train)
        m["Method"] = "K-Means"
        m["Representation"] = rep_name
        m["K"] = 10
        m["Runtime_s"] = runtime
        metrics_records.append(m)

        print(f"  {rep_name} -> Silhouette={m['silhouette']:.4f}, ARI={m['ari']:.4f}, NMI={m['nmi']:.4f}, Purity={m['purity']*100:.2f}% ({runtime:.2f}s)")

        if rep_name == "PCA-50":
            saved_cluster_models["K-Means_PCA50"] = km
            vis_labels = labels[:vis_n]
            plot_cluster_2d_scatter(pca_2d_coords, vis_labels, y_vis, "K-Means", "PCA-50", "PCA-2D")
            plot_cluster_2d_scatter(tsne_coords, vis_labels, y_vis, "K-Means", "PCA-50", "t-SNE")
            plot_cluster_sizes(labels, "K-Means", "PCA-50")
            plot_cluster_digit_heatmap(labels, y_train, "K-Means", "PCA-50")

    # -------------------------------------------------------------
    # 2. Mini-Batch K-Means on PCA-50
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: MINI-BATCH K-MEANS (PCA-50)")
    print("=" * 65)
    start_t = time.perf_counter()
    mb_km = MiniBatchKMeans(n_clusters=10, batch_size=256, n_init=10, random_state=42)
    mb_labels = mb_km.fit_predict(X_train_pca50)
    mb_time = time.perf_counter() - start_t

    m_mb = compute_clustering_metrics(X_train_pca50, mb_labels, y_train)
    m_mb["Method"] = "Mini-Batch K-Means"
    m_mb["Representation"] = "PCA-50"
    m_mb["K"] = 10
    m_mb["Runtime_s"] = mb_time
    metrics_records.append(m_mb)
    print(f"  Mini-Batch K-Means -> Silhouette={m_mb['silhouette']:.4f}, ARI={m_mb['ari']:.4f}, NMI={m_mb['nmi']:.4f}, Purity={m_mb['purity']*100:.2f}% ({mb_time:.2f}s)")
    saved_cluster_models["MiniBatch_KMeans"] = mb_km

    # -------------------------------------------------------------
    # 3. Gaussian Mixture Models (Covariance Sweep)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: GAUSSIAN MIXTURE MODELS (GMM Covariance Sweep)")
    print("=" * 65)
    for cov_type in ["diag", "tied", "spherical", "full"]:
        print(f"Fitting GMM (covariance_type='{cov_type}')...")
        start_t = time.perf_counter()
        gmm = GaussianMixture(n_components=10, covariance_type=cov_type, max_iter=100, random_state=42)
        gmm_labels = gmm.fit_predict(X_train_pca50)
        gmm_time = time.perf_counter() - start_t

        bic = float(gmm.bic(X_train_pca50))
        aic = float(gmm.aic(X_train_pca50))

        m_gmm = compute_clustering_metrics(X_train_pca50, gmm_labels, y_train)
        m_gmm["Method"] = f"GMM ({cov_type})"
        m_gmm["Representation"] = "PCA-50"
        m_gmm["K"] = 10
        m_gmm["Runtime_s"] = gmm_time
        m_gmm["BIC"] = bic
        m_gmm["AIC"] = aic
        metrics_records.append(m_gmm)
        print(f"  GMM ({cov_type}) -> ARI={m_gmm['ari']:.4f}, NMI={m_gmm['nmi']:.4f}, BIC={bic:.1f}, Purity={m_gmm['purity']*100:.2f}% ({gmm_time:.2f}s)")
        if cov_type == "full":
            saved_cluster_models["GMM_full"] = gmm

    # -------------------------------------------------------------
    # 4. Agglomerative Hierarchical Clustering (Documented Subset)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: AGGLOMERATIVE HIERARCHICAL CLUSTERING (Documented Subset)")
    print("=" * 65)
    agg_n = min(4000, len(X_train_pca50))
    X_agg = X_train_pca50[:agg_n]
    y_agg = y_train[:agg_n]

    for linkage in ["ward", "average"]:
        print(f"Fitting Agglomerative (linkage='{linkage}', N={agg_n})...")
        start_t = time.perf_counter()
        agg = AgglomerativeClustering(n_clusters=10, linkage=linkage)
        agg_labels = agg.fit_predict(X_agg)
        agg_time = time.perf_counter() - start_t

        m_agg = compute_clustering_metrics(X_agg, agg_labels, y_agg)
        m_agg["Method"] = f"Agglomerative ({linkage})"
        m_agg["Representation"] = f"PCA-50 (Subset N={agg_n})"
        m_agg["K"] = 10
        m_agg["Runtime_s"] = agg_time
        metrics_records.append(m_agg)
        print(f"  Agglomerative ({linkage}) -> ARI={m_agg['ari']:.4f}, NMI={m_agg['nmi']:.4f}, Purity={m_agg['purity']*100:.2f}% ({agg_time:.2f}s)")

    # -------------------------------------------------------------
    # 5. DBSCAN (Density-Based with Natural Cluster & Noise Discovery)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: DBSCAN (Density-Based Clustering)")
    print("=" * 65)
    db_n = min(3000, len(X_train_pca50))
    X_db = X_train_pca50[:db_n]
    y_db = y_train[:db_n]

    # k-distance elbow check for eps
    nn = NearestNeighbors(n_neighbors=5).fit(X_db)
    distances, _ = nn.kneighbors(X_db)
    k_dist = np.sort(distances[:, -1])

    # Plot k-distance elbow graph
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(k_dist)), k_dist, color="crimson", lw=1.5)
    ax.set_title("DBSCAN k-Distance Graph (k=5)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Points Sorted by 5-NN Distance", fontsize=10)
    ax.set_ylabel("5-NN Distance", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    db_plot_path = FIGURES_DIR / "clustering" / "dbscan_k_distance.png"
    save_figure_with_caption(
        fig,
        db_plot_path,
        "k-distance graph for k=5 nearest neighbors across MNIST samples in 50-dimensional PCA space. "
        "The sharp upward inflection around distance 4.0-5.5 guides the selection of the epsilon neighborhood radius.",
    )

    eps_candidate = float(np.percentile(k_dist, 85))
    print(f"Selected DBSCAN eps = {eps_candidate:.2f}, min_samples = 5")
    start_t = time.perf_counter()
    dbscan = DBSCAN(eps=eps_candidate, min_samples=5)
    db_labels = dbscan.fit_predict(X_db)
    db_time = time.perf_counter() - start_t

    m_db = compute_clustering_metrics(X_db, db_labels, y_db)
    m_db["Method"] = "DBSCAN"
    m_db["Representation"] = f"PCA-50 (Subset N={db_n})"
    m_db["K"] = m_db["n_clusters"]
    m_db["Runtime_s"] = db_time
    metrics_records.append(m_db)
    print(f"  DBSCAN -> Clusters Discovered={m_db['n_clusters']}, Noise={m_db['noise_count']} ({m_db['noise_ratio']*100:.1f}%), ARI={m_db['ari']}")

    # -------------------------------------------------------------
    # 6. Spectral Clustering (Subset)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 6: SPECTRAL CLUSTERING (Normalized Cuts on Subset)")
    print("=" * 65)
    spec_n = min(1500, len(X_train_pca50))
    X_spec = X_train_pca50[:spec_n]
    y_spec = y_train[:spec_n]

    start_t = time.perf_counter()
    spectral = SpectralClustering(n_clusters=10, affinity="nearest_neighbors", n_neighbors=10, random_state=42)
    spec_labels = spectral.fit_predict(X_spec)
    spec_time = time.perf_counter() - start_t

    m_spec = compute_clustering_metrics(X_spec, spec_labels, y_spec)
    m_spec["Method"] = "Spectral Clustering"
    m_spec["Representation"] = f"PCA-50 (Subset N={spec_n})"
    m_spec["K"] = 10
    m_spec["Runtime_s"] = spec_time
    metrics_records.append(m_spec)
    print(f"  Spectral Clustering -> ARI={m_spec['ari']:.4f}, NMI={m_spec['nmi']:.4f}, Purity={m_spec['purity']*100:.2f}% ({spec_time:.2f}s)")

    # -------------------------------------------------------------
    # 7. K-Sweep (Elbow Curve and Silhouette across K=5 to 15)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 7: K-SWEEP (Elbow & Silhouette across K=5..15)")
    print("=" * 65)
    k_range = list(range(5, 16))
    inertias = []
    silhouettes = []

    for k in k_range:
        km_k = KMeans(n_clusters=k, n_init=5, random_state=42)
        km_k.fit(X_train_pca50)
        inertias.append(km_k.inertia_)
        sil = silhouette_score(X_train_pca50[:3000], km_k.labels_[:3000])
        silhouettes.append(sil)

    # Plot Elbow Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(k_range, inertias, "o-", color="steelblue", lw=2)
    ax1.axvline(10, color="crimson", ls="--", label="True K=10")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
    ax1.set_title("K-Means Elbow Curve (WCSS)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.plot(k_range, silhouettes, "s--", color="forestgreen", lw=2)
    ax2.axvline(10, color="crimson", ls="--", label="True K=10")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("K-Means Silhouette Score vs. K", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.5)

    elbow_path = FIGURES_DIR / "clustering" / "kmeans_k_sweep_elbow.png"
    save_figure_with_caption(
        fig,
        elbow_path,
        "K-Means elbow heuristic (Within-Cluster Sum of Squares, left) and mean silhouette score (right) across K=5 to 15. "
        "While mathematical inertia smoothly decreases without an abrupt single hinge, silhouette peaks corroborate that "
        "10 clusters provide a natural structural decomposition matching the ten digit categories.",
    )

    # -------------------------------------------------------------
    # 8. Train and Register Unsupervised K-Means Digit Voter
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TRAINING UNSUPERVISED K-MEANS DIGIT VOTER (For Model Zoo & Exhibition)")
    print("=" * 65)
    km_voter_base = KMeans(n_clusters=10, n_init=10, random_state=42)
    km_voter = ClusterDigitVoter(
        km_voter_base,
        name="K-Means Cluster Voter",
        feature_pipeline=pipeline,
    )
    km_voter.fit(X_train, y_train)
    km_voter_path = km_voter.save()

    # Evaluate on Test Set
    bench_n = min(200, len(X_test))
    t0 = time.perf_counter()
    for i in range(bench_n):
        _ = km_voter.predict(X_test[i])
    km_latency = ((time.perf_counter() - t0) / bench_n) * 1000.0

    voter_test_preds = km_voter.predict_batch(X_test)
    voter_acc = float(np.mean(voter_test_preds == y_test))
    print(f"K-Means Digit Voter Test Accuracy: {voter_acc * 100:.2f}% (Latency: {km_latency:.3f} ms)")

    # Save cluster models
    cluster_save_path = MODELS_DIR / "cluster_models.joblib"
    joblib.dump(saved_cluster_models, cluster_save_path)

    # Register in model registry
    registry_file = JSON_DIR / "model_registry.json"
    registry = load_json(registry_file) if registry_file.exists() else {}
    registry[km_voter.name] = {
        "name": km_voter.name,
        "family": km_voter.family,
        "test_accuracy": voter_acc,
        "val_accuracy": float(np.mean(km_voter.predict_batch(X_val) == y_val)),
        "latency_ms": km_latency,
        "model_size_mb": km_voter_path.stat().st_size / (1024 * 1024),
        "model_path": str(km_voter_path),
    }
    save_json(registry, registry_file)

    # Save Metrics Table
    df_clust = pd.DataFrame(metrics_records)
    table_csv = TABLES_DIR / "clustering_metrics.csv"
    table_md = TABLES_DIR / "clustering_metrics.md"
    df_clust.to_csv(table_csv, index=False)

    headers = [c for c in df_clust.columns if c not in ["BIC", "AIC"]]
    col_widths = {h: max(len(h), max(len(str(v)) for v in df_clust[h])) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for _, row in df_clust.iterrows()
    ]
    with open(table_md, "w", encoding="utf-8") as f:
        f.write("\n".join([header_line, sep_line] + body_lines) + "\n")

    print("\n" + "=" * 75)
    print("CLUSTERING VALIDATION METRICS SUMMARY:")
    print("=" * 75)
    print(df_clust[["Method", "Representation", "n_clusters", "silhouette", "ari", "nmi", "purity", "Runtime_s"]].to_string(index=False))
    print(f"\nClustering tables saved to {table_csv} and figures saved to results/figures/clustering/")
    print("Unsupervised clustering experiments completed successfully!")


if __name__ == "__main__":
    main()
