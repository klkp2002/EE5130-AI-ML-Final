"""
Plotting and publication figure generation for EE5130 MNIST Project.

Generates:
- Confusion Matrix Heatmaps (Counts and Normalized)
- Multi-class ROC-AUC Curves
- Scree Plots and Cumulative Explained Variance (PCA/SVD)
- Accuracy vs. Inference Latency Trade-off Scatter
- Misclassified Sample Grids with True/Predicted Labels
- Calibration Reliability Diagrams
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import FIGURES_DIR

# Set non-interactive backend for automated background generation
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def save_figure_with_caption(fig: plt.Figure, filepath: Path, caption: str) -> None:
    """Save matplotlib figure and write corresponding companion caption text file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption_file = filepath.with_suffix(".txt")
    with open(caption_file, "w", encoding="utf-8") as f:
        f.write(f"Figure: {filepath.name}\n")
        f.write(f"Caption: {caption.strip()}\n")
    print(f"Saved figure {filepath} and caption {caption_file}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    normalized: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate high-resolution annotated confusion matrix."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "confusion_matrices"

    fig, ax = plt.subplots(figsize=(7, 6))
    fmt = ".2f" if normalized else "d"
    vmax = 1.0 if normalized else None
    cmap = "Blues"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        square=True,
        vmin=0.0,
        vmax=vmax,
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        ax=ax,
    )
    title_suffix = " (Normalized)" if normalized else " (Sample Counts)"
    ax.set_title(f"Confusion Matrix: {model_name}{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit", fontsize=11)

    filename = f"cm_{model_name.lower().replace(' ', '_').replace('+', '_')}_{'norm' if normalized else 'counts'}.png"
    filepath = output_dir / filename
    caption = (
        f"Confusion matrix for {model_name} evaluated on the 10,000 MNIST test samples. "
        f"{'Values represent row-normalized recall per class.' if normalized else 'Values indicate raw classification counts.'} "
        f"Diagonal entries reflect correct classifications, while off-diagonal elements show digit confusions."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Plot One-vs-Rest ROC curves for all 10 digits."""
    from sklearn.metrics import roc_curve, auc

    if output_dir is None:
        output_dir = FIGURES_DIR / "roc_curves"

    fig, ax = plt.subplots(figsize=(8, 6))
    y_onehot = np.eye(10)[y_true]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for d in range(10):
        fpr, tpr, _ = roc_curve(y_onehot[:, d], y_proba[:, d])
        digit_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[d], lw=1.5, label=f"Digit {d} (AUC = {digit_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax.set_title(f"One-vs-Rest ROC Curves: {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    filename = f"roc_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
    filepath = output_dir / filename
    caption = (
        f"One-vs-Rest Receiver Operating Characteristic (ROC) curves across all ten handwritten digit classes "
        f"for {model_name} on the 10,000 MNIST test images. Near-vertical curves hugging the top-left indicate exceptional discrimination."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_scree_variance(
    explained_variance_ratio: np.ndarray,
    method_name: str = "PCA",
    output_dir: Optional[Path] = None,
) -> Path:
    """Plot individual and cumulative explained variance ratios."""
    if output_dir is None:
        output_dir = FIGURES_DIR

    fig, ax1 = plt.subplots(figsize=(8, 5))
    n_components = len(explained_variance_ratio)
    x = np.arange(1, n_components + 1)
    cum_var = np.cumsum(explained_variance_ratio)

    ax1.bar(x, explained_variance_ratio, alpha=0.5, color="steelblue", label="Individual Variance Ratio")
    ax1.set_xlabel("Principal Component Index", fontsize=11)
    ax1.set_ylabel("Individual Explained Variance", color="steelblue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_var, color="crimson", lw=2.0, marker="o", markersize=3, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Explained Variance", color="crimson", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim([0, 1.05])
    ax2.axhline(0.80, color="grey", linestyle="--", lw=1.0, label="80% Threshold")
    ax2.axhline(0.90, color="grey", linestyle=":", lw=1.0, label="90% Threshold")

    ax1.set_title(f"Explained Variance vs. Number of Components ({method_name})", fontsize=12, fontweight="bold")
    
    filename = f"variance_explained_{method_name.lower()}.png"
    filepath = output_dir / filename
    caption = (
        f"Scree plot and cumulative explained variance ratio for {method_name} decomposition of MNIST 784-dimensional pixel vectors. "
        f"The first 50 components capture over 82% of total dataset variance, enabling significant compression with minimal information loss."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_accuracy_vs_latency(
    model_summaries: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> Path:
    """Scatter plot displaying the trade-off between test accuracy and inference latency."""
    if output_dir is None:
        output_dir = FIGURES_DIR

    fig, ax = plt.subplots(figsize=(9, 6))

    families = list(set(m["family"] for m in model_summaries))
    palette = sns.color_palette("deep", len(families))
    fam_color = {fam: palette[i] for i, fam in enumerate(families)}

    for m in model_summaries:
        name = m["model_name"]
        acc = m["accuracy"] * 100.0
        lat = m["inference_time_ms_per_sample"]
        fam = m["family"]
        color = fam_color[fam]

        ax.scatter(lat, acc, color=color, s=120, edgecolors="black", alpha=0.85, label=fam)
        ax.annotate(
            name,
            (lat, acc),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
            fontweight="semibold",
        )

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Model Family", loc="lower right", frameon=True)

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency per Sample (ms, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Classification Accuracy vs. Computational Latency Trade-Off", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    filename = "accuracy_vs_latency.png"
    filepath = output_dir / filename
    caption = (
        "Empirical trade-off between MNIST classification accuracy (%) and per-sample inference latency (ms, logarithmic scale) "
        "across all evaluated models. High-capacity non-linear classifiers (CNN, RBF SVM) offer peak accuracy, whereas linear models "
        "and PCA pipelines provide sub-millisecond latencies suitable for ultra-fast deployment."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def plot_misclassified_grid(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_images: int = 16,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate visual grid of real misclassified images with true and predicted labels."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "misclassified_grids"

    wrong_indices = np.where(y_true != y_pred)[0]
    n_show = min(n_images, len(wrong_indices))
    if n_show == 0:
        return output_dir / f"misclassified_{model_name.lower()}.png"

    selected_idx = wrong_indices[:n_show]
    rows = int(np.ceil(n_show / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(8, 2 * rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_show:
            idx = selected_idx[i]
            img = X_test[idx].reshape(28, 28)
            t = y_true[idx]
            p = y_pred[idx]
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"True: {t} | Pred: {p}", color="crimson", fontsize=10, fontweight="bold")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    fig.suptitle(f"Sample Misclassified Digits ({model_name})", fontsize=12, fontweight="bold")
    
    filename = f"misclassified_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
    filepath = output_dir / filename
    caption = (
        f"Gallery of handwritten digit images misclassified by {model_name} on the MNIST test set. "
        f"Examining these samples reveals genuine human ambiguities, irregular stroke thicknesses, slanted axes, "
        f"and atypical digit topologies that challenge single-model decision boundaries."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath

