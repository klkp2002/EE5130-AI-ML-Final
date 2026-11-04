"""
Probability Calibration and Reliability Analysis for EE5130 MNIST Project.

Features:
- Temperature Scaling (optimizing temperature T on validation logits via NLL loss)
- Expected Calibration Error (ECE) calculation
- Maximum Calibration Error (MCE)
- Publication Reliability Diagrams (Confidence vs. Empirical Accuracy)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.plots import save_figure_with_caption
from src.utils import FIGURES_DIR


def compute_ece_and_reliability(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE), MCE, and binned accuracy/confidence stats."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)
            bin_stats.append({
                "lower": float(bin_lower),
                "upper": float(bin_upper),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "sample_count": int(np.sum(in_bin)),
            })
        else:
            bin_stats.append({
                "lower": float(bin_lower),
                "upper": float(bin_upper),
                "accuracy": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_stats": bin_stats,
    }


class TemperatureScaler:
    """Optimizes scalar temperature parameter T on validation data logits."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, val_logits: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        N = len(y_val)
        y_val_arr = np.asarray(y_val, dtype=int)

        def nll_loss(t_param):
            T = t_param[0]
            scaled = val_logits / max(T, 1e-4)
            # Log-sum-exp
            max_s = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_s.squeeze() + np.log(np.sum(np.exp(scaled - max_s), axis=1))
            correct_logits = scaled[np.arange(N), y_val_arr]
            return float(np.mean(log_sum_exp - correct_logits))

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        print(f"Optimized Temperature Scaling: T = {self.temperature:.4f}")
        return self

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate reliability diagram comparing model confidence with empirical accuracy."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = compute_ece_and_reliability(y_true, y_proba, n_bins=n_bins)
    ece = calib["ece"]
    bin_stats = calib["bin_stats"]

    fig, ax = plt.subplots(figsize=(6, 5))
    centers = [(b["lower"] + b["upper"]) / 2 for b in bin_stats]
    accs = [b["accuracy"] for b in bin_stats]
    widths = [b["upper"] - b["lower"] for b in bin_stats]

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")

    # Bar plot of empirical accuracy
    ax.bar(centers, accs, width=widths, alpha=0.6, color="royalblue", edgecolor="black", label="Outputs")

    # Error gaps
    confs = [b["confidence"] for b in bin_stats]
    for c, a, w in zip(centers, accs, widths):
        if c > 0 and a > 0:
            ax.bar(c, abs(a - c), bottom=min(a, c), width=w, color="crimson", alpha=0.3, hatch="//")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title(f"Reliability Diagram: {model_name} (ECE = {ece*100:.2f}%)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, ls="--", alpha=0.5)

    filename = f"reliability_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
    filepath = output_dir / filename
    caption = (
        f"Reliability diagram and Expected Calibration Error (ECE) for {model_name}. "
        f"Bars matching the diagonal indicate that confidence accurately mirrors true empirical probability. "
        f"Shaded red gaps highlight miscalibration."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath

