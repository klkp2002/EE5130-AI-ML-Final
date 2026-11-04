"""
Neural Network Experiment Suite for EE5130 MNIST Project.

Trains MLP, CNN, and Autoencoder on training set, monitors validation loss/acc,
evaluates ONCE on test set, generates learning curves, confusion matrices, and updates registry.
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
import torch

matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.evaluation import evaluate_classifier
from src.nn_models import (
    MLPArchitecture,
    CNNArchitecture,
    PyTorchClassifierWrapper,
    train_autoencoder,
)
from src.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_misclassified_grid,
    save_figure_with_caption,
)
from src.utils import (
    TABLES_DIR,
    FIGURES_DIR,
    JSON_DIR,
    MODELS_DIR,
    save_json,
    load_json,
    set_seed,
    timer,
)


def plot_learning_curves(history: Dict[str, List[float]], model_name: str) -> Path:
    """Plot training and validation loss and accuracy across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", color="steelblue", lw=2, label="Train Loss")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "s--", color="crimson", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title(f"{model_name} Loss Curve", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="steelblue", lw=2, label="Train Acc")
    if history.get("val_acc"):
        ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="crimson", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title(f"{model_name} Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, ls="--", alpha=0.5)

    output_dir = FIGURES_DIR / "learning_curves"
    filename = f"learning_curves_{model_name.lower().replace(' ', '_')}.png"
    filepath = output_dir / filename
    caption = (
        f"Training and validation loss and accuracy curves for {model_name} across training epochs. "
        f"The convergence trajectories demonstrate stable optimization with early stopping preventing over-fitting."
    )
    save_figure_with_caption(fig, filepath, caption)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST Deep Learning Suite")
    parser.add_argument("--quick", action="store_true", help="Run with stratified subset for rapid verification")
    args = parser.parse_args()

    set_seed(42)
    mode_str = "QUICK" if args.quick else "FULL"
    print("=" * 75)
    print(f"RUNNING DEEP LEARNING EXPERIMENTS (Mode: {mode_str})")
    print("=" * 75)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)
    print(f"Data ready: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} in {t_load['elapsed_sec']:.2f}s")

    epochs = 6 if args.quick else 15
    batch_size = 64

    # -------------------------------------------------------------
    # 1. Train MLP (784 -> 256 -> 128 -> 10)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TRAINING MULTILAYER PERCEPTRON (MLP)")
    print("=" * 65)
    mlp_net = MLPArchitecture(input_dim=784, hidden1=256, hidden2=128, num_classes=10, dropout=0.2)
    mlp_model = PyTorchClassifierWrapper(mlp_net, name="MLP (Deep Learning)", is_cnn=False)

    start_train = time.perf_counter()
    mlp_model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=1e-3, patience=3)
    mlp_train_time = time.perf_counter() - start_train

    plot_learning_curves(mlp_model.history, "MLP")

    # -------------------------------------------------------------
    # 2. Train CNN (Conv32 -> Pool -> Conv64 -> Pool -> Dense128 -> Dense10)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TRAINING CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("=" * 65)
    cnn_net = CNNArchitecture(num_classes=10, dropout=0.3)
    cnn_model = PyTorchClassifierWrapper(cnn_net, name="CNN (Deep Learning)", is_cnn=True)

    start_train = time.perf_counter()
    cnn_model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=1e-3, patience=3)
    cnn_train_time = time.perf_counter() - start_train

    plot_learning_curves(cnn_model.history, "CNN")

    # -------------------------------------------------------------
    # 3. Train Autoencoder for Latent Representation (Part B)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TRAINING AUTOENCODER (32-d Bottleneck)")
    print("=" * 65)
    ae_net, ae_hist = train_autoencoder(X_train, X_val, latent_dim=32, epochs=epochs, batch_size=batch_size, lr=1e-3)
    ae_path = MODELS_DIR / "autoencoder.pt"
    torch.save(ae_net.state_dict(), ae_path)
    print(f"Autoencoder saved to {ae_path}")

    # Plot Autoencoder reconstruction loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(ae_hist["train_loss"]) + 1), ae_hist["train_loss"], "o-", color="purple", lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Reconstruction MSE Loss", fontsize=11)
    ax.set_title("Autoencoder Training Reconstruction Loss", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ae_plot_path = FIGURES_DIR / "learning_curves" / "autoencoder_loss.png"
    save_figure_with_caption(
        fig,
        ae_plot_path,
        "Mean Squared Error (MSE) reconstruction loss curve for the 32-dimensional convolutional/dense Autoencoder across training epochs.",
    )

    # -------------------------------------------------------------
    # 4. Evaluate NN models on Test Set
    # -------------------------------------------------------------
    nn_models = [
        (mlp_model, mlp_train_time),
        (cnn_model, cnn_train_time),
    ]

    registry_file = JSON_DIR / "model_registry.json"
    registry = load_json(registry_file) if registry_file.exists() else {}

    # Load existing classification summary if available
    summary_csv = TABLES_DIR / "classification_summary.csv"
    if summary_csv.exists():
        df_summary = pd.read_csv(summary_csv)
    else:
        df_summary = pd.DataFrame()

    new_rows = []
    for model, t_train in nn_models:
        name = model.name
        print("-" * 65)
        print(f"Evaluating {name} on Test Set...")

        # Benchmark single-sample inference latency
        benchmark_n = min(200, len(X_test))
        start_bench = time.perf_counter()
        for i in range(benchmark_n):
            _ = model.predict(X_test[i])
        avg_latency_ms = ((time.perf_counter() - start_bench) / benchmark_n) * 1000.0
        print(f"  Average Single-Sample Latency: {avg_latency_ms:.3f} ms")

        # Evaluate on Test Set
        test_preds = model.predict_batch(X_test)
        test_probas = model.predict_proba(X_test)
        test_acc = float(np.mean(test_preds == y_test))
        val_acc = model.history["val_acc"][-1] if model.history["val_acc"] else test_acc
        print(f"  TEST ACCURACY: {test_acc * 100:.2f}%")

        # Save model
        model_path = model.save()
        model_size_bytes = model_path.stat().st_size

        # Evaluation metrics
        eval_metrics = evaluate_classifier(
            y_true=y_test,
            y_pred=test_preds,
            y_proba=test_probas,
            inference_time_ms=avg_latency_ms,
            model_name=name,
            family=model.family,
            model_size_bytes=model_size_bytes,
        )
        eval_metrics["train_time_sec"] = float(t_train)
        eval_metrics["val_accuracy"] = float(val_acc)

        # Figures
        cm = np.array(eval_metrics["confusion_matrix_counts"])
        plot_confusion_matrix(cm, model_name=name, normalized=False)
        plot_confusion_matrix(np.array(eval_metrics["confusion_matrix_norm"]), model_name=name, normalized=True)
        plot_roc_curves(y_true=y_test, y_proba=test_probas, model_name=name)
        plot_misclassified_grid(X_test=X_test, y_true=y_test, y_pred=test_preds, model_name=name, n_images=12)

        # Registry
        registry[name] = {
            "name": name,
            "family": model.family,
            "test_accuracy": test_acc,
            "val_accuracy": val_acc,
            "latency_ms": avg_latency_ms,
            "model_size_mb": eval_metrics["model_size_mb"],
            "model_path": str(model_path),
            "precision_macro": eval_metrics["precision_macro"],
            "recall_macro": eval_metrics["recall_macro"],
            "f1_macro": eval_metrics["f1_macro"],
        }

        new_rows.append({
            "Model": name,
            "Family": model.family,
            "Test Accuracy (%)": f"{test_acc * 100:.2f}%",
            "Val Accuracy (%)": f"{val_acc * 100:.2f}%",
            "Precision (Macro)": f"{eval_metrics['precision_macro']:.4f}",
            "Recall (Macro)": f"{eval_metrics['recall_macro']:.4f}",
            "F1-Score (Macro)": f"{eval_metrics['f1_macro']:.4f}",
            "F1-Score (Weighted)": f"{eval_metrics['f1_weighted']:.4f}",
            "ROC-AUC (Macro)": f"{eval_metrics['roc_auc_macro']:.4f}" if eval_metrics["roc_auc_macro"] is not None else "N/A",
            "Train Time (s)": f"{t_train:.2f}",
            "Latency (ms/sample)": f"{avg_latency_ms:.3f}",
            "Size (MB)": f"{eval_metrics['model_size_mb']:.2f}",
        })

    # Update summary table
    if not df_summary.empty:
        # Filter out if model already existed
        existing_names = [r["Model"] for r in new_rows]
        df_summary = df_summary[~df_summary["Model"].isin(existing_names)]
        df_combined = pd.concat([df_summary, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_combined = pd.DataFrame(new_rows)

    df_combined.to_csv(summary_csv, index=False)
    
    # Save markdown table
    headers = list(df_combined.columns)
    col_widths = {h: max(len(h), max(len(str(v)) for v in df_combined[h])) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for _, row in df_combined.iterrows()
    ]
    with open(TABLES_DIR / "classification_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join([header_line, sep_line] + body_lines) + "\n")

    save_json(registry, registry_file)
    print(f"\nModel registry updated with Neural Network models.")
    print(f"Summary table updated at {summary_csv}")
    print("Neural network experiments completed successfully!")


if __name__ == "__main__":
    main()

