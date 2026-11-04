"""
Supervised Classification Experiment Suite for EE5130 MNIST Project.

Fits all classical models on training set, tunes/evaluates on validation set,
evaluates ONCE on test set, generates publication figures and tables.
Supports --quick and --full execution modes.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

# Set MPLCONFIGDIR to avoid permission warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers import build_model_zoo, MNISTBaseClassifier
from src.data import load_mnist
from src.evaluation import evaluate_classifier, find_most_confused_pairs
from src.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_misclassified_grid,
    plot_accuracy_vs_latency,
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


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST Supervised Classification Suite")
    parser.add_argument("--quick", action="store_true", help="Run with stratified subset for rapid verification")
    args = parser.parse_args()

    set_seed(42)
    mode_str = "QUICK" if args.quick else "FULL"
    print("=" * 75)
    print(f"RUNNING SUPERVISED CLASSIFICATION EXPERIMENTS (Mode: {mode_str})")
    print("=" * 75)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)
    print(f"Data ready: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} in {t_load['elapsed_sec']:.2f}s")

    model_zoo = build_model_zoo(seed=42)
    print(f"Instantiated {len(model_zoo)} models across Classical and PCA/SVD Pipeline families.")

    results_list: List[Dict[str, Any]] = []
    registry: Dict[str, Any] = {}

    for name, model in model_zoo.items():
        print("-" * 65)
        print(f"Training: {name} (Family: {model.family})")

        # 1. Fit on training set only
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train
        print(f"  Trained in {train_time:.2f}s")

        # 2. Validation check (for parameter selection / calibration verification)
        val_preds = model.predict_batch(X_val)
        val_acc = float(np.mean(val_preds == y_val))
        model.val_accuracy = val_acc
        print(f"  Validation Accuracy: {val_acc * 100:.2f}%")

        # 3. Benchmark inference latency on 200 test samples
        benchmark_n = min(200, len(X_test))
        start_bench = time.perf_counter()
        for i in range(benchmark_n):
            _ = model.predict(X_test[i])
        avg_latency_ms = ((time.perf_counter() - start_bench) / benchmark_n) * 1000.0
        print(f"  Average Single-Sample Latency: {avg_latency_ms:.3f} ms")

        # 4. FINAL EVALUATION ONCE ON TEST SET
        start_test = time.perf_counter()
        test_preds = model.predict_batch(X_test)
        test_probas = model.predict_proba(X_test)
        test_eval_time = time.perf_counter() - start_test
        test_acc = float(np.mean(test_preds == y_test))
        model.test_accuracy = test_acc
        print(f"  TEST ACCURACY: {test_acc * 100:.2f}%")

        # 5. Save model to disk
        model_path = model.save()
        model_size_bytes = model_path.stat().st_size

        # 6. Comprehensive metrics calculation
        eval_metrics = evaluate_classifier(
            y_true=y_test,
            y_pred=test_preds,
            y_proba=test_probas,
            inference_time_ms=avg_latency_ms,
            model_name=name,
            family=model.family,
            model_size_bytes=model_size_bytes,
        )
        eval_metrics["train_time_sec"] = float(train_time)
        eval_metrics["val_accuracy"] = float(val_acc)
        results_list.append(eval_metrics)

        # 7. Generate diagnostic figures
        cm = np.array(eval_metrics["confusion_matrix_counts"])
        plot_confusion_matrix(cm, model_name=name, normalized=False)
        plot_confusion_matrix(np.array(eval_metrics["confusion_matrix_norm"]), model_name=name, normalized=True)
        plot_roc_curves(y_true=y_test, y_proba=test_probas, model_name=name)
        plot_misclassified_grid(X_test=X_test, y_true=y_test, y_pred=test_preds, model_name=name, n_images=12)

        # 8. Record in registry for Web App
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

    # Summary Table Construction
    summary_rows = []
    for r in results_list:
        summary_rows.append({
            "Model": r["model_name"],
            "Family": r["family"],
            "Test Accuracy (%)": f"{r['accuracy'] * 100:.2f}%",
            "Val Accuracy (%)": f"{r['val_accuracy'] * 100:.2f}%",
            "Precision (Macro)": f"{r['precision_macro']:.4f}",
            "Recall (Macro)": f"{r['recall_macro']:.4f}",
            "F1-Score (Macro)": f"{r['f1_macro']:.4f}",
            "F1-Score (Weighted)": f"{r['f1_weighted']:.4f}",
            "ROC-AUC (Macro)": f"{r['roc_auc_macro']:.4f}" if r["roc_auc_macro"] is not None else "N/A",
            "Train Time (s)": f"{r['train_time_sec']:.2f}",
            "Latency (ms/sample)": f"{r['inference_time_ms_per_sample']:.3f}",
            "Size (MB)": f"{r['model_size_mb']:.2f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    table_csv = TABLES_DIR / "classification_summary.csv"
    table_md = TABLES_DIR / "classification_summary.md"
    df_summary.to_csv(table_csv, index=False)
    
    # Write markdown table directly without external tabulate dependency
    headers = list(df_summary.columns)
    col_widths = {h: max(len(h), max(len(str(v)) for v in df_summary[h])) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for _, row in df_summary.iterrows()
    ]
    with open(table_md, "w", encoding="utf-8") as f:
        f.write("\n".join([header_line, sep_line] + body_lines) + "\n")

    print("\n" + "=" * 75)
    print("CLASSIFICATION SUMMARY TABLE:")
    print("=" * 75)
    print(df_summary.to_string(index=False))

    # Accuracy vs Latency Scatter
    plot_accuracy_vs_latency(results_list)

    # Save registry to json
    registry_file = JSON_DIR / "model_registry.json"
    save_json(registry, registry_file)
    print(f"\nModel registry updated and saved to {registry_file}")
    print("Supervised classification experiments completed successfully!")


if __name__ == "__main__":
    main()
