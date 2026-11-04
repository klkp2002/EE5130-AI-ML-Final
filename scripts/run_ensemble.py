"""
All-Model Ensemble and Ablation Study for EE5130 MNIST Project.

Evaluates:
- Hard Voting, Soft Voting, Weighted Soft Voting, and Stacking Meta-Classifier
- Full Ablation Study across model subsets
- Tri-fold 'Most Precise Answer' metrics
- Probability calibration (ECE) and reliability diagrams
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

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import compute_ece_and_reliability, plot_reliability_diagram
from src.classifiers import MNISTBaseClassifier
from src.data import load_mnist
from src.ensemble import AllModelEnsemble
from src.evaluation import evaluate_classifier
from src.plots import plot_confusion_matrix, plot_roc_curves
from src.utils import TABLES_DIR, FIGURES_DIR, JSON_DIR, MODELS_DIR, save_json, load_json, set_seed, timer


def load_all_trained_models() -> Dict[str, MNISTBaseClassifier]:
    """Discover and load all serialized project models from models/."""
    models: Dict[str, MNISTBaseClassifier] = {}
    for joblib_file in MODELS_DIR.glob("*.joblib"):
        if joblib_file.stem in ["feature_pipeline", "cluster_models", "ensemble_model"]:
            continue
        try:
            m = joblib.load(joblib_file)
            if hasattr(m, "predict_proba") and hasattr(m, "name"):
                models[m.name] = m
        except Exception as e:
            print(f"Skipping {joblib_file.name}: {e}")
    return models


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST All-Model Ensemble & Ablation Suite")
    parser.add_argument("--quick", action="store_true", help="Run with stratified subset for rapid verification")
    args = parser.parse_args()

    set_seed(42)
    mode_str = "QUICK" if args.quick else "FULL"
    print("=" * 75)
    print(f"RUNNING ENSEMBLE COMBINATION & ABLATION EXPERIMENTS (Mode: {mode_str})")
    print("=" * 75)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)

    loaded_models = load_all_trained_models()
    print(f"Loaded {len(loaded_models)} trained models for ensemble combination:")
    for name in sorted(loaded_models.keys()):
        print(f"  - {name} ({loaded_models[name].family})")

    # 1. Validation Tuning & Weights Calculation
    print("\n--- Tuning Ensemble Weights on Validation Set (No Test Peeking) ---")
    weights = {}
    val_accs = {}
    for name, m in loaded_models.items():
        v_preds = m.predict_batch(X_val)
        v_acc = float(np.mean(v_preds == y_val))
        val_accs[name] = v_acc
        # Weight exponentially proportional to validation accuracy above random chance
        weights[name] = max(0.01, float((v_acc - 0.10) ** 3))

    best_single_name = max(val_accs, key=val_accs.get)
    print(f"Best single model on validation set: {best_single_name} ({val_accs[best_single_name]*100:.2f}%)")

    # 2. Build and Fit Ensemble (fit stacking on validation set)
    ensemble = AllModelEnsemble(loaded_models, weights=weights, combination_method="weighted")
    ensemble.best_model_overall_name = best_single_name
    ensemble.fit_stacker(X_val, y_val)

    # 3. Compare 4 Combination Strategies on Validation Set
    comb_methods = ["hard", "soft", "weighted", "stacking"]
    val_comb_results = {}
    for cm in comb_methods:
        p_val = ensemble.predict_batch(X_val, method=cm)
        acc_val = float(np.mean(p_val == y_val))
        val_comb_results[cm] = acc_val
        print(f"  Strategy '{cm}' Validation Accuracy: {acc_val * 100:.2f}%")

    best_strategy = max(val_comb_results, key=val_comb_results.get)
    print(f"\nSelected Best Combination Strategy: '{best_strategy.upper()}' (Val Acc: {val_comb_results[best_strategy]*100:.2f}%)")
    ensemble.combination_method = best_strategy

    # 4. Comprehensive Ablation Study on Test Set (EVALUATE ONCE)
    print("\n" + "=" * 65)
    print("RUNNING ABLATION STUDY ON 10,000 TEST SAMPLES")
    print("=" * 65)

    weak_members = ["Linear Regression", "Naive Bayes", "Decision Tree", "K-Means Cluster Voter", "Q-Learning Agent"]
    all_names = list(loaded_models.keys())
    trimmed_names = [m for m in all_names if m not in weak_members]
    classical_names = [m for m in all_names if loaded_models[m].family == "Supervised Classical"]
    nn_names = [m for m in all_names if loaded_models[m].family == "Deep Learning"]

    ablation_configurations = [
        ("Full Ensemble (All Models)", all_names, best_strategy),
        ("Full Ensemble (Soft Voting)", all_names, "soft"),
        ("Full Ensemble (Hard Majority)", all_names, "hard"),
        ("Trimmed Ensemble (No Weak Members)", trimmed_names, "weighted"),
        ("Classical Supervised Only", classical_names, "weighted"),
        ("Deep Learning Only (CNN + MLP)", nn_names, "weighted"),
        (f"Best Single Model ({best_single_name})", [best_single_name], "weighted"),
    ]

    ablation_rows = []
    for cfg_name, model_subset, method in ablation_configurations:
        if not model_subset:
            continue

        start_t = time.perf_counter()
        test_preds = ensemble.predict_batch(X_test, active_models=model_subset, method=method)
        test_probas = ensemble.predict_proba(X_test, active_models=model_subset, method=method)
        eval_time = time.perf_counter() - start_t

        acc = float(np.mean(test_preds == y_test))
        calib_stats = compute_ece_and_reliability(y_test, test_probas)
        ece = calib_stats["ece"]

        # Benchmarking latency
        bench_n = min(150, len(X_test))
        t0 = time.perf_counter()
        for i in range(bench_n):
            _ = ensemble.predict(X_test[i], active_models=model_subset, method=method)
        avg_lat = ((time.perf_counter() - t0) / bench_n) * 1000.0

        eval_m = evaluate_classifier(y_test, test_preds, test_probas, inference_time_ms=avg_lat, model_name=cfg_name)

        ablation_rows.append({
            "Configuration": cfg_name,
            "Members Count": len(model_subset),
            "Method": method.capitalize(),
            "Test Accuracy (%)": f"{acc * 100:.2f}%",
            "Precision (Macro)": f"{eval_m['precision_macro']:.4f}",
            "Recall (Macro)": f"{eval_m['recall_macro']:.4f}",
            "F1 (Macro)": f"{eval_m['f1_macro']:.4f}",
            "ECE (%)": f"{ece * 100:.2f}%",
            "Latency (ms)": f"{avg_lat:.2f}",
        })
        print(f"  {cfg_name.ljust(40)} -> Acc: {acc * 100:.2f}% | ECE: {ece * 100:.2f}% | Latency: {avg_lat:.2f} ms")

        if cfg_name.startswith("Full Ensemble"):
            plot_confusion_matrix(np.array(eval_m["confusion_matrix_norm"]), model_name="Ensemble Final Answer", normalized=True)
            plot_reliability_diagram(y_test, test_probas, model_name="Ensemble (Final Answer)")

    # 5. Measure "Most Confident Model Correctness" (High Confidence != Correctness)
    print("\n--- Investigating 'Most Confident Model' Correctness ---")
    most_conf_correct = 0
    test_eval_n = len(X_test)

    for i in range(test_eval_n):
        high_conf = -1.0
        high_pred = -1
        for m_name in all_names:
            prob = loaded_models[m_name].predict_proba(X_test[i:i+1])[0]
            c = float(np.max(prob))
            if c > high_conf:
                high_conf = c
                high_pred = int(np.argmax(prob))
        if high_pred == y_test[i]:
            most_conf_correct += 1

    most_conf_acc = float(most_conf_correct / test_eval_n)
    print(f"Accuracy of 'Most Confident Model per Input': {most_conf_acc * 100:.2f}%")
    print(f"Selected Ensemble 'Final Answer' Accuracy:    {ablation_rows[0]['Test Accuracy (%)']}")
    print(f"Discrepancy demonstrates that raw high confidence does not equal empirical correctness!")

    # Save Ablation Table
    df_abl = pd.DataFrame(ablation_rows)
    table_csv = TABLES_DIR / "ensemble_ablation.csv"
    table_md = TABLES_DIR / "ensemble_ablation.md"
    df_abl.to_csv(table_csv, index=False)

    headers = list(df_abl.columns)
    col_widths = {h: max(len(h), max(len(str(v)) for v in df_abl[h])) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for _, row in df_abl.iterrows()
    ]
    with open(table_md, "w", encoding="utf-8") as f:
        f.write("\n".join([header_line, sep_line] + body_lines) + "\n")

    print("\n" + "=" * 75)
    print("ENSEMBLE ABLATION TABLE:")
    print("=" * 75)
    print(df_abl.to_string(index=False))

    # Save Ensemble Model
    ens_path = MODELS_DIR / "ensemble_model.joblib"
    joblib.dump(ensemble, ens_path)
    print(f"\nEnsemble model saved to {ens_path}")

    # Register in Model Registry
    registry_file = JSON_DIR / "model_registry.json"
    registry = load_json(registry_file) if registry_file.exists() else {}
    registry["ENSEMBLE – FINAL ANSWER"] = {
        "name": f"ENSEMBLE – FINAL ANSWER (method: {best_strategy})",
        "family": "Ensemble",
        "test_accuracy": float(ablation_rows[0]["Test Accuracy (%)"].replace("%", "")) / 100.0,
        "val_accuracy": val_comb_results[best_strategy],
        "latency_ms": float(ablation_rows[0]["Latency (ms)"]),
        "model_size_mb": ens_path.stat().st_size / (1024 * 1024),
        "model_path": str(ens_path),
        "best_method_overall": best_single_name,
        "most_confident_accuracy": most_conf_acc,
    }
    save_json(registry, registry_file)
    print("Model registry updated with Ensemble Final Answer.")
    print("Ensemble and calibration experiments completed successfully!")


if __name__ == "__main__":
    main()

