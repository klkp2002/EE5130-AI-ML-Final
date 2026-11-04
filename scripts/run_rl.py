"""
Reinforcement Learning Training & Evaluation Suite for EE5130 MNIST Project.

Simulates Contextual Bandit MNIST episodes for:
- Q-Learning
- SARSA
- TD(0)
- Monte Carlo
Plots learning curves across multiple random seeds, evaluates test accuracy,
and saves results table and trained model checkpoints.
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
from typing import Tuple

matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.features import FeaturePipeline
from src.plots import save_figure_with_caption
from src.rl_agents import (
    ContextualBanditMNISTEnv,
    QLearningAgent,
    SARSAAgent,
    TD0Agent,
    MonteCarloAgent,
    BaseRLAgent,
)
from src.utils import TABLES_DIR, FIGURES_DIR, JSON_DIR, MODELS_DIR, save_json, load_json, timer


def train_single_agent(
    agent: BaseRLAgent,
    env: ContextualBanditMNISTEnv,
    n_episodes: int = 10000,
    seed: int = 42,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a single RL agent on the bandit environment and log running metrics."""
    rng = np.random.default_rng(seed)
    decay_rate = (eps_end / eps_start) ** (1.0 / n_episodes)
    epsilon = eps_start

    rewards = np.zeros(n_episodes, dtype=np.float32)
    accuracies = np.zeros(n_episodes, dtype=np.float32)
    window = 200

    for ep in range(n_episodes):
        s = env.reset(rng)
        a = agent.select_action(s, epsilon, rng)
        s_next, r, done, info = env.step(a)

        agent.update(s, a, r, s_next, done)
        rewards[ep] = r
        accuracies[ep] = 1.0 if info["correct"] else 0.0

        epsilon = max(eps_end, epsilon * decay_rate)

    # Compute smoothed running metrics
    smooth_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_acc = np.convolve(accuracies, np.ones(window) / window, mode="valid")
    agent.is_fitted = True
    return smooth_rewards, smooth_acc


def main():
    parser = argparse.ArgumentParser(description="EE5130 MNIST Reinforcement Learning Suite")
    parser.add_argument("--quick", action="store_true", help="Run with fewer episodes for quick verification")
    args = parser.parse_args()

    mode_str = "QUICK" if args.quick else "FULL"
    n_episodes = 3000 if args.quick else 15000
    seeds = [42, 123, 999] if not args.quick else [42, 123]

    print("=" * 75)
    print(f"RUNNING REINFORCEMENT LEARNING EXPERIMENTS (Mode: {mode_str}, Episodes={n_episodes})")
    print("=" * 75)

    with timer("Data Loading") as t_load:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(quick=args.quick)

    # Load fitted feature pipeline for 50-d PCA states
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    if pipeline_path.exists():
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline(pca_components=50, svd_components=50).fit(X_train)
        pipeline.save()

    env = ContextualBanditMNISTEnv(X_train, y_train, feature_pipeline=pipeline)

    agent_classes = [
        ("Q-Learning", QLearningAgent),
        ("SARSA", SARSAAgent),
        ("TD(0)", TD0Agent),
        ("Monte Carlo", MonteCarloAgent),
    ]

    all_curves: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"rewards": [], "accs": []} for name, _ in agent_classes
    }
    final_results = []
    trained_agents = {}

    for name, cls_agent in agent_classes:
        print(f"\n--- Training {name} across {len(seeds)} seeds ---")
        seed_test_accs = []
        seed_train_times = []

        for s in seeds:
            agent = cls_agent(state_dim=50, lr=0.02)
            agent.feature_pipeline = pipeline

            start_t = time.perf_counter()
            r_curve, a_curve = train_single_agent(agent, env, n_episodes=n_episodes, seed=s)
            elapsed = time.perf_counter() - start_t

            all_curves[name]["rewards"].append(r_curve)
            all_curves[name]["accs"].append(a_curve)

            # Test evaluation
            test_preds = agent.predict_batch(X_test)
            acc = float(np.mean(test_preds == y_test))
            seed_test_accs.append(acc)
            seed_train_times.append(elapsed)
            trained_agents[name] = agent

        mean_acc = float(np.mean(seed_test_accs))
        std_acc = float(np.std(seed_test_accs))
        mean_time = float(np.mean(seed_train_times))
        print(f"  {name} Test Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}% (Avg Time: {mean_time:.2f}s)")

        final_results.append({
            "Algorithm": name,
            "Episodes": n_episodes,
            "Seeds": len(seeds),
            "Mean Test Accuracy (%)": f"{mean_acc * 100:.2f}%",
            "Std Dev (%)": f"{std_acc * 100:.2f}%",
            "Final Mean Reward": f"{np.mean(all_curves[name]['rewards'][-1][-50:]):.3f}",
            "Avg Train Time (s)": f"{mean_time:.2f}",
        })

        # Save model
        agent_path = trained_agents[name].save()

    # Save summary table
    df_rl = pd.DataFrame(final_results)
    table_csv = TABLES_DIR / "rl_performance.csv"
    table_md = TABLES_DIR / "rl_performance.md"
    df_rl.to_csv(table_csv, index=False)
    
    headers = list(df_rl.columns)
    col_widths = {h: max(len(h), max(len(str(v)) for v in df_rl[h])) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for _, row in df_rl.iterrows()
    ]
    with open(table_md, "w", encoding="utf-8") as f:
        f.write("\n".join([header_line, sep_line] + body_lines) + "\n")

    print("\n" + "=" * 70)
    print("REINFORCEMENT LEARNING PERFORMANCE SUMMARY:")
    print("=" * 70)
    print(df_rl.to_string(index=False))

    # Generate Learning Curves Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"Q-Learning": "crimson", "SARSA": "darkorange", "TD(0)": "forestgreen", "Monte Carlo": "royalblue"}

    for name, data in all_curves.items():
        arr_r = np.array(data["rewards"])
        arr_a = np.array(data["accs"]) * 100.0
        x = np.arange(len(arr_r[0]))

        mean_r = np.mean(arr_r, axis=0)
        std_r = np.std(arr_r, axis=0)
        ax1.plot(x, mean_r, color=colors[name], lw=1.8, label=name)
        ax1.fill_between(x, mean_r - std_r, mean_r + std_r, color=colors[name], alpha=0.15)

        mean_a = np.mean(arr_a, axis=0)
        std_a = np.std(arr_a, axis=0)
        ax2.plot(x, mean_a, color=colors[name], lw=1.8, label=name)
        ax2.fill_between(x, mean_a - std_a, mean_a + std_a, color=colors[name], alpha=0.15)

    ax1.set_xlabel("Episode Window", fontsize=11)
    ax1.set_ylabel("Rolling Average Reward", fontsize=11)
    ax1.set_title("RL Agent Reward Learning Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.set_xlabel("Episode Window", fontsize=11)
    ax2.set_ylabel("Rolling Accuracy (%)", fontsize=11)
    ax2.set_title("RL Agent Classification Accuracy Curves", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, ls="--", alpha=0.5)

    rl_plot_path = FIGURES_DIR / "learning_curves" / "rl_learning_curves.png"
    save_figure_with_caption(
        fig,
        rl_plot_path,
        "Learning curves (mean ± standard deviation across independent random seeds) for contextual bandit RL agents "
        "(Q-Learning, SARSA, TD(0), and Monte Carlo) trained on MNIST 50-dimensional PCA state representations. "
        "Because episodes terminate after a single step (T=1), the temporal bootstrapping term gamma * Q(s', a') vanishes, "
        "causing the mathematical update rules for Q-Learning, SARSA, and Monte Carlo to converge to equivalent gradient updates.",
    )

    # Register best RL agent in model registry
    registry_file = JSON_DIR / "model_registry.json"
    registry = load_json(registry_file) if registry_file.exists() else {}
    best_agent_name = "Q-Learning"
    best_agent = trained_agents[best_agent_name]
    bench_n = min(200, len(X_test))
    t0 = time.perf_counter()
    for i in range(bench_n):
        _ = best_agent.predict(X_test[i])
    rl_latency = ((time.perf_counter() - t0) / bench_n) * 1000.0

    registry[best_agent.name] = {
        "name": best_agent.name,
        "family": best_agent.family,
        "test_accuracy": float(np.mean(best_agent.predict_batch(X_test) == y_test)),
        "val_accuracy": float(np.mean(best_agent.predict_batch(X_val) == y_val)),
        "latency_ms": rl_latency,
        "model_size_mb": (MODELS_DIR / f"{best_agent.name.lower().replace(' ', '_').replace('+', '_')}.joblib").stat().st_size / (1024 * 1024),
        "model_path": str(MODELS_DIR / f"{best_agent.name.lower().replace(' ', '_').replace('+', '_')}.joblib"),
    }
    save_json(registry, registry_file)
    print("\nReinforcement learning experiments completed successfully!")


if __name__ == "__main__":
    main()

