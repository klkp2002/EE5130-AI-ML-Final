"""
EE5130 Mini Project – Master Pipeline Orchestrator.

Sequentially executes all phases of the project:
1. Dimensionality Reduction (PCA / SVD)
2. Classical Supervised Classifiers
3. Deep Learning (MLP, CNN, Autoencoder)
4. Reinforcement Learning (Contextual Bandit Q-learning, SARSA, TD(0), MC)
5. Unsupervised Clustering Suite & Cluster Voter
6. All-Model Calibrated Ensemble & Ablation Study
7. Web Application Data Precomputation & Offline Bundling

Usage:
  python scripts/run_all.py --quick     # Fast sanity run on stratified subsets
  python scripts/run_all.py --full      # Official run on 60,000 / 10,000 MNIST dataset
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_phase(name: str, cmd_args: list) -> float:
    print("\n" + "=" * 70)
    print(f"STARTING PHASE: {name}")
    print(f"Command: {sys.executable} {' '.join(cmd_args)}")
    print("=" * 70)

    t0 = time.time()
    env = dict(sys.modules.get("os", {}).environ if "os" in sys.modules else {})
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

    cmd = [sys.executable] + cmd_args
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)

    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Phase '{name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)

    print(f"\n[COMPLETED] Phase '{name}' finished in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for EE5130 MNIST Pipeline")
    parser.add_argument("--quick", action="store_true", default=True, help="Run on fast stratified subsets")
    parser.add_argument("--full", action="store_true", help="Run full 60,000 training evaluation")
    args = parser.parse_args()

    mode_flag = "--full" if args.full else "--quick"
    mode_name = "FULL (60k/10k)" if args.full else "QUICK (Stratified Subsets)"

    print("=" * 75)
    print(f"EE5130 MNIST MINI PROJECT: MASTER EXPERIMENT PIPELINE")
    print(f"Execution Mode: {mode_name}")
    print(f"Python Executable: {sys.executable}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 75)

    start_total = time.time()
    timing_log = []

    phases = [
        ("Phase 1: Dimensionality Reduction (PCA & SVD)", ["scripts/run_dimred.py", mode_flag]),
        ("Phase 2: Classical Supervised Classifiers", ["scripts/run_classification.py", mode_flag]),
        ("Phase 3: Deep Neural Networks (MLP, CNN, Autoencoder)", ["scripts/run_nn.py", mode_flag]),
        ("Phase 4: Reinforcement Learning Contextual Bandits", ["scripts/run_rl.py", mode_flag]),
        ("Phase 5: Unsupervised Clustering Suite & Voters", ["scripts/run_clustering.py", mode_flag]),
        ("Phase 6: Calibrated All-Model Ensemble & Ablation", ["scripts/run_ensemble.py", mode_flag]),
        ("Phase 7: Web Application Precomputation", ["scripts/precompute_web_data.py", mode_flag]),
    ]

    for phase_name, args_list in phases:
        elapsed = run_phase(phase_name, args_list)
        timing_log.append((phase_name, elapsed))

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print("ALL PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print("Phase Runtime Summary:")
    for p_name, t_elapsed in timing_log:
        print(f"  - {p_name:<55}: {t_elapsed:>6.1f}s")
    print(f"\nTotal Pipeline Wall-Clock Time: {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("Artifacts, tables, figures, and models are fully saved in results/ and models/.")
    print("=" * 75)


if __name__ == "__main__":
    main()

