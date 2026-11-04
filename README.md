# EE5130 Mini Project: MNIST Classification, Clustering & Live Exhibition Suite

> **Comparative Study of Supervised Classification and Unsupervised Clustering Techniques Using the MNIST Dataset**  
> *Course Weight:* 30% | *Platform:* Windows (PowerShell) | *Execution:* CPU-only | *Offline:* Zero CDN Dependencies

---

## 1. Project Overview
This repository contains the complete, reproducible research codebase, technical report draft, and localhost exhibition web application for the EE5130 Mini Project. 

### Core Highlights:
- **22-Model Zoo:** Full coverage of the course diagram:
  - *Classical Supervised:* Logistic Regression, Linear Regression (classifier via Ridge), Linear SVM, RBF SVM, Random Forest, k-NN ($k=3$), Decision Tree, Naive Bayes, LDA, HistGradientBoosting.
  - *Dimensionality Reduction Pipelines:* PCA+SVM, PCA+kNN, PCA+LogReg, SVD+LogReg, SVD+RF.
  - *Deep Learning:* Multilayer Perceptron (MLP), 2-layer Convolutional Neural Network (CNN with Dropout/BatchNorm), Convolutional Autoencoder (32-d bottleneck).
  - *Reinforcement Learning Contextual Bandits:* Q-Learning, SARSA, TD(0), Monte Carlo with empirical and mathematical proof of equivalence in $T=1$ terminal episodes.
  - *Unsupervised Clustering Suite:* K-Means, Mini-Batch K-Means, GMM (spherical, tied, diag, full), Agglomerative Hierarchical (Ward & Average), DBSCAN (density-based with noise detection), Spectral Clustering, and post-hoc ClusterDigitVoter.
  - *All-Model Calibrated Ensemble:* Hard voting, soft voting, weighted confidence voting, and stacking meta-classifier with Temperature Scaling (ECE = 0.74%).
- **Zero Data Leakage:** Strict 54k train / 6k val / 10k test split (`mnist_test.csv`). Scalers, PCA, SVD, autoencoders, temperature calibration, and stacking meta-models are fitted strictly on train/val.
- **Localhost Exhibition Web Application (`http://127.0.0.1:8000`):**
  - Dark-themed UI (`#121212` / `#1e1e1e`) matching exhibition guidelines.
  - 280×280 drawing canvas with automatic bounding-box crop, aspect-preserving 20×20 downscaling, and center-of-mass centering into a 28×28 normalized tensor.
  - Live prediction with calibrated probability bar chart, inference latency readout, and all-model comparison table.
  - 10-sample random test set benchmark modal.
- **Cluster Spread Explorer (`http://127.0.0.1:8000/clusters`):**
  - Offline bundled Plotly.js (`webapp/static/vendor/plotly.min.js`, 4.55 MB).
  - Interactive 2D projections (PCA-2D, t-SNE, Side-by-Side Dual View).
  - Within-cluster distance-to-centroid spread boxplots quantifying cluster compactness vs. dispersion.
  - Cluster-vs-digit contingency heatmaps and cluster size balance bars.
  - **Dual-monitor live sync:** Real-time communication via `BroadcastChannel` and `localStorage`—drawing a digit in the main demo plots that exact query sample as a glowing gold star marker in the cluster latent space.

---

## 2. Directory Structure

```
mnist_project/
├── data/
│   ├── mnist_test.csv            # Official 10,000-sample test set (copied, unmodified)
│   ├── mnist_train.csv           # 60,000-sample training set
│   └── splits.json               # Disjoint 54,000 train / 6,000 validation stratified split
├── docs/                         # Official course guidelines, diagram, UI references
├── models/                       # Serialized model checkpoints, pipelines, autoencoder (.joblib, .pt)
├── report/
│   ├── report.md                 # Full technical report draft (Sections 7.1 - 7.8)
│   └── exhibition_checklist.md   # Live demo script, walkthrough, evaluator defense Q&A
├── results/
│   ├── figures/                  # Publication-ready plots (.png) and companion captions (.txt)
│   │   ├── ui_main_demo.png      # Screenshot of live exhibition demo
│   │   ├── ui_clusters_page.png  # Screenshot of cluster spread explorer
│   │   ├── learning_curves/      # CNN, MLP, Autoencoder, RL reward learning curves
│   │   └── ...
│   ├── json/                     # Precomputed benchmark and cluster explorer caches
│   └── tables/                   # Empirical summary metrics (.csv, .md)
├── scripts/
│   ├── run_all.py                # Master orchestrator (--quick and --full modes)
│   ├── run_dimred.py             # Phase 1: PCA & SVD dimension reduction sweeps
│   ├── run_classification.py     # Phase 2: Classical classifiers & PCA pipelines
│   ├── run_nn.py                 # Phase 3: MLP, CNN, and Autoencoder
│   ├── run_rl.py                 # Phase 4: Reinforcement learning contextual bandits
│   ├── run_clustering.py         # Phase 5: Unsupervised clustering suite & voters
│   ├── run_ensemble.py           # Phase 6: All-Model Ensemble, calibration & ablation
│   ├── precompute_web_data.py    # Phase 7: Precompute 2D embeddings and thumbnails
│   └── test_web_smoke_playwright.py # Automated browser smoke test with screenshot capture
├── src/
│   ├── calibration.py            # Temperature scaling, ECE calculation, reliability plots
│   ├── classifiers.py            # Unified MNISTBaseClassifier zoo (15 classical models)
│   ├── clustering.py             # Clustering algorithms, metrics & ClusterDigitVoter
│   ├── data.py                   # Data loader, splits, canvas image preprocessing
│   ├── ensemble.py               # Hard, soft, weighted & stacking ensembles
│   ├── evaluation.py             # Macro, weighted, micro metrics & confusion matrices
│   ├── features.py               # FeaturePipeline (PCA, SVD, Scalers) & vectorized HOG
│   ├── nn_models.py              # PyTorch MLP, CNN, Autoencoder architectures
│   ├── plots.py                  # Heatmaps, ROC curves, scree plots & captions
│   ├── rl_agents.py              # Q-Learning, SARSA, TD(0), MC contextual bandits
│   └── utils.py                  # Seed management, directory paths, JSON serialization
├── tests/
│   ├── test_preprocessing.py     # Preprocessing, normalization & HOG tests
│   ├── test_no_leakage.py        # Verification of disjoint train/val/test partitions
│   ├── test_metrics.py           # Metric calculation tests (macro/micro, ECE, ARI)
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── webapp/
│   ├── app.py                    # FastAPI server application
│   ├── api/routes.py             # Inference, benchmark, and clustering endpoints
│   ├── static/                   # CSS dark theme, JS controllers, local Plotly bundle
│   └── templates/                # index.html (Main Demo) and clusters.html (Explorer)
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

## 3. Environment Setup (Windows PowerShell)

Ensure Python 3.10+ (tested on Python 3.14.2) is installed.

```powershell
# 1. Navigate to project root
cd "C:\Users\Victus\Desktop\AI & ML Project\mnist_project"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install CPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Install Playwright browser (for automated screenshot smoke tests)
playwright install chromium
```

---

## 4. Master Orchestrator: Running Experiments

The entire pipeline can be reproduced using a single command:

### Quick Mode (Fast Sanity Verification on Stratified Subsets, ~3-4 minutes)
```powershell
$env:PYTHONPATH = "mnist_project"
python scripts/run_all.py --quick
```

### Full Mode (Official 60,000 / 10,000 Dataset Run)
```powershell
$env:PYTHONPATH = "mnist_project"
python scripts/run_all.py --full
```

The orchestrator sequentially executes:
1. `run_dimred.py`: Sweeps PCA/SVD components (10, 20, 30, 50, 100), outputs explained variance tables and scree plots.
2. `run_classification.py`: Trains and evaluates all 15 classical supervised models, saves checkpoints to `models/` and summary tables to `results/tables/`.
3. `run_nn.py`: Trains MLP, CNN, and Autoencoder, saves loss curves and models.
4. `run_rl.py`: Runs multi-seed Contextual Bandit RL agents (Q-learning, SARSA, TD(0), MC) and logs learning curves.
5. `run_clustering.py`: Executes K-Means, Mini-Batch, GMM (4 covariances), Agglomerative, DBSCAN, and Spectral clustering across representations; generates elbow, silhouette, and k-distance knee plots.
6. `run_ensemble.py`: Calibrates probabilities via Temperature Scaling, constructs Stacking meta-learner, and executes ablation study.
7. `precompute_web_data.py`: Generates cached 2D t-SNE/PCA coordinates, centroids, within-cluster spread percentiles, and sample thumbnails for the webapp.

---

## 5. Launching the Exhibition Web Application

To run the interactive exhibition interface locally:

```powershell
$env:PYTHONPATH = "mnist_project"
python webapp/app.py
```

Open your browser to:
- **Main Exhibition Demo:** `http://127.0.0.1:8000` (Press **F11** for fullscreen mode).
- **Cluster Spread Explorer:** `http://127.0.0.1:8000/clusters` (Can be opened side-by-side or on a second monitor).

*Offline Notice:* All vendor dependencies (Plotly.js 4.55MB) and static assets are bundled locally in `webapp/static/vendor/`. The server runs 100% offline without internet.

---

## 6. Running Unit & Automated End-to-End Tests

### Unit Test Suite (17 Tests)
Verifies preprocessing shapes, zero data leakage in partitions, metric formulas, and API routes:
```powershell
$env:PYTHONPATH = "mnist_project"
$env:MPLCONFIGDIR = "mnist_project/.cache/matplotlib"
pytest -v tests
```
*Result:* 17 passed in ~35 seconds.

### Playwright Automated Browser Smoke Test & Screenshot Capture
Spins up the web server, draws digits, triggers predictions, runs benchmarks, navigates to the cluster explorer, and captures fresh high-res screenshots:
```powershell
$env:PYTHONPATH = "mnist_project"
python scripts/test_web_smoke_playwright.py
```
*Outputs:* `results/figures/ui_main_demo.png` and `results/figures/ui_clusters_page.png`.

---

## 7. Key Findings & Empirical Results

| Model / Ensemble | Family | Test Accuracy (%) | F1 (Macro) | ECE (%) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CNN (Deep Learning)** | Deep Learning | **97.15%** | **0.9715** | 1.13% | 1.65 ms |
| **All-Model Stacking Ensemble** | Ensemble (Stacking) | **96.30%** | **0.9630** | **0.74%** | 397.10 ms |
| **PCA + SVM (k=50)** | PCA/SVD Pipeline | 95.85% | 0.9585 | 4.15% | 9.10 ms |
| **RBF SVM (Full Pixels)** | Supervised Classical | 95.25% | 0.9524 | 4.75% | 15.41 ms |
| **PCA + k-NN (k=3)** | PCA/SVD Pipeline | 94.75% | 0.9475 | 5.25% | 3.20 ms |
| **Random Forest (n=100)** | Supervised Classical | 94.20% | 0.9419 | 5.80% | 49.24 ms |
| **HistGradientBoosting** | Supervised Classical | 94.10% | 0.9410 | 5.90% | 11.39 ms |
| **MLP (Deep Learning)** | Deep Learning | 92.65% | 0.9265 | 7.35% | 0.64 ms |
| **Logistic Regression** | Supervised Classical | 89.85% | 0.8980 | 10.15% | 0.12 ms |
| **Linear Regression (Ridge Class.)**| Supervised Classical | 84.50% | 0.8431 | 15.50% | 0.03 ms |
| **Q-Learning / SARSA / TD(0) / MC**| Contextual Bandits | 76.65% | 0.7665 | N/A | 0.05 ms |
| **Spectral Clustering (Voter)** | Unsupervised Clustering| 66.67% | 0.6667 | N/A | 0.18 ms |
| **K-Means Voter (PCA-50)** | Unsupervised Clustering| 58.14% | 0.5814 | N/A | 0.20 ms |

1. **Dimensionality Reduction Advantage:** PCA ($k=50$) reduces input dimensions by **93.6%**, improves SVM accuracy from 95.25% to 95.85%, cuts training time by 82.4%, and reduces disk footprint by 91.7%.
2. **Bandit Equivalence:** In one-step terminal MDPs ($T=1$), discounted future returns vanish ($\gamma \max Q \equiv 0$), mathematically collapsing Q-learning, SARSA, and Monte Carlo updates to identical gradient steps (verified empirically at 76.65% $\pm 1.45\%$).
3. **Confidence $\neq$ Correctness:** Selecting the "most confident model per sample" yields only **77.95% accuracy** due to overconfident weak models (Naive Bayes, Decision Trees). In contrast, Calibrated Stacking achieves **96.30% accuracy** with an ECE of **0.74%**.

---

## 8. AI Assistance Acknowledgement
In compliance with course guidelines:
- **AI Pair-Programming Assistant:** Antigravity (Google DeepMind Advanced Agentic Coding).
- **Contribution:** Code architecture design, automated test suites, vectorized HOG feature extraction, Jinja2/Plotly web templates, and technical documentation drafting.
- **Verification Protocol:** All empirical numbers, training runs, figures, tables, and unit tests were executed locally on CPU, logged in `results/`, verified via automated test suites, and independently inspected by the authors.

