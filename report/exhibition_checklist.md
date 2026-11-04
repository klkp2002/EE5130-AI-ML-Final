# EE5130 Mini Project: Exhibition Defense & Live Demonstration Checklist

**Project:** Comparative Study of Supervised Classification and Unsupervised Clustering Techniques Using the MNIST Dataset  
**Target:** Mini Exhibition Live Demonstration & Evaluator Defense  
**URL:** `http://127.0.0.1:8000` (Main Demo) and `http://127.0.0.1:8000/clusters` (Cluster Explorer)

---

## 1. Quick Startup Guide (Before Evaluators Arrive)

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "C:\Users\Victus\Desktop\AI & ML Project\mnist_project"
   ```
2. Activate virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Launch the FastAPI Exhibition Server:
   ```powershell
   $env:PYTHONPATH = "."
   python webapp/app.py
   ```
4. Verify server output:
   ```
   ============================================================
   WARMING UP EE5130 MNIST APPLICATION SERVER...
   ============================================================
   - Loaded FeaturePipeline (k=50)
   - Loaded model: CNN (Deep Learning)
   - Loaded model: Ensemble (Weighted)
   ...
   Server ready at http://127.0.0.1:8000
   ```
5. Open your browser:
   - Primary Window: `http://127.0.0.1:8000/` (Press **F11** for fullscreen exhibition dark mode).
   - Secondary Window / Tab (or second monitor): `http://127.0.0.1:8000/clusters` (Cluster Spread Explorer).

---

## 2. 5-Minute Live Demonstration Script

### Step 1: Real-Time Drawing & Strict MNIST Preprocessing
- **Action:** Draw a digit (e.g., '3' or '8') on the 280×280 black canvas using your mouse or touch screen. Click **PREDICT DIGIT**.
- **Talking Point:**  
  *"Notice our two-stage preprocessing preview on the center top panel: first, an automatic bounding-box crop isolates the drawn stroke; second, aspect-ratio preserving downsampling fits it inside a 20×20 box, which is centered by center-of-mass inside a standard 28×28 tensor, identical to LeCun's original MNIST preprocessing."*

### Step 2: Calibrated Probability Distribution & Latency
- **Action:** Point to the Output Probability bar chart and the live latency readout.
- **Talking Point:**  
  *"The probability vector reflects Temperature-Scaled softmax calibration. The predicted bar is color-coded green if correct. Notice our inference latency is under 5 ms for individual models and ~400 ms when querying all 22 models simultaneously."*

### Step 3: Model Zoo Comparison Table
- **Action:** Scroll down the All-Model Comparison Table in the center panel.
- **Talking Point:**  
  *"Our table displays predictions from all models simultaneously: CNN, MLP, PCA+SVM, Random Forests, HistGradientBoosting, Linear Regression, and even Contextual Bandit RL agents. We highlight model agreement and divergence in real time."*

### Step 4: 10-Sample Random Test Set Benchmark
- **Action:** Click the purple **BENCHMARK 10** button in the left sidebar.
- **Talking Point:**  
  *"This pops up an interactive 10-sample random test set challenge. The system pulls 10 unseen images from mnist_test.csv, runs full inference, and calculates live empirical accuracy and latency per sample. (We include an explicit academic disclaimer that 10 samples is an exhibition demo, while our statistical evaluations are conducted on the full 10,000 test set)."*

### Step 5: Dual-Monitor Live Sync with /clusters
- **Action:** Switch to the `/clusters` tab (or show side-by-side on two screens). Draw a digit in the main demo.
- **Talking Point:**  
  *"Watch what happens in our Cluster Spread Explorer: via a local BroadcastChannel and localStorage event bus, our drawn digit is projected into PCA 2-D space in real time as a large glowing gold star marker! We can see exactly which unsupervised cluster it falls into, its Euclidean distance to the centroid, and whether it lands near clean cluster cores or diffuse decision boundaries."*

---

## 3. Anticipated Evaluator Questions & High-Scoring Defense Answers

### Q1: "Why did you use Linear Regression for classification? Isn't it a regression model?"
> **Answer:**  
> *"Linear regression can be adapted for multi-class classification via one-hot target encoding $\mathbf{Y} \in \{0, 1\}^{N \times K}$ (Ridge Regression). The model solves for optimal weights $\mathbf{W} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}$ in closed form. The prediction is $\hat{y} = \arg\max_k \mathbf{x}^T \mathbf{W}_k$. It achieved **84.50%** accuracy on test data in just 0.22 seconds of training time, serving as a clean linear baseline."*

### Q2: "Why do your Q-Learning, SARSA, TD(0), and Monte Carlo RL agents have identical accuracies?"
> **Answer:**  
> *"Because MNIST classification in a contextual bandit setting has an episode length of exactly one step ($T=1$). In one-step terminal MDPs, the next state $s'$ is absorbing, so the discounted future value $\gamma \max_{a'} Q(s', a') \equiv 0$ in Q-learning and $\gamma Q(s', a') \equiv 0$ in SARSA. Both updates collapse to $\Delta Q = \alpha [r - Q(s, a)]$, which mathematically equals the Monte Carlo return update with $G_t = r$. Our empirical experiments verified this mathematical proof, yielding identical **76.65% $\pm 1.45\%$** test accuracy across seeds."*

### Q3: "Why does the 'Most Confident Model per Sample' have worse accuracy (77.95%) than the Ensemble (96.30%)?"
> **Answer:**  
> *"This is one of our most important empirical findings: **high confidence does not equal correctness**. Uncalibrated models like Naive Bayes and unregularized Decision Trees output probabilities of 1.0 (100% confidence) even when making wrong predictions due to independence violations and leaf overfitting. When you pick the 'most confident model', you disproportionately select overconfident weak models. Our Stacking Meta-Learner solves this by learning cross-validated trust weights, achieving **96.30%** accuracy with an ultra-low Expected Calibration Error of **0.74%**."*

### Q4: "Why does PCA + SVM beat raw pixel SVM in both accuracy and speed?"
> **Answer:**  
> *"Raw 784-dimensional pixels contain high-frequency noise, stroke thickness variations, and empty border margins. PCA ($k=50$) acts as an optimal orthogonal noise filter, capturing 82.5% of variance while discarding uncorrelated noise. This increased test accuracy from **95.25% to 95.85%**, reduced training time by **82.4%** (1.86s vs 10.57s), and cut memory footprint from **33.28 MB to 2.77 MB**."*

### Q5: "Why did unsupervised clustering achieve only ~58-66% purity? Why did it fail on digits 4 and 9?"
> **Answer:**  
> *"Unsupervised clustering groups data purely by geometric proximity in feature space, without knowing semantic labels. An open '4' and a curved '9' share nearly identical pixel overlap and directional gradients, so K-Means naturally groups them into cluster #2 (purity 29.3%). Similarly, digits '3', '5', and '8' share curved lower lobes. Only digits with unique global topologies—such as '0' (large hollow ring) and '1' (vertical line)—achieve >85% purity unsupervised."*

### Q6: "How did you guarantee zero data leakage?"
> **Answer:**  
> *"We enforced strict data isolation: the official 60,000 training set was split into 54,000 train and 6,000 validation samples (stratified, verified disjoint by automated unit test `test_train_val_split_disjoint`). All scalers, PCA projections, SVD transforms, HOG parameters, autoencoders, temperature scaling, and stacking meta-models were fitted strictly on train/val data. The official 10,000 test set (`mnist_test.csv`) was evaluated strictly **once** at the end of each experiment."*

---

## 4. Emergency Backup Checklist
- If WiFi or internet is down: **Everything is 100% offline.** Plotly is locally bundled (`webapp/static/vendor/plotly.min.js`), models are locally serialized in `models/`, and data is in `data/`. No internet required!
- If the port is in use: Run `python webapp/app.py` with an alternative port (e.g. `port=8080`).
- To rerun all quick sanity tests:
  ```powershell
  $env:PYTHONPATH = "mnist_project"
  python scripts/run_all.py --quick
  pytest -v tests
  ```

