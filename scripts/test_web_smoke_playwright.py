"""
Playwright Automated End-to-End Browser Smoke Test and Screenshot Capture.

Automates:
1. Launching FastAPI server locally on http://127.0.0.1:8000
2. Navigating to Main Exhibition Demo page
3. Loading a random test sample, evaluating inference across all models, and running 10-sample benchmark
4. Capturing high-resolution exhibition screenshot: results/figures/ui_main_demo.png
5. Navigating to /clusters Cluster Spread Explorer
6. Testing interactive Plotly projections and capturing: results/figures/ui_clusters_page.png
"""

import sys
import threading
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import requests
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from webapp.app import app

BASE_URL = "http://127.0.0.1:8000"


def run_smoke_test():
    print("=" * 70)
    print("STARTING PLAYWRIGHT AUTOMATED BROWSER TEST & SCREENSHOT CAPTURE")
    print("=" * 70)

    # 1. Start Server in background
    print("[1/5] Starting FastAPI exhibition server on http://127.0.0.1:8000...")
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    t0 = time.time()
    ready = False
    while time.time() - t0 < 30.0:
        try:
            r = requests.get(f"{BASE_URL}/", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not ready:
        print("[ERROR] Server failed to start within 30 seconds.")
        sys.exit(1)
    print("  - Server is healthy and responding.")

    with sync_playwright() as p:
        # 2. Launch Chromium Browser
        print("[2/5] Launching headless Chromium browser (1440x950)...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 950})
        page = context.new_page()

        # 3. Main Exhibition Page
        print("[3/5] Testing Main Exhibition Demo page...")
        page.goto(f"{BASE_URL}/")
        page.wait_for_selector("#drawCanvas")
        page.wait_for_selector(".model-option")
        time.sleep(1.0)

        # Trigger Random Test Sample
        print("  - Clicking 'RANDOM TEST SAMPLE'...")
        page.click("#randomSampleBtn")
        page.wait_for_function("document.getElementById('statusPredDigit').textContent !== '-'", timeout=10000)
        time.sleep(1.0)

        pred_text = page.inner_text("#statusPredDigit")
        outcome_text = page.inner_text("#statusOutcome")
        print(f"  - Inference complete: Prediction = {pred_text}, Outcome = {outcome_text}")

        # Trigger 10-Sample Benchmark Modal
        print("  - Clicking 'BENCHMARK 10'...")
        page.click("#benchmarkBtn")
        page.wait_for_selector(".bench-card", timeout=15000)
        time.sleep(1.5)

        # Save Main Exhibition Screenshot
        main_screenshot_path = FIGURES_DIR / "ui_main_demo.png"
        page.screenshot(path=str(main_screenshot_path))
        print(f"  - Saved screenshot: {main_screenshot_path}")

        # Write Caption
        with open(FIGURES_DIR / "ui_main_demo.txt", "w", encoding="utf-8") as f:
            f.write(
                "Figure UI-1: EE5130 MNIST Mini Project Exhibition Interface. "
                "Demonstrates interactive drawing canvas (280x280), dual preprocessing inspection "
                "(bounding-box crop and centered 28x28 tensor), calibrated probability bar chart, "
                "all-model breakdown comparison table, real-time unsupervised PCA cluster topology profiler, "
                "and 10-sample random test set benchmark modal.\n"
            )

        # Close Modal
        page.click("#closeModalBtn")
        time.sleep(0.5)

        # 4. Cluster Spread Explorer Page
        print("[4/5] Testing /clusters Cluster Spread Explorer...")
        page.goto(f"{BASE_URL}/clusters")
        page.wait_for_selector("#scatterContainer")
        page.wait_for_selector("#spreadPlotContainer")
        time.sleep(2.0)

        # Switch projection to Dual View (PCA & t-SNE)
        print("  - Selecting Dual View (Side-by-Side PCA & t-SNE)...")
        page.select_option("#clusterProjectionSelect", "Side-by-Side")
        time.sleep(1.5)

        # Switch method to GMM
        print("  - Selecting Gaussian Mixture Model (GMM)...")
        page.select_option("#clusterMethodSelect", "GMM")
        time.sleep(2.0)

        # Save Clusters Explorer Screenshot
        clusters_screenshot_path = FIGURES_DIR / "ui_clusters_page.png"
        page.screenshot(path=str(clusters_screenshot_path))
        print(f"  - Saved screenshot: {clusters_screenshot_path}")

        # Write Caption
        with open(FIGURES_DIR / "ui_clusters_page.txt", "w", encoding="utf-8") as f:
            f.write(
                "Figure UI-2: EE5130 Cluster Spread Explorer Interface. "
                "Displays interactive dual 2D manifold scatters (PCA-2D alongside t-SNE-2D) with cluster centroids, "
                "within-cluster distance-to-centroid spread boxplots (measuring cluster tightness and dispersion), "
                "cluster point distribution bar charts, cluster vs. true digit contingency heatmap, "
                "and representative centroid exemplar thumbnails.\n"
            )

        browser.close()

    # 5. Terminate Server
    print("[5/5] Shutting down exhibition server...")
    server.should_exit = True
    print("=" * 70)
    print("PLAYWRIGHT SMOKE TEST & SCREENSHOT GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()

