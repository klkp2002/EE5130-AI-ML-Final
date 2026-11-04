"""
Integration and End-to-End HTTP Tests for EE5130 FastAPI Server and Exhibition Endpoints.
"""

import threading
import time
import pytest
import requests
import uvicorn
from webapp.app import app

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module", autouse=True)
def live_server():
    """Start uvicorn server in a background thread for live HTTP testing."""
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll until server is warmed up and responding
    t0 = time.time()
    started = False
    while time.time() - t0 < 30.0:
        try:
            r = requests.get(f"{BASE_URL}/", timeout=1.0)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.5)

    if not started:
        raise RuntimeError("FastAPI server failed to warm up within 30 seconds.")

    yield server

    server.should_exit = True
    thread.join(timeout=3.0)


def test_index_page_returns_html():
    """Verify that root URL returns the Exhibition Interface HTML."""
    res = requests.get(f"{BASE_URL}/")
    assert res.status_code == 200
    assert "EE5130 MNIST MINI PROJECT" in res.text
    assert "drawCanvas" in res.text


def test_clusters_page_returns_html():
    """Verify that /clusters URL returns the Cluster Spread Explorer HTML."""
    res = requests.get(f"{BASE_URL}/clusters")
    assert res.status_code == 200
    assert "EE5130 CLUSTER SPREAD EXPLORER" in res.text
    assert "scatterContainer" in res.text


def test_api_models_endpoint():
    """Verify /api/models returns model zoo with ENSEMBLE included."""
    res = requests.get(f"{BASE_URL}/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) > 0
    assert "ENSEMBLE – FINAL ANSWER" in data["models"]



def test_api_random_sample_endpoint():
    """Verify /api/random_sample delivers a 784-d pixel list with valid label."""
    res = requests.get(f"{BASE_URL}/api/random_sample")
    assert res.status_code == 200
    sample = res.json()
    assert "pixels" in sample
    assert len(sample["pixels"]) == 784
    assert "true_label" in sample
    assert 0 <= sample["true_label"] <= 9


def test_api_predict_endpoint_with_sample():
    """Verify /api/predict correctly evaluates a test vector and returns calibrated outputs."""
    sample_res = requests.get(f"{BASE_URL}/api/random_sample")
    sample = sample_res.json()

    payload = {
        "pixels": sample["pixels"],
        "true_label": sample["true_label"],
        "model_name": "ENSEMBLE – FINAL ANSWER",
        "combination_method": "stacking",
    }
    res = requests.post(f"{BASE_URL}/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "digit" in data
    assert 0 <= data["digit"] <= 9
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 10
    assert sum(data["probabilities"]) == pytest.approx(1.0, abs=0.02)
    assert "top3" in data
    assert len(data["top3"]) == 3
    assert "inference_latency_ms" in data
    assert data["true_label"] == sample["true_label"]
    assert "is_correct" in data


def test_api_benchmark_endpoint():
    """Verify /api/benchmark returns 10 processed test cards."""
    payload = {
        "model_name": "ENSEMBLE – FINAL ANSWER",
        "combination_method": "stacking",
    }
    res = requests.post(f"{BASE_URL}/api/benchmark", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["total_count"] == 10
    assert len(data["benchmark_samples"]) == 10
    assert "accuracy_percent" in data
    assert "average_latency_ms" in data


def test_api_clusters_endpoints():
    """Verify cluster methods and spread endpoints."""
    # Methods list
    res = requests.get(f"{BASE_URL}/api/clusters/methods")
    assert res.status_code == 200
    data = res.json()
    assert "K-Means" in data["methods"]
    assert "projections" in data

    # Spread data for K-Means
    res_spread = requests.get(f"{BASE_URL}/api/clusters/K-Means/spread")
    assert res_spread.status_code == 200
    spread = res_spread.json()
    assert spread["method_name"] == "K-Means"
    assert "cluster_data" in spread
    assert "metrics" in spread["cluster_data"]
    assert "projections" in spread
