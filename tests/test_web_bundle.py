"""The deployable ensemble must work when PyTorch cannot be imported."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_web_bundle_runs_without_pytorch():
    root = Path(__file__).resolve().parent.parent
    if not (root / "models/web/inference_bundle.joblib").exists():
        pytest.skip("Run scripts/export_web_models.py first.")
    script = r'''
import asyncio
import importlib.abc
import sys
import numpy as np

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("PyTorch is deliberately unavailable during this test")

sys.meta_path.insert(0, BlockTorch())
from webapp.app import app, lifespan
from webapp.api import routes

async def verify():
    async with lifespan(app):
        assert len(routes.LOADED_MODELS) == 23
        assert routes.FEATURE_PIPELINE is not None
        ensemble = routes.LOADED_MODELS["ENSEMBLE – FINAL ANSWER"]
        assert len(ensemble.models) == 22
        sample = routes.get_random_sample()
        for name in routes.LOADED_MODELS:
            result = routes.predict_digit(routes.PredictRequest(pixels=sample["pixels"], model_name=name))
            probabilities = np.asarray(result["probabilities"])
            assert probabilities.shape == (10,) and np.isfinite(probabilities).all(), name
            assert abs(float(probabilities.sum()) - 1) < 0.02, name
            assert 0 <= result["digit"] <= 9, name
        assert "K-Means" in routes.get_cluster_methods()["methods"]
        assert "cluster_data" in routes.get_cluster_spread("K-Means")
        assert len(routes.assign_sample_cluster(routes.PredictRequest(pixels=sample["pixels"]))["pca_2d"]) == 2
        assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

asyncio.run(verify())
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VERCEL="1")
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True, timeout=120)
