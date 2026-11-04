"""
FastAPI Server for EE5130 MNIST Mini Project Exhibition Web Application.

Runs locally at http://127.0.0.1:8000
Offline capability: Zero CDN dependencies.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.features import FeaturePipeline
from src.utils import MODELS_DIR, JSON_DIR, load_json
from webapp.api.routes import router as api_router
import webapp.api.routes as routes_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "webapp" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "webapp" / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models and feature pipelines at server launch."""
    print("=" * 60)
    print("WARMING UP EE5130 MNIST APPLICATION SERVER...")
    print("=" * 60)

    # 1. Load Feature Pipeline
    pipeline_file = MODELS_DIR / "feature_pipeline.joblib"
    if pipeline_file.exists():
        routes_module.FEATURE_PIPELINE = FeaturePipeline.load(pipeline_file)
        print("  - Loaded FeaturePipeline (k=50)")

    # 2. Discover and Load all Serialized Models
    for model_path in MODELS_DIR.glob("*.joblib"):
        if model_path.stem in ["feature_pipeline", "cluster_models"]:
            continue
        try:
            m = joblib.load(model_path)
            if hasattr(m, "predict_proba"):
                if model_path.stem == "ensemble_model":
                    routes_module.LOADED_MODELS["ENSEMBLE – FINAL ANSWER"] = m
                else:
                    routes_module.LOADED_MODELS[m.name] = m
                print(f"  - Loaded model: {m.name} ({m.family})")
        except Exception as e:
            print(f"  Warning loading {model_path.name}: {e}")

    # 3. Load Caches
    cache_file = JSON_DIR / "cluster_explorer_cache.json"
    if cache_file.exists():
        routes_module.CLUSTER_CACHE = load_json(cache_file)
        print("  - Preloaded Cluster Explorer Cache")

    bench_file = JSON_DIR / "benchmark_cache.json"
    if bench_file.exists():
        routes_module.BENCHMARK_CACHE = load_json(bench_file)
        print("  - Preloaded Benchmark Test Cache (100 samples)")

    # 4. Warm-up Inference execution
    dummy_input = np.zeros((1, 784), dtype=np.float32)
    for name, m in routes_module.LOADED_MODELS.items():
        try:
            _ = m.predict(dummy_input)
        except Exception:
            pass
    print("  - Executed warm-up inference across all models.")
    print("Server ready at http://127.0.0.1:8000\n")
    yield


app = FastAPI(
    title="EE5130 MNIST Classifier & Topology Explorer",
    description="Interactive Multi-Model Supervised Classification and Unsupervised Topology Profiler",
    lifespan=lifespan,
)

# Mount static files (CSS, JS, local vendor assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register API routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main Exhibition Live Demo page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/clusters", response_class=HTMLResponse)
async def clusters_page(request: Request):
    """External Cluster Spread Explorer page."""
    return templates.TemplateResponse(request=request, name="clusters.html")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=False)
