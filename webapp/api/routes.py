"""
FastAPI Routes and Endpoints for EE5130 MNIST Web Application.

Endpoints:
- GET /api/models: List of registered models with test accuracies and families
- POST /api/predict: Predict digit for drawing or test sample
- POST /api/predict_all: Full ensemble & comparison table
- GET /api/random_sample: Retrieve random test image with true label
- POST /api/benchmark: Benchmark 10 random samples on active model
- GET /api/clusters/methods: List clustering methods and metadata
- GET /api/clusters/{method}/spread: Precomputed 2D scatter and cluster profiles
- POST /api/clusters/assign: Assign cluster to user sample and return 2D coordinates
"""

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from scipy.ndimage import center_of_mass
from typing import Tuple

from src.features import FeaturePipeline
from src.utils import JSON_DIR, MODELS_DIR, load_json, save_json

router = APIRouter(prefix="/api")

# In-memory model cache
LOADED_MODELS: Dict[str, Any] = {}
FEATURE_PIPELINE: Optional[FeaturePipeline] = None
CLUSTER_MODELS: Dict[str, Any] = {}
CLUSTER_CACHE: Optional[Dict[str, Any]] = None
BENCHMARK_CACHE: Optional[List[Dict[str, Any]]] = None


def preprocess_canvas_image(image_b64_or_bytes: Any) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Standard MNIST Preprocessing Pipeline:
    1. Decode PNG image to grayscale
    2. Extract tight bounding box of digit pixels
    3. Resize maintaining aspect ratio to fit inside 20x20
    4. Center into 28x28 black canvas using center-of-mass of pixel intensities
    5. Return (28x28 normalized float array, 28x28 uint8 array, bbox base64, tensor base64)
    """
    if isinstance(image_b64_or_bytes, str):
        if "," in image_b64_or_bytes:
            image_b64_or_bytes = image_b64_or_bytes.split(",")[1]
        img_bytes = base64.b64decode(image_b64_or_bytes)
    else:
        img_bytes = image_b64_or_bytes

    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_arr = np.array(img, dtype=np.uint8)

    # Invert if image background is white (MNIST requires white digit on black background)
    if np.mean(img_arr) > 127:
        img_arr = 255 - img_arr

    # Threshold noise
    img_arr[img_arr < 35] = 0

    # Find bounding box
    rows = np.any(img_arr > 0, axis=1)
    cols = np.any(img_arr > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty canvas
        blank = np.zeros((28, 28), dtype=np.uint8)
        blank_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
        return np.zeros((1, 784), dtype=np.float32), blank, blank_b64, blank_b64

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop bounding box
    crop_arr = img_arr[rmin : rmax + 1, cmin : cmax + 1]
    crop_img = Image.fromarray(crop_arr)

    # Generate crop preview base64
    buf_crop = io.BytesIO()
    crop_img.save(buf_crop, format="PNG")
    crop_b64 = "data:image/png;base64," + base64.b64encode(buf_crop.getvalue()).decode("utf-8")

    # Resize bounding box to fit inside 20x20 maintaining aspect ratio
    w, h = crop_img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized, dtype=np.float32)

    # Place in center of 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas_28[y_off : y_off + new_h, x_off : x_off + new_w] = resized_arr

    # Shift by Center of Mass (standard MNIST centering)
    cy, cx = center_of_mass(canvas_28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(round(14.0 - cy))
        shift_x = int(round(14.0 - cx))
        # Bounded roll
        canvas_28 = np.roll(canvas_28, shift_y, axis=0)
        canvas_28 = np.roll(canvas_28, shift_x, axis=1)

    canvas_uint8 = np.clip(canvas_28, 0, 255).astype(np.uint8)
    tensor_img = Image.fromarray(canvas_uint8)
    buf_tensor = io.BytesIO()
    tensor_img.save(buf_tensor, format="PNG")
    tensor_b64 = "data:image/png;base64," + base64.b64encode(buf_tensor.getvalue()).decode("utf-8")

    norm_vector = (canvas_uint8.reshape(1, 784).astype(np.float32) / 255.0)
    return norm_vector, canvas_uint8, crop_b64, tensor_b64


class PredictRequest(BaseModel):
    image_b64: Optional[str] = None
    pixels: Optional[List[int]] = None
    model_name: Optional[str] = None
    true_label: Optional[int] = None
    combination_method: Optional[str] = "stacking"
    active_models: Optional[List[str]] = None


@router.get("/models")
def get_models() -> Dict[str, Any]:
    """Return model registry with real validated test accuracies and families."""
    reg_path = JSON_DIR / "model_registry.json"
    if not reg_path.exists():
        raise HTTPException(status_code=404, detail="Model registry not found.")
    data = load_json(reg_path)
    return {"models": data, "default_active": "ENSEMBLE – FINAL ANSWER"}


@router.get("/random_sample")
def get_random_sample() -> Dict[str, Any]:
    """Retrieve random official MNIST test image with ground truth label."""
    global BENCHMARK_CACHE
    if BENCHMARK_CACHE is None:
        cache_path = JSON_DIR / "benchmark_cache.json"
        if cache_path.exists():
            BENCHMARK_CACHE = load_json(cache_path)
        else:
            raise HTTPException(status_code=404, detail="Benchmark cache not found.")

    rng = np.random.default_rng()
    sample = rng.choice(BENCHMARK_CACHE)
    return sample


@router.post("/predict")
def predict_digit(req: PredictRequest) -> Dict[str, Any]:
    """Execute prediction on drawing or sample for selected model or ensemble."""
    global FEATURE_PIPELINE, LOADED_MODELS

    # 1. Image preprocessing
    if req.pixels is not None and len(req.pixels) == 784:
        raw_arr = np.array(req.pixels, dtype=np.uint8)
        norm_vector = raw_arr.reshape(1, 784).astype(np.float32) / 255.0
        # For test samples, create direct 28x28 preview
        img = Image.fromarray(raw_arr.reshape(28, 28))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        crop_b64 = b64
        tensor_b64 = b64
    elif req.image_b64:
        norm_vector, canvas_uint8, crop_b64, tensor_b64 = preprocess_canvas_image(req.image_b64)
    else:
        raise HTTPException(status_code=400, detail="Must provide image_b64 or pixels array.")

    model_key = req.model_name or "ENSEMBLE – FINAL ANSWER"
    ensemble_model = LOADED_MODELS.get("ENSEMBLE – FINAL ANSWER")

    # 2. Inference
    start = time.perf_counter()
    if "ENSEMBLE" in model_key.upper() and ensemble_model is not None:
        res = ensemble_model.predict(
            norm_vector,
            active_models=req.active_models,
            method=req.combination_method or "stacking",
        )
        server_latency_ms = (time.perf_counter() - start) * 1000.0
    else:
        target_model = LOADED_MODELS.get(model_key)
        if target_model is None:
            # Case-insensitive or partial match fallback
            for k, m in LOADED_MODELS.items():
                if k.lower() == model_key.lower():
                    target_model = m
                    break
        if target_model is None:
            # Fallback to ensemble if model not loaded
            target_model = ensemble_model or next(iter(LOADED_MODELS.values()))

        single_start = time.perf_counter()
        single_res = target_model.predict(norm_vector)
        server_latency_ms = (time.perf_counter() - single_start) * 1000.0

        # Construct clean model breakdown without circular reference
        breakdown_items = []
        if ensemble_model is not None and hasattr(ensemble_model, "models"):
            try:
                ens_res = ensemble_model.predict(norm_vector, method=req.combination_method or "stacking")
                breakdown_items = ens_res.get("model_breakdown", [])
            except Exception:
                breakdown_items = []

        if not breakdown_items:
            breakdown_items = [{
                "method_name": single_res.get("method_name", model_key),
                "family": single_res.get("family", "Supervised"),
                "digit": int(single_res["digit"]),
                "confidence": float(single_res["confidence"]),
                "inference_time_ms": float(single_res.get("inference_time_ms", server_latency_ms)),
            }]

        res = {
            "digit": int(single_res["digit"]),
            "probabilities": [float(p) for p in single_res["probabilities"]],
            "confidence": float(single_res["confidence"]),
            "inference_time_ms": round(server_latency_ms, 2),
            "method_name": single_res.get("method_name", model_key),
            "family": single_res.get("family", "Supervised"),
            "final_answer": int(single_res["digit"]),
            "agreement_summary": f"Single model inference: {single_res.get('method_name', model_key)}",
            "model_breakdown": breakdown_items,
        }


    # 3. Top 3 likelihood breakdown
    probs = res["probabilities"]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"digit": int(d), "probability": float(probs[d]), "percent": f"{probs[d]*100:.1f}%"}
        for d in top3_idx
    ]

    # 4. Check true label if provided
    is_correct = None
    if req.true_label is not None:
        is_correct = (res["digit"] == req.true_label)

    # 5. Unsupervised PCA + K-Means Topology details
    cluster_info = None
    if FEATURE_PIPELINE is not None:
        pca_50 = FEATURE_PIPELINE.transform_pca(norm_vector)[0]
        pca_2d = pca_50[:2].tolist()
        pca_3d = pca_50[:3].tolist()

        km_model = LOADED_MODELS.get("K-Means Cluster Voter")
        if km_model is not None and km_model.centroids is not None:
            # Nearest cluster
            dists = np.linalg.norm(km_model.centroids - pca_50, axis=1)
            nearest_cluster = int(np.argmin(dists))
            centroid_dist = float(dists[nearest_cluster])
            dominant_digit = km_model.cluster_to_digit.get(nearest_cluster, nearest_cluster)
            purity = km_model.cluster_purities.get(nearest_cluster, 0.5)

            cluster_info = {
                "cluster_id": nearest_cluster,
                "total_clusters": 10,
                "dominant_digit": dominant_digit,
                "purity_percent": f"{purity * 100:.1f}%",
                "distance_to_centroid": round(centroid_dist, 3),
                "pca_2d_coords": [round(pca_2d[0], 2), round(pca_2d[1], 2)],
                "pca_3d_coords": [round(pca_3d[0], 2), round(pca_3d[1], 2), round(pca_3d[2], 2)],
            }


    return {
        "digit": res["digit"],
        "confidence": res["confidence"],
        "confidence_percent": f"{res['confidence'] * 100:.1f}%",
        "probabilities": res["probabilities"],
        "top3": top3,
        "inference_latency_ms": round(server_latency_ms, 2),
        "method_name": res.get("method_name", model_key),
        "family": res.get("family", "Supervised"),
        "true_label": req.true_label,
        "is_correct": is_correct,
        "crop_b64": crop_b64,
        "tensor_b64": tensor_b64,
        "cluster_topology": cluster_info,
        "most_confident_method": res.get("most_confident_method"),
        "best_method_overall": res.get("best_method_overall"),
        "agreement_summary": res.get("agreement_summary"),
        "model_breakdown": res.get("model_breakdown", []),
    }


@router.post("/benchmark")
def benchmark_batch(req: PredictRequest) -> Dict[str, Any]:
    """Run benchmark on 10 random test samples."""
    global BENCHMARK_CACHE
    if BENCHMARK_CACHE is None:
        cache_path = JSON_DIR / "benchmark_cache.json"
        BENCHMARK_CACHE = load_json(cache_path)

    rng = np.random.default_rng()
    selected_samples = rng.choice(BENCHMARK_CACHE, size=10, replace=False)

    cards = []
    correct_count = 0
    total_latency = 0.0

    for s in selected_samples:
        pred_req = PredictRequest(
            pixels=s["pixels"],
            model_name=req.model_name,
            true_label=s["true_label"],
            combination_method=req.combination_method,
            active_models=req.active_models,
        )
        res = predict_digit(pred_req)
        if res["is_correct"]:
            correct_count += 1
        total_latency += res["inference_latency_ms"]

        cards.append({
            "index": s["index"],
            "image_b64": s["image_b64"],
            "true_label": s["true_label"],
            "predicted_digit": res["digit"],
            "confidence_percent": res["confidence_percent"],
            "is_correct": res["is_correct"],
            "latency_ms": res["inference_latency_ms"],
        })

    accuracy = float(correct_count / 10.0)
    avg_latency = float(total_latency / 10.0)

    return {
        "benchmark_samples": cards,
        "accuracy_percent": f"{accuracy * 100:.1f}%",
        "correct_count": correct_count,
        "total_count": 10,
        "average_latency_ms": round(avg_latency, 2),
        "active_model": req.model_name or "ENSEMBLE – FINAL ANSWER",
        "disclaimer": "Note: 10 samples is an interactive mini exhibition demonstration, not a rigorous statistical evaluation.",
    }


@router.get("/clusters/methods")
def get_cluster_methods() -> Dict[str, Any]:
    """Return available clustering methods and representations."""
    global CLUSTER_CACHE
    if CLUSTER_CACHE is None:
        c_path = JSON_DIR / "cluster_explorer_cache.json"
        if c_path.exists():
            CLUSTER_CACHE = load_json(c_path)
        else:
            raise HTTPException(status_code=404, detail="Cluster cache not found.")

    return {
        "methods": list(CLUSTER_CACHE["methods"].keys()),
        "representations": ["PCA-50", "PCA-10", "Autoencoder-32", "Normalized Pixels"],
        "projections": ["PCA-3D", "t-SNE-3D", "PCA-2D", "t-SNE"],
    }



@router.get("/clusters/{method}/spread")
def get_cluster_spread(method: str) -> Dict[str, Any]:
    """Return full precomputed 2D scatter coordinates and metrics for the cluster spread page."""
    global CLUSTER_CACHE
    if CLUSTER_CACHE is None:
        c_path = JSON_DIR / "cluster_explorer_cache.json"
        CLUSTER_CACHE = load_json(c_path)

    if method not in CLUSTER_CACHE["methods"]:
        # Case insensitive search
        matched = [m for m in CLUSTER_CACHE["methods"] if m.lower() == method.lower()]
        if matched:
            method = matched[0]
        else:
            method = "K-Means"

    method_data = CLUSTER_CACHE["methods"][method]
    return {
        "method_name": method,
        "reference_points_count": CLUSTER_CACHE["reference_points_count"],
        "true_labels": CLUSTER_CACHE["true_labels"],
        "projections": CLUSTER_CACHE["projections"],
        "thumbnails": CLUSTER_CACHE["thumbnails"],
        "cluster_data": method_data,
    }


@router.post("/clusters/assign")
def assign_sample_cluster(req: PredictRequest) -> Dict[str, Any]:
    """Project query sample into PCA 2D space and determine cluster membership."""
    global FEATURE_PIPELINE, LOADED_MODELS

    if req.pixels is not None and len(req.pixels) == 784:
        norm_vector = np.array(req.pixels, dtype=np.float32).reshape(1, 784) / 255.0
    elif req.image_b64:
        norm_vector, _, _, _ = preprocess_canvas_image(req.image_b64)
    else:
        raise HTTPException(status_code=400, detail="Must provide pixels or image_b64")

    if FEATURE_PIPELINE is None:
        FEATURE_PIPELINE = FeaturePipeline.load()

    pca_50 = FEATURE_PIPELINE.transform_pca(norm_vector)[0]
    pca_2d = [float(pca_50[0]), float(pca_50[1])]
    pca_3d = [float(pca_50[0]), float(pca_50[1]), float(pca_50[2])]

    km_voter = LOADED_MODELS.get("K-Means Cluster Voter")
    if km_voter is not None and km_voter.centroids is not None:
        dists = np.linalg.norm(km_voter.centroids - pca_50, axis=1)
        cluster_id = int(np.argmin(dists))
        dist_to_center = float(dists[cluster_id])
    else:
        cluster_id = 0
        dist_to_center = 0.0

    return {
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "assigned_cluster": cluster_id,
        "distance_to_centroid": round(dist_to_center, 3),
    }


