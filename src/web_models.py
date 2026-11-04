"""Load the compact web bundle, while supporting original local checkpoints."""

from functools import lru_cache

import joblib

from src.features import FeaturePipeline
from src.utils import MODELS_DIR

INFERENCE_BUNDLE_PATH = MODELS_DIR / "web" / "inference_bundle.joblib"


@lru_cache(maxsize=1)
def load_inference_bundle():
    bundle = joblib.load(INFERENCE_BUNDLE_PATH)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported web bundle format; rerun scripts/export_web_models.py.")
    return bundle


def load_feature_pipeline():
    if INFERENCE_BUNDLE_PATH.exists():
        return load_inference_bundle()["feature_pipeline"]
    return FeaturePipeline.load()
