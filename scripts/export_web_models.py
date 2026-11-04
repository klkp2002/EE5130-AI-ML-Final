"""Export the trained ensemble for web inference without a PyTorch dependency.

Run locally after training; do not run this export during the Vercel build.
All 22 models and the fitted stacking classifier are retained. The export is
written only after neural and ensemble probability comparisons pass.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import sklearn
import torch
from sklearn.exceptions import InconsistentVersionWarning
import warnings

from src.inference_nn import NumpyNeuralClassifier
from src.utils import MODELS_DIR, JSON_DIR


def export(source, output, cache):
    # Never rewrite persistence metadata using a different sklearn version.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        original = joblib.load(source / "ensemble_model.joblib")
        pipeline = joblib.load(source / "feature_pipeline.joblib")
    ensemble = copy.copy(original)
    ensemble.models = original.models.copy()
    torch.set_num_threads(1)
    samples = json.loads(cache.read_text(encoding="utf-8"))
    X = np.asarray([sample["pixels"] for sample in samples], dtype=np.float32) / 255
    rng = np.random.default_rng(42)
    neural_inputs = np.concatenate([X, rng.random((32, 784), dtype=np.float32), np.zeros((1, 784), dtype=np.float32), np.ones((1, 784), dtype=np.float32)])
    errors = {}
    for name, model in original.models.items():
        if not hasattr(model, "net"):
            continue
        if type(model.net).__name__ not in {"CNNArchitecture", "MLPArchitecture"}:
            raise ValueError(f"Unsupported network architecture: {type(model.net).__name__}")
        weights = {key: value.detach().cpu().numpy().copy() for key, value in model.net.state_dict().items()}
        eps = {name: module.eps for name, module in model.net.named_modules() if isinstance(module, torch.nn.BatchNorm2d)}
        converted = NumpyNeuralClassifier(name, weights, model.is_cnn, eps)
        converted.test_accuracy = model.test_accuracy
        converted.val_accuracy = model.val_accuracy
        expected = model.predict_proba(neural_inputs)
        actual = converted.predict_proba(neural_inputs)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        errors[name] = float(np.max(np.abs(actual - expected)))
        ensemble.models[name] = converted
        print(f"Validated {name}: max probability difference {errors[name]:.3g}", flush=True)
    for method in ["stacking", "soft", "weighted", "hard"]:
        if method == "hard":
            np.testing.assert_array_equal(ensemble.predict_batch(X[:20], method=method), original.predict_batch(X[:20], method=method))
        else:
            np.testing.assert_allclose(ensemble.predict_proba(X[:20], method=method), original.predict_proba(X[:20], method=method), atol=1e-5, rtol=1e-4)
    # The old web server loaded standalone checkpoints for individual choices.
    # Verify those agree before sharing the ensemble's copies in the web bundle.
    standalone_names = set()
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        for path in source.glob("*.joblib"):
            if path.stem in {"feature_pipeline", "cluster_models", "ensemble_model"}:
                continue
            standalone = joblib.load(path)
            if standalone.name not in ensemble.models:
                raise ValueError(f"Standalone model missing from ensemble: {standalone.name}")
            expected = standalone.predict_proba(X[:20])
            actual = ensemble.models[standalone.name].predict_proba(X[:20])
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
            standalone_names.add(standalone.name)
    if standalone_names != set(ensemble.models):
        raise ValueError("Standalone checkpoints do not cover all ensemble models.")
    bundle = {"format_version": 1, "ensemble": ensemble, "feature_pipeline": pipeline}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(output)
    metadata = {"format_version": 1, "model_count": len(ensemble.models), "sklearn_version": sklearn.__version__, "neural_validation_samples": len(neural_inputs), "neural_max_probability_differences": errors, "standalone_models_validated": len(standalone_names), "ensemble_validation_samples": 20, "ensemble_methods": ["stacking", "soft", "weighted", "hard"], "bundle_bytes": output.stat().st_size}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MODELS_DIR)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "web" / "inference_bundle.joblib")
    parser.add_argument("--cache", type=Path, default=JSON_DIR / "benchmark_cache.json")
    args = parser.parse_args()
    export(args.source, args.output, args.cache)
