# Vercel deployment

The FastAPI site uses a pre-exported inference bundle at
`models/web/inference_bundle.joblib`. It includes all 22 classifiers, the fitted
stacking classifier, and the fitted feature pipeline. The CNN and MLP use their
original float32 weights through NumPy inference. Training still uses PyTorch in
`src/nn_models.py`; the original checkpoints remain in `models/`.

## Deploy

Import this repository into Vercel using the FastAPI preset. The entry point and
Python 3.12 version are declared in `pyproject.toml` and `.python-version`.
Runtime dependencies are pinned in `uv.lock`. No custom install/build command or
Large Functions environment variable is needed for this compact bundle.

`.vercelignore` and `vercel.json` exclude training inputs, original checkpoints,
and reports from deployment. Keep `models/web/`, `results/json/`, `src/`, and
`webapp/` in the deployment. Missing exports fail clearly at startup on Vercel.

## Regenerate after training

Use a local Python environment with CPU PyTorch installed and the same
scikit-learn version as the saved checkpoints (currently 1.9.1). The exporter
refuses checkpoints with incompatible scikit-learn persistence metadata.

```powershell
python scripts/export_web_models.py
```

Commit both generated files under `models/web/` after the export passes. The
exporter checks CNN/MLP probabilities and digit predictions against PyTorch on
cached MNIST samples, random inputs, an all-black image, and an all-white image.
It also compares stacking, soft, weighted, and hard-voting ensemble outputs.
The generated JSON records the sample counts, errors, version, and bundle size.

## Validation for this export

- Both neural models matched digit predictions on 134 inputs, with maximum
  probability differences below 0.0000007.
- All four ensemble methods and all 22 standalone classifiers matched on 20 cached test samples.
- The compressed model bundle is 19,963,483 bytes (19.04 MiB).
- Locked Linux x86_64 Python 3.12 wheels plus application files total about
  249 MiB before Vercel platform runtime overhead. This is a local package-size
  measurement; the final deployment size is determined by Vercel.
- Existing API tests and a subprocess test that blocks PyTorch imports pass.
  The subprocess exercises all 22 classifiers plus the ensemble and clustering.

These checks establish local numerical and API compatibility, not acceptance
by Vercel or equivalence over every possible input.
