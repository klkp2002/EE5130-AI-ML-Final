"""
Unit Tests for Preprocessing and Feature Extraction Pipelines.
"""

import base64
import io
import numpy as np
import pytest
from PIL import Image
from webapp.api.routes import preprocess_canvas_image
from src.features import FeaturePipeline, extract_hog_features


def test_preprocess_canvas_image_output_shape():
    """Verify that preprocess_canvas_image produces exact 1x784 normalized vector and base64 crops."""
    # Create synthetic 280x280 grayscale image with a digit in center
    img = Image.new("L", (280, 280), color=0)
    # Draw stroke
    arr = np.array(img)
    arr[100:180, 120:140] = 255
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    norm_vector, canvas_uint8, crop_b64, tensor_b64 = preprocess_canvas_image(b64)

    assert isinstance(norm_vector, np.ndarray)
    assert norm_vector.shape == (1, 784)
    assert norm_vector.dtype == np.float32
    assert norm_vector.min() >= 0.0
    assert norm_vector.max() <= 1.0
    assert crop_b64.startswith("data:image/png;base64,")
    assert tensor_b64.startswith("data:image/png;base64,")


def test_preprocess_blank_canvas_graceful():
    """Verify that an empty/blank canvas returns zero vector without crashing."""
    img = Image.new("L", (280, 280), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    norm_vector, canvas_uint8, crop_b64, tensor_b64 = preprocess_canvas_image(b64)
    assert norm_vector.shape == (1, 784)
    assert np.all(norm_vector == 0.0)


def test_hog_feature_extraction_shape():
    """Verify HOG feature extraction vector dimensions."""
    sample = np.random.uniform(0.0, 1.0, size=(5, 784)).astype(np.float32)
    hog_feats = extract_hog_features(sample, cell_size=7, n_bins=8)

    assert hog_feats.ndim == 2
    assert hog_feats.shape[0] == 5
    assert hog_feats.shape[1] > 0
    assert not np.isnan(hog_feats).any()


def test_feature_pipeline_fit_and_transform():
    """Verify FeaturePipeline fit and transform shapes."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(100, 784)).astype(np.float32)

    pipeline = FeaturePipeline(pca_components=20, svd_components=20)
    pipeline.fit(X)


    pca_out = pipeline.transform_pca(X)
    svd_out = pipeline.transform_svd(X)
    std_out = pipeline.transform_standardized(X)

    assert pca_out.shape == (100, 20)
    assert svd_out.shape == (100, 20)
    assert std_out.shape == (100, 784)

    non_zero_cols = np.std(X, axis=0) > 1e-5
    np.testing.assert_allclose(np.mean(std_out[:, non_zero_cols], axis=0), 0.0, atol=1e-5)
