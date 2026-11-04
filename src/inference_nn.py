"""Inference-only versions of the trained exhibition networks, using NumPy.

Training remains in nn_models.py. Exported weights retain their float32 values;
dropout is omitted and batch normalization uses the saved evaluation statistics.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.classifiers import MNISTBaseClassifier


class NumpyNeuralClassifier(MNISTBaseClassifier):
    def __init__(self, name, weights, is_cnn=False, batch_norm_eps=None):
        super().__init__(name=name, family="Deep Learning")
        self.weights = weights
        self.is_cnn = is_cnn
        self.batch_norm_eps = batch_norm_eps or {}
        self.is_fitted = True

    def fit(self, X, y):
        raise RuntimeError("Use src.nn_models to train, then export the web bundle.")

    def _linear(self, x, prefix):
        return x @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]

    def _conv(self, x, prefix):
        weights = self.weights[prefix + ".weight"]
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = sliding_window_view(padded, (3, 3), axis=(-2, -1))
        n, c, h, w, _, _ = windows.shape
        columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, c * 9)
        output = columns @ weights.reshape(weights.shape[0], -1).T
        output += self.weights[prefix + ".bias"]
        return output.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def _batch_norm(self, x, prefix):
        def channel(key):
            return self.weights[prefix + key][None, :, None, None]

        scale = channel(".weight") / np.sqrt(
            channel(".running_var") + self.batch_norm_eps[prefix]
        )
        return (x - channel(".running_mean")) * scale + channel(".bias")

    @staticmethod
    def _pool(x):
        n, c, h, w = x.shape
        return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))

    def _logits(self, x):
        if not self.is_cnn:
            x = np.maximum(self._linear(x, "net.0"), 0)
            x = np.maximum(self._linear(x, "net.3"), 0)
            return self._linear(x, "net.6")

        x = x.reshape(-1, 1, 28, 28)
        x = self._pool(np.maximum(self._batch_norm(self._conv(x, "features.0"), "features.1"), 0))
        x = self._pool(np.maximum(self._batch_norm(self._conv(x, "features.4"), "features.5"), 0))
        x = np.maximum(self._linear(x.reshape(len(x), -1), "classifier.1"), 0)
        return self._linear(x, "classifier.4")

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        x = np.asarray(X, dtype=np.float32).reshape(-1, 784)
        if not len(x):
            return np.empty((0, 10), dtype=np.float32)
        if x.max() > 1:
            x = x / np.float32(255)
        batches = []
        # Bound the convolution workspace for benchmark and batch predictions.
        for start in range(0, len(x), 32):
            logits = self._logits(x[start:start + 32])
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            batches.append(exp / exp.sum(axis=1, keepdims=True))
        return np.concatenate(batches)
