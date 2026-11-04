"""
Utility functions for EE5130 MNIST Project:
- Seeding and reproducibility
- High-precision execution timing
- JSON serialization with NumPy type handling
- Directory and path helpers
"""

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import json
import numpy as np

# Project Root is two levels up from src/utils.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
JSON_DIR = RESULTS_DIR / "json"


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str = "Operation") -> Generator[Dict[str, float], None, None]:
    """Context manager to measure and log execution duration in milliseconds."""
    info: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        elapsed_sec = time.perf_counter() - start
        elapsed_ms = elapsed_sec * 1000.0
        info["elapsed_sec"] = elapsed_sec
        info["elapsed_ms"] = elapsed_ms


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder handling NumPy scalar and array types seamlessly."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, filepath: Path) -> None:
    """Save serializable data to JSON file with indentation and UTF-8 encoding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2)


def load_json(filepath: Path) -> Any:
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
