"""Small shared utilities."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings


def ensure_directories() -> None:
    """Create all runtime output directories."""
    for path in (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.models_dir,
        settings.figures_dir,
        settings.metrics_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = settings.random_seed) -> None:
    """Seed Python, NumPy, and PyTorch (when installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
