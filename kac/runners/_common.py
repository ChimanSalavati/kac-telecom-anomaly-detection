"""Helpers shared by all runners: device resolution, seeding, metrics.

Metric and threshold utilities are re-exported from
``experiments.sota_models.common`` so every runner reports numbers the exact
same way the baseline notebooks did.
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np

# Reuse the canonical metric/threshold code already in the repo.
from experiments.sota_models.common import (  # noqa: F401
    best_f1_threshold,
    compute_metrics,
    reduce_window_score,
)


def resolve_device(device: str) -> str:
    """Map ``auto`` to the best available backend; pass others through."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def metric_row(name_key: str, name: str, seed: int, metrics: Dict[str, float]) -> Dict:
    """Standardize a per-(method/variant, seed) result row."""
    return {
        name_key: name,
        "seed": seed,
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "ap": float(metrics.get("ap", float("nan"))),
    }
