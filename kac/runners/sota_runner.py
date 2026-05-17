"""Deep MTSAD / one-class SOTA baselines runner.

Replaces notebooks:
    E3_TelecomTS_SOTA_baselines (+ E3b imbalanced), E8_SpotLight_SOTA_baselines
        -> ``main.py sota ...``

Reuses ``experiments/sota_models/training.evaluate_method`` (DCdetector, TimesNet,
ModernTCN, MEMTO, D3R) verbatim. In smoke mode every method runs for a single
epoch on tiny synthetic windows so the full train/score/threshold/metric path is
exercised offline.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.sota_models.training import evaluate_method

from ..config import ExperimentConfig
from ..data import get_data
from ._common import metric_row, resolve_device

#: Per-method kwargs used only in smoke mode (each train fn has a different
#: epoch-argument name, so they cannot share a single ``epochs`` kwarg).
_SMOKE_KWARGS = {
    "DCdetector": dict(epochs=1, d_model=32, e_layers=1, batch_size=8),
    "TimesNet": dict(epochs=1, d_model=16, d_ff=16, e_layers=1, batch_size=8),
    "ModernTCN": dict(epochs=1, d_model=16, num_blocks=1, batch_size=8),
    "MEMTO": dict(epochs_phase1=1, epochs_phase2=1, d_model=32, e_layers=1, d_ff=32, batch_size=8),
    "D3R": dict(epochs=1, model_dim=32, ff_dim=32, block_num=1, batch_size=8),
}


def run(cfg: ExperimentConfig, logger) -> List[Dict]:
    data = get_data(cfg)
    X_tr, y_tr = np.asarray(data["X_train"]), np.asarray(data["y_train"])
    X_va, y_va = np.asarray(data["X_val"]), np.asarray(data["y_val"])
    X_te, y_te = np.asarray(data["X_test"]), np.asarray(data["y_test"])
    device = resolve_device(cfg.device)

    rows: List[Dict] = []
    seeds = [0] if cfg.smoke else cfg.seeds
    for method in cfg.sota_methods:
        logger.info("--- SOTA method: %s ---", method)
        for seed in seeds:
            kwargs = _SMOKE_KWARGS.get(method, {}) if cfg.smoke else {}
            res = evaluate_method(
                method, X_tr, y_tr, X_va, y_va, X_te, y_te,
                seed=seed, device=device, verbose=cfg.smoke is False, **kwargs,
            )
            logger.info("  seed=%-5d F1=%.4f AUROC=%.4f", seed, res["f1"], res.get("auroc", float("nan")))
            rows.append(metric_row("method", method, seed, res))
    return rows


GROUP_KEY = "method"
