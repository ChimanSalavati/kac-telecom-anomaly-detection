"""SpotLight JVGAN/MRPI pipeline baseline runner.

Replaces notebooks:
    E5_TelecomTS_SpotLight_baseline (+ E5b imbalanced)
        -> ``main.py spotlight-baseline ...``

The paper reports the SpotLight JVGAN--MRPI pipeline (Sun et al., MobiCom'24)
applied to our windows. That pipeline ships with the upstream SpotLight release
and is not vendored here.

* Real mode requires the upstream SpotLight pipeline checkpoints/code; if they
  are not wired in, it raises a clear error.
* Smoke mode uses an offline residual-energy scorer (mean |residual| per window,
  or mean |X| when no residual cache is present) so the threshold/metric path is
  exercised without the external pipeline.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import ExperimentConfig
from ..data import get_data
from ._common import best_f1_threshold, compute_metrics, metric_row


def _energy_scores(data: Dict[str, object], split: str) -> np.ndarray:
    key = f"R_{split}" if f"R_{split}" in data else f"X_{split}"
    arr = np.abs(np.asarray(data[key], dtype=np.float64))
    return arr.reshape(arr.shape[0], -1).mean(axis=1)


def run(cfg: ExperimentConfig, logger) -> List[Dict]:
    if not cfg.smoke:
        raise NotImplementedError(
            "The SpotLight JVGAN/MRPI pipeline is an external dependency (upstream "
            "SpotLight MobiCom'24 release). Wire its checkpoints/inference in here, "
            "or run with --smoke for the offline residual-energy stand-in."
        )
    data = get_data(cfg)
    va = _energy_scores(data, "val")
    te = _energy_scores(data, "test")
    y_va = np.asarray(data["y_val"])
    y_te = np.asarray(data["y_test"])
    thr, _ = best_f1_threshold(va, y_va, verbose=False)
    metrics = compute_metrics(te, y_te, thr)
    logger.info("SpotLight pipeline (stand-in) F1=%.4f AUROC=%.4f", metrics["f1"], metrics["auroc"])
    return [metric_row("method", "SpotLight pipeline", 0, metrics)]


GROUP_KEY = "method"
