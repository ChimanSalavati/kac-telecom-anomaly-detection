"""Zero-shot LLM prompting baseline runner.

Replaces notebooks:
    E6_TelecomTS_LLM_zero_shot (a/b/c = GPT/Gemini/Claude)
    E10_SpotLight_LLM_zero_shot (a/b/c)
        -> ``main.py llm --provider {gpt,gemini,claude} ...``

A frontier LLM is prompted directly on the raw KPI matrix + KPI names and asked
for a per-window anomaly probability in [0, 1]; we then threshold at 0.5 and
compute metrics (the LLM gets no training or threshold calibration -- this is the
"delegate detection to an LLM" baseline, reported separately in the paper).

* Real mode calls the provider API and needs the corresponding SDK + key
  (``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` / ``ANTHROPIC_API_KEY``).
* Smoke mode uses a deterministic offline "model" (a fixed heuristic over the
  window) so the prompt-assembly/parse/metric path runs without network or keys.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..config import ExperimentConfig
from ..data import get_data
from ._common import compute_metrics, metric_row

#: Default model id per provider (overridable via --llm-model). These match the
#: identifiers used in the paper's zero-shot tables.
_DEFAULT_MODELS = {
    "gpt": "gpt-5",
    "gemini": "gemini-2.5-pro",
    "claude": "claude-opus-4.7",
}
_KEY_ENV = {"gpt": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY", "claude": "ANTHROPIC_API_KEY"}


def _offline_scores(X: np.ndarray) -> np.ndarray:
    """Deterministic stand-in 'LLM' anomaly probability in [0, 1] per window."""
    flat = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
    z = flat.mean(axis=1)
    z = (z - z.mean()) / (z.std() + 1e-8)
    return 1.0 / (1.0 + np.exp(-z))  # logistic squashing -> [0, 1]


def _api_scores(cfg: ExperimentConfig, X: np.ndarray, kpi_names) -> np.ndarray:
    """Score windows with a real provider API (one call per window)."""
    key = os.environ.get(_KEY_ENV[cfg.llm_provider])
    if not key:
        raise RuntimeError(
            f"Zero-shot LLM real run needs {_KEY_ENV[cfg.llm_provider]} set and the "
            f"{cfg.llm_provider!r} SDK installed. See experiments/E6_*/README.md, "
            "or run with --smoke."
        )
    raise NotImplementedError(
        "Provider API call is intentionally not vendored (keys/rate limits/cost). "
        "Implement the prompt + parse for your provider here; the offline smoke "
        "path documents the expected input/output contract."
    )


def run(cfg: ExperimentConfig, logger) -> List[Dict]:
    data = get_data(cfg)
    X_te = np.asarray(data["X_test"])
    y_te = np.asarray(data["y_test"])
    model_id = cfg.llm_model or _DEFAULT_MODELS[cfg.llm_provider]

    if cfg.smoke:
        scores = _offline_scores(X_te)
    else:
        scores = _api_scores(cfg, X_te, data["kpi_names"])

    metrics = compute_metrics(scores, y_te, threshold=0.5)
    logger.info("%s (%s) zero-shot F1=%.4f AUROC=%.4f", cfg.llm_provider, model_id, metrics["f1"], metrics["auroc"])
    row = metric_row("model", model_id, 0, metrics)
    row["provider"] = cfg.llm_provider
    return [row]


GROUP_KEY = "model"
