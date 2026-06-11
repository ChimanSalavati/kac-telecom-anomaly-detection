"""Shared inference helpers for the KAC pre-production shadow scorer.

This module is intentionally framework-light (no FastAPI import) so it can be
reused by the FastAPI app (:mod:`deployment.serve_app`), the shadow-mode harness
(:mod:`deployment.shadow_runner`), and the latency benchmark
(:mod:`deployment.benchmark_latency`).

The scorer reproduces the paper's deployment configuration (Section
"Pre-Production Shadow Integration and Operational Cost"):

* CPU-only inference on **cached** Chronos-2 residual features and **cached**
  operator-style text summaries -- the online scoring path makes no LLM calls.
* The KAC-specific fusion/heads add only ~0.21% of parameters on top of the
  DistilBERT backbone (see :func:`param_breakdown`).

Set ``KAC_SMOKE=1`` to build a tiny, randomly-initialized DistilBERT so the
service can be exercised fully offline (CI / plumbing tests) without downloading
the pretrained backbone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Make ``experiments/_shared`` importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._shared import kac_ablation as KA  # noqa: E402

DEFAULT_N_KPIS = int(os.environ.get("KAC_N_KPIS", "64"))
DEFAULT_FEATS_PER_KPI = int(os.environ.get("KAC_FEATS_PER_KPI", "5"))


def git_sha() -> str:
    """Best-effort short git SHA for traceability in /model-info."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _build_smoke_model(n_kpis: int, feats_per_kpi: int) -> torch.nn.Module:
    """Tiny offline KAC model (random-init DistilBERT) for CI / plumbing tests."""
    from transformers import DistilBertConfig, DistilBertModel

    enc = DistilBertModel(
        DistilBertConfig(
            vocab_size=1000, dim=32, hidden_dim=64,
            n_layers=2, n_heads=2, max_position_embeddings=64,
        )
    )
    for p in enc.parameters():
        p.requires_grad = False
    for layer in enc.transformer.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    return KA.KPIAwareContrastiveModel(
        text_encoder=enc, resid_dim=n_kpis * feats_per_kpi, text_dim=32,
        d_model=64, proj_dim=128, n_kpis=n_kpis, feats_per_kpi=feats_per_kpi,
        num_heads=4, dropout=0.1,
    )


def build_inference_model(
    n_kpis: int = DEFAULT_N_KPIS,
    feats_per_kpi: int = DEFAULT_FEATS_PER_KPI,
    state_dict_path: str | None = None,
    smoke: bool | None = None,
) -> torch.nn.Module:
    """Build the KAC scorer on CPU and load weights if provided.

    Parameters
    ----------
    state_dict_path
        Optional ``torch.save`` checkpoint compatible with
        ``KPIAwareContrastiveModel.load_state_dict``. If omitted, the model
        starts with the pretrained-backbone + random heads (useful for latency
        measurement, which is weight-independent).
    smoke
        Force the tiny offline encoder. Defaults to the ``KAC_SMOKE`` env var.
    """
    if smoke is None:
        smoke = os.environ.get("KAC_SMOKE", "") not in ("", "0", "false", "False")

    model = (
        _build_smoke_model(n_kpis, feats_per_kpi)
        if smoke
        else KA.build_model(n_kpis, feats_per_kpi)
    )

    ckpt = state_dict_path or os.environ.get("KAC_STATE_DICT")
    if ckpt and Path(ckpt).is_file():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:  # older torch
            state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def param_breakdown(model: torch.nn.Module) -> Dict[str, object]:
    """Parameter breakdown matching Table "deployment_cost" in the paper."""
    te = model.text_encoder
    te_total = sum(p.numel() for p in te.parameters())
    te_train = sum(p.numel() for p in te.parameters() if p.requires_grad)
    te_frozen = te_total - te_train
    queries = int(model.kpi_queries.numel())
    others = sum(
        p.numel()
        for n, p in model.named_parameters()
        if not n.startswith("text_encoder") and n != "kpi_queries"
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def pct(x: int) -> float:
        return round(100.0 * x / total, 3) if total else 0.0

    return {
        "distilbert_frozen": te_frozen,
        "distilbert_last2": te_train,
        "kac_fusion_heads": others,
        "learnable_queries": queries,
        "total": total,
        "trainable": trainable,
        "kac_specific_pct": pct(others + queries),
        "breakdown_pct": {
            "distilbert_frozen": pct(te_frozen),
            "distilbert_last2": pct(te_train),
            "kac_fusion_heads": pct(others),
            "trainable": pct(trainable),
        },
    }


@torch.no_grad()
def score_batch(
    model: torch.nn.Module,
    input_ids: List[List[int]],
    attention_mask: List[List[int]],
    residuals: List[List[List[float]]],
) -> Dict[str, List[float]]:
    """Score a batch of cached (input_ids, attention_mask, residuals) windows.

    Returns window-level anomaly ``logits`` and ``probs`` in [0, 1].
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    mask = torch.as_tensor(attention_mask, dtype=torch.long)
    resid = torch.as_tensor(residuals, dtype=torch.float32)
    logits = model(ids, mask, resid, use_text=True)[0]
    probs = torch.sigmoid(logits)
    return {
        "logits": logits.detach().cpu().reshape(-1).tolist(),
        "probs": probs.detach().cpu().reshape(-1).tolist(),
    }
