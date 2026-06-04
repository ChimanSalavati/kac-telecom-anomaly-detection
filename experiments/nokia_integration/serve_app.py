"""
Minimal FastAPI surface for KAC inference (Nokia ``aad-traditional-ml`` style).

Environment:
    KAC_STATE_DICT   optional path to torch state_dict (CPU weights)
    KAC_N_KPIS       default 64 (LabTrace-SA)
    KAC_FEATS_PER_KPI default 5
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "_shared"))
import kac_ablation as KA  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install fastapi (and uvicorn) to run the scorer: pip install fastapi uvicorn"
    ) from e


class ScoreRequest(BaseModel):
    """One batch of windows."""

    input_ids: list[list[int]] = Field(..., description="[batch, L] tokenizer ids")
    attention_mask: list[list[int]] = Field(..., description="[batch, L] mask")
    residuals: list[list[list[float]]] = Field(
        ...,
        description="[batch, T_r, K*feats_per_kpi] z-normalised Chronos residuals",
    )


class ScoreResponse(BaseModel):
    logits: list[float]
    probs: list[float]


def _load_model() -> torch.nn.Module:
    n_kpis = int(os.environ.get("KAC_N_KPIS", "64"))
    feats = int(os.environ.get("KAC_FEATS_PER_KPI", "5"))
    m = KA.build_model(n_kpis, feats)
    ckpt = os.environ.get("KAC_STATE_DICT")
    if ckpt and Path(ckpt).is_file():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state, strict=False)
    m.eval()
    return m


app = FastAPI(title="KAC score", version="0.1")
_model: torch.nn.Module | None = None


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = _load_model()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if _model is None:
        raise HTTPException(500, "model not loaded")
    ids = torch.tensor(req.input_ids, dtype=torch.long)
    mask = torch.tensor(req.attention_mask, dtype=torch.long)
    resid = torch.tensor(req.residuals, dtype=torch.float32)
    if ids.shape[0] != resid.shape[0] or mask.shape != ids.shape:
        raise HTTPException(400, "shape mismatch on input_ids / attention_mask")
    with torch.no_grad():
        logits_t = _model(ids, mask, resid)[0]
        logits = logits_t.squeeze(-1).cpu().tolist()
    probs = torch.sigmoid(torch.tensor(logits)).tolist()
    return ScoreResponse(logits=[float(x) for x in logits], probs=[float(x) for x in probs])
