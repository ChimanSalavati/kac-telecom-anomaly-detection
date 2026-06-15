"""FastAPI scorer for the KAC pre-production shadow deployment.

This is the online scoring surface described in the paper's deployment section.
In Nokia's pre-production shadow configuration it sits behind a Kong route and
is fed cached residual/summary tensors built by the GKE KPI-construction
pipeline; here it is packaged as a portable, CPU-only FastAPI service so the
deployment path is inspectable and runnable outside Nokia infrastructure.

Endpoints
---------
* ``GET  /healthz``    liveness (process up)
* ``GET  /readyz``     readiness (model loaded)
* ``GET  /model-info`` parameter breakdown (Table "deployment_cost") + git SHA
* ``POST /v1/score``   batch scoring -> per-window anomaly logits/probabilities

Run locally::

    pip install -r deployment/requirements.txt
    export KAC_STATE_DICT=/path/to/kac_state_dict.pt   # optional
    uvicorn deployment.serve_app:app --host 0.0.0.0 --port 8765

For an offline plumbing test (tiny random-init backbone, no download)::

    KAC_SMOKE=1 KAC_N_KPIS=6 uvicorn deployment.serve_app:app --port 8765
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .kac_service import (
    DEFAULT_FEATS_PER_KPI,
    DEFAULT_N_KPIS,
    build_inference_model,
    git_sha,
    param_breakdown,
    score_batch,
)

try:
    from fastapi import FastAPI, HTTPException
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install the serving extras: pip install -r deployment/requirements.txt"
    ) from e


class ScoreRequest(BaseModel):
    """One batch of cached KPI windows."""

    input_ids: List[List[int]] = Field(..., description="[batch, L] tokenizer ids of the cached summary")
    attention_mask: List[List[int]] = Field(..., description="[batch, L] attention mask")
    residuals: List[List[List[float]]] = Field(
        ..., description="[batch, T_r, K*feats_per_kpi] z-normalized foundation-model residuals (Chronos-2 in this pipeline)",
    )


class ScoreResponse(BaseModel):
    logits: List[float]
    probs: List[float]


app = FastAPI(
    title="KAC shadow scorer",
    version="1.0",
    description="CPU-only KAC anomaly scorer for pre-production shadow integration.",
)

_model = None  # lazily loaded on startup


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = build_inference_model(DEFAULT_N_KPIS, DEFAULT_FEATS_PER_KPI)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/model-info")
def model_info() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    info = param_breakdown(_model)
    info.update({
        "git_sha": git_sha(),
        "n_kpis": DEFAULT_N_KPIS,
        "feats_per_kpi": DEFAULT_FEATS_PER_KPI,
        "device": "cpu",
        "online_llm_calls": False,
    })
    return info


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not (len(req.input_ids) == len(req.attention_mask) == len(req.residuals)):
        raise HTTPException(status_code=400, detail="batch size mismatch across fields")
    out = score_batch(_model, req.input_ids, req.attention_mask, req.residuals)
    return ScoreResponse(**out)
