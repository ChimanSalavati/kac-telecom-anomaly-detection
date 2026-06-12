"""Offline smoke tests for the deployment artifact.

Exercises the shadow scorer end-to-end on tiny synthetic data with a
randomly-initialized backbone (``KAC_SMOKE``), so no weights/data/network are
needed. Covers:

* the FastAPI surface (`/healthz`, `/readyz`, `/model-info`, `POST /v1/score`)
* the shadow-mode harness (`shadow_runner.run_shadow`)
* the operational-cost benchmark (`benchmark_latency.benchmark`)

Run with:  pytest -q tests/test_deployment_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Build the tiny offline backbone and a small KPI count for all deployment code.
os.environ["KAC_SMOKE"] = "1"
os.environ["KAC_N_KPIS"] = "6"
os.environ["KAC_FEATS_PER_KPI"] = "5"

K, FEATS, T_R, L = 6, 5, 12, 8


def _batch(n=4):
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(0, 1000, size=(n, L)).tolist(),
        "attention_mask": np.ones((n, L), dtype=int).tolist(),
        "residuals": rng.normal(0, 1, size=(n, T_R, K * FEATS)).astype(float).tolist(),
    }


def test_serve_app_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from deployment import serve_app

    with TestClient(serve_app.app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/readyz").json()["status"] == "ready"

        info = client.get("/model-info").json()
        assert info["online_llm_calls"] is False
        assert info["total"] > 0
        # Sanity-check the breakdown is a valid percentage. With the real
        # DistilBERT backbone the KAC-specific share is ~0.21%; the tiny
        # random-init smoke backbone used here makes that share much larger,
        # so we only assert it is a well-formed percentage.
        assert 0.0 < info["kac_specific_pct"] <= 100.0
        assert info["trainable"] <= info["total"]

        resp = client.post("/v1/score", json=_batch(4))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["probs"]) == 4
        assert all(0.0 <= p <= 1.0 for p in body["probs"])


def test_shadow_runner(tmp_path):
    from deployment import shadow_runner

    out = tmp_path / "shadow.jsonl"
    summary = shadow_runner.run_shadow(smoke=True, out_log=str(out))
    assert summary["windows_scored"] > 0
    assert summary["promoted_incidents"] == 0
    assert out.exists() and out.stat().st_size > 0


def test_benchmark_latency(tmp_path):
    from deployment import benchmark_latency

    res = benchmark_latency.benchmark(smoke=True, out_dir=str(tmp_path / "dep"))
    assert res["params"]["total"] > 0
    assert len(res["latency"]) >= 1
    assert (tmp_path / "dep" / "latency.csv").exists()
    assert (tmp_path / "dep" / "params.csv").exists()
