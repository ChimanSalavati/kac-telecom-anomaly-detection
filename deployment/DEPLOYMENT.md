# KAC deployment artifact (pre-production shadow mode)

This directory is the **portable, CPU-only** representation of the deployment
reported in the paper. It lets a reader inspect and run the KAC scoring path,
the shadow-mode harness, and the operational-cost benchmark without access to
Nokia's internal infrastructure or data.

> **Status and timeline (read this).** The *deployment* — KAC integrated in
> **pre-production shadow mode** with Nokia's GKE-based KPI-construction pipeline —
> is the industrial contribution reported in the paper and predates submission.
> This `deployment/` directory is the **public code artifact** that mirrors that
> integration in a sanitized, runnable form, and (like most reproducibility
> artifacts) is published/polished as part of the artifact-release process. The
> integration runs alongside Nokia's internal packet-level detector and logs
> scores for offline comparison only; KAC does **not** promote incidents, raise
> operator-facing alerts, or trigger mitigation. Operator-facing rollout is
> explicitly future work (paper, Limitations).

## Contents

| File | Role |
|---|---|
| `kac_service.py` | Model build/load, parameter breakdown, batch scoring (no web framework) |
| `serve_app.py` | FastAPI scorer: `/healthz`, `/readyz`, `/model-info`, `POST /v1/score` |
| `shadow_runner.py` | Shadow-mode loop: score cached windows, append JSONL shadow log, no promotion |
| `benchmark_latency.py` | Regenerates the operational-cost table (params + CPU latency) |
| `Dockerfile` | CPU-only container image for the scorer |
| `docker-compose.yml` | Local stack: scorer + MinIO (cached-artifact store) |
| `k8s/` | `deployment.yaml` (probes, CPU resources), `service.yaml`, `hpa.yaml` |
| `architecture.md` | Shadow data-flow diagram + repo↔Nokia component mapping |

## Model card (online scorer)

- **Task:** window-level anomaly probability for 5G/Open RAN KPI telemetry.
- **Inputs (cached):** z-normalized Chronos-2 residual tensor `[T_r, K*5]` + a
  tokenized operator-style summary (`input_ids`, `attention_mask`). The online
  path makes **no LLM calls**.
- **Output:** anomaly logit + probability in `[0, 1]`.
- **Backbone:** DistilBERT (frozen + last 2 blocks fine-tuned) ≈ 66.5M params;
  KAC fusion/heads + queries add ≈ 0.21%.
- **Compute:** CPU; batch-1 median well under the 100 ms telemetry cadence.
- **Not for:** standalone operator alerting; it is an evaluation-only shadow
  scorer in its current integration.

## Quick start

```bash
# 1) Serving extras (on top of the repo's requirements.txt)
pip install -r deployment/requirements.txt

# 2) Offline plumbing check (tiny random-init backbone, no download/data)
KAC_SMOKE=1 KAC_N_KPIS=6 uvicorn deployment.serve_app:app --port 8765 &
curl -s localhost:8765/healthz
curl -s localhost:8765/model-info

# 3) Shadow-mode loop (offline synthetic windows) -> logs/shadow_scores.jsonl
python -m deployment.shadow_runner --smoke

# 4) Operational-cost benchmark (params + CPU latency) -> artifacts/deployment/
python -m deployment.benchmark_latency --smoke      # fast offline
python -m deployment.benchmark_latency              # full (batch 1/16/64)
```

### Real weights

Point `KAC_STATE_DICT` at a `torch.save` checkpoint compatible with
`KPIAwareContrastiveModel.load_state_dict` (produced by a `main.py kac` run).
Without it the service still starts (random heads) for latency/plumbing tests.

### Container / Kubernetes

```bash
docker compose -f deployment/docker-compose.yml up --build   # scorer + MinIO
# or, on a cluster:
kubectl apply -f deployment/k8s/
```

## What is and isn't shipped

- **Shipped:** the scorer, shadow harness, benchmark, container, and manifests —
  everything needed to run and evaluate the KAC scoring path on public-benchmark
  or your own cached tensors.
- **Not shipped (Nokia-internal):** the GKE KPI-construction pipeline,
  ProdTrace-SA PCAPs/derived NPZs, and the internal packet-level detector used
  as the offline comparison baseline. See `docs/data_availability.md`.
