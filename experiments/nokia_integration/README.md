# Nokia-style integration (KAC)

This folder holds **portable** artifacts for staging KAC next to traditional
detectors (e.g. in Nokia `aad-traditional-ml`): a minimal FastAPI scorer and
pointers to revision benchmarks.

## Layout

| Path | Role |
|------|------|
| `serve_app.py` | FastAPI `POST /v1/score` — `input_ids`, `attention_mask`, `residuals` → logits/probs |
| `../revision_pipeline/runner_labtrace_shape_bench.py` | Latency on $K{=}64$, $T_r{=}24$ tensors (no internal NPZ) |
| `../revision_pipeline/compute_shadow_labtrace_stats.py` | Summarise `experiments/_shared/kac_ablation_labtrace_sa.csv` → LaTeX macros |

## Quick start (scorer)

```bash
cd kac-telecom-anomaly-detection
pip install fastapi uvicorn  # if not already installed
export KAC_STATE_DICT=path/to/kac_state_dict.pt   # optional
uvicorn experiments.nokia_integration.serve_app:app --host 0.0.0.0 --port 8765
```

`KAC_STATE_DICT` should be a `torch.save` dict compatible with
`KPIAwareContrastiveModel.load_state_dict`. If unset, the service starts with
**random** weights (useful only for plumbing tests).

## Copying into `aad-traditional-ml`

Copy `experiments/_shared/kac_ablation.py` (model + `build_model`) and this
`nokia_integration/` tree under `src/anomaly_detection/kac/`, then wire your
dependency injection / Kong route to `serve_app:app`.
