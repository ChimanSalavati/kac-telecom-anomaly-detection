# KPI-Aware Multimodal Anomaly Detection for 5G and Open RAN Telemetry

[![Project website](https://img.shields.io/badge/Project%20website-Open-2ea44f?style=for-the-badge)](https://chimansalavati.github.io/kac-telecom-anomaly-detection/)

> [!IMPORTANT]
> **Project website:** **https://chimansalavati.github.io/kac-telecom-anomaly-detection/**
> — paper, key results, deployment, and reproduction commands.

Reproducibility artefact for the IEEE ICDM 2026 Applied Track paper *"KPI-Aware Multimodal Anomaly Detection for 5G and Open RAN Telemetry"*.

## What is reproducible

- **TelecomTS** and **SpotLight** are public benchmarks: every number on these two datasets is fully reproducible from this repository (data download/preprocessing scripts + training/evaluation code + fixed seeds 42/123/456/789/1337).
- **ProdTrace-SA** is built from real 5G RAN production packet captures processed through Nokia's KPI-construction pipeline and **cannot be redistributed** (see `docs/data_availability.md`). All ProdTrace-SA training/evaluation **code** is included so the pipeline can be inspected and rerun by collaborators with data access, but the raw PCAPs and derived NPZ splits are not shipped.

KAC combines full-KPI Chronos-2 residual evidence with compact, KPI-specific operator-style text summaries and aligns the two views with an uncertainty-weighted KPI-level contrastive objective. The detector itself stays lightweight: the LLM only writes the text summaries offline, the alerting hot path never calls an LLM at inference time.

On the two public benchmarks released here, KAC achieves the best threshold-based F1; some baselines remain competitive on ranking or precision metrics (see the paper for the full tables):

| Benchmark | Best baseline F1 | KAC F1 | Metrics where KAC is best | Where a baseline leads |
|---|---|---|---|---|
| TelecomTS (balanced) | 0.918 (QR-TAN, residual-only) | **0.960** | F1, Precision, AUROC, AP | Recall (rule-based) |
| SpotLight (Open RAN) | 0.940 (iTransformer) | **0.950** | F1, Recall | Precision (ModernTCN-cls); AUROC/AP (PatchTST) |

Zero-shot frontier LLMs prompted directly on the raw KPI matrix reach only F1 in `[0.250, 0.641]` across the same benchmarks, which is why KAC keeps the LLM offline and uses it as a side-channel rather than as the detector.

The third benchmark in the paper (*ProdTrace-SA*) is built from real 5G RAN production packet captures held by Nokia and cannot be redistributed. The code path is identical to the public-benchmark runs (`main.py` with `--dataset production`); see [`docs/data_availability.md`](docs/data_availability.md).

## Contents

```
.
├── main.py                               # Unified CLI entry point for every experiment
├── kac/                                  # Centralized experiment package
│   ├── config.py                         # ExperimentConfig dataclass + per-dataset presets
│   ├── io.py                             # Run dirs, logging, metric/summary writers
│   ├── data.py                           # Public-split loaders + synthetic smoke data
│   └── runners/                          # One runner per experiment family
├── experiments/
│   ├── _shared/                          # Loaders, KAC model + ablation driver, helpers
│   └── sota_models/                      # DCdetector, D3R, MEMTO, ModernTCN, TimesNet
├── scripts/
│   ├── download_telecomts.py
│   ├── download_spotlight.py
│   └── compute_chronos_residuals.py
├── tests/
│   └── test_smoke.py                     # Offline smoke test for every experiment
├── tables/                               # Snapshot .tex tables matching the paper
├── figures/                              # Figures used by the LaTeX source
├── artifacts/                            # Centralized run outputs (git-ignored)
├── docs/
│   └── data_availability.md
├── main_kac_paper.tex                    # LaTeX source of the paper
├── references.bib
├── requirements.txt
├── CITATION.cff
├── LICENSE                               # Apache-2.0
└── README.md
```

## Experiment to paper-artefact map

All experiments run through a single command-line entry point, `main.py`. Each
experiment family is a subcommand; the dataset and scenario are flags.

| Command | Paper artefact | What it produces |
|---|---|---|
| `main.py kac --dataset telecomts --scenario balanced` | **Table 2** (KAC row) + **Figure 2** | Multi-seed KAC headline numbers on the balanced TelecomTS split |
| `main.py kac --dataset telecomts --scenario imbalanced` | **Table 2** (imbalanced KAC row) | Same pipeline on the imbalanced stress split |
| `main.py kac --dataset spotlight` | **Table 4** (KAC row) | KAC headline numbers on SpotLight Open RAN |
| `main.py sota --dataset telecomts` | **Table 2** (DCdetector, D3R, MEMTO, ModernTCN, TimesNet) | Trained deep MTSAD baselines |
| `main.py sota --dataset spotlight` | **Table 4** (SOTA block) | Deep MTSAD baselines on SpotLight |
| `main.py foundation --dataset telecomts --foundation-model {moment,toto,mantis}` | **Table 2** (foundation block) | Frozen-encoder + linear probe |
| `main.py foundation --dataset spotlight --foundation-model {moment,toto,mantis}` | **Table 4** (foundation block) | TSFM baselines on SpotLight |
| `main.py spotlight-baseline --dataset telecomts` | **Table 2** (SpotLight row) | SpotLight JVGAN/MRPI pipeline baseline |
| `main.py llm --dataset telecomts --provider {gpt,gemini,claude}` | **Table 3** (TelecomTS rows) | Zero-shot LLM scoring |
| `main.py llm --dataset spotlight --provider {gpt,gemini,claude}` | **Table 3** (SpotLight rows) | Zero-shot LLM scoring |
| `main.py ablation --dataset telecomts --variants V1 V2 V3` | **Table 6** (TelecomTS) | V1 → V3 KAC component ablation (5 seeds) |
| `main.py ablation --dataset spotlight --variants V1 V2 V3` | **Table 6** (SpotLight) | V1 → V3 KAC component ablation (5 seeds) |

## Quick start

```bash
git clone https://github.com/ChimanSalavati/kac-telecom-anomaly-detection.git
cd kac-telecom-anomaly-detection

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0. (Optional) verify the whole pipeline offline in seconds, no data needed:
python main.py kac --dataset telecomts --smoke
pytest -q tests/

# 1. Download the two public corpora (TelecomTS, SpotLight).
python scripts/download_telecomts.py
python scripts/download_spotlight.py

# 2. Build the Chronos-2 residual cache once per dataset
#    (consumed by the KAC and KAC-ablation runs).
python scripts/compute_chronos_residuals.py --dataset telecomts
python scripts/compute_chronos_residuals.py --dataset spotlight

# 3. Run any experiment. Example: the TelecomTS headline KAC run (5 seeds).
python main.py kac --dataset telecomts --scenario balanced \
    --seeds 42 123 456 789 1337
```

## Run experiments

Every run writes to `artifacts/<run_id>/` (`config.json`, `metrics.csv`,
`summary.csv`) and a transcript to `logs/<run_id>.log`.

```bash
# Headline KAC, balanced and imbalanced TelecomTS, and SpotLight.
python main.py kac --dataset telecomts --scenario balanced
python main.py kac --dataset telecomts --scenario imbalanced
python main.py kac --dataset spotlight

# Component ablation V1/V2/V3 (paper Table 6).
python main.py ablation --dataset telecomts --variants V1 V2 V3
python main.py ablation --dataset spotlight

# Deep MTSAD baselines (subset selectable).
python main.py sota --dataset telecomts --methods DCdetector TimesNet D3R

# Frozen TSFM encoder + linear probe.
python main.py foundation --dataset spotlight --foundation-model moment

# Zero-shot LLM (needs the provider SDK + key:
#   gpt -> OPENAI_API_KEY, gemini -> GOOGLE_API_KEY, claude -> ANTHROPIC_API_KEY).
python main.py llm --dataset telecomts --provider gpt
```

### Overriding the config

Defaults come from per-dataset presets in [`kac/config.py`](kac/config.py) and
match the paper's Section 5.1.5. Override any field with a dedicated flag or the
generic `--set key=value`:

```bash
# Dedicated flags:
python main.py kac --dataset telecomts --epochs 40 --lr-head 3e-4 \
    --batch-size 32 --alpha-supcon 0.5 --beta-kpi 0.5 --device cpu

# Generic escape hatch for any other field (repeatable):
python main.py kac --dataset telecomts --set proj_dim=64 --set max_len=256
```

`--smoke` swaps in tiny synthetic data and a randomly-initialized text encoder so
every subcommand runs end-to-end offline (no datasets, GPU, or API keys); this is
exactly what `tests/test_smoke.py` and the `smoke` CI workflow exercise.

## Deployment (pre-production shadow mode)

The paper's applied contribution includes KAC's **pre-production shadow integration**
with Nokia's GKE-based KPI-construction pipeline. A portable, CPU-only version of
that deployment lives in [`deployment/`](deployment/): a FastAPI scorer, a
shadow-mode harness that logs scores for offline comparison (and never promotes
incidents), an operational-cost benchmark that regenerates the latency/parameter
table, plus a `Dockerfile`, `docker-compose.yml`, and Kubernetes manifests. See
[`deployment/DEPLOYMENT.md`](deployment/DEPLOYMENT.md) and
[`deployment/architecture.md`](deployment/architecture.md).

```bash
pip install -r deployment/requirements.txt

# Offline plumbing checks (tiny random-init backbone; no data/GPU/keys):
KAC_SMOKE=1 KAC_N_KPIS=6 uvicorn deployment.serve_app:app --port 8765
python -m deployment.shadow_runner --smoke         # -> logs/shadow_scores.jsonl
python -m deployment.benchmark_latency --smoke      # -> artifacts/deployment/

# Containerized scorer + cached-artifact store (MinIO), or a cluster:
docker compose -f deployment/docker-compose.yml up --build
kubectl apply -f deployment/k8s/
```

## Project website

**Live site: https://chimansalavati.github.io/kac-telecom-anomaly-detection/**

A static showcase of the paper with key result tables, figures, the deployment,
reproduction commands, and the logs/artifacts layout lives in [`docs/`](docs/).
Preview locally with `python -m http.server -d docs 8000`, or publish via GitHub
Pages (Settings → Pages → branch `main`, folder `/docs`); see
[`docs/README.md`](docs/README.md).

## Data availability

See [`docs/data_availability.md`](docs/data_availability.md). The short version: TelecomTS and SpotLight are public and auto-downloaded by the scripts above; ProdTrace-SA is built from real 5G RAN production packet captures held by Nokia and is not redistributable.

## Citation

```bibtex
@inproceedings{salavati2026kac,
  title     = {{KPI}-Aware Multimodal Anomaly Detection for 5G and Open RAN Telemetry},
  author    = {Salavati, Chiman and Wu, Liang and Wan, Kelly and Darbari, Mayank and Hong, Liangjie},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Data Mining (ICDM), Applied Track},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE). The bundled SOTA baseline implementations under `experiments/sota_models/` are minor re-implementations of the publicly released models cited in their docstrings; the upstream licenses are preserved in the module-level comments.

## Acknowledgements

- TelecomTS by Feng et al. ([arXiv:2510.06063](https://arxiv.org/abs/2510.06063)).
- SpotLight by Sun et al. (MobiCom 2024).
- Chronos-2 by Ansari et al.; MOMENT by Goswami et al.; Mantis and TOTO as cited in the paper.
- KAC is being prepared for evaluation against Nokia's internal packet-level anomaly detector on the same 5G RAN production telemetry.
