# KAC: KPI-Aware Multimodal Anomaly Detection for 5G and Open RAN

Reproducibility artefact for the IEEE ICDM 2026 Applied Track paper *"KAC: KPI-Aware Multimodal Anomaly Detection for High-Dimensional 5G and Open RAN Telemetry"*.

## What is reproducible

- **TelecomTS** and **SpotLight** are public benchmarks: every number on these two datasets is fully reproducible from this repository (data download/preprocessing scripts + training/evaluation code + fixed seeds 42/123/456/789/1337).
- **LabTrace-SA** is built from Nokia-internal Samsung 5G RAN lab PCAPs and **cannot be redistributed** (see `docs/data_availability.md`). All LabTrace-SA training/evaluation **code** is included so the pipeline can be inspected and rerun by collaborators with data access, but the raw PCAPs and derived NPZ splits are not shipped.

KAC combines full-KPI Chronos-2 residual evidence with compact, KPI-specific operator-style text summaries and aligns the two views with an uncertainty-weighted KPI-level contrastive objective. The detector itself stays lightweight: the LLM only writes the text summaries offline, the alerting hot path never calls an LLM at inference time.

On the two public benchmarks released here, KAC matches or beats every baseline we evaluated:

| Benchmark | Best baseline F1 | KAC F1 | Best baseline metric beaten by KAC |
|---|---|---|---|
| TelecomTS (balanced) | 0.918 (QR-TAN, residual-only) | **0.960** | F1, AUROC, AP |
| SpotLight (Open RAN) | 0.941 (QR-TAN) | **0.950** | F1, Precision, Recall, AUROC, AP |

Zero-shot frontier LLMs prompted directly on the raw KPI matrix reach only F1 in `[0.250, 0.641]` across the same benchmarks, which is why KAC keeps the LLM offline and uses it as a side-channel rather than as the detector.

The third benchmark in the paper (*LabTrace-SA*) is built from Samsung 5G RAN lab PCAP captures held by Nokia and cannot be redistributed. The code path is identical to the public-benchmark notebooks; see [`docs/data_availability.md`](docs/data_availability.md).

## Contents

```
.
├── main_kac.tex                          # LaTeX source of the paper
├── references.bib
├── figures/                              # Figures used by the LaTeX source
├── tables/                               # Auto-generated .tex tables
├── experiments/
│   ├── revision_pipeline/                # Regression runners (latency, shadow stats, leakage)
│   ├── nokia_integration/                # FastAPI scorer stub for aad-traditional-ml-style staging
│   ├── _shared/                          # Loaders, KAC model + ablation driver, helpers
│   ├── sota_models/                      # DCdetector, D3R, MEMTO, ModernTCN, TimesNet
│   ├── E1_TelecomTS_KAC_main/            # Table 2 (KAC row) + Figure 2 (ROC)
│   ├── E2_TelecomTS_imbalanced_KAC/      # Table 2 Scenario 2 (KAC row)
│   ├── E3_TelecomTS_SOTA_baselines/      # Table 2 SOTA block
│   ├── E4_TelecomTS_Foundation_Models/   # Table 2 foundation-model block
│   ├── E5_TelecomTS_SpotLight_baseline/  # Table 2 SpotLight-as-method row
│   ├── E6_TelecomTS_LLM_zero_shot/       # Table 3 TelecomTS rows
│   ├── E7_SpotLight_KAC_main/            # Table 4 (KAC row)
│   ├── E8_SpotLight_SOTA_baselines/      # Table 4 SOTA block
│   ├── E9_SpotLight_Foundation_Models/   # Table 4 foundation-model block
│   ├── E10_SpotLight_LLM_zero_shot/      # Table 3 SpotLight rows
│   ├── E11_KAC_ablation_TelecomTS/       # Table 6 (TelecomTS columns)
│   └── E12_KAC_ablation_SpotLight/       # Table 6 (SpotLight columns)
├── scripts/
│   ├── download_telecomts.py
│   ├── download_spotlight.py
│   └── compute_chronos_residuals.py
├── docs/
│   └── data_availability.md
├── requirements.txt
├── CITATION.cff
├── LICENSE                                # Apache-2.0
└── README.md
```

## Notebook to paper-artefact map

| Notebook | Paper artefact | What it produces |
|---|---|---|
| `E1_TelecomTS_KAC_main` | **Table 2** (KAC + QR-TAN + Chronos-2+LR + rule-based rows) and **Figure 2** | Multi-seed KAC headline numbers on the balanced TelecomTS split, plus the ROC export `roc_zoomed_top4.pdf` |
| `E2_TelecomTS_imbalanced_KAC` | **Table 2 (Scenario 2)** | Same pipeline on the 190/10 anomaly-rate split |
| `E3_TelecomTS_SOTA_baselines` | **Table 2** (DCdetector, D3R, MEMTO, ModernTCN, TimesNet, LOF, IsolationForest) | Trained SOTA baselines on TelecomTS |
| `E4_TelecomTS_Foundation_Models` | **Table 2** (MOMENT, TOTO, Mantis rows) | Frozen-encoder + linear probe |
| `E5_TelecomTS_SpotLight_baseline` | **Table 2** (SpotLight row) | SpotLight JVGAN/MRPI pipeline applied to TelecomTS windows |
| `E6_TelecomTS_LLM_zero_shot` | **Table 3** TelecomTS rows | Claude / GPT / Gemini direct zero-shot scoring |
| `E7_SpotLight_KAC_main` | **Table 4** (KAC row) | KAC headline numbers on SpotLight Open RAN |
| `E8_SpotLight_SOTA_baselines` | **Table 4** (SOTA block) | DCdetector / D3R / MEMTO / ModernTCN / TimesNet on SpotLight |
| `E9_SpotLight_Foundation_Models` | **Table 4** (foundation-model block) | TSFM baselines on SpotLight |
| `E10_SpotLight_LLM_zero_shot` | **Table 3** SpotLight rows | Zero-shot LLM probes on SpotLight |
| `E11_KAC_ablation_TelecomTS` | **Table 6** (TelecomTS) | V1 → V4 KAC component ablation (5 seeds) |
| `E12_KAC_ablation_SpotLight` | **Table 6** (SpotLight) | V1 → V4 KAC component ablation (5 seeds) |

## Quick start

```bash
git clone https://github.com/ChimanSalavati/kac-telecom-anomaly-detection.git
cd kac-telecom-anomaly-detection

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download the two public corpora (TelecomTS, SpotLight).
python scripts/download_telecomts.py
python scripts/download_spotlight.py

# 2. Build the Chronos-2 residual cache once per dataset.
#    The cache is what every KAC and KAC-ablation notebook consumes.
python scripts/compute_chronos_residuals.py --dataset telecomts
python scripts/compute_chronos_residuals.py --dataset spotlight

# 3. Run any experiment notebook. Example: the TelecomTS headline run.
jupyter nbconvert --to notebook --execute \
    experiments/E1_TelecomTS_KAC_main/E1_TelecomTS_KAC_main.ipynb \
    --inplace
```

The ablation table can be reproduced from the command line without opening a notebook:

```bash
cd experiments/E11_KAC_ablation_TelecomTS
python -m experiments._shared.kac_ablation --dataset TelecomTS --seeds 42 123 456 789 1337
```

## Hardware and runtime

KAC trains on a single GPU. We trained the headline runs on an A100; an Apple M-series GPU is also enough for the TelecomTS notebooks (slower but reproducible). The SOTA baselines and foundation-model notebooks are also feasible on a single A100. Approximate per-dataset wall-clock for the full ablation (4 variants × 5 seeds):

| Dataset | A100 | Apple M2 Pro (MPS) | CPU only |
|---|---|---|---|
| TelecomTS | ~3.5 h | ~15 h | ~17 h |
| SpotLight | ~8 h | ~40 h | not recommended |

Each notebook checkpoints per-seed results so an interrupted run can resume without recomputing.

## Data availability

See [`docs/data_availability.md`](docs/data_availability.md). The short version: TelecomTS and SpotLight are public and auto-downloaded by the scripts above; LabTrace-SA uses Samsung 5G RAN lab data held by Nokia and is not redistributable.

## Citation

```bibtex
@inproceedings{salavati2026kac,
  title     = {{KAC}: {KPI}-Aware Multimodal Anomaly Detection for High-Dimensional 5G and Open RAN Telemetry},
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
- KAC is being prepared for evaluation against an internal Nokia anomaly detector on the same Samsung 5G RAN telemetry.
