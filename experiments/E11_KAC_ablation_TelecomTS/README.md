# E11 - KAC component ablation on TelecomTS (Table 6)

Runs the four KAC variants V1 -> V4 on TelecomTS with five seeds
(42, 123, 456, 789, 1337):

| Variant | Text branch | KPI contrastive loss | Uncertainty weights |
|---|---|---|---|
| V1 | off | off | off |
| V2 | on  | off | off |
| V3 | on  | on (uniform) | off |
| V4 (full KAC) | on | on | on |

The driver lives at `experiments/_shared/kac_ablation.py` and is shared
with E12. The notebook is a thin wrapper that calls the driver with the
TelecomTS configuration.

## How to run

```bash
cd experiments/E11_KAC_ablation_TelecomTS
python -m experiments._shared.kac_ablation --dataset TelecomTS \
    --seeds 42 123 456 789 1337
```

or, equivalently:

```bash
jupyter nbconvert --to notebook --execute E11_KAC_ablation_TelecomTS.ipynb --inplace
```

Both routes write `results/kac_ablation.csv` (one row per seed and
variant) and `results/kac_ablation_summary.csv` (mean and std per
variant), which together populate the TelecomTS columns of Table 6.
