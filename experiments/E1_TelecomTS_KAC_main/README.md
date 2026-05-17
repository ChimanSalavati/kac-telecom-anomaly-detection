# E1 - TelecomTS KAC headline run (Table 2 main row + Figure 2)

Reproduces the KAC, QR-TAN, Chronos-2+LR, and rule-based rows of Table 2
on the balanced TelecomTS split, plus the ROC curve exported as
`results/roc_zoomed_top4.pdf` and reused as Figure 2 of the paper.

## What you need

- The TelecomTS NPZ splits in `../_shared/cache/telecomts/` (created by
  `python scripts/download_telecomts.py`).
- The Chronos-2 residual cache in
  `../_shared/cache/telecomts/features_cache/` (created by
  `python scripts/compute_chronos_residuals.py --dataset telecomts`).

## How to run

```bash
cd experiments/E1_TelecomTS_KAC_main
# ln -s ../_shared/cache/telecomts data    # optional; see Quick start in the top-level README
jupyter nbconvert --to notebook --execute E1_TelecomTS_KAC_main.ipynb --inplace
```

The notebook writes per-seed metrics under `results/` and the ROC PDF
that the paper renders as Figure 2.
