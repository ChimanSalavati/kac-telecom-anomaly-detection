# E7 - SpotLight KAC headline run (Table 4)

KAC on the SpotLight Open RAN corpus, producing the KAC row of Table 4
and the SpotLight half of every multi-seed number quoted in the paper.

## What you need

- The SpotLight NPZ splits in `../_shared/cache/spotlight/` (created by
  `python scripts/download_spotlight.py`).
- The Chronos-2 residual cache in
  `../_shared/cache/spotlight/features_cache/` (created by
  `python scripts/compute_chronos_residuals.py --dataset spotlight`).

## How to run

```bash
cd experiments/E7_SpotLight_KAC_main
jupyter nbconvert --to notebook --execute E7_SpotLight_KAC_main.ipynb --inplace
```
