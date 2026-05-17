# E2 - TelecomTS imbalanced (Scenario 2)

Same KAC pipeline as E1 but on the 190/10 imbalanced TelecomTS split
described in the paper as Scenario 2. Populates the Scenario 2 columns
of Table 2.

## What you need

- The TelecomTS NPZ splits (see E1 README).
- The Chronos-2 residual cache (see E1 README).
- The imbalanced split index lists produced by the loader in the
  notebook's first section.

## How to run

```bash
cd experiments/E2_TelecomTS_imbalanced_KAC
jupyter nbconvert --to notebook --execute E2_TelecomTS_imbalanced_KAC.ipynb --inplace
```
