# E3 - TelecomTS SOTA baselines (Table 2 SOTA block)

Trained baselines from the vendored `sota_models/` package: DCdetector,
D3R, MEMTO, ModernTCN, TimesNet, plus the classical LOF and Isolation
Forest references. Populates the SOTA block of Table 2 on TelecomTS.

`E3b_TelecomTS_imbalanced_SOTA_baselines.ipynb` repeats the same suite
on the Scenario 2 split used by E2.

## How to run

```bash
cd experiments/E3_TelecomTS_SOTA_baselines
jupyter nbconvert --to notebook --execute E3_TelecomTS_SOTA_baselines.ipynb --inplace
```

Both notebooks import the SOTA implementations from
`experiments/sota_models/`. The model code is a re-implementation of the
official repositories cited in
`experiments/sota_models/__init__.py`.
