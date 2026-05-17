# E4 - TelecomTS foundation-model baselines (Table 2 TSFM block)

Frozen-encoder + linear-probe baselines on TelecomTS using MOMENT, TOTO,
and Mantis. The companion notebook
`E4b_TelecomTS_imbalanced_Foundation_Models.ipynb` repeats them on the
Scenario 2 imbalanced split.

## How to run

```bash
cd experiments/E4_TelecomTS_Foundation_Models
jupyter nbconvert --to notebook --execute E4_TelecomTS_Foundation_Models.ipynb --inplace
```

The notebooks load each foundation model from HuggingFace at runtime;
no checkpoints are shipped in the repository.
