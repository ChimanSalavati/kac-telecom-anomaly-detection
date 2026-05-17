# E5 - SpotLight pipeline as a method, evaluated on TelecomTS

Applies the SpotLight (MobiCom 2024) JVGAN/MRPI pipeline to TelecomTS
windows, populating the "SpotLight" row of Table 2.
`E5b_TelecomTS_imbalanced_SpotLight_baseline.ipynb` repeats the same on
the Scenario 2 split.

## How to run

```bash
cd experiments/E5_TelecomTS_SpotLight_baseline
jupyter nbconvert --to notebook --execute E5_TelecomTS_SpotLight_baseline.ipynb --inplace
```

Checkpoints are not shipped; the notebooks retrain JVGAN/MRPI from
scratch on the TelecomTS train split.
