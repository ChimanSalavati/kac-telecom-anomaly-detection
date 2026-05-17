# E12 - KAC component ablation on SpotLight (Table 6)

Same protocol as E11 but on SpotLight; populates the SpotLight columns
of Table 6.

## How to run

```bash
cd experiments/E12_KAC_ablation_SpotLight
python -m experiments._shared.kac_ablation --dataset SpotLight \
    --seeds 42 123 456 789 1337
```

or, equivalently:

```bash
jupyter nbconvert --to notebook --execute E12_KAC_ablation_SpotLight.ipynb --inplace
```
