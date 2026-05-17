# Artifacts

This directory is the **centralized output root** for every `main.py` run.
It is created/populated at run time and is otherwise empty in version control
(everything except this README is git-ignored).

## Layout

Each run writes to `artifacts/<run_id>/`, where `<run_id>` encodes the
experiment, dataset, and scenario (e.g. `kac__telecomts__balanced`,
`ablation__spotlight__balanced`, `foundation__spotlight__balanced__moment`,
`llm__telecomts__balanced__gpt`):

```
artifacts/<run_id>/
├── config.json   # the exact resolved ExperimentConfig for the run
├── metrics.csv   # one row per (method/variant, seed)
└── summary.csv   # mean ± std across seeds
```

Full run transcripts are written to `logs/<run_id>.log`.

## Regenerating

```bash
python main.py kac --dataset telecomts --scenario balanced \
    --seeds 42 123 456 789 1337
```

See the repository `README.md` ("Run experiments") for the full command list.
