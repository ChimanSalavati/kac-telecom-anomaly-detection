# E6 - TelecomTS zero-shot frontier LLM probes (Table 3 TelecomTS rows)

Three vendor notebooks score TelecomTS windows directly from the raw
KPI matrix without training. Each one needs the corresponding API key
in the environment:

| Notebook | Environment variable |
|---|---|
| `E6a_TelecomTS_GPT_zero_shot.ipynb`    | `OPENAI_API_KEY` |
| `E6b_TelecomTS_Gemini_zero_shot.ipynb` | `GOOGLE_API_KEY` |
| `E6c_TelecomTS_Claude_zero_shot.ipynb` | `ANTHROPIC_API_KEY` |

The model IDs documented in the notebook headers match the model
families reported in the paper. None of the API keys are stored in
this repository; the notebooks read them from the environment.

## How to run

```bash
cd experiments/E6_TelecomTS_LLM_zero_shot
export OPENAI_API_KEY=...
jupyter nbconvert --to notebook --execute E6a_TelecomTS_GPT_zero_shot.ipynb --inplace
```

Each notebook writes per-window predictions to `results/` and the
aggregated metrics that feed Table 3.
