# E10 - SpotLight zero-shot frontier LLM probes (Table 3 SpotLight rows)

Same protocol as E6 but on SpotLight. API keys are read from the
environment; no secrets live in this repository.

| Notebook | Environment variable |
|---|---|
| `E10a_SpotLight_GPT_zero_shot.ipynb`    | `OPENAI_API_KEY` |
| `E10b_SpotLight_Gemini_zero_shot.ipynb` | `GOOGLE_API_KEY` |
| `E10c_SpotLight_Claude_zero_shot.ipynb` | `ANTHROPIC_API_KEY` |

## How to run

```bash
cd experiments/E10_SpotLight_LLM_zero_shot
export OPENAI_API_KEY=...
jupyter nbconvert --to notebook --execute E10a_SpotLight_GPT_zero_shot.ipynb --inplace
```
