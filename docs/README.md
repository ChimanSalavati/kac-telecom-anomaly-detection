# Project website (`docs/`)

A static, dependency-free site that showcases the KAC paper ("KPI-Aware
Multimodal Anomaly Detection for 5G and Open RAN Telemetry", IEEE ICDM 2026
Applied Track), its key results, the deployment, the reproduction commands, and
the logs/artifacts layout.

```
docs/
├── index.html      # the page
├── styles.css      # styling (dark theme)
├── .nojekyll       # serve files as-is (no Jekyll processing)
└── assets/         # bundled figures
    ├── kac_overview.pdf
    └── kac_roc.pdf
```

## Preview locally

```bash
python -m http.server -d docs 8000
# open http://localhost:8000
```

## Publish with GitHub Pages

In the GitHub repo: **Settings → Pages → Build and deployment → Source: Deploy
from a branch**, then select **Branch: `main`**, **Folder: `/docs`**, and save.
The site will be served at:

```
https://chimansalavati.github.io/kac-telecom-anomaly-detection/
```

Equivalently, via the GitHub CLI (requires appropriate permissions):

```bash
gh api -X POST repos/ChimanSalavati/kac-telecom-anomaly-detection/pages \
  -f 'source[branch]=main' -f 'source[path]=/docs'
```
