# Data availability

The paper reports results on three benchmarks. Two are public and reproduced end-to-end by the experiments in this repository. The third, **LabTrace-SA**, is a Nokia collaboration benchmark derived from controlled Samsung 5G RAN lab PCAPs with injected SYN-flood anomalies. It cannot be redistributed; the repository contains no LabTrace-SA data, no LabTrace-SA checkpoints, and no LabTrace-SA derivatives.

## TelecomTS (public, reproducible from this repo)

- Source: HuggingFace Datasets, dataset id `AliMaatouk/TelecomTS` (Feng et al., 2025).
- Acquisition: `python scripts/download_telecomts.py` fetches the corpus into `experiments/_shared/cache/telecomts/`.
- Footprint after caching: roughly 150 MB.
- Used by every `E1`-`E6` and `E11` notebook.

## SpotLight (public, reproducible from this repo)

- Source: the public release that accompanies the MobiCom 2024 paper by Sun et al. The repository ships a download helper that pulls the same NPZ split layout (`SpotLight_train.npz`, `SpotLight_val.npz`, `SpotLight_test.npz`) the paper uses.
- Acquisition: `python scripts/download_spotlight.py`. After the splits are present, `python scripts/compute_chronos_residuals.py --dataset spotlight` builds the Chronos-2 residual cache consumed by `E7`-`E10` and `E12`.
- Footprint: ~1 GB of residual cache on disk after the script finishes. The cache is deterministic given the Chronos-2 release pinned in `requirements.txt`.

## LabTrace-SA (Nokia lab benchmark, not redistributable)

- Source: Samsung 5G RAN conformance-test PCAP captures collected in a controlled Nokia lab setting. The captures contain no native attacks; SYN-flood anomalies are injected and the same GKE-based KPI-construction infrastructure used in Nokia internal lab workflows converts packet traces into KPI windows.
- Restriction: the PCAP captures and derived KPI-window NPZ files are Nokia-internal and cannot be released. The exact figures and tables that depend on LabTrace-SA are reported in the paper from the authors' internal runs.
- Reproducibility claim: the KAC architecture, the SOTA baseline list, the foundation-model probes, the ablation, and the LLM zero-shot evaluation are run by the public-benchmark notebooks in this repository using exactly the same code path. Readers can therefore audit the methodology end-to-end on TelecomTS and SpotLight; only the LabTrace-SA-specific numbers are not re-runnable from a clean checkout.

If you are a Nokia collaborator with appropriate access and want to reproduce the LabTrace-SA rows, contact Chiman Salavati (chiman.salavati@uconn.edu) and Liang Wu (liang.wu@nokia.com).
