# Data availability

The paper reports results on three benchmarks. Two are public and reproduced end-to-end by the experiments in this repository. The third, **ProdTrace-SA**, is a Nokia collaboration benchmark built from real 5G RAN production packet captures with realism-hardened injected anomaly families. It cannot be redistributed; the repository contains no ProdTrace-SA data, no ProdTrace-SA checkpoints, and no ProdTrace-SA derivatives.

## TelecomTS (public, reproducible from this repo)

- Source: HuggingFace Datasets, dataset id `AliMaatouk/TelecomTS` (Feng et al., 2025).
- Acquisition: `python scripts/download_telecomts.py` fetches the corpus into `experiments/_shared/cache/telecomts/`.
- Footprint after caching: roughly 150 MB.
- Used by every `E1`-`E6` and `E11` notebook.

## SpotLight (public, reproducible from this repo)

- Source: the public release that accompanies the MobiCom 2024 paper by Sun et al. The repository ships a download helper that pulls the same NPZ split layout (`SpotLight_train.npz`, `SpotLight_val.npz`, `SpotLight_test.npz`) the paper uses.
- Acquisition: `python scripts/download_spotlight.py`. After the splits are present, `python scripts/compute_chronos_residuals.py --dataset spotlight` builds the Chronos-2 residual cache consumed by `E7`-`E10` and `E12`.
- Footprint: ~1 GB of residual cache on disk after the script finishes. The cache is deterministic given the Chronos-2 release pinned in `requirements.txt`.

## ProdTrace-SA (Nokia production-traffic benchmark, not redistributable)

- Source: real 5G RAN production packet captures from a major telecom equipment vendor, collected from operational network environments. Because naturally labeled attack incidents are unavailable, eight realism-hardened anomaly families are injected into the production traffic (DataExfiltration, SignalingFlood, PortScan, SMPPSpam, TCPAnomalies, ExactPacketReplay, SeqNumberManipulation, and DuplicateBurst), and the same Nokia GKE-based KPI-construction infrastructure converts packet traces into 64 KPI windows. Train, validation, and test windows are split by disjoint PCAPs to avoid capture-level leakage.
- Restriction: the PCAP captures and derived KPI-window NPZ files are Nokia-internal and cannot be released. The exact figures and tables that depend on ProdTrace-SA are reported in the paper from the authors' internal runs.
- Reproducibility claim: the KAC architecture, the SOTA baseline list, the foundation-model probes, the ablation, and the LLM zero-shot evaluation are run by the public-benchmark notebooks in this repository using exactly the same code path. Readers can therefore audit the methodology end-to-end on TelecomTS and SpotLight; only the ProdTrace-SA-specific numbers are not re-runnable from a clean checkout.

If you are a Nokia collaborator with appropriate access and want to reproduce the ProdTrace-SA rows, contact Chiman Salavati (chiman.salavati@uconn.edu) and Liang Wu (liang.wu@nokia.com).
