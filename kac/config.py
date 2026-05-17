"""Centralized experiment configuration with per-dataset presets.

Everything that the old notebooks hard-coded -- learning rates, loss weights,
Chronos context length, batch size, seeds, split file names -- now lives in a
single :class:`ExperimentConfig` dataclass. ``main.py`` builds one of these per
run by:

1. starting from the preset for the chosen ``(dataset)`` (see :data:`PRESETS`),
2. applying scenario-specific tweaks (balanced vs. imbalanced split names),
3. overriding any field from the command line (``--epochs``, ``--lr-head`` ...)
   or from the generic ``--set key=value`` escape hatch.

The dataclass is intentionally flat and JSON-serializable so a copy of the
resolved config is written next to every run's artifacts for provenance.

Design notes
------------
* No heavy imports here (no torch/transformers). This module is imported by the
  smoke tests and by tooling that should stay lightweight.
* The hyper-parameter defaults mirror Section 5.1.5 ("Implementation Details")
  of the paper and ``experiments/_shared/kac_ablation.py``'s ``DATASET_CONFIGS``
  so that ``main.py kac`` reproduces the published headline numbers.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical experiment / dataset / scenario vocabularies
# ---------------------------------------------------------------------------

#: Experiment families. Each maps to a runner in :mod:`kac.runners` and to the
#: notebook(s) it replaces.
EXPERIMENTS: Dict[str, str] = {
    "kac": "Headline KAC model (replaces E1/E2/E7).",
    "ablation": "KAC component ablation V1/V2/V3 (replaces E11/E12).",
    "sota": "Deep MTSAD / one-class baselines (replaces E3/E8).",
    "foundation": "Frozen TSFM encoder + linear probe (replaces E4/E9).",
    "spotlight-baseline": "SpotLight JVGAN/MRPI pipeline baseline (replaces E5).",
    "llm": "Zero-shot LLM prompting baseline (replaces E6/E10).",
}

#: Public datasets. ``production`` (ProdTrace-SA) is Nokia-internal and not
#: shipped, but the code path is kept (see docs/data_availability.md).
DATASETS: List[str] = ["telecomts", "spotlight", "production"]

#: Evaluation scenarios. Only TelecomTS ships an imbalanced stress split
#: (paper Table 2, "Imbalanced test set"); other datasets ignore the flag.
SCENARIOS: List[str] = ["balanced", "imbalanced"]

#: Default 5-seed protocol used everywhere in the paper.
DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1337]


# ---------------------------------------------------------------------------
# The config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Fully-resolved configuration for a single ``main.py`` invocation."""

    # --- what to run --------------------------------------------------------
    experiment: str = "kac"
    dataset: str = "telecomts"
    scenario: str = "balanced"
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    # --- KAC / forecasting hyper-parameters (paper Section 5.1.5) -----------
    chronos_context: int = 20          # forecasting context length C
    alpha_supcon: float = 1.0          # weight on supervised-contrastive loss
    beta_kpi: float = 1.0              # weight on KPI-level contrastive loss
    lr_head: float = 5e-4              # LR for all non-encoder KAC params
    lr_encoder: float = 2e-5           # LR for the unfrozen DistilBERT blocks
    hidden: int = 64                   # model hidden dim d
    proj_dim: int = 128                # contrastive projection dim p
    num_heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.07
    max_len: int = 128                 # max text tokens L
    batch_size: int = 16
    epochs: int = 80
    patience: int = 15
    text_model_name: str = "distilbert-base-uncased"

    # --- experiment-specific knobs -----------------------------------------
    variants: List[str] = field(default_factory=lambda: ["V1", "V2", "V3"])  # ablation
    foundation_model: str = "moment"   # foundation: moment | toto | mantis
    llm_provider: str = "gpt"          # llm: gpt | gemini | claude
    llm_model: Optional[str] = None    # explicit model id; None -> provider default
    sota_methods: List[str] = field(
        default_factory=lambda: ["DCdetector", "TimesNet", "ModernTCN", "MEMTO", "D3R"]
    )

    # --- paths --------------------------------------------------------------
    data_root: Optional[str] = None    # override for the dataset cache directory
    output_root: str = "artifacts"     # all metrics/checkpoints/figures land here
    log_root: str = "logs"             # structured per-run logs land here

    # --- execution ----------------------------------------------------------
    device: str = "auto"               # auto | cpu | cuda | mps
    smoke: bool = False                # tiny synthetic run for CI / sanity checks

    # ------------------------------------------------------------------ utils
    def validate(self) -> "ExperimentConfig":
        if self.experiment not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment {self.experiment!r}; choose from {list(EXPERIMENTS)}."
            )
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {self.dataset!r}; choose from {DATASETS}.")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}; choose from {SCENARIOS}.")
        if self.llm_provider not in {"gpt", "gemini", "claude"}:
            raise ValueError(f"Unknown llm_provider {self.llm_provider!r}.")
        return self

    def split_filenames(self) -> Dict[str, str]:
        """NPZ file names for train/val/test under the dataset cache dir.

        The TelecomTS imbalanced stress split uses a ``_imbalanced`` infix; all
        other dataset/scenario combinations use the plain names. These are
        conventions consumed by :mod:`kac.data`; the raw files are produced by
        ``scripts/download_*.py``.
        """
        prefix = {"telecomts": "TelecomTS", "spotlight": "SpotLight", "production": "Production"}[
            self.dataset
        ]
        infix = "_imbalanced" if (self.dataset == "telecomts" and self.scenario == "imbalanced") else ""
        return {s: f"{prefix}{infix}_{s}.npz" for s in ("train", "val", "test")}

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def run_id(self) -> str:
        """Short, filesystem-safe identifier used for artifact/log directories."""
        bits = [self.experiment, self.dataset, self.scenario]
        if self.experiment == "foundation":
            bits.append(self.foundation_model)
        if self.experiment == "llm":
            bits.append(self.llm_provider)
        if self.smoke:
            bits.append("smoke")
        return "__".join(bits)


# ---------------------------------------------------------------------------
# Per-dataset presets (values match the paper / kac_ablation DATASET_CONFIGS)
# ---------------------------------------------------------------------------

#: Only the fields that differ from :class:`ExperimentConfig` defaults are listed.
PRESETS: Dict[str, Dict[str, Any]] = {
    "telecomts": dict(chronos_context=20, alpha_supcon=1.0, beta_kpi=1.0, lr_head=5e-4),
    "spotlight": dict(chronos_context=20, alpha_supcon=1.0, beta_kpi=1.0, lr_head=5e-4),
    # ProdTrace-SA uses a shorter context and damped contrastive weights.
    "production": dict(chronos_context=8, alpha_supcon=0.1, beta_kpi=0.3, lr_head=1e-3),
}


def _coerce(value: str, like: Any) -> Any:
    """Coerce a CLI string ``value`` to the type of an existing field ``like``."""
    if isinstance(like, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, list):
        parts = [p for p in str(value).replace(",", " ").split() if p]
        if like and isinstance(like[0], int):
            return [int(p) for p in parts]
        return parts
    return value


def build_config(
    experiment: str,
    dataset: str,
    *,
    scenario: str = "balanced",
    overrides: Optional[Dict[str, Any]] = None,
    raw_set: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Construct an :class:`ExperimentConfig` from preset + overrides.

    Parameters
    ----------
    experiment, dataset, scenario
        The core selectors.
    overrides
        Mapping of dataclass-field name -> value coming from explicit CLI flags
        (e.g. ``{"epochs": 5}``). ``None`` values are ignored so unset flags do
        not clobber the preset.
    raw_set
        Generic ``key=value`` strings from ``--set`` for any field not exposed
        as a dedicated flag. Values are coerced to the field's type.
    """
    cfg = ExperimentConfig(experiment=experiment, dataset=dataset, scenario=scenario)

    # 1) apply the per-dataset preset
    for k, v in PRESETS.get(dataset, {}).items():
        setattr(cfg, k, v)

    # 2) apply explicit, typed overrides (skip None == "flag not provided")
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config field {k!r}.")
        setattr(cfg, k, v)

    # 3) apply generic --set key=value overrides last
    for item in raw_set or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}.")
        key, raw = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field {key!r} in --set.")
        setattr(cfg, key, _coerce(raw, getattr(cfg, key)))

    return cfg.validate()
