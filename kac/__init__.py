"""KAC: centralized package for the ICDM 2026 Applied Track artifact.

This package replaces the per-experiment Jupyter notebooks with a single,
configuration-driven entry point (:mod:`main`) plus a small set of importable
runners. Every experiment that used to be a notebook is now a subcommand of
``main.py`` and reuses the same model, loss, and data code under
``experiments/_shared`` and ``experiments/sota_models``.

Sub-modules
-----------
* :mod:`kac.config`  -- dataclass config + per-dataset presets, fully overridable.
* :mod:`kac.io`      -- centralized run directories, logging, and artifact writers.
* :mod:`kac.data`    -- real public-split loaders + tiny synthetic data for smoke runs.
* :mod:`kac.runners` -- one runner per experiment family (kac / ablation / sota /
  foundation / spotlight_baseline / llm).
"""

__all__ = ["config", "io", "data", "runners"]

__version__ = "1.0.0"
