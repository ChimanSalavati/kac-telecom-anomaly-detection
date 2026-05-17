"""Shared utilities for the KAC reproducibility notebooks.

Every per-experiment notebook adds the parent ``experiments/`` directory to
``sys.path`` via :func:`notebook_helpers.add_experiments_to_path` so that
both this package and :mod:`sota_models` resolve cleanly without any
hard-coded absolute paths.
"""
