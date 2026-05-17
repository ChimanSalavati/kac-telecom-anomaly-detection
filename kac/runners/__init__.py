"""Runner registry: maps an experiment name to its runner module.

Every runner module exposes:
* ``run(cfg, logger) -> list[dict]`` -- per-(method/variant, seed) metric rows.
* ``GROUP_KEY`` -- the row column to group on for the across-seed summary.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from ..config import ExperimentConfig


def get_runner(experiment: str) -> Tuple[Callable[[ExperimentConfig, object], List[dict]], str]:
    """Return ``(run_fn, group_key)`` for the given experiment name."""
    if experiment in ("kac", "ablation"):
        from . import kac_runner as m
    elif experiment == "sota":
        from . import sota_runner as m
    elif experiment == "foundation":
        from . import foundation_runner as m
    elif experiment == "spotlight-baseline":
        from . import spotlight_runner as m
    elif experiment == "llm":
        from . import llm_runner as m
    else:
        raise ValueError(f"No runner for experiment {experiment!r}.")
    return m.run, m.GROUP_KEY
