"""Path resolution helpers for the per-experiment notebooks.

The notebooks should never use absolute filesystem paths. They locate the
repository root by walking up from their own folder until they find the
``experiments`` directory, and then expose three reusable Paths:

* ``REPO_ROOT`` -- the top of the cloned repository.
* ``EXPERIMENTS_ROOT`` -- ``REPO_ROOT / "experiments"``.
* ``SHARED_CACHE`` -- ``EXPERIMENTS_ROOT / "_shared" / "cache"``, the
  default location where the download/preprocessing scripts place the
  TelecomTS and SpotLight artefacts.

Typical use at the top of a notebook::

    from pathlib import Path
    import sys

    HERE = Path.cwd()
    while HERE.name != "experiments" and HERE.parent != HERE:
        HERE = HERE.parent
    sys.path.insert(0, str(HERE))

    from _shared.notebook_helpers import REPO_ROOT, SHARED_CACHE
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_experiments_root(start: Path) -> Path:
    cur = start.resolve()
    while cur != cur.parent:
        if cur.name == "experiments":
            return cur
        if (cur / "experiments").is_dir():
            return cur / "experiments"
        cur = cur.parent
    raise RuntimeError(
        "Could not locate the experiments/ directory from "
        f"{start!s}. Run notebooks from inside the cloned repo."
    )


EXPERIMENTS_ROOT: Path = _find_experiments_root(Path(__file__).resolve().parent)
REPO_ROOT: Path = EXPERIMENTS_ROOT.parent
SHARED_CACHE: Path = EXPERIMENTS_ROOT / "_shared" / "cache"


def add_experiments_to_path() -> Path:
    """Insert ``experiments/`` at the front of ``sys.path``.

    Returns the path that was inserted so notebooks can log it.
    """
    p = str(EXPERIMENTS_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return EXPERIMENTS_ROOT


def dataset_cache_dir(dataset: str) -> Path:
    """Return the canonical cache directory for one of the public datasets.

    Parameters
    ----------
    dataset:
        ``"telecomts"`` or ``"spotlight"``. The directory is created on
        first access.
    """
    key = dataset.strip().lower()
    if key not in {"telecomts", "spotlight"}:
        raise ValueError(
            f"Unknown public dataset {dataset!r}. "
            "Production data is not redistributed; see docs/data_availability.md."
        )
    path = SHARED_CACHE / key
    path.mkdir(parents=True, exist_ok=True)
    return path
