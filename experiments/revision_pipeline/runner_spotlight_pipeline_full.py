"""Run the paper-aligned SpotLight pipeline on the *full* SpotLight
benchmark split (Workstream K), so that the headline SpotLight table
contains a directly comparable result.

This thin wrapper imports the existing SpotLight runner code and points
it at the full split that KAC and the other supervised baselines use.

Usage::

    cd evaluation_ver2/SpotLight
    python ../../ICDM_2026_Applied_Track/paper/revision_pipeline/runner_spotlight_pipeline_full.py
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1] / "evaluation_ver2"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "SpotLight"))

# Re-use the existing runner; we only override the dataset split.
import run_spotlight_baseline_spotlight as runner  # noqa: E402


def main() -> None:
    # The existing runner has a `main()` that reads CLI args; we override
    # them in-process to point at the full SpotLight split.
    sys.argv = [
        "run_spotlight_baseline_spotlight.py",
        "--split", "full",
        "--out-csv", "results/spotlight_pipeline_full.csv",
    ]
    runner.main()


if __name__ == "__main__":
    main()
