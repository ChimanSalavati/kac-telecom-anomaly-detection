"""Offline smoke tests for every experiment subcommand.

Each test invokes ``main.main([...])`` with ``--smoke`` so the full
config -> data -> runner -> artifact path executes on tiny synthetic data with
no downloaded datasets, GPU, or API keys. We assert the run returns 0 and writes
the standardized artifacts (config.json, metrics.csv, summary.csv).

These cover the notebook-equivalent runs:
    kac                -> E1 / E2 / E7
    ablation           -> E11 / E12
    sota               -> E3 / E8
    foundation         -> E4 / E9
    spotlight-baseline -> E5
    llm                -> E6 / E10

Run with:  pytest -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Run everything offline and from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import main as cli  # noqa: E402


# (argv, run_id) for each experiment family. Small datasets/scenarios keep CI fast.
CASES = [
    (["kac", "--dataset", "telecomts", "--scenario", "balanced"], "kac__telecomts__balanced__smoke"),
    (["kac", "--dataset", "telecomts", "--scenario", "imbalanced"], "kac__telecomts__imbalanced__smoke"),
    (["kac", "--dataset", "spotlight"], "kac__spotlight__balanced__smoke"),
    (["ablation", "--dataset", "telecomts", "--variants", "V1", "V2", "V3"], "ablation__telecomts__balanced__smoke"),
    (["ablation", "--dataset", "spotlight"], "ablation__spotlight__balanced__smoke"),
    (["sota", "--dataset", "telecomts"], "sota__telecomts__balanced__smoke"),
    (["foundation", "--dataset", "spotlight", "--foundation-model", "moment"], "foundation__spotlight__balanced__moment__smoke"),
    (["spotlight-baseline", "--dataset", "telecomts"], "spotlight-baseline__telecomts__balanced__smoke"),
    (["llm", "--dataset", "telecomts", "--provider", "gpt"], "llm__telecomts__balanced__gpt__smoke"),
    (["llm", "--dataset", "spotlight", "--provider", "claude"], "llm__spotlight__balanced__claude__smoke"),
]


@pytest.fixture(autouse=True)
def _in_tmp_workspace(tmp_path, monkeypatch):
    """Run each test in a temp dir so artifacts/logs do not pollute the repo."""
    monkeypatch.chdir(tmp_path)
    yield


@pytest.mark.parametrize("argv,run_id", CASES, ids=[c[1] for c in CASES])
def test_smoke_experiment(argv, run_id, tmp_path):
    rc = cli.main(argv + ["--smoke"])
    assert rc == 0

    run_dir = tmp_path / "artifacts" / run_id
    assert (run_dir / "config.json").exists(), "config.json not written"
    metrics = run_dir / "metrics.csv"
    summary = run_dir / "summary.csv"
    assert metrics.exists() and metrics.stat().st_size > 0, "metrics.csv missing/empty"
    assert summary.exists() and summary.stat().st_size > 0, "summary.csv missing/empty"
    assert (tmp_path / "logs" / f"{run_id}.log").exists(), "run log not written"


def test_ablation_emits_three_variants(tmp_path):
    """The ablation must report exactly V1/V2/V3 (paper Table 6)."""
    rc = cli.main(["ablation", "--dataset", "telecomts", "--smoke"])
    assert rc == 0
    summary = (tmp_path / "artifacts" / "ablation__telecomts__balanced__smoke" / "summary.csv").read_text()
    for code in ("V1", "V2", "V3"):
        assert code in summary
    assert "V4" not in summary
