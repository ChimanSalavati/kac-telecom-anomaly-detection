#!/usr/bin/env python3
"""KAC unified experiment entry point.

This single script replaces all of the per-experiment Jupyter notebooks. It
takes a benchmark name and an experiment family, builds a fully-resolved
:class:`kac.config.ExperimentConfig` (preset + CLI overrides), runs the matching
runner, and writes every artifact to a centralized location.

Quick examples
--------------
Headline KAC on the balanced TelecomTS split, 5 seeds (reproduces Table 2 KAC):

    python main.py kac --dataset telecomts --scenario balanced \
        --seeds 42 123 456 789 1337

KAC component ablation V1/V2/V3 on SpotLight (Table 6):

    python main.py ablation --dataset spotlight --variants V1 V2 V3

Deep MTSAD baselines on TelecomTS (Table 2 SOTA block):

    python main.py sota --dataset telecomts --methods DCdetector TimesNet D3R

Frozen MOMENT encoder + linear probe on SpotLight (Table 4 foundation block):

    python main.py foundation --dataset spotlight --foundation-model moment

Zero-shot GPT on TelecomTS (Table 3):

    python main.py llm --dataset telecomts --provider gpt

Override any config field with a dedicated flag or the generic --set:

    python main.py kac --dataset telecomts --epochs 40 --lr-head 3e-4 \
        --set beta_kpi=0.5 --set proj_dim=64

Fast offline sanity check of every experiment (no data/GPU/keys needed):

    python main.py kac --dataset telecomts --smoke

Outputs
-------
    artifacts/<run_id>/config.json   resolved configuration
    artifacts/<run_id>/metrics.csv   one row per (method/variant, seed)
    artifacts/<run_id>/summary.csv   mean +/- std across seeds
    logs/<run_id>.log                full run transcript
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from kac.config import DATASETS, EXPERIMENTS, SCENARIOS, build_config
from kac.io import run_dir, save_config, setup_logging, write_metrics, write_summary
from kac.runners import get_runner


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand (each maps to an ExperimentConfig field)."""
    p.add_argument("--dataset", choices=DATASETS, default="telecomts",
                   help="Benchmark to run on (default: telecomts).")
    p.add_argument("--scenario", choices=SCENARIOS, default="balanced",
                   help="Evaluation scenario; 'imbalanced' uses the TelecomTS stress split.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Random seeds (default: 42 123 456 789 1337).")
    p.add_argument("--data-root", default=None,
                   help="Override the dataset cache directory.")
    p.add_argument("--output-root", default=None, help="Artifacts root (default: artifacts).")
    p.add_argument("--log-root", default=None, help="Logs root (default: logs).")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device (default: auto).")
    # Common training overrides (None == use preset/default).
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr-head", type=float, default=None, dest="lr_head")
    p.add_argument("--lr-encoder", type=float, default=None, dest="lr_encoder")
    p.add_argument("--alpha-supcon", type=float, default=None, dest="alpha_supcon")
    p.add_argument("--beta-kpi", type=float, default=None, dest="beta_kpi")
    p.add_argument("--chronos-context", type=int, default=None, dest="chronos_context")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny offline synthetic run (no data/GPU/keys); for CI and sanity checks.")
    p.add_argument("--set", dest="raw_set", action="append", default=[], metavar="key=value",
                   help="Override any other config field, e.g. --set proj_dim=64 (repeatable).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified entry point for the KAC ICDM 2026 experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="experiment", required=True)

    for name in EXPERIMENTS:
        sp = sub.add_parser(name, help=EXPERIMENTS[name])
        _add_common_args(sp)
        if name == "ablation":
            sp.add_argument("--variants", nargs="+", default=None,
                            help="Ablation variants to run (default: V1 V2 V3).")
        if name == "sota":
            sp.add_argument("--methods", nargs="+", default=None, dest="sota_methods",
                            help="SOTA methods (default: DCdetector TimesNet ModernTCN MEMTO D3R).")
        if name == "foundation":
            sp.add_argument("--foundation-model", default=None, dest="foundation_model",
                            choices=["moment", "toto", "mantis"],
                            help="Frozen TSFM backbone (default: moment).")
        if name == "llm":
            sp.add_argument("--provider", default=None, dest="llm_provider",
                            choices=["gpt", "gemini", "claude"],
                            help="LLM provider (default: gpt).")
            sp.add_argument("--llm-model", default=None, dest="llm_model",
                            help="Explicit model id (default: provider default).")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict:
    """Collect the dedicated-flag overrides (skip selectors handled elsewhere)."""
    skip = {"experiment", "dataset", "scenario", "raw_set"}
    out = {}
    for k, v in vars(args).items():
        if k in skip:
            continue
        out[k] = v
    return out


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = build_config(
        experiment=args.experiment,
        dataset=args.dataset,
        scenario=args.scenario,
        overrides=_overrides_from_args(args),
        raw_set=args.raw_set,
    )

    logger = setup_logging(cfg)
    logger.info("Resolved config:\n%s", cfg.to_json())
    save_config(cfg)

    run_fn, group_key = get_runner(cfg.experiment)
    rows = run_fn(cfg, logger)

    metrics_path = write_metrics(cfg, rows)
    summary_path = write_summary(cfg, rows, group_key)
    logger.info("Wrote %s (%d rows)", metrics_path, len(rows))
    logger.info("Wrote %s", summary_path)
    logger.info("Artifacts in %s", run_dir(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
