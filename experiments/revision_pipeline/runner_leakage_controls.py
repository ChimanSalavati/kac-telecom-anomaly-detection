"""TelecomTS text-leakage control experiments (Workstream F).

Three conditions, each trained for 5 seeds with the same KAC architecture
and hyperparameters as the headline runs:

  L1: shuffled-text   - replace each window's text with a random other
                        window's text (text retains style but loses
                        ground-truth correlation).
  L2: no-text         - zero out the text branch entirely (variant V1
                        from the ablation, promoted into this section).
  L3: regen-text      - re-generate TelecomTS summaries using the
                        Production/SpotLight LLM pipeline so the
                        benchmark-provided descriptions are no longer in
                        the training input. Requires ``--regen-text-csv``
                        with a CSV of (window_id, summary) for the
                        TelecomTS train+val+test splits.

Usage::

    cd evaluation_ver2/TelecomTS
    python ../../ICDM_2026_Applied_Track/paper/revision_pipeline/runner_leakage_controls.py \
        --condition L1 --seeds 42 123 456 789 1337
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation_ver2"))
import kac_ablation as KA


def shuffle_texts(texts: List[str], rng: np.random.Generator) -> List[str]:
    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["L1", "L2", "L3"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 456, 789, 1337])
    ap.add_argument("--regen-text-csv", default=None,
                    help="CSV of regenerated summaries for L3")
    args = ap.parse_args()

    cfg = KA.DATASET_CONFIGS["TelecomTS"]
    data = KA.load_dataset(cfg)

    if args.condition == "L3":
        if not args.regen_text_csv:
            raise SystemExit("L3 requires --regen-text-csv")
        import pandas as pd
        df = pd.read_csv(args.regen_text_csv)
        # Expect columns: split (train/val/test), idx, summary
        for split in ("train", "val", "test"):
            sub = df[df.split == split].sort_values("idx")
            data[f"texts_{split}"] = [KA.mask_text(s) for s in sub.summary]

    out_path = Path("results") / f"leakage_{args.condition}.csv"
    out_path.parent.mkdir(exist_ok=True)
    write_header = not out_path.exists()
    fields = ["condition", "seed", "precision", "recall", "f1",
              "auroc", "ap", "wall_seconds"]
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(KA.TEXT_MODEL_NAME)

    for s in args.seeds:
        rng = np.random.default_rng(s)
        local_texts = {}
        for split in ("train", "val", "test"):
            t = data[f"texts_{split}"]
            if args.condition == "L1":
                t = shuffle_texts(t, rng)
            elif args.condition == "L2":
                t = ["" for _ in t]
            local_texts[split] = t

        def make(texts, R, y, W, shuffle):
            enc = tok(texts, padding="max_length", truncation=True,
                      max_length=KA.MAX_LEN, return_tensors="pt")
            ds = KA.ContrastiveDataset(enc["input_ids"], enc["attention_mask"],
                                       R, y, W)
            return DataLoader(ds, batch_size=KA.BATCH_SIZE, shuffle=shuffle)

        train = make(local_texts["train"], data["R_train"],
                     data["y_train"], data["W_train"], True)
        val = make(local_texts["val"], data["R_val"],
                   data["y_val"], data["W_val"], False)
        test = make(local_texts["test"], data["R_test"],
                    data["y_test"], data["W_test"], False)

        model = KA.build_model(data["n_kpis"], data["feats_per_kpi"])
        spec = KA.VARIANTS[3]  # V4 full KAC
        # For L2 (no text) we mimic V1 by setting use_text=False at training
        # time via VariantSpec replacement
        if args.condition == "L2":
            spec = KA.VARIANTS[0]
        t0 = time.perf_counter()
        out = KA.train_variant(
            model, train, val, test, data["y_train"], spec,
            cfg.alpha_supcon, cfg.beta_kpi, cfg.lr_head, seed=s,
        )
        out["condition"] = args.condition
        out["wall_seconds"] = round(time.perf_counter() - t0, 1)
        writer.writerow({k: out[k] for k in fields if k in out}); f.flush()
    f.close()


if __name__ == "__main__":
    main()
