"""KAC headline + component-ablation runner.

Replaces notebooks:
    E1_TelecomTS_KAC_main, E2_TelecomTS_imbalanced_KAC, E7_SpotLight_KAC_main
        -> ``main.py kac ...``         (runs variant V3 = full KAC)
    E11_KAC_ablation_TelecomTS, E12_KAC_ablation_SpotLight
        -> ``main.py ablation ...``    (runs V1, V2, V3)

The actual model, losses, training loop, and data loading are reused verbatim
from ``experiments/_shared/kac_ablation.py`` -- this runner only wires that code
to the centralized config + artifact IO and (in smoke mode) swaps the pretrained
DistilBERT for a tiny randomly-initialized encoder so it runs fully offline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments._shared import kac_ablation as KA

from ..config import ExperimentConfig
from ..data import _cache_dir, make_synthetic
from ._common import resolve_device


def _dataset_config(cfg: ExperimentConfig) -> "KA.DatasetConfig":
    """Build a shared DatasetConfig with absolute paths from our config."""
    cache = _cache_dir(cfg)
    names = cfg.split_filenames()
    return KA.DatasetConfig(
        name={"telecomts": "TelecomTS", "spotlight": "SpotLight", "production": "Production"}[
            cfg.dataset
        ],
        npz_train=str(cache / names["train"]),
        npz_val=str(cache / names["val"]),
        npz_test=str(cache / names["test"]),
        cache_dir=str(cache / "features_cache"),
        npz_kpis_kind="kpis_dict" if cfg.dataset == "production" else "X_array",
        chronos_context=cfg.chronos_context,
        alpha_supcon=cfg.alpha_supcon,
        beta_kpi=cfg.beta_kpi,
        lr_head=cfg.lr_head,
    )


def _selected_variants(cfg: ExperimentConfig) -> List["KA.VariantSpec"]:
    if cfg.experiment == "kac":
        wanted = ["V3"]  # headline KAC == full model
    else:
        wanted = cfg.variants
    return [v for v in KA.VARIANTS if v.code in wanted]


def _build_smoke_model(K: int, feats_per_kpi: int, cfg: ExperimentConfig):
    """Tiny offline KAC model (random-init DistilBERT) for smoke runs."""
    from transformers import DistilBertConfig, DistilBertModel

    enc = DistilBertModel(
        DistilBertConfig(
            vocab_size=1000, dim=32, hidden_dim=64,
            n_layers=2, n_heads=2, max_position_embeddings=64,
        )
    )
    for p in enc.parameters():
        p.requires_grad = False
    for layer in enc.transformer.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    return KA.KPIAwareContrastiveModel(
        text_encoder=enc, resid_dim=K * feats_per_kpi, text_dim=32,
        d_model=cfg.hidden, proj_dim=cfg.proj_dim, n_kpis=K,
        feats_per_kpi=feats_per_kpi, num_heads=cfg.num_heads, dropout=cfg.dropout,
    )


def _run_smoke(cfg: ExperimentConfig, logger) -> List[Dict]:
    data = make_synthetic(cfg)
    K = len(data["kpi_names"])
    feats_per_kpi = 5
    N_tr = len(data["y_train"])
    L = 8
    g = torch.Generator().manual_seed(0)
    tok = {
        split: {
            "input_ids": torch.randint(0, 1000, (len(data[f"y_{split}"]), L), generator=g),
            "attention_mask": torch.ones(len(data[f"y_{split}"]), L, dtype=torch.long),
        }
        for split in ("train", "val", "test")
    }
    KA.DEVICE = "cpu"
    rows: List[Dict] = []
    for v in _selected_variants(cfg):
        loaders = KA.make_loaders(data, tok["train"], tok["val"], tok["test"], seed=0)
        model = _build_smoke_model(K, feats_per_kpi, cfg)
        test = KA.train_variant(
            model, *loaders, data["y_train"], variant=v,
            alpha_supcon=cfg.alpha_supcon, beta_kpi=cfg.beta_kpi,
            lr_head=cfg.lr_head, epochs=1, patience=1, seed=0,
        )
        logger.info("smoke variant %s: F1=%.3f AUROC=%.3f", v.code, test["f1"], test["auroc"])
        rows.append(_variant_row(cfg, v, 0, test, 0.0))
    return rows


def _variant_row(cfg, v, seed, test, elapsed) -> Dict:
    return {
        "dataset": cfg.dataset,
        "scenario": cfg.scenario,
        "variant": v.code,
        "variant_name": v.name,
        "seed": seed,
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "auroc": test["auroc"],
        "ap": test["ap"],
        "wall_seconds": round(elapsed, 1),
    }


def _run_real(cfg: ExperimentConfig, logger) -> List[Dict]:
    from transformers import AutoTokenizer

    KA.DEVICE = resolve_device(cfg.device)
    dscfg = _dataset_config(cfg)
    data = KA.load_dataset(dscfg)
    logger.info(
        "loaded %s: train/val/test=%d/%d/%d K=%d",
        cfg.dataset, len(data["y_train"]), len(data["y_val"]),
        len(data["y_test"]), data["n_kpis"],
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    def _tok(texts):
        return tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=cfg.max_len, return_tensors="pt",
        )

    tok = {s: _tok(data[f"texts_{s}"]) for s in ("train", "val", "test")}

    rows: List[Dict] = []
    for v in _selected_variants(cfg):
        logger.info("--- variant %s: %s ---", v.code, v.name)
        for seed in cfg.seeds:
            t0 = time.time()
            loaders = KA.make_loaders(data, tok["train"], tok["val"], tok["test"], seed=seed)
            model = KA.build_model(data["n_kpis"], data["feats_per_kpi"])
            test = KA.train_variant(
                model, *loaders, data["y_train"], variant=v,
                alpha_supcon=dscfg.alpha_supcon, beta_kpi=dscfg.beta_kpi,
                lr_head=dscfg.lr_head, epochs=cfg.epochs, patience=cfg.patience, seed=seed,
            )
            elapsed = time.time() - t0
            logger.info(
                "  seed=%-5d F1=%.4f AUROC=%.4f AP=%.4f (%.0fs)",
                seed, test["f1"], test["auroc"], test["ap"], elapsed,
            )
            rows.append(_variant_row(cfg, v, seed, test, elapsed))
    return rows


def run(cfg: ExperimentConfig, logger) -> List[Dict]:
    """Entry point used by ``main.py``. Returns per-(variant, seed) rows."""
    return _run_smoke(cfg, logger) if cfg.smoke else _run_real(cfg, logger)


#: Column used to group the per-seed rows into the across-seed summary table.
GROUP_KEY = "variant"
