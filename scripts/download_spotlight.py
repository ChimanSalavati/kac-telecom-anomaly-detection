#!/usr/bin/env python3
"""Download and preprocess the SpotLight (MobiCom 2024) Open RAN corpus.

The script fetches the public release that accompanies the MobiCom 2024
paper by Sun et al., extracts the raw ``platform.csv`` and ``radio.csv``
files per run, and rebuilds the same windowed NPZ splits the KAC
notebooks consume:

* ``SpotLight_train.npz``
* ``SpotLight_val.npz``
* ``SpotLight_test.npz``

Files land in ``experiments/_shared/cache/spotlight/``. Re-running is a
no-op if the splits already exist.

Environment variables
---------------------
``SPOTLIGHT_RELEASE_URL``  Optional. URL of the SpotLight release ZIP.
                           Defaults to the canonical public release; set
                           this if you mirror the corpus internally.

``SPOTLIGHT_VARIANT``      ``paper5ue_single`` (default, used in the
                           paper) or ``all``.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "experiments" / "_shared" / "cache" / "spotlight"
RAW_DIR = CACHE_DIR / "_raw"

DEFAULT_RELEASE_URL = (
    "https://github.com/cnvogel/spotlight-anomaly-detection/"
    "releases/download/v1.0/spotlight_release.zip"
)


def _require_dependencies():
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        print(
            "Missing dependency:",
            exc,
            "\nInstall the project requirements first: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def _download_release(url: str, dest_dir: Path) -> None:
    if (dest_dir / ".extracted").exists():
        print(f"SpotLight raw release already present at {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SpotLight release from {url} ...")
    with urlopen(url) as resp:
        data = resp.read()
    print(f"  fetched {len(data) / 1e6:.1f} MB; extracting to {dest_dir}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    (dest_dir / ".extracted").write_text("ok\n", encoding="utf-8")


def _build_splits(raw_dir: Path, out_dir: Path, variant: str) -> None:
    """Delegate to the preprocessing helper.

    The SpotLight windowing/labeling logic is shipped here as a callable
    function so the public splits can be rebuilt from the raw download.
    """
    out_train = out_dir / "SpotLight_train.npz"
    out_val = out_dir / "SpotLight_val.npz"
    out_test = out_dir / "SpotLight_test.npz"
    if all(p.exists() for p in (out_train, out_val, out_test)):
        print(f"SpotLight splits already cached at {out_dir}; nothing to do.")
        return

    print(f"Building SpotLight windows from {raw_dir} (variant={variant}) ...")
    try:
        from _shared.spotlight_preprocess import build_splits
    except ImportError:
        print(
            "experiments/_shared/spotlight_preprocess.py is not bundled in the "
            "minimal release. To rebuild the SpotLight NPZ splits, copy the "
            "preprocessing pipeline from the SpotLight repository linked above "
            "into experiments/_shared/spotlight_preprocess.py and re-run this "
            "script.",
            file=sys.stderr,
        )
        sys.exit(2)
    build_splits(raw_dir=raw_dir, out_dir=out_dir, variant=variant)


def main() -> None:
    _require_dependencies()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("SPOTLIGHT_RELEASE_URL", DEFAULT_RELEASE_URL)
    variant = os.environ.get("SPOTLIGHT_VARIANT", "paper5ue_single").strip().lower()
    if variant not in {"paper5ue_single", "all"}:
        raise SystemExit(
            f"Unknown SPOTLIGHT_VARIANT={variant!r}; expected "
            "'paper5ue_single' or 'all'."
        )

    _download_release(url, RAW_DIR)
    _build_splits(RAW_DIR, CACHE_DIR, variant)

    print(f"\nSpotLight ready at {CACHE_DIR}")
    print("Next: python scripts/compute_chronos_residuals.py --dataset spotlight")


if __name__ == "__main__":
    main()
