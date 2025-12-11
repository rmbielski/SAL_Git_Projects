# =============================================================================
# VALIDITY MASK DATASET BUILDER
# =============================================================================
# Clone an existing dataset tree that contains *_data.npy / *_mask.npy files and
# augment it with *_valid.npy files indicating observed/finite pixels.
#
# Valid pixel definition:
#   * all bands are finite (no NaNs / infs), AND
#   * not equal to nodata sentinel (default -9999), AND
#   * at least one band is non-zero (sum |band| > 0)
#
# Env overrides:
#   CHEATGRASS_SRC_ROOT  -> source dataset root (default: quality filtered split)
#   CHEATGRASS_VALID_ROOT -> destination root for validity-mask copy
# =============================================================================
from __future__ import annotations

import os
import shutil
from pathlib import Path
import numpy as np

DEFAULT_SRC_ROOT = Path("/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered_split")
DEFAULT_VALID_ROOT = Path("/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_validmask_copy")
DEFAULT_NODATA = -9999.0


def _discover_data_files(split_dir: Path):
    return sorted(split_dir.glob("*_data.npy"))


def _as_bool(val, default=False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _compute_valid_mask(data: np.ndarray, nodata_value: float) -> np.ndarray:
    """Return uint8 validity mask for (T,H,W,C) data."""
    if data.ndim != 4:
        raise ValueError(f"Expected data with 4 dims (T,H,W,C), got shape {data.shape}")
    finite = np.isfinite(data).all(axis=-1)      # (T,H,W)
    not_nodata = (data > (nodata_value + 1e-3)).all(axis=-1)  # reject sentinel fills like -9999
    nonzero = (np.abs(data).sum(axis=-1) > 0.0)  # reject all-zero slices
    return (finite & not_nodata & nonzero).astype(np.uint8)


def build_validmask_dataset(
    src_root: str | Path | None = None,
    dst_root: str | Path | None = None,
    nodata_value: float | None = None,
    reset_dst: bool | None = None,
) -> int:
    """
    Clone *_data.npy / *_mask.npy tree and add *_valid.npy files.

    Returns
    -------
    int
        Number of samples written.
    """
    src_root = Path(os.getenv("CHEATGRASS_SRC_ROOT", src_root or DEFAULT_SRC_ROOT))
    dst_root = Path(os.getenv("CHEATGRASS_VALID_ROOT", dst_root or DEFAULT_VALID_ROOT))
    nodata_value = float(os.getenv("CHEATGRASS_NODATA_VALUE", nodata_value if nodata_value is not None else DEFAULT_NODATA))
    if reset_dst is None:
        reset_dst = _as_bool(os.getenv("CHEATGRASS_RESET_VALID_DST"), default=False)
    else:
        reset_dst = _as_bool(reset_dst, default=bool(reset_dst))

    print("\n--- Building validity-mask dataset ---")
    print(f"Source root:      {src_root}")
    print(f"Destination root: {dst_root}")

    if reset_dst and dst_root.exists():
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    n_samples = 0
    for split_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        split_name = split_dir.name
        out_split = dst_root / split_name
        out_split.mkdir(parents=True, exist_ok=True)
        print(f"\n[split: {split_name}]")

        data_files = _discover_data_files(split_dir)
        if not data_files:
            print("  (no *_data.npy files found)")
            continue

        for data_path in data_files:
            base = data_path.stem.replace("_data", "")
            mask_path = data_path.with_name(f"{base}_mask.npy")
            meta_path = data_path.with_name(f"{base}_metadata.json")
            if not mask_path.exists():
                print(f"  ⚠ skipping {base}: no mask file found")
                continue

            data = np.load(data_path).astype(np.float32)
            mask = np.load(mask_path)
            if data.ndim != 4:
                raise ValueError(f"{data_path.name}: expected data shape (T,H,W,C), got {data.shape}")

            valid = _compute_valid_mask(data, nodata_value)

            out_data = out_split / f"{base}_data.npy"
            out_mask = out_split / f"{base}_mask.npy"
            out_valid = out_split / f"{base}_valid.npy"

            np.save(out_data, data)
            np.save(out_mask, mask)
            np.save(out_valid, valid)
            if meta_path.exists():
                shutil.copy2(meta_path, out_split / meta_path.name)

            n_samples += 1

        print(f"  -> wrote {n_samples} samples (cumulative)")

    print(f"\n✅ Done. Total samples with validity masks: {n_samples}")
    print(f"   New dataset root: {dst_root}")
    return n_samples
