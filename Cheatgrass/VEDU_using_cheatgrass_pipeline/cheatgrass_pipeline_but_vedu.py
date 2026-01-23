# =================================================================================
# CHEATGRASS DATA PROCESSING PIPELINE (refactored & robust)
# - Stable, unique global_id
# - "cheatgrass" naming (no VEDU)
# - Diagnostics + experimental set generation
# =================================================================================

import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split

from validmask_utils import (
    build_validmask_dataset,
    DEFAULT_SRC_ROOT as DEFAULT_VALIDMASK_SRC_ROOT,
    DEFAULT_VALID_ROOT,
)

warnings.filterwarnings("ignore", message="The get_cmap function was deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

# ----------------------------- CONFIG -------------------------------------------
DEFAULT_INPUT_PATH = "/home/rbielski/SAL_Git_Projects/Ventenata/Ventenata_Files/merge2324.shp"
INPUT_PATH = os.getenv("CHEATGRASS_INPUT_PATH", DEFAULT_INPUT_PATH)
print(f"Input dataset: {INPUT_PATH} (override with CHEATGRASS_INPUT_PATH)")

PERCENT_VARIATIONS = [50, 75, 90, 100]
AREA_VARIATIONS = [900, 8100]  # m² thresholds
TARGET_CRS = "EPSG:32611"

# Treat <= this as "missing" InfestSqM so we can fallback to geometry area
MIN_GEOFALLBACK_AREA = 1.0

CHEATGRASS_ALIASES = {
    "VEDU",
    "VENTENATA",
    "VENTENATA DUBIA",
    "VENTENATA  DUBIA",
}

def _normalize_sciname(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()  # squeeze spaces
    return s

CHEATGRASS_ALIASES_NORMALIZED = {_normalize_sciname(a) for a in CHEATGRASS_ALIASES}


# ----------------------------- ID HELPERS ---------------------------------------
import numpy as np
import pandas as pd
import re

def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_-]+', '-', s)
    s = s.strip('-')
    return s or 'X'

def _uniqify_series(s: pd.Series) -> pd.Series:
    counts = {}
    out = []
    for v in s.astype(str):
        if v in counts:
            counts[v] += 1
            out.append(f"{v}-{counts[v]:02d}")
        else:
            counts[v] = 0
            out.append(v)
    return pd.Series(out, index=s.index)

def _build_gid(gdf: 'gpd.GeoDataFrame') -> pd.Series:
    # start from existing column if present
    if 'global_id' in gdf.columns:
        gid = gdf['global_id'].astype('string')
    else:
        gid = pd.Series([pd.NA] * len(gdf), index=gdf.index, dtype='string')

    # normalize and null-out null-like strings
    gid = gid.str.strip()
    gid = gid.where(~gid.str.lower().isin(['', 'none', 'nan', 'null']), pd.NA)

    # fallback 1: any common ID-ish column
    for cand in ['OBJECTID', 'objectid', 'ObjectID', 'FID', 'fid']:
        if cand in gdf.columns:
            gid = gid.fillna(gdf[cand].astype('string'))
            break

    # fallback 2: stable row-based id  (*** FIX: use a Series, not an Index ***)
    row_ids = gdf.index.to_series().map(lambda i: f"ROW-{int(i):06d}")
    gid = gid.fillna(row_ids.astype('string'))

    # sanitize and ensure uniqueness
    gid = gid.map(_sanitize_token)
    gid = _uniqify_series(gid)

    return gid.astype('string')



# ----------------------------- LOAD & PREP --------------------------------------
def load_and_prepare_cheatgrass(filepath: str) -> gpd.GeoDataFrame:
    print("\n--- Loading Cleaned Cheatgrass Dataset ---")
    gdf = gpd.read_file(filepath)
    print(f"✅ Loaded {len(gdf)} rows | {len(gdf.columns)} columns")

    # Species name normalized
    if "SciName" in gdf.columns:
        gdf["primary_sp"] = gdf["SciName"].apply(_normalize_sciname)
    else:
        gdf["primary_sp"] = "UNKNOWN"

    # % cover
    if "primary_sp_percent" not in gdf.columns:
        if "Percentcov_num" in gdf.columns:
            gdf["primary_sp_percent"] = gdf["Percentcov_num"].fillna(0)
        elif "Percentcov" in gdf.columns:
            gdf["primary_sp_percent"] = pd.to_numeric(gdf["Percentcov"], errors="coerce").fillna(0)
        else:
            gdf["primary_sp_percent"] = 0.0

    # Area in m² from acres (if provided)
    if "InfestSqM" not in gdf.columns:
        if "InfestAcre_clean" in gdf.columns:
            gdf["InfestSqM"] = gdf["InfestAcre_clean"].fillna(0) * 4046.86
        else:
            gdf["InfestSqM"] = 0.0

    # Cheatgrass presence flag
    if "is_cheatgrass" not in gdf.columns:
        gdf["is_cheatgrass"] = gdf["primary_sp"].apply(
            lambda x: 1 if any(a in x for a in CHEATGRASS_ALIASES_NORMALIZED) else 0
        )
    gdf["is_cheatgrass"] = gdf["is_cheatgrass"].fillna(0).astype(int)

    # Valid geometries only
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # Always (re)build robust global_id
    gdf["global_id"] = _build_gid(gdf)

    # Diagnostics
    print(f"✅ Geometry-valid rows: {len(gdf)}")
    print("Alias match positives (is_cheatgrass==1):", int((gdf["is_cheatgrass"] == 1).sum()), "/", len(gdf))
    preview_cols = [
        c
        for c in ["global_id", "primary_sp", "primary_sp_percent", "InfestSqM", "is_cheatgrass"]
        if c in gdf.columns
    ]
    print(gdf[preview_cols].head())
    return gdf


def _diagnostic_counts(df: pd.DataFrame, label: str):
    if df.empty:
        return
    print(f"[{label}] class balance is_cheatgrass ->", df["is_cheatgrass"].value_counts().to_dict())
    if "primary_sp_percent" in df.columns:
        print(
            f"[{label}] primary_sp_percent stats -> "
            f"min={df.primary_sp_percent.min()} max={df.primary_sp_percent.max()}"
        )
    if "InfestSqM" in df.columns:
        print(
            f"[{label}] InfestSqM summary -> "
            f"count>0={(df.InfestSqM > 0).sum()} zeros/NaN={(df.InfestSqM.isna() | (df.InfestSqM <= 0)).sum()}"
        )


# ----------------------- EXPERIMENTAL SET GENERATION ----------------------------
def generate_experimental_datasets(
    filepath: str,
    percent_thresholds: list,
    area_thresholds: list,
    target_crs: str = TARGET_CRS,
    include_controls: bool = False,   # default False (your file appears to be all cheatgrass)
    test_size: float = 0.2,
    random_state: int = 42,
):
    base_gdf = load_and_prepare_cheatgrass(filepath)
    _diagnostic_counts(base_gdf, "BASE")

    # Split (fallback to random if only one class or tiny dataset)
    if base_gdf["is_cheatgrass"].nunique() < 2 or len(base_gdf) < 4:
        print("⚠ Not enough class diversity or too few rows for stratified split; using random split.")
        master_train_gdf, final_test_set = train_test_split(
            base_gdf, test_size=min(0.5, test_size), random_state=random_state
        )
    else:
        master_train_gdf, final_test_set = train_test_split(
            base_gdf,
            test_size=test_size,
            stratify=base_gdf["is_cheatgrass"],
            random_state=random_state,
        )

    print("\n--- Master Sets Created ---")
    print(f"Master Training Set: {len(master_train_gdf)} | Final Test Set: {len(final_test_set)}")
    _diagnostic_counts(master_train_gdf, "TRAIN")

    # Reproject + robust area fallback
    reprojected = master_train_gdf.to_crs(target_crs)
    reprojected["geom_area_sqm"] = reprojected.geometry.area

    if "InfestSqM" in reprojected.columns:
        # use InfestSqM unless missing/too small, otherwise geometry area
        reprojected["area_sqm"] = reprojected["InfestSqM"].where(
            ~(reprojected["InfestSqM"].isna() | (reprojected["InfestSqM"] <= MIN_GEOFALLBACK_AREA)),
            reprojected["geom_area_sqm"],
        )
    else:
        reprojected["area_sqm"] = reprojected["geom_area_sqm"]

    print("\nArea diagnostics (first 5):")
    print(reprojected[["global_id", "InfestSqM", "geom_area_sqm", "area_sqm"]].head())
    print("area_sqm stats ->", reprojected["area_sqm"].describe())

    # Build experimental datasets
    training_datasets = {}
    print("\n--- Generating Experimental Datasets ---")
    all_ctrl_rows = reprojected[reprojected["is_cheatgrass"] == 0]
    if include_controls and all_ctrl_rows.empty:
        print("⚠ include_controls=True but no controls found in this file (is_cheatgrass==0). Proceeding with positives only.")

    for pct in percent_thresholds:
        for area in area_thresholds:
            name = f"percent_{pct}_area_{area}"

            pos_mask = (
                (reprojected["is_cheatgrass"] == 1)
                & (reprojected["primary_sp_percent"] >= pct)
                & (reprojected["area_sqm"] >= area)
            )
            pos_subset = reprojected[pos_mask]

            if include_controls and not all_ctrl_rows.empty:
                ctrl_subset = all_ctrl_rows[all_ctrl_rows["area_sqm"] >= area]
                combined = pd.concat([pos_subset, ctrl_subset], ignore_index=True)
            else:
                ctrl_subset = reprojected.iloc[0:0].copy()  # empty like reprojected
                combined = pos_subset.copy()

            print(f"\n>> {name}")
            print(f"  - Pos (cheatgrass) count: {len(pos_subset)}")
            print(f"  - Ctrl count: {len(ctrl_subset)}")
            if len(reprojected):
                print(
                    f"  - Pos keep %: {100 * len(pos_subset) / len(reprojected):.1f}% | "
                    f"Ctrl keep %: {100 * len(ctrl_subset) / len(reprojected):.1f}%"
                )

            if len(pos_subset) == 0 and len(ctrl_subset) == 0:
                failing = reprojected[(reprojected["is_cheatgrass"] == 1)]
                if not failing.empty:
                    print("    Diagnostics (positive class candidates):")
                    print(failing[["global_id", "primary_sp_percent", "area_sqm"]])

            training_datasets[name] = combined.reset_index(drop=True)

    print("\n✅ Dataset generation complete.")
    return training_datasets, final_test_set


# ------------------ OPTIONAL: BUILD DATASET WITH VALIDITY MASK ------------------
# See validmask_utils.build_validmask_dataset for details. Expose a simple
# wrapper here so notebooks or scripts importing this module can call it.
def build_validmask_copy(
    src_root: str | Path | None = None,
    dst_root: str | Path | None = None,
) -> int:
    """Wrapper around validmask_utils.build_validmask_dataset."""
    return build_validmask_dataset(
        src_root=src_root or DEFAULT_VALIDMASK_SRC_ROOT,
        dst_root=dst_root or DEFAULT_VALID_ROOT,
    )


# -------------------------------- EXECUTION -------------------------------------
if __name__ == "__main__":
    experimental_sets, final_test_set = generate_experimental_datasets(
        filepath=INPUT_PATH,
        percent_thresholds=PERCENT_VARIATIONS,
        area_thresholds=AREA_VARIATIONS,
        target_crs=TARGET_CRS,
        include_controls=False,  # your file appears to be all cheatgrass
    )

    print(f"\nSummary of generated experimental sets: {len(experimental_sets)} total")
    if experimental_sets:
        first_key = next(iter(experimental_sets.keys()))
        sample_df = experimental_sets[first_key]
        print(f"Example set '{first_key}' rows: {len(sample_df)} | Columns: {list(sample_df.columns)[:12]} ...")

    # Optional: build validity-mask copy of the split dataset
    if os.getenv("BUILD_VALIDMASK_COPY", "0") == "1":
        build_validmask_copy()
