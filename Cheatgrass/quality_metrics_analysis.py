# =============================================================================
# QUALITY METRICS DIAGNOSTIC ANALYSIS (EFFICIENT WITH PROGRESS UPDATES)
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import sys
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict, List, Tuple

# ------------------------------- CONFIG ---------------------------------

TENSOR_DIR = Path("cheatgrass_data")
EPS = 1e-8
VERBOSE_SAMPLES = 3
PROGRESS_EVERY = 5

# How many polygons do we want to accept, approximately?
TARGET_PASSES = 100  # global Top-N acceptance target

# ULTRA-PERMISSIVE requirements for very small polygons
MIN_DATES = 1
MIN_INTERIOR = 1  # Accept 1-pixel polygons
MIN_RING = 1      # Accept 1-pixel buffer rings
DEFAULT_NODATA_VALUE = -9999.0
DEFAULT_MIN_VALID_FRACTION = 0.6
DEFAULT_MIN_VALID_TIMESLICE = 0.6


# ----------------------------- CONFIG CLASS -----------------------------

@dataclass
class QualityConfig:
    # required
    tensor_dir: Path
    save_dir: Path
    buffer_radii_px: Sequence[int]
    ndvi: Tuple[int, int]                   # (red_idx, nir_idx)

    # basic minimums
    min_dates:      int   = 1
    min_interior_pixels: int = 1
    min_ring_pixels:     int = 1
    min_ring_interior_ratio: float = 0.01

    # NEW: geometric / label filters
    min_mask_pixels: int = 1
    max_mask_pixels: Optional[int] = None
    min_percent_cover: Optional[float] = None
    max_percent_cover: Optional[float] = None
    percent_cover_col: str = "primary_sp_percent"

    # thresholds & weights
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights:    Dict[str, float] = field(default_factory=dict)

    # selection / output behavior
    composite_cutoff: float = 0.10
    plots:  bool = True
    export_json: bool = True
    verbose: bool = False

    # validity mask / nodata handling
    nodata_value: float = DEFAULT_NODATA_VALUE
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION   # fraction of patch that must be valid overall
    min_valid_fraction_timeslice: float = DEFAULT_MIN_VALID_TIMESLICE  # fraction of patch per time slice
    min_valid_fraction_poly: float = DEFAULT_MIN_VALID_FRACTION        # fraction of polygon interior that must be valid

    # soft-pass knobs
    min_metrics_met: int = 3
    critical_metrics: List[str] = field(default_factory=lambda: ['temporal'])
    pass_slack: float = 0.05

    # NEW: Top-N selection (optional)
    use_topn: bool = False
    target_passes: int = 100

    def __post_init__(self):
        # Paths
        self.tensor_dir = Path(self.tensor_dir)
        self.save_dir   = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Types / coercions
        self.buffer_radii_px = [int(r) for r in self.buffer_radii_px]
        if len(self.ndvi) != 2:
            raise ValueError("ndvi must be a (red_idx, nir_idx) 2-tuple.")
        self.ndvi = (int(self.ndvi[0]), int(self.ndvi[1]))

        # Minimums / knobs
        self.min_dates              = int(self.min_dates)
        self.min_interior_pixels    = int(self.min_interior_pixels)
        self.min_ring_pixels        = int(self.min_ring_pixels)
        self.min_ring_interior_ratio = float(self.min_ring_interior_ratio)
        self.min_metrics_met        = int(self.min_metrics_met)
        self.pass_slack             = float(self.pass_slack)
        self.nodata_value           = float(self.nodata_value)
        self.min_valid_fraction     = float(self.min_valid_fraction)
        self.min_valid_fraction_timeslice = float(self.min_valid_fraction_timeslice)
        self.min_valid_fraction_poly = float(self.min_valid_fraction_poly)

        # Selection flags
        self.use_topn      = bool(self.use_topn)
        self.target_passes = int(self.target_passes)

        # Fill defaults if empty
        if not self.thresholds:
            self.thresholds = {
                'contrast':    0.01,
                'effect_size': 0.15,
                'temporal_cv': 1.50,
                'ndvi_diff':   0.005,
                'snr':         0.80,
            }
        if not self.weights:
            # Keep your original weighting if none provided
            self.weights = {
                'contrast': 0.35,
                'effect':   0.10,
                'temporal': 0.15,
                'ndvi':     0.25,
                'snr':      0.15,
            }

        # Light sanity checks
        if not self.tensor_dir.exists():
            raise ValueError(f"Tensor directory does not exist: {self.tensor_dir}")

        missing_thr = {'contrast','effect_size','temporal_cv','ndvi_diff','snr'} - set(self.thresholds)
        if missing_thr:
            raise ValueError(f"Missing thresholds for: {sorted(missing_thr)}")

        missing_wts = {'contrast','effect','temporal','ndvi','snr'} - set(self.weights)
        if missing_wts:
            raise ValueError(f"Missing weights for: {sorted(missing_wts)}")

        # Light patch for percent_cover_col (new field)
        if not hasattr(self, 'percent_cover_col') or not self.percent_cover_col:
            self.percent_cover_col = 'primary_sp_percent'

        # (Optional) ensure weights sum to ~1.0
        s = sum(float(v) for v in self.weights.values())
        if s <= 0:
            raise ValueError("Weights must sum to a positive value.")
        # normalize slightly to be safe (keeps your ratios)
        self.weights = {k: float(v)/s for k, v in self.weights.items()}

    def validate(self) -> bool:
        if not self.tensor_dir.exists():
            raise ValueError(f"Tensor directory does not exist: {self.tensor_dir}")
        return True



# ------------------------------- UTILS ----------------------------------

def disk_struct(radius_px: int) -> np.ndarray:
    """Create circular structuring element for morphological operations"""
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    return (x*x + y*y) <= (radius_px * radius_px)

def make_ring(mask2d: np.ndarray, radius_px: int) -> np.ndarray:
    """Create ring buffer around polygon mask"""
    if radius_px <= 0:
        return np.zeros_like(mask2d, dtype=bool)
    selem = disk_struct(radius_px)
    dil = binary_dilation(mask2d.astype(bool), structure=selem)
    ring = np.logical_and(dil, ~mask2d.astype(bool))
    return ring

def compute_contrast_and_snr(interior: np.ndarray, buffer: np.ndarray) -> Tuple[List[float], float]:
    mean_in = interior.mean(axis=0)
    mean_out = buffer.mean(axis=0)
    std_out = buffer.std(axis=0) + EPS
    contrast = (np.abs(mean_in - mean_out) / (np.abs(mean_out) + EPS)).tolist()
    snr = float(np.mean(np.abs(mean_in - mean_out) / std_out))
    return contrast, snr

def compute_t_stats_and_effect(interior: np.ndarray, buffer: np.ndarray) -> Tuple[List[float], List[float]]:
    effect_sizes, p_values = [], []
    for b in range(interior.shape[1]):
        x, y = interior[:, b], buffer[:, b]
        pooled_std = np.sqrt((x.std()**2 + y.std()**2) / 2.0) + EPS
        effect_sizes.append(abs(x.mean() - y.mean()) / pooled_std)
    pvals = []
    for b in range(interior.shape[1]):
        x, y = interior[:, b], buffer[:, b]
        _, p = stats.ttest_ind(x, y, equal_var=False)
        pvals.append(float(p))
    return effect_sizes, pvals

def compute_temporal_cv_on_interior(data_TxHxWxC: np.ndarray, poly_mask_2d: np.ndarray) -> float:
    T, H, W, C = data_TxHxWxC.shape
    series = []
    for t in range(T):
        pix = data_TxHxWxC[t][poly_mask_2d]
        series.append(float(pix.mean()) if pix.size else 0.0)
    series = np.asarray(series)
    mu, sd = float(np.mean(series)), float(np.std(series))
    return float(sd / (abs(mu) + EPS))

def compute_ndvi_diff(interior: np.ndarray, buffer: np.ndarray, red_idx: int, nir_idx: int) -> float:
    nir_i, red_i = interior[:, nir_idx], interior[:, red_idx]
    nir_o, red_o = buffer[:, nir_idx], buffer[:, red_idx]
    ndvi_i = (nir_i - red_i) / (nir_i + red_i + EPS)
    ndvi_o = (nir_o - red_o) / (nir_o + red_o + EPS)
    return float(abs(ndvi_i.mean() - ndvi_o.mean()))


def compute_valid_mask(data: np.ndarray, nodata_value: float) -> np.ndarray:
    """Return validity mask for data shaped (T,H,W,C)."""
    finite = np.isfinite(data).all(axis=-1)  # (T,H,W)
    not_nodata = (data > (nodata_value + 1e-3)).all(axis=-1)  # reject sentinel fills like -9999
    nonzero = (np.abs(data).sum(axis=-1) > 0.0)
    return (finite & not_nodata & nonzero)

def scale_metric(value: float, threshold: float, *, higher_is_better: bool) -> float:
    if higher_is_better:
        return float(min(1.0, value / (threshold + EPS)))
    else:
        return float(min(1.0, threshold / (value + EPS)))


# ---------- TOP-N CALIBRATION (use ranking; avoids cutoff tie problems) ----------

def calibrate_topn(summary_df: pd.DataFrame, target_pass: int = 100) -> float:
    if summary_df.empty:
        summary_df['rank'] = []
        return 0.0

    scores = summary_df['composite_score'].to_numpy(copy=False)
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    summary_df['rank'] = ranks

    N = max(1, min(target_pass, len(scores)))
    nth_score = float(scores[order[N - 1]])
    return nth_score





# --------------------------- CORE EVALUATION ----------------------------

def evaluate_polygon(
    exp_name: str,
    gid: str,
    data: np.ndarray,  # (T,H,W,C)
    mask: np.ndarray,  # (T,H,W)
    cfg: QualityConfig,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Soft-pass refactor:
      - Pass if composite >= cutoff AND (>=K of 5 metrics meet threshold within 'pass_slack')
        AND all 'critical_metrics' meet threshold.
      - Adds 'num_metrics_met' and 'critical_ok' to each row.
    """
    T, H, W, C = data.shape
    poly_mask = (mask.max(axis=0) > 0)
    n_interior = int(poly_mask.sum())

    if T < cfg.min_dates:
        return [], None
    if n_interior < cfg.min_interior_pixels:
        return [], None

    # --- validity / nodata filtering ---
    valid_mask = compute_valid_mask(data, cfg.nodata_value)  # (T,H,W)
    overall_valid_fraction = float(valid_mask.mean())
    if overall_valid_fraction < cfg.min_valid_fraction:
        # Too much missing/invalid data across the patch
        return [], None

    # drop time slices with poor valid coverage
    keep_indices = []
    for t in range(T):
        frac_t = float(valid_mask[t].mean())
        if frac_t >= cfg.min_valid_fraction_timeslice:
            keep_indices.append(t)
    if not keep_indices:
        return [], None
    if len(keep_indices) < T:
        data = data[keep_indices]
        mask = mask[keep_indices]
        valid_mask = valid_mask[keep_indices]
        T = data.shape[0]
        if T < cfg.min_dates:
            return [], None

    # polygon coverage check using aggregated valid mask
    poly_valid_fraction = float(valid_mask.max(axis=0)[poly_mask].mean()) if n_interior else 0.0
    if poly_valid_fraction < cfg.min_valid_fraction_poly:
        return [], None

    # --- soft-pass knobs with safe defaults ---
    K = getattr(cfg, "min_metrics_met", 3)                         # pass if >=K metrics meet
    critical_metrics = getattr(cfg, "critical_metrics", ["temporal"])
    pass_slack = getattr(cfg, "pass_slack", 0.05)                  # allow small shortfall

    # --- thresholds with safe get() fallbacks ---
    thr = cfg.thresholds or {}
    thr_contrast   = float(thr.get("contrast",   0.01))
    thr_effect     = float(thr.get("effect_size",0.15))
    thr_temporal   = float(thr.get("temporal_cv",1.50))
    thr_ndvi       = float(thr.get("ndvi_diff",  0.005))
    thr_snr        = float(thr.get("snr",        0.80))

    per_ring_rows: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, Dict[str, Any]]] = None

    # Stack interior pixels across time
    interior_stack = [data[t][poly_mask] for t in range(T)]
    if not interior_stack or interior_stack[0].size == 0:
        return [], None
    interior_all = np.concatenate(interior_stack, axis=0)

    temporal_cv = compute_temporal_cv_on_interior(data, poly_mask)

    for radius in cfg.buffer_radii_px:
        ring_mask = make_ring(poly_mask, radius)
        n_ring = int(ring_mask.sum())
        if n_ring < cfg.min_ring_pixels or n_ring < int(cfg.min_ring_interior_ratio * n_interior):
            continue

        ring_stack = [data[t][ring_mask] for t in range(T)]
        if not ring_stack or ring_stack[0].size == 0:
            continue
        ring_all = np.concatenate(ring_stack, axis=0)
        if ring_all.size == 0:
            continue

        # --- core metrics ---
        contrast, snr_val = compute_contrast_and_snr(interior_all, ring_all)
        effect, pvals     = compute_t_stats_and_effect(interior_all, ring_all)
        ndvi_delta        = compute_ndvi_diff(interior_all, ring_all, cfg.ndvi[0], cfg.ndvi[1])

        # --- scale each metric against its threshold (1.0 == meets/exceeds) ---
        scaled = dict(
            contrast=float(np.mean([scale_metric(c, thr_contrast,   higher_is_better=True)  for c in contrast])),
            effect  =float(np.mean([scale_metric(abs(d), thr_effect, higher_is_better=True)  for d in effect])),
            temporal=scale_metric(temporal_cv, thr_temporal, higher_is_better=False),
            ndvi    =scale_metric(ndvi_delta, thr_ndvi,     higher_is_better=True),
            snr     =scale_metric(snr_val,    thr_snr,      higher_is_better=True),
        )

        # --- weighted composite (unchanged) ---
        composite = (
            cfg.weights['contrast'] * scaled['contrast'] +
            cfg.weights['effect']   * scaled['effect']   +
            cfg.weights['temporal'] * scaled['temporal'] +
            cfg.weights['ndvi']     * scaled['ndvi']     +
            cfg.weights['snr']      * scaled['snr']
        )

        # --- soft pass logic: K-of-N with critical metrics and slack ---
        meets = {k: (v >= 1.0 - pass_slack) for k, v in scaled.items()}
        num_meet = int(sum(meets.values()))
        critical_ok = all(meets.get(m, False) for m in critical_metrics)
        reasons = [f"{name} < threshold" for name, ok in meets.items() if not ok]

        passed = (composite >= cfg.composite_cutoff) and (num_meet >= K) and critical_ok

        # robust p<0.05 fraction
        p_arr = np.asarray(pvals, dtype=float)
        p_arr = p_arr[~np.isnan(p_arr)]
        p_lt = float((p_arr < 0.05).mean()) if p_arr.size else 0.0

        row = dict(
            experiment=exp_name,
            global_id=str(gid),
            radius_px=int(radius),
            n_interior=n_interior,
            n_ring=n_ring,
            temporal_cv=float(temporal_cv),
            contrast_mean=float(np.mean(contrast)) if len(contrast) else 0.0,
            effect_size_mean=float(np.mean(np.abs(effect))) if len(effect) else 0.0,
            ndvi_diff=float(ndvi_delta),
            snr=float(snr_val),
            composite_score=float(composite),
            passed=bool(passed),  # will be overridden by Top-N calibration below
            p_lt_0_05=p_lt,
            failure_reasons=", ".join(reasons),
            num_metrics_met=num_meet,
            critical_ok=bool(critical_ok),
        )
        per_ring_rows.append(row)

        if (best is None) or (composite > best[0]):
            best = (composite, row)

    return per_ring_rows, (best[1] if best is not None else None)


# ------------------------------- DRIVER ---------------------------------

def run_quality_filter(experimental_sets: Dict[str, Any], cfg: QualityConfig):
    cfg.validate()

    per_pair_rows: List[Dict[str, Any]] = []
    per_poly_rows: List[Dict[str, Any]] = []

    for exp_name, gdf in experimental_sets.items():
        if gdf.empty:
            continue
        gids = gdf['global_id'].astype(str).tolist()

        for gid in tqdm(gids, desc=f"Evaluating {exp_name}"):
            data_path = cfg.tensor_dir / f"{gid}_data.npy"
            mask_path = cfg.tensor_dir / f"{gid}_mask.npy"
            if not (data_path.exists() and mask_path.exists()):
                continue
            try:
                data = np.load(data_path)
                mask = np.load(mask_path)
            except Exception:
                continue

            if data.ndim != 4 or mask.ndim != 3:
                continue

            ring_rows, best_row = evaluate_polygon(exp_name, gid, data, mask, cfg)
            per_pair_rows.extend(ring_rows)
            if best_row is not None:
                per_poly_rows.append(best_row)

    results_df = pd.DataFrame(per_pair_rows)
    summary_df = pd.DataFrame(per_poly_rows)

    # === Auto-calibrate by Top-N (global) ===
    # === Auto-calibrate by Top-N (ranking only; optional selection) ===
    if not summary_df.empty:
        nth = calibrate_topn(summary_df, target_pass=cfg.target_passes)
        cfg.composite_cutoff = nth  # informational

        if cfg.use_topn:
            summary_df['passed'] = summary_df['rank'] <= cfg.target_passes
            print(f"🎯 Ranked by composite; passing Top-{cfg.target_passes}. Nth score ≈ {cfg.composite_cutoff:.3f}")
        else:
            # Keep threshold/soft-pass decision from evaluate_polygon
            print(f"🎯 Ranked by composite for reference. Threshold/soft-pass mode. Nth score (Top-{cfg.target_passes}) ≈ {cfg.composite_cutoff:.3f}")


    # Save after pass/fail is final
    if not results_df.empty:
        results_df.to_csv(cfg.save_dir / 'per_polygon_results.csv', index=False)
    if not summary_df.empty:
        summary_df.to_csv(cfg.save_dir / 'summary_by_experiment.csv', index=False)

    if cfg.export_json and len(per_pair_rows) > 0:
        with open(cfg.save_dir / 'per_polygon_results.json', 'w', encoding='utf-8') as f:
            json.dump(per_pair_rows, f, indent=2)

    if cfg.plots and not results_df.empty and not summary_df.empty:
        fig1, ax1 = plt.subplots()
        ax1.hist(results_df['composite_score'], bins=30)
        ax1.set_title('Composite Score Distribution')
        ax1.set_xlabel('Composite (0–1)'); ax1.set_ylabel('Count')
        fig1.savefig(cfg.save_dir / 'plot_composite_hist.png', dpi=150)

        fig2, ax2 = plt.subplots()
        counts = summary_df['passed'].value_counts()
        ax2.pie([counts.get(True, 0), counts.get(False, 0)], labels=['Pass','Fail'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Pass/Fail (best radius, Top-N)')
        fig2.savefig(cfg.save_dir / 'plot_pass_fail_pie.png', dpi=150)

        fig3, ax3 = plt.subplots()
        bar_vals = [
            results_df.get('contrast_mean', pd.Series([0])).mean(),
            results_df.get('effect_size_mean', pd.Series([0])).mean(),
            results_df.get('temporal_cv', pd.Series([0])).mean(),
            results_df.get('ndvi_diff', pd.Series([0])).mean(),
            results_df.get('snr', pd.Series([0])).mean(),
        ]
        ax3.bar(['Contrast','Effect','TempCV','NDVIΔ','SNR'], bar_vals)
        ax3.set_title('Average Sub-metrics (All Pairs)')
        ax3.set_ylim(0, max(bar_vals)*1.15 if len(bar_vals) and max(bar_vals)>0 else 1)
        fig3.savefig(cfg.save_dir / 'plot_metric_bars.png', dpi=150)

        reasons = results_df['failure_reasons'].str.split(', ').explode().dropna()
        if not reasons.empty:
            fig4, ax4 = plt.subplots()
            counts = reasons.value_counts()
            ax4.bar(counts.index, counts.values)
            ax4.set_title('Failure Reasons (All Pairs)')
            plt.xticks(rotation=30, ha='right')
            fig4.savefig(cfg.save_dir / 'plot_failure_reasons.png', dpi=150)

    # Build filtered sets from final 'passed'
    filtered_experimental_sets: Dict[str, Any] = {}
    if not summary_df.empty:
        for exp_name, gdf in experimental_sets.items():
            sub = summary_df[summary_df["experiment"] == exp_name].copy()
            if sub.empty:
                filtered_experimental_sets[exp_name] = gdf.iloc[0:0].copy()
                continue
            # mask size filters
            if getattr(cfg, "min_mask_pixels", None) is not None:
                sub = sub[sub["n_interior"] >= cfg.min_mask_pixels]
            if getattr(cfg, "max_mask_pixels", None) is not None:
                sub = sub[sub["n_interior"] <= cfg.max_mask_pixels]
            # keep only passed quality
            sub = sub[sub["passed"]]
            if sub.empty:
                filtered_experimental_sets[exp_name] = gdf.iloc[0:0].copy()
                continue
            keep_ids = sub["global_id"].astype(str)
            merged = gdf[gdf["global_id"].astype(str).isin(keep_ids)].copy()
            # Safe fallback if an older cfg exists in memory without the field
            col = getattr(cfg, 'percent_cover_col', 'primary_sp_percent')
            if col in merged.columns:
                if getattr(cfg, 'min_percent_cover', None) is not None:
                    merged = merged[merged[col] >= cfg.min_percent_cover]
                if getattr(cfg, 'max_percent_cover', None) is not None:
                    merged = merged[merged[col] <= cfg.max_percent_cover]
            filtered_experimental_sets[exp_name] = merged
    return results_df, summary_df, filtered_experimental_sets
