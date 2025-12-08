# =============================================================================
# CREATE QUALITY-FILTERED TENSOR DATASET (ROBUST)
# =============================================================================
import os, shutil, json, sys
from pathlib import Path
import pandas as pd
import numpy as np

SOURCE_DIR = Path("cheatgrass_data")
FILTERED_DIR = Path("/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered")
RETAIN_DUPLICATES = True  # set False to keep current unique-only behavior
DUP_SUFFIX_FMT = "__rep{idx:03d}"  # adjust if desired

def _discover_triplets(folder: Path):
    data_ids = {p.stem[:-5] for p in folder.glob('*_data.npy')}
    mask_ids = {p.stem[:-5] for p in folder.glob('*_mask.npy')}
    meta_ids = {p.stem[:-9] for p in folder.glob('*_metadata.json')}
    return sorted(data_ids & mask_ids & meta_ids)

def _sanitize_ids(id_list):
    s = pd.Series(id_list, dtype="object")
    s = s.replace({None: np.nan})
    s = s.dropna()
    s = s.astype(str).str.strip()
    bad_mask = s.str.lower().isin(["", "none", "nan", "null"])
    invalid_ids = s[bad_mask].tolist()
    s = s[~bad_mask]
    return s.tolist(), invalid_ids

def create_filtered_dataset(
    summary_df=None,
    summary_csv=None,
    config=None,
    source_dir=None,
    filtered_dir=None,
    retain_duplicates=None,
    dup_suffix_fmt=None,
):
    global FILTERED_COPIED

    # Resolve notebook / module globals so the function works when called from a notebook
    main_mod = sys.modules.get("__main__")
    if summary_df is None:
        if 'summary_df_interactive' in globals():
            summary_df = globals()['summary_df_interactive']
        elif main_mod and hasattr(main_mod, 'summary_df_interactive'):
            summary_df = getattr(main_mod, 'summary_df_interactive')

    cfg_obj = config or globals().get('cfg')
    if cfg_obj is None and main_mod and hasattr(main_mod, 'cfg'):
        cfg_obj = getattr(main_mod, 'cfg')

    # Fallback: try to load saved summary CSV so users don't have to re-run the quality cell
    if summary_df is None or getattr(summary_df, "empty", True):
        fallback_paths = []
        if summary_csv:
            fallback_paths.append(Path(summary_csv))
        if cfg_obj is not None and hasattr(cfg_obj, "save_dir"):
            fallback_paths.append(Path(cfg_obj.save_dir) / "summary_by_experiment.csv")
        fallback_paths.append(Path("cheatgrass_quality_diagnostics/summary_by_experiment.csv"))
        for cand in fallback_paths:
            if cand.exists():
                try:
                    summary_df = pd.read_csv(cand)
                    # Propagate back to notebook globals for downstream cells
                    globals()['summary_df_interactive'] = summary_df
                    if main_mod:
                        setattr(main_mod, 'summary_df_interactive', summary_df)
                    print(f"ℹ️ Loaded quality results from {cand}")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {cand}: {e}")
        if summary_df is None or getattr(summary_df, "empty", True):
            print("❌ No quality filter results found. Run the interactive quality filter cell first.")
            return None, 0, {}

    source_dir = Path(source_dir or getattr(main_mod, 'SOURCE_DIR', SOURCE_DIR))
    filtered_dir = Path(filtered_dir or getattr(main_mod, 'FILTERED_DIR', FILTERED_DIR))
    retain_duplicates = RETAIN_DUPLICATES if retain_duplicates is None else bool(retain_duplicates)
    dup_suffix_fmt = dup_suffix_fmt or getattr(main_mod, 'DUP_SUFFIX_FMT', DUP_SUFFIX_FMT)

    raw_pass = summary_df.loc[
        summary_df['passed'], 'global_id'
    ].astype(str).tolist()

    total_evaluated = int(len(summary_df))
    raw_pass_count = int(len(raw_pass))

    clean_ids, tossed_ids = _sanitize_ids(raw_pass)
    clean_unique = pd.unique(pd.Series(clean_ids, dtype="object"))

    available = set(_discover_triplets(source_dir))

    # === REPLACE original valid_ids construction with the new variant-aware logic ===
    # old: valid_ids = [gid for gid in clean_ids if gid in available]
    valid_ids = []
    for base_gid in clean_ids:
        # Find any exact matches or variant ids (e.g., "BASE__rep001", "BASE__aug01", etc.)
        variants = [a for a in available if a == base_gid or a.startswith(base_gid + "__")]
        if variants:
            if retain_duplicates:
                valid_ids.extend(variants)
            else:
                # keep only the first variant if duplicates not requested
                valid_ids.append(variants[0])
        # If no variant match, skip (we report missing later)
    # Preserve order and unique values (in case of repeated matches)
    unique_valid_ids = []
    for v in valid_ids:
        if v not in unique_valid_ids:
            unique_valid_ids.append(v)
    valid_ids = unique_valid_ids
    # === END replace ===

    # ...existing code where missing_in_source is computed...
    missing_in_source = sorted(set(clean_unique) - available)
    duplicate_estimate = raw_pass_count - len(pd.unique(pd.Series(raw_pass, dtype="object")))
    print("🎯 Quality Filter Results Summary:")
    print(f"   • Total polygons evaluated: {total_evaluated}")
    print(f"   • Marked as passed (raw):   {raw_pass_count}")
    print(f"   • Sanitized (kept order):   {len(clean_ids)}")
    print(f"   • Unique sanitized IDs:     {len(clean_unique)}")
    # Updated line to reflect variant inclusion
    print(f"   • Valid & present on disk (including variants):  {len(valid_ids)}")
    if duplicate_estimate > 0:
        print(f"   • Duplicates in pass list:  ~{duplicate_estimate}")
    if tossed_ids:
        print(f"   • Dropped invalid IDs (first 10): {tossed_ids[:10]}")
    if missing_in_source:
        print(f"   • Passed but missing on disk (first 10): {missing_in_source[:10]}")

    if len(valid_ids) == 0:
        print("❌ No valid IDs to copy after sanitation/existence checks.")
        return None, 0, {}

    if filtered_dir.exists():
        print(f"🗂️  Removing existing directory: {filtered_dir}")
        shutil.rmtree(filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created filtered dataset directory: {filtered_dir}")

    # Duplicate handling
    occurrence_counts = {}
    copied = 0
    expanded_final_ids = []
    for i, gid in enumerate(valid_ids, 1):
        occurrence_counts[gid] = occurrence_counts.get(gid, 0) + 1
        occ_idx = occurrence_counts[gid] - 1  # 0-based
        if retain_duplicates and occ_idx > 0:
            new_gid = f"{gid}{dup_suffix_fmt.format(idx=occ_idx)}"
        else:
            new_gid = gid

        # Source paths (original gid)
        src_data = source_dir / f"{gid}_data.npy"
        src_mask = source_dir / f"{gid}_mask.npy"
        src_meta = source_dir / f"{gid}_metadata.json"

        if not (src_data.exists() and src_mask.exists() and src_meta.exists()):
            print(f"   ⚠️  Skipping (files missing): {gid}")
            continue

        # Destination paths (new_gid may include suffix)
        dst_data = filtered_dir / f"{new_gid}_data.npy"
        dst_mask = filtered_dir / f"{new_gid}_mask.npy"
        dst_meta = filtered_dir / f"{new_gid}_metadata.json"

        # Copy (or hardlink if you want): shutil.copy2 to duplicate bytes
        shutil.copy2(src_data, dst_data)
        shutil.copy2(src_mask, dst_mask)

        # Patch metadata global_id if duplicated
        try:
            meta = json.loads(src_meta.read_text())
            meta['global_id'] = new_gid  # reflect new id
            meta['duplicate_source_id'] = gid if new_gid != gid else None
            meta['duplicate_index'] = occ_idx if new_gid != gid else 0
            meta['duplicate_total_for_source'] = None  # filled later
            dst_meta.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            shutil.copy2(src_meta, dst_meta)
            print(f"   ⚠️  Metadata patch failed for {new_gid}: {e}")

        copied += 1
        expanded_final_ids.append(new_gid)
        if i <= 10 or i % 50 == 0:
            print(f"   [{i:3d}/{len(valid_ids)}] Copied as: {new_gid}")

    # After loop: finalize duplicate_total_for_source for those with suffix
    # (optional second pass to annotate counts)
    dup_totals = {k: v for k, v in occurrence_counts.items() if v > 1}
    if dup_totals:
        for meta_file in filtered_dir.glob("*_metadata.json"):
            try:
                m = json.loads(meta_file.read_text())
                src_id = m.get('duplicate_source_id') or m.get('global_id')
                if src_id in dup_totals:
                    m['duplicate_total_for_source'] = dup_totals[src_id]
                    meta_file.write_text(json.dumps(m, indent=2))
            except Exception:
                pass

    target_passes = getattr(cfg_obj, 'target_passes', None)
    if target_passes is None:
        target_passes = globals().get('TARGET_PASSES')
    if target_passes is None and main_mod and hasattr(main_mod, 'TARGET_PASSES'):
        target_passes = getattr(main_mod, 'TARGET_PASSES')
    if target_passes is None:
        target_passes = 100

    # Summary report
    summary_report = {
        "creation_timestamp": pd.Timestamp.now().isoformat(),
        "source_directory": str(source_dir),
        "filtered_directory": str(filtered_dir),
        "retain_duplicates": retain_duplicates,
        "duplicate_suffix_format": dup_suffix_fmt if retain_duplicates else None,
        "quality_filter_settings": {
            "buffer_radii_px": getattr(cfg_obj, 'buffer_radii_px', [3, 5]),
            "thresholds": getattr(cfg_obj, 'thresholds', {}),
            "composite_cutoff": getattr(cfg_obj, 'composite_cutoff', 0.10),
            "target_passes": target_passes,
        },
        "statistics": {
            "total_evaluated": total_evaluated,
            "passed_raw": raw_pass_count,
            "sanitized_total": len(clean_ids),
            "unique_sanitized": len(clean_unique),
            "valid_on_disk_source": len(valid_ids),
            "final_files_copied": copied,
            "duplicates_detected": int(sum(v > 1 for v in occurrence_counts.values())),
            "expanded_ids_count": len(expanded_final_ids),
            "unique_final_ids": len(set(expanded_final_ids)),
            "average_composite_score": float(summary_df['composite_score'].mean()),
        },
        "original_pass_ids_sample": raw_pass[:50],
        "final_ids_sample": expanded_final_ids[:50],
    }

    with open(filtered_dir / "quality_filter_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # Save filtered results restricted to expanded IDs (strip suffixes when matching originals)
    summary_df.assign(
        expanded_id=lambda d: d['global_id'].apply(str)
    ).to_csv(filtered_dir / "quality_results_used.csv", index=False)

    FILTERED_COPIED = copied

    print(f"\n✅ FILTERED DATASET CREATED WITH DUPLICATES={'YES' if retain_duplicates else 'NO'}:")
    print(f"   📁 Directory: {filtered_dir}")
    print(f"   📊 Expanded triplets: {len(set(expanded_final_ids))} unique IDs, {copied} total copies")
    print(f"   🔎 Discover triplets now: {len(_discover_triplets(filtered_dir))}")
