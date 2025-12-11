# --- TRAIN/TEST SPLIT FROM QUALITY-FILTERED DIRECTORY ---
import os
from pathlib import Path
import shutil, random, pandas as pd
import numpy as np
from validmask_utils import build_validmask_dataset, DEFAULT_VALID_ROOT, DEFAULT_NODATA

SOURCE_DIR = Path('/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered')
DEST_ROOT  = Path('/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered_split')
TRAIN_RATIO = 0.8
SEED = 42
RESET_DEST = True  # set False to keep and extend existing split
BUILD_VALIDMASK_COPY = True  # set False to skip validity-mask copy
VALIDMASK_DEST_ROOT = DEFAULT_VALID_ROOT
NODATA_VALUE = DEFAULT_NODATA

def _discover_triplets(folder: Path):
    data_ids = {p.stem[:-5] for p in folder.glob('*_data.npy')}
    mask_ids = {p.stem[:-5] for p in folder.glob('*_mask.npy')}
    meta_ids = {p.stem[:-9] for p in folder.glob('*_metadata.json')}
    return sorted(data_ids & mask_ids & meta_ids)

def _copy_triplet(gid: str, src: Path, dst: Path):
    for suf in ('_data.npy','_mask.npy','_metadata.json'):
        sp = src / f'{gid}{suf}'
        if not sp.exists():
            raise FileNotFoundError(f'Missing {sp}')
        shutil.copy2(sp, dst / sp.name)

def _peek_shapes(folder: Path, gid: str):
    data_path = folder / f"{gid}_data.npy"
    mask_path = folder / f"{gid}_mask.npy"
    data = np.load(data_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"{data_path.name}: expected data shape (T,H,W,C), got {data.shape}")
    if mask.ndim == 3:
        if mask.shape[-2:] != tuple(data.shape[1:3]):
            raise ValueError(f"{mask_path.name}: expected spatial shape {(data.shape[1], data.shape[2])}, got {mask.shape}")
        if mask.shape[0] not in (data.shape[0], 1):
            raise ValueError(f"{mask_path.name}: expected T dimension {data.shape[0]} (or 1), got {mask.shape[0]}")
    elif mask.ndim == 2:
        if mask.shape != tuple(data.shape[1:3]):
            raise ValueError(f"{mask_path.name}: expected spatial shape {(data.shape[1], data.shape[2])}, got {mask.shape}")
    else:
        raise ValueError(f"{mask_path.name}: unsupported mask ndim={mask.ndim}")
    return data.shape, mask.shape


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def run_split_from_quality(
    source_dir: Path | str = SOURCE_DIR,
    dest_root: Path | str = DEST_ROOT,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    reset_dest: bool = RESET_DEST,
    build_validmask_copy: bool = BUILD_VALIDMASK_COPY,
    validmask_dest_root: Path | str = VALIDMASK_DEST_ROOT,
    nodata_value: float = NODATA_VALUE,
):
    """Split quality-filtered tensors into train/test and optionally build validity masks."""
    source_dir = Path(source_dir)
    dest_root = Path(dest_root)
    validmask_dest_root = Path(validmask_dest_root)
    train_ratio = float(train_ratio)
    seed = int(seed)
    reset_dest = bool(reset_dest)
    build_validmask_copy = bool(build_validmask_copy)

    ids = _discover_triplets(source_dir)
    if not ids:
        raise RuntimeError(f"No complete triplets in {source_dir}")

    sample_shape, mask_shape = _peek_shapes(source_dir, ids[0])
    print("\n--- Train/Test split from quality-filtered dataset ---")
    print(f"Source: {source_dir} (found {len(ids)} samples)")
    print(f"Sample tensor shapes -> data: {sample_shape}, mask: {mask_shape}")

    if reset_dest and dest_root.exists():
        shutil.rmtree(dest_root)
    train_dir = dest_root / 'train'; train_dir.mkdir(parents=True, exist_ok=True)
    test_dir  = dest_root / 'test';  test_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed); rng.shuffle(ids)
    split_idx = max(1, int(len(ids)*train_ratio))
    train_ids = ids[:split_idx]
    test_ids  = ids[split_idx:] or [train_ids.pop()]

    for target, subset in ((train_dir, train_ids),(test_dir, test_ids)):
        for gid in subset: _copy_triplet(gid, source_dir, target)

    manifest = pd.DataFrame([('train',g) for g in train_ids] + [('test',g) for g in test_ids], columns=['split','global_id'])
    manifest_path = dest_root / 'split_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    print('Split complete')
    print(f'Train: {len(train_ids)} -> {train_dir}')
    print(f'Test : {len(test_ids)} -> {test_dir}')
    print(f'Manifest: {manifest_path}')

    # Optional: build validity-mask copy (adds *_valid.npy per sample)
    do_validmask = _bool_env("CHEATGRASS_BUILD_VALIDMASK", build_validmask_copy)
    if do_validmask:
        valid_dst = Path(os.getenv("CHEATGRASS_VALID_ROOT", validmask_dest_root))
        n_valid = build_validmask_dataset(
            src_root=dest_root,
            dst_root=valid_dst,
            nodata_value=nodata_value,
            reset_dst=True,
        )
        print(f"Validity masks built: {n_valid} samples -> {valid_dst}")
    else:
        print("Validity-mask copy skipped (set CHEATGRASS_BUILD_VALIDMASK=1 to enable).")

    # Append diagnostic block (end of the CREATE QUALITY-FILTERED TENSOR DATASET cell)
    print("\n[Post-run diagnostic] Summary counts & missing IDs check")

    try:
        raw_pass_len = len(raw_pass) if 'raw_pass' in globals() else None
        unique_len = len(clean_unique) if 'clean_unique' in globals() else None
        valid_len = len(valid_ids) if 'valid_ids' in globals() else None
        copied_val = globals().get('FILTERED_COPIED', None)

        print(f" - raw_pass_count: {raw_pass_len}")
        print(f" - unique_sanitized_ids: {unique_len}")
        print(f" - valid_on_disk (valid_ids len): {valid_len}")
        print(f" - files_copied (reported): {copied_val}")

        # list missing unique sanitized IDs vs available files (quick check)
        if 'clean_unique' in globals() and 'available' in globals():
            clean_set = set(clean_unique)
            avail_set = set(available)
            # also compute 'base' of available by removing suffixes like __rep###
            avail_bases = set([a.split("__")[0] for a in avail_set])
            missing_exact = sorted(clean_set - avail_set)
            missing_base = sorted(clean_set - avail_bases)
            print(f" - exact missing (passed but no exact file): {len(missing_exact)} (sample 10): {missing_exact[:10]}")
            print(f" - base-miss (passed but no base-match among available names): {len(missing_base)} (sample 10): {missing_base[:10]}")

            # Try case-insensitive / punctuation-insensitive fuzzy lookup for missed ids
            def normalize_id(s):
                import re
                return re.sub(r'[^a-z0-9]', '', str(s).lower())
            norm_avail_map = {}
            for a in avail_set:
                norm = normalize_id(a)
                norm_avail_map.setdefault(norm, []).append(a)

            fuzzy_candidates = {}
            import difflib
            for mid in missing_base[:200]:  # limit length for speed
                nm = normalize_id(mid)
                # check direct normalized matches
                if nm in norm_avail_map:
                    fuzzy_candidates[mid] = norm_avail_map[nm]
                    continue
                # fuzzy match using difflib
                key_list = list(norm_avail_map.keys())
                matches = difflib.get_close_matches(nm, key_list, n=3, cutoff=0.7)
                if matches:
                    # map back to actual avail names
                    cand = []
                    for k in matches:
                        cand.extend(norm_avail_map[k])
                    fuzzy_candidates[mid] = cand[:6]

            print(f" - Fuzzy candidates for missing (sample 10):")
            for i, (k, v) in enumerate(fuzzy_candidates.items()):
                if i >= 10: break
                print(f"    {k} -> {v}")

        else:
            print(" - Skipped detailed diff (clean_unique / available not present in globals).")

        # Provide a user-run suggestion to attempt copying near matches
        print("\nSuggested next steps:")
        print("  - Inspect missing IDs: run the above code to list them.")
        print("  - If many are formatting variants, consider normalizing IDs in the upstream pipeline or add mapping.")
        print("  - To attempt an automatic mapping/copy for near matches (manual consent required), set:")
        print("       AUTO_COPY_NEAR_MATCHES = True")
        print("    and re-run this cell. (I will not enable it automatically.)")
    except Exception as e:
        print(f"[Post-run diagnostic] failed: {e}")

    return manifest_path


if __name__ == "__main__":
    run_split_from_quality()
