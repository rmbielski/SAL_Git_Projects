# --- TRAIN/TEST SPLIT FROM QUALITY-FILTERED DIRECTORY ---
import os
from pathlib import Path
import shutil, random, pandas as pd
from validmask_utils import build_validmask_dataset, DEFAULT_VALID_ROOT

SOURCE_DIR = Path('/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered')
DEST_ROOT  = Path('/home/rbielski/SAL_Git_Projects/Cheatgrass/cheatgrass_data_quality_filtered_split')
TRAIN_RATIO = 0.8
SEED = 42
RESET_DEST = True  # set False to keep and extend existing split
BUILD_VALIDMASK_COPY = True  # set False to skip validity-mask copy
VALIDMASK_DEST_ROOT = DEFAULT_VALID_ROOT

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

ids = _discover_triplets(SOURCE_DIR)
if not ids:
    raise RuntimeError(f'No complete triplets in {SOURCE_DIR}')

if RESET_DEST and DEST_ROOT.exists(): shutil.rmtree(DEST_ROOT)
train_dir = DEST_ROOT / 'train'; train_dir.mkdir(parents=True, exist_ok=True)
test_dir  = DEST_ROOT / 'test';  test_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED); rng.shuffle(ids)
split_idx = max(1, int(len(ids)*TRAIN_RATIO))
train_ids = ids[:split_idx]
test_ids  = ids[split_idx:] or [train_ids.pop()]

for target, subset in ((train_dir, train_ids),(test_dir, test_ids)):
    for gid in subset: _copy_triplet(gid, SOURCE_DIR, target)

manifest = pd.DataFrame([('train',g) for g in train_ids] + [('test',g) for g in test_ids], columns=['split','global_id'])
manifest_path = DEST_ROOT / 'split_manifest.csv'
manifest.to_csv(manifest_path, index=False)
print('Split complete')
print(f'Train: {len(train_ids)} -> {train_dir}')
print(f'Test : {len(test_ids)} -> {test_dir}')
print(f'Manifest: {manifest_path}')

# Optional: build validity-mask copy (adds *_valid.npy per sample)
do_validmask = str(os.getenv("CHEATGRASS_BUILD_VALIDMASK", str(BUILD_VALIDMASK_COPY))).lower() in {"1","true","yes","on"}
if do_validmask:
    valid_dst = Path(os.getenv("CHEATGRASS_VALID_ROOT", VALIDMASK_DEST_ROOT))
    build_validmask_dataset(src_root=DEST_ROOT, dst_root=valid_dst)

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
