# === Set Earthdata token for downstream cells ===
import os

# *** PASTE YOUR ACTIVE EARTHDATA TOKEN HERE ***
EARTHDATA_LOGIN_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InJ1c3RpbmJpZWxza2kiLCJleHAiOjE3NzAwNTk0MzMsImlhdCI6MTc2NDg3NTQzMywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.4iF2QMd0-4f2EA9NePVlZG14l9O0wB2fWUW6O5iVlbe8tD6_EV15LGFOqYKVmJ6mXYTMJFsJ5h9iDBk64RPtiIHaSnEijSj2LFxM9vW7VuF2dvDTEBQecr1uZQDCWkq7-IAw-YVk6JWoaX2N2sXbhHlJg1h3O9Jt9I1IY7N4TNhpWt2i2BFgEHVtRYFtR_5STb56UgQLleM91XXNKSWu3dCYsEof9rgytouvgbDfIiSu2UUbT2GlWopfucEWLqBGdHjY2ZvccO3ed0d2MGmPHqGR9_YCs0z5otsLEQkKOka5OqnQlLVCwdGQt9nNnzAPYKl8AvSnx4Of3m-kjkFKog"

# Make it available both as:
#  1) An environment variable (for load_earthdata_token / os.getenv calls)
#  2) A plain variable `token` (for any code that accesses `token` directly)
os.environ["EARTHDATA_LOGIN_TOKEN"] = EARTHDATA_LOGIN_TOKEN
token = EARTHDATA_LOGIN_TOKEN

print("Earthdata token set from this cell (env var + token variable).")


# IMPORTANT: Set your Earthdata Login token here or as an environment variable
import os
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import shutil
import rasterio
from rasterio.features import rasterize
from urllib.parse import urlparse
import re
import time
from pathlib import Path
from datetime import datetime
import json
import logging
from urllib3.util.retry import Retry  # added
from requests.adapters import HTTPAdapter  # added
import sys  # NEW

# Initialize Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: Set your Earthdata Login token here or as an environment variable
import os

try:
    from config import EARTHDATA_LOGIN_TOKEN
except ImportError:
    EARTHDATA_LOGIN_TOKEN = ""

# Prefer environment variable, fall back to config.py
token = os.getenv("EARTHDATA_LOGIN_TOKEN", EARTHDATA_LOGIN_TOKEN)

headers = {"Authorization": f"Bearer {token}"} if token else {}

if not token or "YOUR_TOKEN" in token:
    logger.error("CRITICAL: Earthdata Login Token is not set. The script will fail.")

# Build retry-capable HTTP session (handles 429/5xx incl. 502)
def build_http_session(base_headers: dict) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, read=5, connect=5,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=1.5,
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update(base_headers or {})
    s.headers.setdefault("User-Agent", "cheatgrass-pipeline/0.1 (+https://example.org)")
    return s

HTTP = build_http_session(headers)

# Bands & sizing
BANDS_OF_INTEREST = ["B02", "B03", "B04", "B05", "B07", "B11"]
CONTEXT_SIZE_CHEATGRASS = 256
CONTEXT_SIZE_CONTROL = 32
TRAINING_WINDOW_SIZE = 32
NODATA_SENTINEL = -9999.0
MIN_VALID_FRACTION_PATCH = 0.6   # overall patch valid fraction to keep a slice
MIN_VALID_FRACTION_POLY = 0.7    # valid fraction inside polygon mask
DATE_RANGE_START = "2024-05-01"
DATE_RANGE_END   = "2024-10-31"
OUTPUT_DIR = Path("cheatgrass_data")  # align with VEDU tooling
OUTPUT_DIR.mkdir(exist_ok=True)
REQUEST_SLEEP = 0.5  # align with VEDU

# URL helper to handle S3 links gracefully
def to_https_if_s3(url: str) -> str:
    if not url:
        return url
    if url.startswith("s3://"):
        return "https://" + url[len("s3://"):]
    return url

# ----------------------- MISSING HELPERS (ADDED) -----------------------
def determine_context_size(is_cheatgrass: bool) -> int:
    return CONTEXT_SIZE_CHEATGRASS if is_cheatgrass else CONTEXT_SIZE_CONTROL

def download_and_extract_data(url: str, polygon_geom, context_size: int):
    url = to_https_if_s3(url)
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.basename(urlparse(url).path) or 'tmp.tif'
        fpath = os.path.join(td, fname)
        try:
            r = HTTP.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(fpath, 'wb') as f: shutil.copyfileobj(r.raw, f)
            with rasterio.open(fpath) as src:
                poly = gpd.GeoSeries([polygon_geom], crs='EPSG:32611').to_crs(src.crs).iloc[0]
                cx, cy = poly.centroid.x, poly.centroid.y
                row, col = src.index(cx, cy)
                half = context_size // 2
                h, w = src.height, src.width
                c0, r0 = max(0, col - half), max(0, row - half)
                c1, r1 = min(w, col + half), min(h, row + half)
                win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                data = src.read(1, window=win)
                # Guard against empty/degenerate reads (polygon outside granule)
                if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
                    return None, None, None
                if data.shape != (context_size, context_size):
                    pad_h = max(0, context_size - data.shape[0])
                    pad_w = max(0, context_size - data.shape[1])
                    if pad_h or pad_w:
                        nodata = src.nodata if src.nodata is not None else NODATA_SENTINEL
                        out = np.full((context_size, context_size), nodata, dtype=data.dtype)
                        out[:data.shape[0], :data.shape[1]] = data
                        data = out
                return data, src.window_transform(win), src.crs
        except Exception as e:
            logger.error(f'download/extract failed for {url}: {e}')
            return None, None, None

def is_pixel_contaminated(fmask_value: int) -> bool:
    v = int(fmask_value)
    is_cloud = (v & (1 << 1)) > 0
    is_shadow = (v & (1 << 3)) > 0
    is_snow = (v & (1 << 4)) > 0
    aerosol_bits = (v >> 6) & 0b11
    return is_cloud or is_shadow or is_snow or (aerosol_bits == 0b11)

def create_segmentation_mask(polygon_geom, image_transform, image_crs, context_size: int):
    poly = gpd.GeoSeries([polygon_geom], crs='EPSG:32611').to_crs(image_crs).iloc[0]
    return rasterize(
        [(poly, 1)], out_shape=(context_size, context_size), transform=image_transform,
        fill=0, all_touched=True, dtype=rasterio.uint8
    )

def compute_valid_mask(data_stack: np.ndarray, nodata_value: float = NODATA_SENTINEL) -> np.ndarray:
    """Return boolean valid mask for data shaped (T,H,W,C)."""
    finite = np.isfinite(data_stack).all(axis=-1)
    not_nodata = (data_stack > (nodata_value + 1e-3)).all(axis=-1)
    nonzero = (np.abs(data_stack).sum(axis=-1) > 0.0)
    return finite & not_nodata & nonzero

def save_metadata(gid: str, is_cheatgrass: bool, ctx_size: int, n_obs: int, quality_info=None):
    meta = {
        'global_id': gid, 'is_cheatgrass': bool(is_cheatgrass),
        'context_size': int(ctx_size), 'observations_saved': int(n_obs),
        'training_window_size': TRAINING_WINDOW_SIZE,
        'processed_timestamp': datetime.now().isoformat(),
        'quality_info': quality_info or {}
    }
    with open(OUTPUT_DIR / f'{gid}_metadata.json', 'w') as f: json.dump(meta, f, indent=2)

# ------------------------- HELPER FUNCTIONS --------------------------------------
def format_bbox_for_cmr(bbox_geometry, input_crs="EPSG:32611"):
    gs = gpd.GeoSeries([bbox_geometry], crs=input_crs).to_crs(epsg=4326).iloc[0]
    minx, miny, maxx, maxy = gs.bounds
    return f"{minx},{miny},{maxx},{maxy}"

def find_granule_urls_by_date(bbox_geometry, start_date, end_date, max_retries=5, initial_delay=2):
    bbox_wgs84_str = format_bbox_for_cmr(bbox_geometry)
    params = {
        "short_name": "HLSL30",
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": bbox_wgs84_str,
        "page_size": 2000
    }
    retries = 0
    delay = float(initial_delay)
    while retries <= max_retries:
        try:
            resp = HTTP.get("https://cmr.earthdata.nasa.gov/search/granules.umm_json", params=params, timeout=45)
            resp.raise_for_status()
            items = (resp.json() or {}).get("items", [])
            date_grouped_urls = {}
            for item in items:
                umm = item.get("umm", {})
                granule_id = umm.get("GranuleUR", "")
                m = re.search(r"\.(\d{4})(\d{3})T", granule_id)
                if not m:
                    continue
                acq_date = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y-%j").strftime("%Y-%m-%d")
                if acq_date not in date_grouped_urls:
                    date_grouped_urls[acq_date] = {}
                for url_info in (umm.get("RelatedUrls", []) or []):  # FIXED
                    url = url_info.get("URL", "")
                    if url_info.get("Type") == "GET DATA" and url.endswith(".tif"):
                        url = to_https_if_s3(url)
                        if "Fmask" in url:
                            date_grouped_urls[acq_date]["Fmask"] = url
                        else:
                            for band in BANDS_OF_INTEREST:
                                if f".{band}." in url or f"_{band}." in url:
                                    date_grouped_urls[acq_date][band] = url
                                    break
            return date_grouped_urls
        except Exception as e:
            logger.warning(f"CMR request failed (attempt {retries+1}/{max_retries+1}): {e}")
            time.sleep(delay)
            retries += 1
            delay *= 2
    logger.error("Failed to retrieve granules after retries")
    return {}

# ------------------------- CORE PROCESSING ---------------------------------------
def process_unique_polygon_list(unique_polygons_gdf: gpd.GeoDataFrame):
    summary = {"cheatgrass_256x256": 0, "control_32x32": 0, "skipped": 0}
    # removed: seen = set()  # allow duplicates/overlaps
    df = unique_polygons_gdf.reset_index(drop=True)  # ensure sequential indexing
    total = len(df)
    for pos, row in df.iterrows():
        last_debug = {}
        try:
            gid = str(row.get('global_id') or row.name)  # NEW: robust gid string
            # removed: if gid in seen: continue
            # removed: seen.add(gid)
            geom = row['geometry']
            val = row.get('is_cheatgrass', row.get('is_vedu_present', 0))
            is_cg = str(val).strip().lower() in ('1', 'true', 't')

            ctx_size = determine_context_size(is_cg)
            ctx_label = "CHEATGRASS (256x256)" if is_cg else "CONTROL (32x32)"
            print(f"[{pos+1}/{total}] {gid} [{ctx_label}]", flush=True)

            data_path = OUTPUT_DIR / f"{gid}_data.npy"
            mask_path = OUTPUT_DIR / f"{gid}_mask.npy"
            meta_path = OUTPUT_DIR / f"{gid}_metadata.json"
            # ---- replace this block in process_unique_polygon_list() ----
            if data_path.exists() and mask_path.exists() and meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    ok_size = int(meta.get("context_size", 0)) == ctx_size
                except Exception:
                    ok_size = False

                if ok_size:
                    print("  already processed (correct context); skipping")
                    continue
                else:
                    print(f"  found existing chips with wrong context -> purging and reprocessing (wanted {ctx_size})")
                    for p in (data_path, mask_path, meta_path):
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass
            # -------------------------------------------------------------


            print("  querying CMR...", flush=True)
            granule_dates = find_granule_urls_by_date(geom, DATE_RANGE_START, DATE_RANGE_END)
            print(f"  dates found: {len(granule_dates)}", flush=True)
            if not granule_dates:
                print("  no granules found; skip polygon", flush=True)
                summary['skipped'] += 1
                continue

            obs_data, obs_masks = [], []
            skipped_partial = 0
            skipped_contaminated = 0
            skipped_invalid = 0

            for date, band_urls in sorted(granule_dates.items()):
                # Light per-date heartbeat for first two polygons only
                if pos < 2:
                    print(f"    date {date}: start", flush=True)
                if "Fmask" not in band_urls:
                    continue
                fmask_chip, _, _ = download_and_extract_data(band_urls['Fmask'], geom, ctx_size)
                if fmask_chip is None:
                    last_debug = {'step': 'fmask_none'}
                    skipped_partial += 1; continue
                fmask_arr = np.asarray(fmask_chip)
                # guard against degenerate shapes
                if fmask_arr.ndim != 2 or fmask_arr.shape[0] == 0 or fmask_arr.shape[1] == 0:
                    last_debug = {'step': 'fmask_shape', 'fmask_shape': getattr(fmask_arr, 'shape', None)}
                    skipped_partial += 1; continue
                center = min(ctx_size // 2, fmask_arr.shape[0]-1, fmask_arr.shape[1]-1)
                try:
                    contaminated = is_pixel_contaminated(fmask_arr[center, center])
                except Exception as ex:
                    last_debug = {'step': 'fmask_index', 'fmask_shape': getattr(fmask_arr, 'shape', None), 'err': str(ex)}
                    skipped_partial += 1; continue
                if contaminated:
                    skipped_contaminated += 1; continue
                bands_stack = []
                final_tr = final_crs = None
                all_ok = True
                for band in BANDS_OF_INTEREST:
                    if band not in band_urls: all_ok = False; break
                    bdata, tr, crs = download_and_extract_data(band_urls[band], geom, ctx_size)
                    if bdata is None or bdata.ndim != 2 or bdata.shape[0] == 0 or bdata.shape[1] == 0:
                        last_debug = {'step': 'band_shape', 'band': band, 'band_shape': (None if bdata is None else np.asarray(bdata).shape)}
                        all_ok = False; break
                    bands_stack.append(bdata)
                    if final_tr is None: final_tr, final_crs = tr, crs
                if not all_ok or final_tr is None:
                    skipped_partial += 1; continue
                chip = np.stack(bands_stack, axis=-1)
                mask = create_segmentation_mask(geom, final_tr, final_crs, ctx_size)

                valid = compute_valid_mask(chip, NODATA_SENTINEL)  # (H,W)
                valid_frac_patch = float(valid.mean())
                valid_frac_poly = float(valid[mask.astype(bool)].mean()) if mask.sum() else 0.0

                if (valid_frac_patch < MIN_VALID_FRACTION_PATCH) or (valid_frac_poly < MIN_VALID_FRACTION_POLY):
                    skipped_invalid += 1
                    continue

                obs_data.append(chip)
                obs_masks.append(mask)
                time.sleep(REQUEST_SLEEP)

            if obs_data:
                data_arr = np.stack(obs_data, axis=0)         # (T,H,W,C)



                mask_arr = np.stack(obs_masks, axis=0)  # (T,H,W)
                np.save(data_path, data_arr)
                np.save(mask_path, mask_arr)
                save_metadata(
                    gid, is_cg, ctx_size, len(obs_data),
                    quality_info={
                        'observations_saved': len(obs_data),
                        'skipped_partial': skipped_partial,
                        'skipped_contaminated': skipped_contaminated,
                        'skipped_invalid': skipped_invalid,
                        'total_dates_processed': len(granule_dates)
                    }
                )
                if is_cg: summary['cheatgrass_256x256'] += 1
                else:      summary['control_32x32'] += 1
                print(f"  saved {len(obs_data)} obs | data {data_arr.shape} mask {mask_arr.shape}", flush=True)
            else:
                print("  no clear observations kept", flush=True)
                summary['skipped'] += 1
        except Exception as e:
            logger.exception(f"Polygon {row.get('global_id','?')} failed: {e} | debug={last_debug}")
            summary['skipped'] += 1
            continue

    logger.info("\n================ SUMMARY ===============")
    for k,v in summary.items(): logger.info(f"{k}: {v}")
    logger.info("========================================")

# ------------------------- EXECUTION WRAPPER -------------------------------------

#### REMOVE COMMENTS TO RUN

'''
if 'experimental_sets' in globals() and 'final_test_set' in globals():
    # Run-once guard to prevent duplicate execution/interleaved prints
    if globals().get('_CHEATGRASS_IMAGERY_RUNNING', False):
        print('Imagery ingestion already running; skipped duplicate invocation.')  # NEW
    else:
        _CHEATGRASS_IMAGERY_RUNNING = True  # NEW
        try:
            print("\n--- Preparing unique polygon list for CHEATGRASS imagery ingestion ---")
            # Combine all experiment GDFs + final test set
            train_gdfs = [gdf for gdf in experimental_sets.values() if not gdf.empty]
            if not train_gdfs:
                print("No experimental sets available; abort imagery step.")
            else:
                # Ensure final_test_set CRS matches experiments
                try:
                    exp_crs = train_gdfs[0].crs or 'EPSG:32611'
                    test_reproj = final_test_set.to_crs(exp_crs)
                except Exception:
                    test_reproj = final_test_set
                master = pd.concat(train_gdfs + [test_reproj], ignore_index=True)
                master = master.drop_duplicates(subset=['global_id']).copy()
               # Removed sampling block (previously limited polygon count)
                # Add/normalize presence column ONCE, never overwrite real values
                if 'is_cheatgrass' not in master.columns:
                    if 'is_vedu_present' in master.columns:
                        master = master.rename(columns={'is_vedu_present': 'is_cheatgrass'})
                    else:
                        # if this file is entirely cheatgrass positives, mark them as 1
                        master['is_cheatgrass'] = 1
                # No else/elif that can overwrite the column!

                print(f"Unique polygons: {len(master)} (Cheatgrass positives: {master['is_cheatgrass'].sum()})")
                # Filter out null/empty geometries defensively
                master = master[master.geometry.notna() & ~master.geometry.is_empty]
                if master.empty:
                    print("No valid geometries to process.")
                else:
                    process_unique_polygon_list(master)
        finally:
            _CHEATGRASS_IMAGERY_RUNNING = False  # NEW
else:
    print("Required variables experimental_sets / final_test_set not found. Run previous pipeline cell first.")

    '''
