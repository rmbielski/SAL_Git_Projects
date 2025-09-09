import os
import re
import ast
import torch
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely import wkt
from shapely.geometry import Polygon
import dask
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar

def download_granule(url, target_crs="EPSG:4326"):
    """Download a granule, reproject if needed, and return as a numpy array."""
    local_filename = url.split('/')[-1]

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with rasterio.open(local_filename) as src:
            band_data = src.read(1)
            transform, crs = src.transform, src.crs

            # Reproject if CRS doesn't match target CRS
            if crs and crs.to_string() != target_crs:
                dst_array = np.empty((src.height, src.width), dtype=band_data.dtype)

                transform, width, height = calculate_default_transform(
                    crs, target_crs, src.width, src.height, *src.bounds
                )

                reproject(
                    source=band_data,
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )

                band_data = dst_array

        os.remove(local_filename)  # Clean up
        return band_data, transform, target_crs

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        return None, None, None

@delayed
def process_granule(url, bbox, target_crs="EPSG:4326"):
    """Download and process a single granule for a given bounding box, ensuring CRS alignment."""
    band_data, transform, crs = download_granule(url, target_crs)

    if band_data is not None:
        # Clip to bounding box
        window = from_bounds(*bbox.bounds, transform=transform)
        row_off, col_off = int(window.row_off), int(window.col_off)
        height, width = int(window.height), int(window.width)

        clipped_band = band_data[row_off: row_off + height, col_off: col_off + width]
        return clipped_band
    return None

def process_granules(granule_urls, bbox, target_crs="EPSG:4326"):
    """Download and process granules for a given bounding box, ensuring CRS alignment."""
    band_arrays = [process_granule(url, bbox, target_crs) for url in granule_urls]
    band_arrays = dask.compute(*band_arrays)
    band_arrays = [ba for ba in band_arrays if ba is not None]
    return band_arrays

def average_bands(band_arrays):
    """Average pixel values across all bands."""
    return np.mean(np.stack(band_arrays, axis=0), axis=0) if band_arrays else None

@delayed
def create_tensor_for_bbox(row):
    """Create a tensor containing averaged bands for a single bounding box."""
    bbox = row['bbox']
    granule_urls = ast.literal_eval(row['data_urls']) if isinstance(row['data_urls'], str) else row['data_urls']
    band_arrays = process_granules(granule_urls, bbox)
    averaged_band = average_bands(band_arrays)
    if averaged_band is not None:
        return torch.tensor(averaged_band, dtype=torch.float32)
    return None

def create_tensor(bbox_df):
    """Create a tensor containing averaged bands for each bounding box."""
    tensors = [create_tensor_for_bbox(row) for _, row in bbox_df.iterrows()]
    tensors = dask.compute(*tensors)
    tensors = [t for t in tensors if t is not None]
    return torch.stack(tensors) if tensors else None

# Load the CSV file
csv_path = "UNET_bboxs_single_bbox.csv"
bbox_df = pd.read_csv(csv_path)

# Ensure CRS is set correctly
bbox_df["bbox"] = bbox_df["bbox"].apply(wkt.loads)

# Create the tensor
with ProgressBar():
    bbox_tensor = create_tensor(bbox_df)

if bbox_tensor is not None:
    print("✅ Tensor shape:", bbox_tensor.shape)
else:
    print("❌ No valid data processed.")