import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import time
import tracemalloc
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- Decorator for Sequential Performance ----------
def monitor_performance(func):
    def wrapper(*args, **kwargs):
        print(f"\nSTARTING: {func.__name__}")
        start_time = time.time()
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.2f} seconds")
        print(f"Peak memory used by {func.__name__}: {peak / 1024 / 1024:.2f} MB")
        print(f"FINISHED: {func.__name__}\n")
        return result
    return wrapper


# ---------- Decorator for Parallel Performance ----------
def monitor_parallel_performance(func):
    def wrapper(*args, **kwargs):
        print(f"\nSTARTING (Parallel): {func.__name__}")
        start_time = time.time()
        process = psutil.Process(os.getpid())
        peak_mem_mb = 0
        stop_flag = False

        def track_memory():
            nonlocal peak_mem_mb
            while not stop_flag:
                total = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                peak_mem_mb = max(peak_mem_mb, total / 1024 / 1024)
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=track_memory)
        monitor_thread.start()

        result = func(*args, **kwargs)

        stop_flag = True
        monitor_thread.join()
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.2f} seconds")
        print(f"Peak total memory used by {func.__name__}: {peak_mem_mb:.2f} MB")
        print(f"FINISHED: {func.__name__}\n")
        return result
    return wrapper


# ---------- Step-A (Parallelized): Apply Band-wise Gain and Offset ----------
def process_band_gain_offset(i, input_path, gain, offset):
    with rasterio.open(input_path) as src:
        band = src.read(i + 1)
        nonzero_mask = band != 0
        band[nonzero_mask] = band[nonzero_mask] * gain - offset
        return i, band


def process_and_write(i, gain, offset, input_path, output_path):
    with rasterio.open(input_path) as src_in:
        band = src_in.read(i + 1)
        nonzero_mask = band != 0
        band[nonzero_mask] = band[nonzero_mask] * gain - offset

    # Write directly to the output band
    with rasterio.open(output_path, 'r+') as dst_out:
        dst_out.write(band, i + 1)

    return f"✅ Band {i} processed and written"


@monitor_parallel_performance
def apply_gain_offset_parallel(input_path, output_path, gain=0.9, offset=5, num_bands=65):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read metadata once
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        total_bands = min(num_bands, src.count)
        profile.update(count=total_bands)

    # Create output file once
    with rasterio.open(output_path, 'w', **profile) as dst:
        pass  # Just create an empty file structure

    # Define a worker that writes directly to output file


    # Run each band in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_and_write, i, gain, offset, input_path, output_path) for i in range(total_bands)]
        for future in as_completed(futures):
            print(future.result())

    print(f"Step-A completed (streaming write): {output_path}")




# ---------- Step-B: Thresholding and Vector Conversion ----------
@monitor_performance
def create_mask_and_geojson(input_path, band_index, threshold=100, output_dir="parallel_results"):
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, "mask.tif")
    geojson_path = os.path.join(output_dir, "mask.geojson")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read(band_index + 1)
        mask_zeros = (data == 0)
        mask_data = np.zeros_like(data, dtype=np.uint8)
        mask_data[data > threshold] = 255
        mask_data[mask_zeros] = 0

        profile.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask_data, 1)

    print(f"Step-B mask created: {mask_path}")

    shapes_gen = shapes(mask_data, mask=(mask_data > 0), transform=src.transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"Step-B GeoJSON created: {geojson_path}")
    else:
        print("⚠️ No pixels above threshold found, empty GeoJSON not created.")


# ---------- Step-C (Parallel): Multi-band mask creation ----------
def process_band(i, input_path, threshold, output_dir, profile, transform, crs):
    with rasterio.open(input_path) as inner_src:
        band = inner_src.read(i + 1)

    mask_zeros = (band == 0)
    mask_data = np.zeros_like(band, dtype=np.uint8)
    mask_data[band > threshold] = 255
    mask_data[mask_zeros] = 0

    mask_path = os.path.join(output_dir, f"mask_{i}.tif")
    geojson_path = os.path.join(output_dir, f"mask_{i}.geojson")

    local_profile = profile.copy()
    local_profile.update(count=1, dtype=rasterio.uint8)
    with rasterio.open(mask_path, 'w', **local_profile) as dst:
        dst.write(mask_data, 1)

    shapes_gen = shapes(mask_data, mask=(mask_data > 0), transform=transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        gdf.to_file(geojson_path, driver="GeoJSON")
        return f"✅ Band {i} done ({len(geoms)} polygons)"
    else:
        return f"⚠️ Band {i}: no pixels above threshold"


@monitor_parallel_performance
def create_masks_for_first_n_bands_parallel(input_path, n, threshold=100, output_dir="parallel_results"):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        total_bands = min(n, src.count)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(process_band, i, input_path, threshold, output_dir, profile, transform, crs): i
            for i in range(total_bands)
        }

        for future in as_completed(futures):
            print(future.result())


# ---------- Main Execution ----------
if __name__ == "__main__":
    input_image = "sample_image.tif"
    output_dir = "parallel_results"
    stepA_output = os.path.join(output_dir, "image_after_stepa.tif")

    # Step A (parallel)
    apply_gain_offset_parallel(input_image, stepA_output, gain=0.9, offset=5, num_bands=65)

    # Step B
    create_mask_and_geojson(stepA_output, band_index=50, threshold=100, output_dir=output_dir)

    # Step C (parallel)
    create_masks_for_first_n_bands_parallel(stepA_output, 20, 100, output_dir=output_dir)
