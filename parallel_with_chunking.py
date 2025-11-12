import os
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import shape
import time
import tracemalloc
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- Decorators ----------
def monitor_performance(func):
    def wrapper(*args, **kwargs):
        print(f"\nSTARTING: {func.__name__}")
        start_time = time.time()
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.2f} s")
        print(f"Peak memory used by {func.__name__}: {peak / 1024 / 1024:.2f} MB")
        print(f"FINISHED: {func.__name__}\n")
        return result
    return wrapper


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
        print(f"Time taken by {func.__name__}: {end_time - start_time:.2f} s")
        print(f"Peak total memory used by {func.__name__}: {peak_mem_mb:.2f} MB")
        print(f"FINISHED: {func.__name__}\n")
        return result
    return wrapper


# ---------- Helper: Get Chunks ----------
def get_chunks(rows, cols, k):
    """Divide image into k×k chunks."""
    if k == 0:
        return [(r, c, r + 1, c + 1) for r in range(rows) for c in range(cols)]

    row_chunks = np.linspace(0, rows, k + 1, dtype=int)
    col_chunks = np.linspace(0, cols, k + 1, dtype=int)
    chunks = []
    for i in range(k):
        for j in range(k):
            chunks.append((row_chunks[i], col_chunks[j], row_chunks[i + 1], col_chunks[j + 1]))
    return chunks


# ---------- Step-A ----------
def process_chunk_band(i, input_path, gain, offset, chunk, output_path):
    with rasterio.open(input_path) as src:
        row_start, col_start, row_end, col_end = chunk
        band = src.read(i + 1, window=((row_start, row_end), (col_start, col_end)))

    nonzero_mask = band != 0
    band[nonzero_mask] = band[nonzero_mask] * gain - offset

    with rasterio.open(output_path, 'r+') as dst:
        dst.write(band, i + 1, window=((row_start, row_end), (col_start, col_end)))


@monitor_parallel_performance
def apply_gain_offset_parallel_chunked(input_path, output_path, gain=0.9, offset=5, num_bands=65, k=2):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        total_bands = min(num_bands, src.count)
        profile.update(count=total_bands)
        rows, cols = src.height, src.width

    with rasterio.open(output_path, 'w', **profile) as dst:
        pass

    chunks = get_chunks(rows, cols, k)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_chunk_band, i, input_path, gain, offset, chunk, output_path)
            for i in range(total_bands)
            for chunk in chunks
        ]
        for _ in as_completed(futures):
            pass

    print(f"Step-A completed with chunking (k={k}): {output_path}")


# ---------- Step-B ----------
@monitor_performance
def create_mask_and_geojson(input_path, band_index, threshold=100, output_dir="parallel_with_chunking_results"):
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
        print("⚠️ No pixels above threshold found.")


# ---------- Step-C (Parallel + Aggregation) ----------
# def process_chunk_mask(i, input_path, threshold, chunk, output_dir, profile):
#     row_start, col_start, row_end, col_end = chunk
#     with rasterio.open(input_path) as src:
#         band = src.read(i + 1, window=((row_start, row_end), (col_start, col_end)))

#     mask_zeros = (band == 0)
#     mask_data = np.zeros_like(band, dtype=np.uint8)
#     mask_data[band > threshold] = 255
#     mask_data[mask_zeros] = 0

#     chunk_path = os.path.join(output_dir, f"mask_chunk_{i}_{row_start}_{col_start}.tif")

#     local_profile = profile.copy()
#     local_profile.update(count=1, dtype=rasterio.uint8)
#     with rasterio.open(chunk_path, 'w', **local_profile) as dst:
#         dst.write(mask_data, 1)

#     return chunk_path


# @monitor_parallel_performance
# def create_masks_for_first_n_bands_parallel_chunked(input_path, n, threshold=100, output_dir="parallel_with_chunking_results", k=2):
#     os.makedirs(output_dir, exist_ok=True)
#     with rasterio.open(input_path) as src:
#         profile = src.profile.copy()
#         crs = src.crs
#         transform = src.transform
#         rows, cols = src.height, src.width
#         total_bands = min(n, src.count)

#     chunks = get_chunks(rows, cols, k)

#     for i in range(total_bands):
#         print(f"\nProcessing Band {i + 1}/{total_bands}")
#         chunk_paths = []

#         with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#             futures = [
#                 executor.submit(process_chunk_mask, i, input_path, threshold, chunk, output_dir, profile)
#                 for chunk in chunks
#             ]
#             for future in as_completed(futures):
#                 chunk_paths.append(future.result())

#         # Merge chunked mask tiles into one full-size mask
#         src_files_to_mosaic = [rasterio.open(p) for p in chunk_paths]
#         mosaic, out_trans = merge(src_files_to_mosaic)
#         for src in src_files_to_mosaic:
#             src.close()

#         # Save merged mask
#         band_mask_path = os.path.join(output_dir, f"mask_band_{i + 1}.tif")
#         profile.update(transform=out_trans, count=1, dtype=rasterio.uint8)
#         with rasterio.open(band_mask_path, 'w', **profile) as dst:
#             dst.write(mosaic[0], 1)

#         # Create GeoJSON from merged mask
#         shapes_gen = shapes(mosaic[0], mask=(mosaic[0] > 0), transform=out_trans)
#         geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
#         if geoms:
#             gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
#             band_geojson_path = os.path.join(output_dir, f"mask_band_{i + 1}.geojson")
#             gdf.to_file(band_geojson_path, driver="GeoJSON")
#             print(f"✅ Created: {band_mask_path} & {band_geojson_path}")
#         else:
#             print(f"⚠️ Band {i + 1}: No pixels above threshold found.")

#         # Optionally remove temporary chunk files
#         for path in chunk_paths:
#             os.remove(path)

#     print(f"\nStep-C completed with chunking (k={k})")


import tempfile

def process_chunk_mask(i, input_path, threshold, chunk, output_dir, profile):
    """
    Worker: read window of band i, threshold it, write a small temporary tile tif,
    and return its path plus the chunk coords (row_start, col_start, row_end, col_end).
    """
    row_start, col_start, row_end, col_end = chunk
    with rasterio.open(input_path) as src:
        # read only the window
        band = src.read(i + 1, window=((row_start, row_end), (col_start, col_end)))

    mask_zeros = (band == 0)
    mask_data = np.zeros_like(band, dtype=np.uint8)
    mask_data[band > threshold] = 255
    mask_data[mask_zeros] = 0

    # temporary filename in output_dir
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=output_dir)
    os.close(tmp_fd)  # we will open with rasterio
    local_profile = profile.copy()
    # adjust the transform for the chunk's window
    # compute window transform using src transform in parent (we do not have it here),
    # so write the chunk with default transform and encode coords in the filename.
    local_profile.update(count=1, dtype=rasterio.uint8, height=mask_data.shape[1], width=mask_data.shape[2] if mask_data.ndim==3 else mask_data.shape[0])

    # write chunk to tmp_path (we will read it back in parent and write to final)
    with rasterio.open(tmp_path, 'w', **local_profile) as dst:
        # mask_data might be 2D or 3D with singleton band dimension
        if mask_data.ndim == 3:
            dst.write(mask_data[0], 1)
        else:
            dst.write(mask_data, 1)

    # return the path and coordinates so parent can put it correctly
    return tmp_path, (row_start, col_start, row_end, col_end)


@monitor_parallel_performance
def create_masks_for_first_n_bands_parallel_chunked(
    input_path,
    n,
    threshold=100,
    output_dir="parallel_with_chunking_results",
    k=2,
    max_workers=None
):
    """
    New implementation that writes the final band mask incrementally to disk as chunks complete.
    This avoids reading all chunk files into memory and avoids rasterio.merge().
    """
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        rows, cols = src.height, src.width
        total_bands = min(n, src.count)

    chunks = get_chunks(rows, cols, k)

    # conservative default: half of CPU count to avoid too many GDAL handles
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) // 2)

    for i in range(total_bands):
        print(f"\nProcessing Band {i + 1}/{total_bands}")
        band_mask_path = os.path.join(output_dir, f"mask_band_{i + 1}.tif")

        # prepare final band file with zeros (so we can write windows to it)
        out_profile = profile.copy()
        out_profile.update(count=1, dtype=rasterio.uint8, height=rows, width=cols, transform=transform)
        with rasterio.open(band_mask_path, 'w', **out_profile) as dst:
            # write an initial zero array to reserve file on disk
            zeros = np.zeros((rows, cols), dtype=np.uint8)
            # write in one shot might be heavy; instead write row-blocks to avoid huge memory usage
            # but here zeros will be large if image huge; so write row-wise in a loop
            row_block = 1024
            for r in range(0, rows, row_block):
                r_end = min(rows, r + row_block)
                dst.write(zeros[r:r_end, :], 1, window=((r, r_end), (0, cols)))

        # process chunks in parallel (each worker writes a small temp tile)
        chunk_paths = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_chunk_mask, i, input_path, threshold, chunk, output_dir, profile): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                try:
                    tmp_path, (r0, c0, r1, c1) = future.result()
                except Exception as e:
                    print("Worker failed:", e)
                    continue

                # Now parent opens tmp_path, reads data and writes to final band_mask_path in window
                with rasterio.open(tmp_path) as tmp_ds, rasterio.open(band_mask_path, 'r+') as dst:
                    data = tmp_ds.read(1)
                    # write into the final mask using the same window coords
                    dst.write(data, 1, window=((r0, r1), (c0, c1)))

                # remove temp tile right away
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Create GeoJSON from the final mask
        with rasterio.open(band_mask_path) as final_ds:
            # read in reasonably sized windows to avoid one-shot huge read if image huge
            mask_full = final_ds.read(1)
            shapes_gen = shapes(mask_full, mask=(mask_full > 0), transform=final_ds.transform)
            geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
            if geoms:
                gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
                band_geojson_path = os.path.join(output_dir, f"mask_band_{i + 1}.geojson")
                gdf.to_file(band_geojson_path, driver="GeoJSON")
                print(f"✅ Created: {band_mask_path} & {band_geojson_path}")
            else:
                print(f"⚠️ Band {i + 1}: No pixels above threshold found.")

    print(f"\nStep-C completed with chunking (k={k})")



# ---------- Main ----------
if __name__ == "__main__":
    input_image = "sample_image.tif"
    output_dir = "parallel_with_chunking_results"
    os.makedirs(output_dir, exist_ok=True)
    stepA_output = os.path.join(output_dir, "image_after_stepa.tif")

    k = int(input("Enter chunking factor k (0 = pixel-level): "))

    apply_gain_offset_parallel_chunked(input_image, stepA_output, gain=0.9, offset=5, num_bands=65, k=k)
    create_mask_and_geojson(stepA_output, band_index=50, threshold=100, output_dir=output_dir)
    create_masks_for_first_n_bands_parallel_chunked(stepA_output, 20, 100, output_dir=output_dir, k=k)
