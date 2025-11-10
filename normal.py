import os
import numpy as np
import rasterio
from rasterio import mask
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

import time
import tracemalloc


# ---------- Decorator for Monitoring ----------
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


# ---------- Step-A: Apply Band-wise Gain and Offset ----------
@monitor_performance
def apply_gain_offset(input_path, output_path, gain=0.9, offset=5, num_bands=65):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()  # shape: (bands, rows, cols)
        
        # Apply transformation for first 65 bands
        for i in range(min(num_bands, src.count)):
            band = data[i]
            mask_zeros = (band == 0)
            band = band * gain - offset
            band[mask_zeros] = 0
            data[i] = band
        
        # Save modified image
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    
    print(f"✅ Step-A completed: Saved as {output_path}")


# ---------- Step-B: Thresholding and Vector Conversion ----------
@monitor_performance
def create_mask_and_geojson(input_path, band_index, threshold=100, output_dir="normal_results"):
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, "mask.tif")
    geojson_path = os.path.join(output_dir, "mask.geojson")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read(band_index + 1)  # band indices are 1-based
        
        mask_zeros = (data == 0)
        mask_data = np.where(data > threshold, 255, 0).astype(np.uint8)
        mask_data[mask_zeros] = 0
        
        profile.update(count=1, dtype=rasterio.uint8)
        
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask_data, 1)
    
    print(f"✅ Step-B mask created: {mask_path}")
    
    # Convert mask to GeoJSON
    shapes_gen = shapes(mask_data, mask=(mask_data > 0), transform=src.transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
    
    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"✅ Step-B GeoJSON created: {geojson_path}")
    else:
        print("⚠️ No pixels above threshold found, empty GeoJSON not created.")


# ---------- Step-C: Apply thresholding for first n bands ----------
@monitor_performance
def create_masks_for_first_n_bands(input_path, n, threshold=100, output_dir="normal_results"):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        
        for i in range(min(n, src.count)):
            band = src.read(i + 1)
            mask_zeros = (band == 0)
            
            mask_data = np.where(band > threshold, 255, 0).astype(np.uint8)
            mask_data[mask_zeros] = 0
            
            profile.update(count=1, dtype=rasterio.uint8)
            mask_path = os.path.join(output_dir, f"mask_{i}.tif")
            geojson_path = os.path.join(output_dir, f"mask_{i}.geojson")
            
            with rasterio.open(mask_path, 'w', **profile) as dst:
                dst.write(mask_data, 1)
            
            # Convert to GeoJSON
            shapes_gen = shapes(mask_data, mask=(mask_data > 0), transform=src.transform)
            geoms = [shape(geom) for geom, val in shapes_gen if val == 255]
            
            if geoms:
                gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
                gdf.to_file(geojson_path, driver="GeoJSON")
            else:
                print(f"⚠️ Band {i}: No pixels above threshold, empty GeoJSON not created.")


# ---------- Main Execution ----------
if __name__ == "__main__":
    input_image = "sample_image.tif"
    output_dir = "normal_results"
    stepA_output = os.path.join(output_dir, "image_after_stepa.tif")
    
    # Step A
    apply_gain_offset(input_image, stepA_output, gain=0.9, offset=5, num_bands=65)
    
    # Step B
    create_mask_and_geojson(stepA_output, band_index=50, threshold=100, output_dir=output_dir)
    
    # Step C
    create_masks_for_first_n_bands(stepA_output, n=20, threshold=100, output_dir=output_dir)
