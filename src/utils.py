"""
Utility functions for QuakeSlide system.

This module contains helper functions for geospatial processing,
raster operations, and geometry validation.
"""

# Standard library imports
import math
import os

# Third-party imports
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.validation import make_valid, explain_validity


# 保存栅格文件的函数
def save_raster(data, output_path, meta, transform):
    """
    保存 NumPy 数组为栅格文件，参考已有栅格的元信息和地理变换
    """
    # 更新 meta 信息，确保数据类型一致
    meta.update({
        "dtype": str(data.dtype),
        "width": data.shape[1],
        "height": data.shape[0],
        "transform": transform,
        "count": 1
    })

    # 写入栅格文件
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data, 1)





def check_and_fix_geometries(ls_inventory):
    """
    检查并修复无效几何体。

    Args:
        ls_inventory: GeoDataFrame 包含滑坡多边形的库存

    Returns:
        fixed_inventory: 修复后的 GeoDataFrame
    """

    # 检查无效的几何体
    invalid_geometries = ls_inventory[~ls_inventory['geometry'].is_valid]
    if invalid_geometries.empty:
        print("没有无效的几何体。")
    else:
        print(f"发现 {len(invalid_geometries)} 个无效的几何体，正在尝试修复...")

        # 输出无效几何体的详细信息
        for idx, row in invalid_geometries.iterrows():
            print(f"无效几何体索引: {idx}, 问题: {explain_validity(row['geometry'])}")

    # 复制 GeoDataFrame，以免修改原始数据
    fixed_inventory = ls_inventory.copy()

    # 遍历所有几何体，修复无效几何体，并简化复杂几何体
    for idx, row in fixed_inventory.iterrows():
        geom = row['geometry']

        # 修复无效几何体
        if not geom.is_valid:
            try:
                fixed_geom = make_valid(geom)
                if not fixed_geom.is_valid:
                    print(f"几何体在索引 {idx} 仍然无效: {explain_validity(fixed_geom)}")
                    fixed_geom = geom.buffer(0)  # 尝试使用 buffer(0) 修复
                    if not fixed_geom.is_valid:
                        print(f"索引 {idx} 的几何仍然无效，缓冲修复失败。")
                        continue  # 如果修复失败，跳过此几何体
            except Exception as e:
                print(f"修复几何体时出错（索引 {idx}）: {e}")
                continue
        else:
            fixed_geom = geom

        # 更新修复后的几何体
        fixed_inventory.at[idx, 'geometry'] = fixed_geom

    return fixed_inventory


def convert_wgs_to_utm(lon, lat):
    """
    Returns UTM code for lon/lat coordinates
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code


def clip_features_raster_files(factor_names, raster_files, buffered_gdf, output_dir):
    """
    Process raster files by clipping them to the geometry of a buffered GeoDataFrame,
    remove borders with values less than -1e5, and save them with new names.

    Args:
    factor_names (list of str): List of factor names.
    raster_files (list of str): List of paths to raster files.
    buffered_gdf (gpd.GeoDataFrame): Buffered GeoDataFrame containing the geometries for clipping.
    output_dir (str): Directory where the processed files will be saved.

    Returns:
    None
    """
    # Ensure the number of factor names matches the number of raster files
    if len(factor_names) != len(raster_files):
        raise ValueError("The number of factor names must match the number of raster files.")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the buffered GeoDataFrame to WGS84 coordinate system
    buffered_gdf = buffered_gdf.to_crs("EPSG:4326")

    # Get the geometries from the GeoDataFrame
    geometries = [geom for geom in buffered_gdf.geometry]

    # Iterate through each factor name and corresponding raster file
    for i, raster_file in enumerate(raster_files):
        # Open the raster file
        with rasterio.open(raster_file) as src:
            # Use the mask function to clip the raster
            out_image, out_transform = mask(src, geometries, crop=True)
            
            # Remove borders where values are less than -1e5
            out_image, out_transform = remove_low_value_borders(out_image, out_transform, threshold=-1e5)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the clipped raster to the output directory
            out_tif = os.path.join(output_dir, f"Clipped_{factor_names[i]}.tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_image)


def remove_low_value_borders(image, transform, threshold):
    """
    Remove rows and columns where all values are less than the given threshold.

    Args:
    image (numpy.ndarray): The raster image array (bands, height, width).
    transform (Affine): The affine transform of the raster.
    threshold (float): The threshold below which values are considered invalid.

    Returns:
    numpy.ndarray: The cropped raster image array.
    Affine: The updated affine transform.
    """
    # Get a boolean mask of valid values (greater or equal to the threshold)
    valid_mask = image > threshold

    # Combine across all bands to find rows and columns that are not entirely invalid
    valid_rows = np.any(valid_mask, axis=(0, 2))
    valid_cols = np.any(valid_mask, axis=(0, 1))

    # Find the first and last valid row/column
    row_start, row_end = np.where(valid_rows)[0][[0, -1]]
    col_start, col_end = np.where(valid_cols)[0][[0, -1]]

    # Crop the image to the valid area
    cropped_image = image[:, row_start:row_end + 1, col_start:col_end + 1]

    # Update the affine transform to match the cropped image
    new_transform = transform * rasterio.Affine.translation(col_start, row_start)

    return cropped_image, new_transform