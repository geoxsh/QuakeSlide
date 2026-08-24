"""
Utility functions for QuakeSlide system.

This module contains helper functions for geospatial processing,
raster operations, and geometry validation.

Date: August 2026
"""

# Standard library imports
import math
import os

# Third-party imports
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.validation import make_valid, explain_validity


def save_raster(data, output_path, meta, transform):
    """Save a NumPy array as a raster using reference metadata and transform."""
    meta.update({
        "dtype": str(data.dtype),
        "width": data.shape[1],
        "height": data.shape[0],
        "transform": transform,
        "count": 1
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data, 1)





def check_and_fix_geometries(ls_inventory):
    """Check and repair invalid geometries in a landslide inventory.

    Args:
        ls_inventory: GeoDataFrame containing landslide polygons.

    Returns:
        GeoDataFrame with repaired geometries.
    """

    invalid_geometries = ls_inventory[~ls_inventory['geometry'].is_valid]
    if invalid_geometries.empty:
        print("No invalid geometries found.")
    else:
        print(
            f"Found {len(invalid_geometries)} invalid geometries; "
            "attempting repairs..."
        )

        for idx, row in invalid_geometries.iterrows():
            print(
                f"Invalid geometry at index {idx}: "
                f"{explain_validity(row['geometry'])}"
            )

    fixed_inventory = ls_inventory.copy()

    for idx, row in fixed_inventory.iterrows():
        geom = row['geometry']

        if not geom.is_valid:
            try:
                fixed_geom = make_valid(geom)
                if not fixed_geom.is_valid:
                    print(
                        f"Geometry at index {idx} remains invalid: "
                        f"{explain_validity(fixed_geom)}"
                    )
                    fixed_geom = geom.buffer(0)
                    if not fixed_geom.is_valid:
                        print(f"Buffer repair failed for geometry at index {idx}.")
                        continue
            except Exception as e:
                print(f"Error repairing geometry at index {idx}: {e}")
                continue
        else:
            fixed_geom = geom

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

    if buffered_gdf.crs is None:
        raise ValueError("Clip geometry has no CRS.")

    # Iterate through each factor name and corresponding raster file
    for i, raster_file in enumerate(raster_files):
        # Open the raster file
        with rasterio.open(raster_file) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {raster_file}")
            clip_gdf = (
                buffered_gdf
                if buffered_gdf.crs == src.crs
                else buffered_gdf.to_crs(src.crs)
            )
            geometries = list(clip_gdf.geometry)
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


# ── Notebook helpers ──────────────────────────────────────────────────────────

def search_and_display_events(starttime, endtime,
                               minmagnitude=7.0, maxmagnitude=10.0,
                               minlatitude=None, maxlatitude=None,
                               minlongitude=None, maxlongitude=None):
    """Search USGS ComCat for earthquakes and display results as an HTML table.

    Returns the raw list of SummaryEvent objects from libcomcat.
    """
    import pandas as pd
    from IPython.display import display, HTML
    from libcomcat.search import search

    events = search(
        starttime=starttime, endtime=endtime,
        minlatitude=minlatitude, maxlatitude=maxlatitude,
        minlongitude=minlongitude, maxlongitude=maxlongitude,
        minmagnitude=minmagnitude, maxmagnitude=maxmagnitude,
    )
    print(f"Found {len(events)} earthquake event(s)")

    rows = []
    for e in events:
        link = f"https://earthquake.usgs.gov/earthquakes/eventpage/{e.id}/executive"
        rows.append({
            "Time (UTC)": e.time.strftime("%Y-%m-%d %H:%M"),
            "Mag":        e.magnitude,
            "Location":   e.location,
            "ID":         e.id,
            "Link":       f'<a href="{link}" target="_blank">{link}</a>',
        })

    df = pd.DataFrame(rows)
    display(HTML(df.to_html(escape=False, index=False)))
    return events
