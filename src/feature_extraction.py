"""
Feature extraction module for QuakeSlide system.

This module extracts predictive features from various geographic datasets,
including landform, lithology, and PGA.

Author: Shihao Xiao
Date: Oct 2025
"""

# Standard library imports
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from shapely.geometry import Point
from tqdm import tqdm

# Local imports
from src.utils import clip_features_raster_files

warnings.filterwarnings("ignore", category=RuntimeWarning)


def feature_extraction(event_id, lithology_data, input_path, output_path):
    """
    Extract and process features for landslide susceptibility modeling.
    
    This function performs comprehensive feature extraction including terrain attributes,
    lithology, and ground motion parameters for each mapping unit.
    
    Args:
        event_id (str): USGS earthquake event ID
        lithology_data (GeoDataFrame): Global lithology dataset
        input_path (dict): Dictionary containing input parameters and file paths
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        GeoDataFrame: Processed mapping units with extracted features
    """
    # print(f"Starting feature extraction for event {event_id}...")

    # Step 1: Extract terrain features (Slope and Geomorphon)
    factor_names = ["Slope", "Geomorphon"]
    raster_files = [
        input_path['GlobalSlopeFile'],
        input_path['GlobalLandformFile']
    ]

    # Load affected area polygon and ensure WGS84 projection
    polygon = gpd.read_file(output_path['LandslideAffectedArea'])
    polygon = polygon.to_crs(epsg=4326)

    # Clip terrain rasters to affected area
    clip_features_raster_files(
        factor_names, raster_files, polygon, output_path['feature_maps']
    )

    # Step 2: Generate point mapping units from clipped rasters
    clipped_raster_paths = [
        output_path['ClippedSlopeFile'],
        output_path['ClippedLandformFile']
    ]

    mapping_unit = generate_point_mapping_unit(
        output_path['ClippedSlopeFile'], 
        clipped_raster_paths, 
        factor_names, 
        output_path['PointMappingUnit'], 
        Is_save=False
    )


    # Step 3: Extract lithology information
    # print("Extracting lithology data...")
    mapping_unit = extract_Lithology_to_points(
        mapping_unit, lithology_data, ["Lithology"]
    )

    # Step 4: Extract ground motion data (PGA)
    # print("Extracting PGA values...")
    mapping_unit = extract_raster_values_to_points(
        mapping_unit, [output_path['PGAFile']], ["PGA"]
    )

    # Step 5: Data cleaning and preprocessing
    # print("Processing and filtering features...")
    
    # Remove rows with missing values
    mapping_unit = mapping_unit.dropna()
    
    # Convert slope from percentage to decimal (divide by 100)
    mapping_unit['Slope'] = mapping_unit['Slope'] / 100
    
    # Remove water body pixels (lithology code 'wb')
    mapping_unit = mapping_unit[mapping_unit['Lithology'] != "wb"]
    
    # Filter out areas with slopes below threshold
    slope_threshold = float(input_path['SlopeThreshold'])
    mapping_unit = mapping_unit[mapping_unit['Slope'] >= slope_threshold]
    
    # Standardize lithology codes: merge 'ev' and 'ig' into 'nd' (No Data)
    ev_count = (mapping_unit['Lithology'] == "ev").sum()
    if ev_count > 0:
        mapping_unit['Lithology'] = mapping_unit['Lithology'].replace("ev", "nd")
        # print(f"Replaced {ev_count} 'ev' (evaporites) with 'nd' (no data)")
    
    ig_count = (mapping_unit['Lithology'] == "ig").sum()
    if ig_count > 0:
        mapping_unit['Lithology'] = mapping_unit['Lithology'].replace("ig", "nd")
        # print(f"Replaced {ig_count} 'ig' (ice/glaciers) with 'nd' (no data)")


    # Optional: Save point mapping units to file
    save_points = input_path.get('IS_save_PointMappingUnit', 'False') == "True"
    if save_points:
        mapping_unit.to_file(output_path['save_PointMappingUnitFile'], driver='ESRI Shapefile')
        print(f"✓ Point mapping unit saved: {output_path['save_PointMappingUnitFile']}")

    # Step 6: Set up categorical variables
    mapping_unit['Lithology'] = mapping_unit['Lithology'].astype('category')
    mapping_unit['Geomorphon'] = mapping_unit['Geomorphon'].astype('category')

    # Step 7: Create readable category labels
    geomorphon_mapping = {
        1: "Flat", 2: "Summit", 3: "Ridge", 4: "Shoulder", 5: "Spur",
        6: "Slope", 7: "Hollow", 8: "Footslope", 9: "Valley", 10: "Depression"
    }
    
    lithology_mapping = {
        'su': "Unconsolidated Sediments", 'ss': "Siliciclastic Sedimentary Rocks",
        'sm': "Mixed Sedimentary Rocks", 'py': "Pyroclastic", 
        'sc': "Carbonate Sedimentary Rocks", 'ev': "Evaporites",
        'mt': "Metamorphic Rocks", 'pa': "Acid Plutonic Rocks",
        'pi': "Intermediate Plutonic Rocks", 'pb': "Basic Plutonic Rocks",
        'va': "Acid Volcanic Rocks", 'vi': "Intermediate Volcanic Rocks",
        'vb': "Basic Volcanic Rocks", 'ig': "Ice and Glaciers",
        'wb': "Water Bodies", 'nd': "No Data"
    }

    # Create categorical labels for better interpretability
    mapping_unit['Geom_cat'] = mapping_unit['Geomorphon'].map(geomorphon_mapping)
    mapping_unit['Litho_cat'] = mapping_unit['Lithology'].map(lithology_mapping)

    print(f"✓ Feature extraction completed.")
    return mapping_unit


def read_raster_data(raster_path):
    """Read raster data and return array, transform, CRS, and nodata value."""
    with rasterio.open(raster_path) as src:
        return src.read(1), src.transform, src.crs, src.nodata


def generate_point_mapping_unit(slope_raster_path, raster_paths, factor_names, output_path, Is_save=False):
    """
    Generate point mapping units from raster data using efficient parallel processing.
    
    Args:
        slope_raster_path (str): Path to slope raster (used as reference)
        raster_paths (list): List of all raster file paths
        factor_names (list): List of factor names corresponding to rasters
        output_path (str): Output path for optional shapefile saving
        Is_save (bool): Whether to save the result as shapefile
        
    Returns:
        GeoDataFrame: Point mapping units with extracted values
    """
    # Read reference slope raster data
    slope_data, slope_transform, slope_crs, slope_nodata = read_raster_data(slope_raster_path)
    slope_shape = slope_data.shape
    # Generate coordinate grid from raster pixels
    cols, rows = np.meshgrid(np.arange(slope_shape[1]), np.arange(slope_shape[0]))
    xs, ys = rasterio.transform.xy(slope_transform, rows, cols, offset='center')
    
    # Flatten coordinate arrays and slope values
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()
    slope_values = slope_data.flatten()

    # Filter out nodata values to create valid points only
    valid_mask = slope_values != slope_nodata
    points = [Point(x, y) for x, y in zip(xs[valid_mask], ys[valid_mask])]
    slope_values = slope_values[valid_mask]

    # Create GeoDataFrame with slope values
    gdf = gpd.GeoDataFrame(geometry=points, crs=slope_crs)
    gdf["Slope"] = slope_values

    def extract_raster_values(factor_raster_path, points, factor_nodata):
        """Extract raster values for given points."""
        with rasterio.open(factor_raster_path) as src:
            factor_data = src.read(1)
            factor_transform = src.transform
            factor_values = []
            
            for point in points:
                # Convert geographic coordinates to pixel indices
                col, row = ~factor_transform * (point.x, point.y)
                col, row = int(col), int(row)
                
                # Check if pixel is within raster bounds
                if 0 <= row < factor_data.shape[0] and 0 <= col < factor_data.shape[1]:
                    value = factor_data[row, col]
                    factor_values.append(value if value != factor_nodata else np.nan)
                else:
                    factor_values.append(np.nan)
                    
            return factor_values

    # Use parallel processing for efficient raster value extraction
    with ThreadPoolExecutor() as executor:
        futures = []
        
        # Submit extraction tasks for additional factors (skip slope as it's already added)
        for i in range(1, len(factor_names)):
            factor_name = factor_names[i]
            factor_raster_path = raster_paths[i]
            _, _, _, factor_nodata = read_raster_data(factor_raster_path)
            
            futures.append((
                factor_name, 
                executor.submit(extract_raster_values, factor_raster_path, points, factor_nodata)
            ))

        # Collect results and add to GeoDataFrame
        for factor_name, future in futures:
            gdf[factor_name] = future.result()

    # Optionally save to shapefile
    if Is_save:
        gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"✓ Point shapefile saved: {output_path}")
    
    return gdf


def extract_raster_values_to_points(mapping_unit, raster_paths, factor_names):
    """
    Extract raster values to point mapping unit with efficient batch processing.
    
    Args:
        mapping_unit (GeoDataFrame): Point mapping units
        raster_paths (list): List of raster file paths
        factor_names (list): List of factor names corresponding to rasters
        
    Returns:
        GeoDataFrame: Updated mapping units with extracted raster values
    """
    points = mapping_unit
    xs = points.geometry.x.values
    ys = points.geometry.y.values

    for raster_path, factor_name in zip(raster_paths, factor_names):
        with rasterio.open(raster_path) as src:
            # Batch convert coordinates to row/column indices
            rows, cols = src.index(xs, ys)
            data = src.read(1)
            nodata = src.nodata
            vals = np.full(xs.shape, np.nan, dtype=float)
            
            # Extract values for each point
            for idx in range(len(xs)):
                r, c = rows[idx], cols[idx]
                
                # Check if pixel is within raster bounds
                if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                    v = data[r, c]
                    # Handle nodata values
                    if nodata is not None and np.isclose(v, nodata):
                        vals[idx] = np.nan
                    else:
                        vals[idx] = v
                        
            points[factor_name] = vals
            
    return points


def extract_Lithology_to_points(mapping_unit, lithology_data, factor_names):
    """
    Extract lithology information to point mapping unit using spatial join.
    
    Args:
        mapping_unit (GeoDataFrame): Point mapping units
        lithology_data (GeoDataFrame): Global lithology dataset
        factor_names (list): List of factor names (should contain 'Lithology')
        
    Returns:
        GeoDataFrame: Updated mapping units with lithology information
    """
    points = mapping_unit

    # Ensure coordinate reference systems match
    if points.crs != lithology_data.crs:
        lithology_data = lithology_data.to_crs(points.crs)

    # Perform spatial join to extract lithology data
    joined_data = gpd.sjoin(
        points, 
        lithology_data[['geometry', 'xx']], 
        how="left", 
        predicate='intersects'
    )

    # Rename lithology column and clean up
    joined_data.rename(columns={'xx': factor_names[0]}, inplace=True)
    joined_data = joined_data.drop(columns='index_right', errors='ignore')

    return joined_data