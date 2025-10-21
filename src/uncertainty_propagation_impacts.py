"""
Uncertainty Propagation Module for Landslide Impacts.

This module evaluates the landslide impacts on
critical infrastructure and populations through Monte Carlo simulation:

Author: Shihao Xiao
Date: Oct 2025
"""

# Standard library imports
import glob
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import MultiPoint, MultiPolygon, LineString, box
from shapely.ops import split, snap
from shapely.validation import make_valid, explain_validity
from tqdm import tqdm

# Local imports
from src.utils import clip_features_raster_files, save_raster, check_and_fix_geometries, convert_wgs_to_utm

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def uncertainty_propagation_impacts(grid_area, 
                                    simulated_counts_store, simulated_area_coverages_store, 
                                    input_path, output_path,
                                    river_network_data, road_network_data):
    """
    Evaluate landslide impacts on infrastructure and population through uncertainty propagation.
    
    This function conducts comprehensive impact assessment including:
    1. Population exposure
    2. River network disruption
    3. Road network disruption
    
    Parameters:
    -----------
    grid_area : float
        Area of each raster grid cell in square meters
    simulated_counts_store : ndarray
        4D array of simulated landslide counts [outer, inner, height, width]
    simulated_area_coverages_store : ndarray
        4D array of simulated areal coverages [outer, inner, height, width]
    input_path : dict
        Dictionary containing input parameters and file paths
    output_path : dict
        Dictionary containing output file paths
    river_network_data : GeoDataFrame
        River network data with discharge information
    road_network_data : networkx.Graph
        Road network data from OSMnx
        
    Returns:
    --------
    None
        Results are saved as raster files, shapefiles, and numpy arrays
    """
    # =================================================================
    # SECTION 1: POPULATION EXPOSURE ASSESSMENT
    # =================================================================
    print("\nEvaluating population exposure ...")
    
    # Step 1.1: Load landslide-affected area polygon
    polygon = gpd.read_file(output_path['LandslideAffectedArea'])

    # Step 1.2: Clip population density raster to study area
    clip_features_raster_files(
        ["Population_density"], 
        [input_path['GlobalPopulationDensityFile']], 
        polygon, 
        output_path['PopulationExposure']
    )

    # Step 1.3: Resample population density to match landslide intensity grid (unit: persons/km²)
    population_density_array = resample_raster(
        output_path['ClippedPopulationDensityFile'], 
        output_path['ClippedSlopeFile'], 
        output_path['ResampledClippedPopulationDensityFile']
    )

    # Step 1.4: Calculate population exposure density (persons/km²)
    population_exposure_density_store = simulated_area_coverages_store * population_density_array

    # Memory optimization: Free intermediate arrays if needed
    # del simulated_area_coverages_store

    # Step 1.5: Calculate total population exposure statistics
    # Sum over spatial dimensions (height, width)
    summed_density = np.nansum(population_exposure_density_store, axis=(-2, -1))  # Shape: (n_outer, n_inner)
    # Flatten simulation dimensions
    flattened_density = summed_density.flatten()  # Shape: (n_total_simulations,)
    # Convert units: density (persons/km²) × area (m²) × (1 km²/1e6 m²) = total persons
    total_population_exposure = flattened_density * grid_area * 1e-6

    # Step 1.6: Calculate spatial statistics maps
    exposure_density_mean_map = np.mean(population_exposure_density_store, axis=(0, 1))

    # Step 1.7: Calculate and display summary statistics
    exposure_mean = np.mean(total_population_exposure)
    exposure_2_5 = np.percentile(total_population_exposure, 2.5)
    exposure_50_0 = np.percentile(total_population_exposure, 50)
    exposure_97_5 = np.percentile(total_population_exposure, 97.5)

    print(
        f"Total population exposure: median={int(round(exposure_50_0))} persons, "
        f"95% prediction interval=({int(round(exposure_2_5))}, {int(round(exposure_97_5))}) persons"
    )

    # Save total exposure values as numpy array
    np.save(output_path['TotalPopulationExposureFile'], total_population_exposure)

    # Read reference raster metadata for spatial output
    with rasterio.open(output_path['ClippedSlopeFile']) as src:
        slope_raster_meta = src.meta.copy()
        slope_raster_transform = src.transform

    # Save mean exposure density raster
    save_raster(exposure_density_mean_map, output_path['MeanPopulationExposureDensityRasterFile'], 
               slope_raster_meta, slope_raster_transform)

    # =================================================================
    # SECTION 2: RIVER NETWORK BLOCKAGE ASSESSMENT
    # =================================================================
    print("\nEvaluating river blockages ...")
    
    # Step 2.1: Prepare study area polygon and coordinate systems
    polygon = gpd.read_file(output_path['LandslideAffectedArea'])
    polygon = polygon.to_crs(epsg=4326)  # Ensure WGS84 projection
    polygon = polygon.geometry.unary_union  # Convert to single polygon geometry

    # Step 2.2: Check for landslide inventory availability
    landslide_polygon = None
    fixed_landslide_polygon = None
    
    if input_path['IS_LS_PolygonAvailable'] == "True":
        # Find landslide polygon shapefile
        shp_files = glob.glob(os.path.join(output_path['mapped_LS_Polygon'], "*.shp"))
        
        if len(shp_files) != 1:
            raise ValueError(f"Expected exactly one .shp file, but found {len(shp_files)} files: {shp_files}")
        
        shp_file = shp_files[0]

        # Load and fix landslide polygons
        landslide_polygon = gpd.read_file(shp_file)
        fixed_landslide_polygon = check_and_fix_geometries(landslide_polygon)

    # Step 2.3: Extract and process river network within study area
    river_edges = river_network_data[river_network_data.intersects(polygon)]

    # Load raster metadata for spatial analysis
    with rasterio.open(output_path['LSIntensityFile']) as src:
        predicted_counts_raster = src.read(1)  # Read landslide intensity data
        raster_meta = src.meta.copy()  # Preserve metadata for processing
        raster_crs = src.crs  # Extract coordinate reference system
    
    # Step 2.4: Check if river network exists in study area
    if river_edges.empty:
        print("  ⚠ No river network found within study area - skipping analysis")
        # Create notification file for missing river data
        with open(output_path['RiverBlockages_NonDataFile'], "w", encoding="utf-8") as f:
            f.write("No river network found within the earthquake-affected area.")
        print("  - Created notification file for missing river network data")
        
    else:
        # Step 2.5: Calculate river widths using hydraulic relationships
        # Width calculation: w = 7.2 * (discharge)^0.5
        # Based on: Moody & Troutman (2002), Andreadis et al. (2013), Frasson et al. (2019)
        river_edges.loc[:, 'width'] = 7.2 * (river_edges['DIS_AV_CMS'] ** 0.5)

        # Step 2.6: Split river edges into analysis segments
        edges_polyline, edges_polygon_river, edges_polygon_riverPlusLSRunout = river_split_edges(
            river_edges, 
            float(input_path['buffer_LS_runout']), 
            max_length=5000., 
            split_length=5000.
        )

        # Step 2.7: Extract observed blockages from mapped landslides (if available)
        total_obs_counts_FromLS_Polygon = 0
        riverblocks = None
        
        if input_path['IS_LS_PolygonAvailable'] == "True":
            # Generate river blockages from landslide-river intersections
            riverblocks = generate_river_or_road_blocks(fixed_landslide_polygon, edges_polygon_river)
            
            # Add blockage information to river edge polygons
            edges_polygon_river = add_LS_block_No(edges_polygon_river, riverblocks)
            
            total_obs_counts_FromLS_Polygon = edges_polygon_river["obs_LS"].sum()
            print(f"    - Found {total_obs_counts_FromLS_Polygon} observed river blockages")

        # Step 2.8: Extract predicted blockages from simulation results
        # Calculate intersecting grid indices for each river segment
        intersecting_indices_and_areas = extract_intersecting_grid_indices_for_each_geometry(
            raster_meta, edges_polygon_riverPlusLSRunout
        )
        
        # Calculate blockage statistics for all simulations
        edges_polygon_riverPlusLSRunout, total_pred_counts = calculate_for_each_geometry(
            simulated_counts_store, intersecting_indices_and_areas, edges_polygon_riverPlusLSRunout
        )

        # Step 2.9: Integrate blockage data into polyline format for analysis
        # Ensure consistent coordinate systems
        edges_polyline = edges_polyline.to_crs(epsg=4326)
        edges_polygon_river = edges_polygon_river.to_crs(epsg=4326)
        edges_polygon_riverPlusLSRunout = edges_polygon_riverPlusLSRunout.to_crs(epsg=4326)

        # Transfer attributes from polygons to polylines
        edges_polyline_obs_pred = edges_polyline.copy()
        
        # Add observed blockages if landslide inventory is available
        if input_path['IS_LS_PolygonAvailable'] == "True":
            edges_polyline_obs_pred.loc[:, 'obs_LS'] = edges_polygon_river['obs_LS'].values
            
        # Add predicted blockage statistics
        edges_polyline_obs_pred.loc[:, ['mean', 'median', 'Q_2.5', 'Q_97.5']] = \
            edges_polygon_riverPlusLSRunout[['mean', 'median', 'Q_2.5', 'Q_97.5']].values

        # Step 2.10: Calculate blockage densities (blockages per km)
        # Fill missing values with zeros to avoid computation errors
        temp_filled = edges_polyline_obs_pred[['mean', 'median', 'Q_2.5', 'Q_97.5']].fillna(0)
        edges_polyline_obs_pred.loc[:, ['mean', 'median', 'Q_2.5', 'Q_97.5']] = temp_filled.values

        # Calculate density statistics per kilometer of river length
        river_interruptions_density = calculate_density(edges_polyline_obs_pred, input_path['IS_LS_PolygonAvailable'])

        # Step 2.11: Save river blockage analysis results
        # Save total predicted blockages as numpy array
        np.save(output_path['TotalRiverBlockagesFile'], total_pred_counts)

        # Step 2.12: Calculate summary statistics for total predicted river blockages
        pred_mean = np.mean(total_pred_counts)
        pred_2_5 = np.percentile(total_pred_counts, 2.5)
        pred_50_0 = np.percentile(total_pred_counts, 50.0)
        pred_97_5 = np.percentile(total_pred_counts, 97.5)
        print(
            f"Total river blockages: "
            f"median={int(round(pred_50_0))}, "
            f"95% prediction interval=({int(round(pred_2_5))}, {int(round(pred_97_5))})"
        )

        # Save river blockage density shapefile for visualization
        river_interruptions_density.to_file(output_path['MeanRiverBlockagesDensityFile'])

        # Save observed blockage data if landslide inventory is available
        if input_path['IS_LS_PolygonAvailable'] == "True":
            # Save observed river blockage polygons
            riverblocks.to_file(output_path['ObservedRiverBlocksFile'], driver="ESRI Shapefile")
            
            # Save total observed blockage count
            np.save(output_path['ObservedTotalRiverBlockagesFile'], 
                   np.array([total_obs_counts_FromLS_Polygon]))
            
    # =================================================================
    # SECTION 3: ROAD NETWORK BLOCKAGE ASSESSMENT
    # =================================================================
    print("\nEvaluating road blockages ...")
    
    # Step 3.1: Extract and process road network data
    G_save = ox.convert.to_undirected(road_network_data.copy())
    road_edges = ox.graph_to_gdfs(G_save, nodes=False, edges=True)

    # Step 3.2: Split road edges into analysis segments
    edges_polyline, edges_polygon_road, edges_polygon_roadPlusLSRunout = road_split_edges(
        road_edges, 
        float(input_path['buffer_LS_runout']), 
        max_length=5000., 
        split_length=5000.
    )

    # Step 3.3: Extract observed road blockages (if landslide inventory available)
    total_obs_counts_FromLS_Polygon = 0
    roadblocks = None
    
    if input_path['IS_LS_PolygonAvailable'] == "True":
        
        # Generate road blockages from landslide-road intersections
        roadblocks = generate_river_or_road_blocks(fixed_landslide_polygon, edges_polygon_road)
        
        # Add blockage information to road edge polygons  
        edges_polygon_road = add_LS_block_No(edges_polygon_road, roadblocks)
        
        total_obs_counts_FromLS_Polygon = edges_polygon_road["obs_LS"].sum()
        print(f"    - Found {total_obs_counts_FromLS_Polygon} observed road blockages")

    # Step 3.4: Extract predicted road blockages from simulation results
    
    # Calculate intersecting grid indices for each road segment
    intersecting_indices_and_areas = extract_intersecting_grid_indices_for_each_geometry(
        raster_meta, edges_polygon_roadPlusLSRunout
    )
    
    # Calculate blockage statistics for all simulations
    edges_polygon_roadPlusLSRunout, total_pred_counts = calculate_for_each_geometry(
        simulated_counts_store, intersecting_indices_and_areas, edges_polygon_roadPlusLSRunout
    )

    # Step 3.5: Integrate road blockage data into polyline format
    
    # Ensure consistent coordinate systems
    edges_polyline = edges_polyline.to_crs(epsg=4326)
    edges_polygon_road = edges_polygon_road.to_crs(epsg=4326)
    edges_polygon_roadPlusLSRunout = edges_polygon_roadPlusLSRunout.to_crs(epsg=4326)

    # Transfer attributes from polygons to polylines
    edges_polyline_obs_pred = edges_polyline.copy()
    
    # Add observed blockages if landslide inventory is available
    if input_path['IS_LS_PolygonAvailable'] == "True":
        edges_polyline_obs_pred.loc[:, 'obs_LS'] = edges_polygon_road['obs_LS'].values
        
    # Add predicted blockage statistics
    edges_polyline_obs_pred.loc[:, ['mean', 'median', 'Q_2.5', 'Q_97.5']] = \
        edges_polygon_roadPlusLSRunout[['mean', 'median', 'Q_2.5', 'Q_97.5']].values

    # Step 3.6: Calculate road blockage densities (blockages per km)
    # Fill missing values with zeros to avoid computation errors
    temp_filled_road = edges_polyline_obs_pred[['mean', 'median', 'Q_2.5', 'Q_97.5']].fillna(0)
    edges_polyline_obs_pred.loc[:, ['mean', 'median', 'Q_2.5', 'Q_97.5']] = temp_filled_road.values

    # Calculate density statistics per kilometer of road length
    road_interruptions_density = calculate_density(edges_polyline_obs_pred, input_path['IS_LS_PolygonAvailable'])

    # Step 3.7: Save road blockage analysis results
    
    # Save total predicted blockages as numpy array
    np.save(output_path['TotalRoadBlockagesFile'], total_pred_counts)

    # Step 3.8: Calculate summary statistics for total predicted road blockages
    pred_mean = np.mean(total_pred_counts)
    pred_2_5 = np.percentile(total_pred_counts, 2.5)
    pred_50_0 = np.percentile(total_pred_counts, 50.0)
    pred_97_5 = np.percentile(total_pred_counts, 97.5)
    print(
        f"Total road blockages: "
        f"median={int(round(pred_50_0))}, "
        f"95% prediction interval=({int(round(pred_2_5))}, {int(round(pred_97_5))})"
    )

    # Save road blockage density shapefile for visualization
    road_interruptions_density.to_file(output_path['MeanRoadBlockagesDensityFile'])

    # Save observed blockage data if landslide inventory is available
    if input_path['IS_LS_PolygonAvailable'] == "True":
        # Save total observed blockage count
        np.save(output_path['ObservedTotalRoadBlockagesFile'], 
               np.array([total_obs_counts_FromLS_Polygon]))
        
        # Save observed road blockage polygons
        roadblocks.to_file(output_path['ObservedRoadBlocksFile'], driver="ESRI Shapefile")
        

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def resample_raster(source_raster_path, target_raster_path, output_raster_path, 
                   resampling_method=Resampling.average):
    """
    Resample a source raster to match target raster resolution and extent.

    Parameters:
    -----------
    source_raster_path : str
        Path to the source raster file
    target_raster_path : str
        Path to target raster (provides resolution, extent, projection)
    output_raster_path : str
        Path for the resampled output raster
    resampling_method : rasterio.enums.Resampling
        Resampling method (default: average)

    Returns:
    --------
    np.ndarray
        Resampled 2D array
    """
    # Read target raster properties for alignment
    with rasterio.open(target_raster_path) as target_src:
        target_transform = target_src.transform
        target_crs = target_src.crs
        target_width = target_src.width
        target_height = target_src.height
        target_meta = target_src.meta.copy()

    # Read and resample source raster
    with rasterio.open(source_raster_path) as src:
        source_data = src.read(1)  # Single band assumption
        source_transform = src.transform
        source_crs = src.crs

        # Check CRS compatibility
        if source_crs != target_crs:
            raise ValueError("Source and target CRS do not match. Reproject before resampling.")

        # Initialize resampled array
        resampled_data = np.empty((target_height, target_width), dtype=source_data.dtype)

        # Perform resampling
        reproject(
            source=source_data,
            destination=resampled_data,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling_method
        )

    # Clean up negative values (often representing water bodies)
    resampled_data[resampled_data < 0] = 0

    # Update metadata and save output raster
    target_meta.update({
        "driver": "GTiff",
        "height": target_height,
        "width": target_width,
        "transform": target_transform,
        "crs": target_crs,
        "dtype": str(resampled_data.dtype),
        "count": 1
    })

    with rasterio.open(output_raster_path, "w", **target_meta) as dst:
        dst.write(resampled_data, 1)

    return resampled_data


def river_split_edges(edges, buffer_LS_runout, max_length=10000., split_length=10000.):
    """
    Split long river segments into smaller analysis units with buffering.

    Parameters:
    -----------
    edges : GeoDataFrame
        River edge geometries with discharge information
    buffer_LS_runout : float
        Buffer distance for landslide runout zone (meters)
    max_length : float
        Maximum segment length threshold (meters)
    split_length : float
        Target length for split segments (meters)

    Returns:
    --------
    tuple
        - edges_split : GeoDataFrame of split polylines
        - edges_polygon_river : GeoDataFrame of buffered river polygons
        - edges_polygon_riverPlusLSRunout : GeoDataFrame with runout buffer
    """
    # Convert to UTM projection for accurate distance calculations
    utm_epsg = int(convert_wgs_to_utm(
        min(edges.bounds.minx), max(edges.bounds.maxy)
    ))
    edges = edges.to_crs(epsg=utm_epsg)
    edges.crs = f"EPSG:{utm_epsg}"

    # Split edges into smaller segments
    edges_split = []
    
    for i, line in edges.iterrows():
        # Check if segment needs splitting
        split_geoms, split_flag = partition_segment(
            line["geometry"], max_length, split_length
        )
        
        if split_flag == 1:  # Segment was split
            for sline in split_geoms.geoms:
                edges_split.append({
                    "geometry": sline,
                    "length": sline.length,
                    "UPLAND_SKM": line.UPLAND_SKM,
                    "DIS_AV_CMS": line.DIS_AV_CMS,
                    "width": line.width
                })
        else:  # Segment kept as-is
            edges_split.append({
                "geometry": split_geoms,
                "length": split_geoms.length,
                "UPLAND_SKM": line.UPLAND_SKM,
                "DIS_AV_CMS": line.DIS_AV_CMS,
                "width": line.width
            })

    # Convert to GeoDataFrame
    edges_split = gpd.GeoDataFrame(edges_split, crs=f"EPSG:{utm_epsg}")
    edges_split.set_crs(epsg=utm_epsg, inplace=True)

    # Create buffered copies
    edges_polygon_river = edges_split.copy(deep=True)
    edges_polygon_riverPlusLSRunout = edges_split.copy(deep=True)

    # Apply buffers
    edges_polygon_river.loc[:, "geometry"] = edges_polygon_river.buffer(
        edges_polygon_river["width"]/2, cap_style=2
    )
    
    edges_polygon_riverPlusLSRunout.loc[:, "geometry"] = edges_polygon_riverPlusLSRunout.buffer(
        edges_polygon_riverPlusLSRunout["width"]/2 + buffer_LS_runout, cap_style=2
    )

    # Convert back to WGS84
    edges_split = edges_split.to_crs(epsg=4326)
    edges_polygon_river = edges_polygon_river.to_crs(epsg=4326)
    edges_polygon_riverPlusLSRunout = edges_polygon_riverPlusLSRunout.to_crs(epsg=4326)
    
    return edges_split, edges_polygon_river, edges_polygon_riverPlusLSRunout


def road_split_edges(edges, buffer_LS_runout, max_length=10000., split_length=10000.):
    """
    Split long road segments into smaller analysis units with classification and buffering.

    Parameters:
    -----------
    edges : GeoDataFrame
        Road edge geometries from OSMnx
    buffer_LS_runout : float
        Buffer distance for landslide runout zone (meters)
    max_length : float
        Maximum segment length threshold (meters)
    split_length : float
        Target length for split segments (meters)

    Returns:
    --------
    tuple
        - edges_split : GeoDataFrame of split road polylines
        - edges_polygon_road : GeoDataFrame of buffered road polygons
        - edges_polygon_roadPlusLSRunout : GeoDataFrame with runout buffer
    """
    edges = edges.copy()
    
    if edges.empty:
        empty = gpd.GeoDataFrame(
            columns=["length", "geometry", "highway", "lanes"],
            geometry="geometry", crs="EPSG:4326"
        )
        return empty, empty.copy(), empty.copy()

    # Standardize highway classification
    edges.loc[:, 'highway'] = np.where(
        edges.highway.apply(type) == list,
        [','.join(map(str, e)) for e in edges['highway']], 
        edges['highway']
    )

    # Reclassify road types into categories
    edges.loc[edges["highway"].str.contains("unclassified|residential", na=False), "highway"] = "local"
    edges.loc[edges["highway"].str.contains("secondary|tertiary", na=False), "highway"] = "arterial"
    edges.loc[edges["highway"].str.contains("primary|motorway|trunk", na=False), "highway"] = "highway"

    # Set CRS if missing
    if edges.crs is None:
        edges.set_crs(epsg=4326, inplace=True)

    # Convert to UTM for accurate measurements
    cx, cy = edges.unary_union.centroid.x, edges.unary_union.centroid.y
    utm_epsg = int(convert_wgs_to_utm(float(cx), float(cy)))
    edges = edges.to_crs(epsg=utm_epsg)

    # Split edges into segments
    split_rows = []
    for _, row in edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        split_geoms, flag = partition_segment(geom, max_length, split_length)
        
        if flag == 1:  # Split occurred
            for sg in getattr(split_geoms, 'geoms', []):
                if sg.is_empty or sg.length <= 0:
                    continue
                split_rows.append({
                    "geometry": sg,
                    "length": float(sg.length),
                    "highway": row.highway
                })
        else:  # No split needed
            if (not split_geoms.is_empty) and split_geoms.length > 0:
                split_rows.append({
                    "geometry": split_geoms,
                    "length": float(split_geoms.length),
                    "highway": row.highway
                })

    # Create GeoDataFrame and filter valid geometries
    edges_split = gpd.GeoDataFrame(
        split_rows, crs=f"EPSG:{utm_epsg}",
        columns=["length", "geometry", "highway"]
    )

    if not edges_split.empty:
        edges_split = edges_split[
            edges_split.geometry.notnull() &
            (~edges_split.geometry.is_empty) &
            (edges_split.length > 0)
        ].reset_index(drop=True)

    if edges_split.empty:
        empty = gpd.GeoDataFrame(
            columns=["length", "geometry", "highway", "lanes"],
            geometry="geometry", crs="EPSG:4326"
        )
        return empty, empty.copy(), empty.copy()

    # Assign lane widths based on road classification
    width_map = {"highway": 14, "arterial": 7, "local": 6}
    edges_split.loc[:, "lanes"] = edges_split["highway"].map(width_map).fillna(6)
    edges_split.loc[edges_split["lanes"] <= 0, "lanes"] = 6
    edges_split.loc[:, "lanes"] = edges_split["lanes"].astype(float)

    # Create buffered copies
    edges_polygon_road = edges_split.copy(deep=True)
    edges_polygon_roadPlusLSRunout = edges_split.copy(deep=True)

    # Ensure buffer_LS_runout is float
    try:
        bru = float(buffer_LS_runout)
        if np.isnan(bru):
            bru = 0.0
    except Exception:
        bru = 0.0

    # Apply buffers
    edges_polygon_road.loc[:, "geometry"] = edges_polygon_road.buffer(
        edges_polygon_road["lanes"]/2.0, cap_style=2
    )
    
    edges_polygon_roadPlusLSRunout.loc[:, "geometry"] = edges_polygon_roadPlusLSRunout.buffer(
        edges_polygon_roadPlusLSRunout["lanes"]/2.0 + bru, cap_style=2
    )

    # Convert back to WGS84
    edges_split = edges_split.to_crs(epsg=4326)
    edges_polygon_road = edges_polygon_road.to_crs(epsg=4326)
    edges_polygon_roadPlusLSRunout = edges_polygon_roadPlusLSRunout.to_crs(epsg=4326)

    return edges_split, edges_polygon_road, edges_polygon_roadPlusLSRunout


def partition_segment(line, max_length, split_length):
    """
    Split LineString geometry into smaller segments if above length threshold.

    Parameters:
    -----------
    line : shapely.geometry.LineString
        Input line geometry
    max_length : float
        Length threshold for splitting
    split_length : float
        Target length for split segments

    Returns:
    --------
    tuple
        - split_geoms : LineString or MultiLineString geometry
        - split_flag : int (1 if split occurred, 0 if not)
    """
    if line.length >= max_length:
        # Calculate number of split points
        n_split_points = np.ceil(line.length / split_length).astype(int)

        # Generate split points along the line
        splitter = MultiPoint([
            line.interpolate((i / n_split_points), normalized=True)
            for i in range(1, n_split_points)
        ])

        # Snap line to split points and perform split
        line_snap = snap(line, splitter, 1e-4)
        split_geoms = split(line_snap, splitter)
        split_flag = 1
    else:
        split_geoms = line
        split_flag = 0

    return split_geoms, split_flag


def generate_river_or_road_blocks(ls_inventory, edges):
    """
    Create blockage polygons from landslide inventory and infrastructure edges.

    Parameters:
    -----------
    ls_inventory : GeoDataFrame
        Landslide polygon inventory
    edges : GeoDataFrame
        Infrastructure edge polygons (roads or rivers)

    Returns:
    --------
    GeoDataFrame
        Infrastructure blockage polygons with intersection geometries
    """
    road_polygons = edges.copy(deep=True)
    temp_inventory = ls_inventory.copy(deep=True)

    # Ensure WGS84 coordinate system
    if road_polygons.crs != "EPSG:4326":
        road_polygons = road_polygons.to_crs(epsg=4326)
    if temp_inventory.crs != "EPSG:4326":
        temp_inventory = temp_inventory.to_crs(epsg=4326)

    # Find intersections between infrastructure and landslides
    road_blocks = gpd.sjoin(road_polygons, temp_inventory, op='intersects')
    ls_intersect = temp_inventory[temp_inventory.index.isin(list(road_blocks.index_right))]

    # Calculate intersection geometries
    road_blocks_clip = []
    
    for i, roads in road_blocks.iterrows():
        for j, ls in ls_intersect.iterrows():
            if roads["geometry"].intersects(ls["geometry"]):
                try:
                    # Validate and fix geometries if needed
                    if not roads["geometry"].is_valid:
                        print(f"Invalid infrastructure geometry at index {i}, attempting fix...")
                        roads["geometry"] = make_valid(roads["geometry"])
                        if not roads["geometry"].is_valid:
                            print(f"Infrastructure geometry at index {i} remains invalid: {explain_validity(roads['geometry'])}")
                            continue

                    if not ls["geometry"].is_valid:
                        print(f"Invalid landslide geometry at index {j}, attempting fix...")
                        ls["geometry"] = make_valid(ls["geometry"])
                        if not ls["geometry"].is_valid:
                            print(f"Landslide geometry at index {j} remains invalid: {explain_validity(ls['geometry'])}")
                            continue

                    # Calculate intersection
                    intersection_geom = roads['geometry'].intersection(ls['geometry'])
                    
                    if not intersection_geom.is_empty:
                        road_blocks_clip.append({
                            'o_index': i,
                            'geometry': intersection_geom
                        })
                        
                except Exception as e:
                    print(f"Error processing intersection: {e}")

    # Create output GeoDataFrame and remove duplicates
    road_blocks_out = gpd.GeoDataFrame(
        road_blocks_clip, crs="EPSG:4326",
        columns=['o_index', 'geometry']
    )
    
    if not road_blocks_out.empty:
        # Remove duplicate geometries using WKB representation
        GO = road_blocks_out["geometry"].apply(lambda geom: geom.wkb)
        road_blocks_out = road_blocks_out.loc[GO.drop_duplicates().index]
    
    return road_blocks_out


def add_LS_block_No(edges, roadblocks):
    """
    Add landslide blockage counts to infrastructure edges using spatial indexing.

    Parameters:
    -----------
    edges : GeoDataFrame
        Infrastructure edge geometries
    roadblocks : GeoDataFrame
        Infrastructure blockage polygons

    Returns:
    --------
    GeoDataFrame
        Updated edges with 'obs_LS' column containing blockage counts
    """
    # Initialize blockage count column
    edges.loc[:, 'obs_LS'] = 0

    # Create spatial index for efficient intersection queries
    roadblocks_sindex = roadblocks.sindex

    # Check intersections for each edge
    for idx, edge in edges.iterrows():
        if edge['geometry'] is None:
            continue
        
        # Use spatial index to find potential matches
        possible_matches_index = list(roadblocks_sindex.intersection(edge['geometry'].bounds))
        
        if possible_matches_index:
            # Filter to actual matches
            possible_matches = roadblocks.iloc[possible_matches_index]
            
            # Count actual intersections
            intersections = possible_matches['geometry'].intersects(edge['geometry'])
            
            if intersections.any():
                edges.loc[idx, 'obs_LS'] = intersections.sum()

    return edges


def _process_single_polygon(args):
    """
    Process a single polygon for grid intersection analysis (helper function).
    """
    idx, polygon, raster_transform, height, width = args
    mask = geometry_mask([polygon], transform=raster_transform, 
                        invert=True, out_shape=(height, width))
    intersecting_indices = np.argwhere(mask)
    polygon_indices_and_areas = []
    
    for row, col in intersecting_indices:
        cell_bounds = rasterio.windows.bounds(
            ((row, row + 1), (col, col + 1)), transform=raster_transform
        )
        cell_polygon = box(*cell_bounds)
        intersection = polygon.intersection(cell_polygon)
        intersection_area = intersection.area
        cell_area = cell_polygon.area
        proportion = intersection_area / cell_area
        polygon_indices_and_areas.append((row, col, proportion))
    
    return idx, polygon_indices_and_areas


def extract_intersecting_grid_indices_for_each_geometry(raster_meta, polygon_gdf, n_workers=4):
    """
    Identify raster grid indices that intersect with polygons and calculate area proportions.

    Parameters:
    -----------
    raster_meta : dict
        Raster metadata including transform, CRS, dimensions
    polygon_gdf : GeoDataFrame
        Polygon geometries for intersection analysis
    n_workers : int
        Number of worker threads for parallel processing

    Returns:
    --------
    dict
        Dictionary mapping polygon indices to lists of (row, col, proportion) tuples
    """
    # Ensure consistent coordinate systems
    if polygon_gdf.crs != raster_meta["crs"]:
        polygon_gdf = polygon_gdf.to_crs(raster_meta["crs"])
    
    raster_transform = raster_meta["transform"]
    height, width = raster_meta["height"], raster_meta["width"]

    intersecting_indices_and_areas = {}

    # Prepare arguments for parallel processing
    args_list = [
        (idx, polygon, raster_transform, height, width)
        for idx, polygon in enumerate(polygon_gdf.geometry)
    ]
    
    # Process polygons in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(_process_single_polygon, args_list),
            total=len(args_list), 
            desc="Processing polygons"
        ))
    
    # Organize results
    for idx, polygon_indices_and_areas in results:
        intersecting_indices_and_areas[idx] = polygon_indices_and_areas
    
    return intersecting_indices_and_areas


def calculate_for_each_geometry(simulated_counts_store, intersecting_indices_and_areas, polygon_gdf):
    """
    Calculate weighted sum of grid values for each simulated map using pre-computed intersections.

    Parameters:
    -----------
    simulated_counts_store : ndarray
        4D array of simulated maps [n_inter, n_intra, height, width]
    intersecting_indices_and_areas : dict
        Pre-computed grid intersections and area proportions
    polygon_gdf : GeoDataFrame
        Polygon geometries to update with statistics

    Returns:
    --------
    tuple
        - Updated GeoDataFrame with statistical columns
        - Array of total counts for each simulation map
    """
    n_inter_simulations, n_intra_simulations = simulated_counts_store.shape[:2]
    total_counts_for_each_map = np.zeros(n_inter_simulations * n_intra_simulations)
    start_time = time.time()

    # Reshape for efficient indexing
    sim_maps = simulated_counts_store.reshape(
        n_inter_simulations * n_intra_simulations, *simulated_counts_store.shape[2:]
    )

    with tqdm(total=len(intersecting_indices_and_areas), desc="Processing polygons") as pbar:
        for polygon_idx, indices_and_areas in intersecting_indices_and_areas.items():
            if not indices_and_areas:
                pbar.update(1)
                continue

            # Extract grid coordinates and area proportions
            rows, cols, proportions = zip(*indices_and_areas)
            rows = np.array(rows, dtype=int)
            cols = np.array(cols, dtype=int)
            proportions = np.array(proportions, dtype=np.float32)

            # Vectorized extraction for all maps
            values = np.stack([sim_maps[:, r, c] for r, c in zip(rows, cols)], axis=1)
            weighted_sums_flat = np.nansum(values * proportions, axis=1)

            total_counts_for_each_map += weighted_sums_flat

            # Calculate and store statistics
            mean_val = float(np.mean(weighted_sums_flat))
            median_val = float(np.median(weighted_sums_flat))
            q_2_5_val = float(np.percentile(weighted_sums_flat, 2.5))
            q_97_5_val = float(np.percentile(weighted_sums_flat, 97.5))

            polygon_gdf.loc[polygon_idx, 'mean'] = mean_val
            polygon_gdf.loc[polygon_idx, 'median'] = median_val
            polygon_gdf.loc[polygon_idx, 'Q_2.5'] = q_2_5_val
            polygon_gdf.loc[polygon_idx, 'Q_97.5'] = q_97_5_val

            pbar.update(1)
            elapsed_time = (time.time() - start_time) / 60
            pbar.set_postfix(elapsed_time=f"{elapsed_time:.2f} min")

    return polygon_gdf, total_counts_for_each_map


def calculate_density(edges_polyline, is_LS_polygon):
    """
    Calculate blockage density statistics per kilometer of infrastructure length.

    Parameters:
    -----------
    edges_polyline : GeoDataFrame
        Infrastructure polylines with blockage statistics
    is_LS_polygon : str
        Flag indicating if landslide polygon data is available ("True"/"False")

    Returns:
    --------
    GeoDataFrame
        Updated GeoDataFrame with density values (blockages per km)
    """
    # Convert to Mercator projection for accurate length calculations
    edges_polyline_mercator = edges_polyline.to_crs(epsg=3857)

    # Calculate lengths in meters and kilometers
    edges_polyline_mercator.loc[:, 'length_m'] = edges_polyline_mercator.geometry.length
    edges_polyline_mercator.loc[:, 'length_km'] = edges_polyline_mercator['length_m'] / 1000

    # Calculate density values (blockages per km)
    if is_LS_polygon == "True":
        # Include observed data if available
        for col in ['obs_LS', 'mean', 'median', 'Q_2.5', 'Q_97.5']:
            edges_polyline_mercator.loc[:, f'{col}_d'] = \
                edges_polyline_mercator[col] / edges_polyline_mercator['length_km']
    else:
        # Only predicted data available
        for col in ['mean', 'median', 'Q_2.5', 'Q_97.5']:
            edges_polyline_mercator.loc[:, f'{col}_d'] = \
                edges_polyline_mercator[col] / edges_polyline_mercator['length_km']

    # Convert back to WGS84
    interruptions_density = edges_polyline_mercator.to_crs(epsg=4326)
    
    return interruptions_density