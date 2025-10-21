"""
Data preloading module for QuakeSlide system.

This module handles the loading and caching of base geographic datasets including
lithology, river networks, and road networks for landslide impact assessment.

Author: Shihao Xiao
Date: Oct 2025
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import time
import geopandas as gpd
import pyogrio
import osmnx as ox


def preload_data(event_id, input_path, output_path,
                lithology_data=None, 
                river_network_data=None, 
                road_network_data=None):
    """
    Load and cache base geographic datasets for landslide impact assessment.
    
    This function efficiently loads large datasets with caching support to avoid
    redundant loading in batch processing scenarios.
    
    Args:
        event_id (str): USGS earthquake event ID
        input_path (dict): Dictionary containing input file paths and parameters
        output_path (dict): Dictionary containing output file paths
        lithology_data (GeoDataFrame, optional): Cached lithology data
        river_network_data (GeoDataFrame, optional): Cached river network data
        road_network_data (NetworkX graph, optional): Cached road network data
        
    Returns:
        tuple: (lithology_data, river_network_data, road_network_data)
    """
    # Load lithology data (Global Lithological Map)
    if lithology_data is None:
        print("Loading lithology data...")
        start_time = time.time()
        
        lithology_data = pyogrio.read_dataframe(
            input_path['GlobalLithologyFile'], 
            layer='GLiM_export'
        )
        lithology_data = gpd.GeoDataFrame(lithology_data, geometry=lithology_data.geometry)
        
        elapsed_time = time.time() - start_time
        print(f"Lithology data loading time: {elapsed_time/60:.2f} mins")
    else:
        print("Skipping lithology data loading")

    # Load river network data
    if river_network_data is None:
        print("Loading river network data...")
        start_time = time.time()
        
        river_network_data = gpd.read_file(input_path['GlobalRiverNetwork'])
        
        elapsed_time = time.time() - start_time
        print(f"River network data loading time: {elapsed_time/60:.2f} mins")
    else:
        print("Skipping river network data loading")

    # Download road network data from OpenStreetMap
    needs_road_download = (
        road_network_data is None or
        not hasattr(road_network_data, "event_id") or
        road_network_data.event_id != event_id
    )
    
    if needs_road_download:
        print("Downloading road network...")
        start_time = time.time()

        # Load affected area polygon for spatial boundary
        polygon = gpd.read_file(output_path['LandslideAffectedArea'])
        polygon = polygon.to_crs(epsg=4326)
        polygon_union = polygon.geometry.unary_union

        # Configure road network filter (focus on major roads for efficiency)
        use_major_roads_only = True
        major_road_filter = '["highway"~"secondary|tertiary|primary|motorway|trunk"]'

        if use_major_roads_only:
            road_network_data = ox.graph_from_polygon(
                polygon_union, 
                custom_filter=major_road_filter
            )
        else:
            road_network_data = ox.graph_from_polygon(
                polygon_union, 
                network_type='drive'
            )

        # Cache event ID to avoid redundant downloads
        road_network_data.event_id = event_id
        
        elapsed_time = time.time() - start_time
        print(f"Road network download time: {elapsed_time/60:.2f} mins")
    else:
        print("Skipping road network download")

    return lithology_data, river_network_data, road_network_data

