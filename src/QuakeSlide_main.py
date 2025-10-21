"""
QuakeSlide Main Module.

This module contains the main QuakeSlide workflow function and all necessary imports
for earthquake-triggered landslide impact assessment.

Author: Shihao Xiao
Date: Oct 2025
"""

import time

# Import all QuakeSlide modules
from src.define_file_path import define_file_path
from src.download_PGA_raster import download_PGA_raster
from src.create_landslide_affected_area import create_landslide_affected_area
from src.preload_data import preload_data
from src.feature_extraction import feature_extraction
from src.predict_landslide_intensity import predict_landslide_intensity
from src.uncertainty_propagation_hazards import uncertainty_propagation_hazards
from src.uncertainty_propagation_impacts import uncertainty_propagation_impacts
from src.report_generation import report_generation


def QuakeSlide(event_id, lithology_data=None, river_network_data=None, road_network_data=None):
    """
    Main QuakeSlide workflow for earthquake-triggered landslide impact assessment.
    
    Parameters:
        event_id (str): USGS earthquake event identifier
        lithology_data (GeoDataFrame, optional): Preloaded lithology data for batch processing
        river_network_data (GeoDataFrame, optional): Preloaded river network data
        road_network_data (GeoDataFrame, optional): Preloaded road network data
    
    Returns:
        tuple: (lithology_data, river_network_data, road_network_data) for reuse in batch processing
    """
    print(f"\n{'='*50}")
    print(f"Processing earthquake event: {event_id}")
    print(f"{'='*50}")
    start_time = time.time()
    
    # Step 1: Initialize file paths and directories
    input_path, output_path = define_file_path("Input.txt", event_id)
    
    # Step 2: Download ground motion data and define affected area
    download_PGA_raster(event_id, output_path)
    create_landslide_affected_area(input_path, output_path)
    
    # Step 3: Load base geographic data (pass the preloaded data correctly)
    lithology_data, river_network_data, road_network_data = preload_data(
        event_id, input_path, output_path,
        lithology_data=lithology_data,
        river_network_data=river_network_data,
        road_network_data=road_network_data
    )
    
    # Step 4: Extract predictive features for each mapping unit
    mapping_unit = feature_extraction(event_id, lithology_data, input_path, output_path)
    
    # Step 5: Predict landslide intensity using trained model
    predict_landslide_intensity(event_id, mapping_unit, input_path, output_path)
    
    # Step 6: Quantify uncertainty in landslide hazards (counts and areas)
    grid_area, simulated_counts_store, simulated_area_coverages_store = uncertainty_propagation_hazards(
        input_path, output_path
    )
    
    # Step 7: Assess landslide impacts with uncertainty propagation
    uncertainty_propagation_impacts(
        grid_area, simulated_counts_store, simulated_area_coverages_store,
        input_path, output_path, river_network_data, road_network_data
    )
    
    # Step 8: Generate comprehensive assessment report
    report_generation(event_id, input_path, output_path)
    
    # Summary
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"✓ QuakeSlide Completed Successfully")
    print(f"  Total Processing Time: {elapsed_time:.2f} minutes")
    print(f"{'='*60}")
    
    return lithology_data, river_network_data, road_network_data


def get_cached_data(var_name, globals_dict):
    """
    Safely retrieve cached data variables if they exist and are not None.
    
    Parameters:
        var_name (str): Name of the variable to check
        globals_dict (dict): globals() dictionary
    
    Returns:
        The variable value if it exists and is not None, otherwise None
    """
    try:
        if var_name in globals_dict and globals_dict[var_name] is not None:
            return globals_dict[var_name]
        else:
            return None
    except:
        return None