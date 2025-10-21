"""
Landslide Intensity Prediction Module for QuakeSlide System.

This module predict landslide intensity using 
pre-trained models and convert predictions to raster format.

Author: Shihao Xiao
Date: Oct 2025
"""

# Standard library imports
import os
import pickle

# Third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def predict_landslide_intensity(event_id, mapping_unit, input_path, output_path):
    """
    Predict landslide intensity using trained model.
    
    This function loads a pre-trained model and associated encoders to predict
    landslide intensity for mapping units, then converts results to raster format.
    
    Args:
        event_id (str): USGS earthquake event ID
        mapping_unit (GeoDataFrame): Mapping units with extracted features
        input_path (dict): Dictionary containing input file paths
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None: Results are saved as raster file
    """
    # print(f"Starting landslide intensity prediction for event {event_id}...")
    
    # Step 1: Load trained model and preprocessing components
    # print("✓ Loading trained model...")
    with open(input_path['LSIntensityModelFile'], 'rb') as file:
        trained_model = pickle.load(file)

    # print("✓ Loading feature encoders...")
    with open(input_path['FeatureLabelEncodersFile'], 'rb') as file:
        label_encoders = pickle.load(file)

    # print("✓ Loading model features...")
    with open(input_path['SelectedModelFeatureFile'], 'rb') as file:
        model_features = pickle.load(file)

    # Step 2: Prepare data for prediction
    # print("✓ Preparing features for model input...")
    
    # Reorder columns according to model feature order
    mapping_unit_processed = mapping_unit[model_features].copy()

    # Encode categorical variables
    for feature, encoder in label_encoders.items():
        if feature in mapping_unit_processed.columns:
            mapping_unit_processed[feature] = encoder.transform(mapping_unit_processed[feature])

    # Step 3: Make predictions
    # print("✓ Running landslide intensity predictions...")
    landslide_intensity = trained_model.predict(mapping_unit_processed)

    # Add predictions back to original data
    mapping_unit['Predicted_Counts'] = landslide_intensity

    # Step 4: Convert predictions to raster format
    # print("✓ Converting predictions to raster format...")
    convert_points_to_raster(
        output_path['ClippedSlopeFile'], 
        mapping_unit, 
        'Predicted_Counts', 
        output_path['LSIntensityFile']
    )
    
    print(f"✓ Landslide intensity prediction completed.")


def convert_points_to_raster(slope_raster_path, mapping_unit, value_column, output_raster_path):
    """
    Convert point predictions to raster format with efficient batch processing.
    
    This function takes point-based predictions and converts them to a raster
    using the spatial reference of an existing slope raster file.
    
    Args:
        slope_raster_path (str): Path to reference raster file for spatial properties
        mapping_unit (GeoDataFrame): Point mapping units with predictions
        value_column (str): Column name containing prediction values
        output_raster_path (str): Output path for the raster file
        
    Returns:
        numpy.ndarray: The created raster array
    """
    # Read reference raster metadata
    with rasterio.open(slope_raster_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        width = src.width
        height = src.height

    # Extract point coordinates and values
    xs = mapping_unit.geometry.x.values
    ys = mapping_unit.geometry.y.values
    values = mapping_unit[value_column].values

    # Convert coordinates to raster row/column indices
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.array(rows)
    cols = np.array(cols)

    # Filter points within raster boundaries
    valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows_valid = rows[valid_mask]
    cols_valid = cols[valid_mask]
    values_valid = values[valid_mask]

    # Create empty raster and populate with values
    raster = np.full((height, width), np.nan, dtype=np.float32)
    raster[rows_valid, cols_valid] = values_valid

    # Update metadata for output
    meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    # Save raster to file
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(raster, 1)

    # print(f"✓ Landslide intensity raster saved: {output_raster_path}")
    return raster