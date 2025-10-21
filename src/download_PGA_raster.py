"""
PGA raster download module for QuakeSlide system.

This module downloads USGS ShakeMap ground motion data (Peak Ground Acceleration)
and processes it for earthquake-triggered landslide assessment.

Author: Shihao Xiao
Date: Oct 2025
"""

import os
import requests
import zipfile
from osgeo import gdal
import numpy as np


def download_PGA_raster(event_id, output_path):
    """
    Download and process USGS ShakeMap PGA raster data for a given earthquake event.
    
    Args:
        event_id (str): USGS earthquake event ID
        output_path (dict): Dictionary containing output directory paths
        
    Returns:
        None
    """

    # Get USGS event details URL
    detail_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
    
    # Fetch event details from USGS API
    response = requests.get(detail_url)
    if response.status_code != 200:
        print(f"ERROR: Failed to fetch event details (HTTP {response.status_code})")
        return
    
    event_data = response.json()
    properties = event_data["properties"]
    
    # Extract ShakeMap product URL
    if "products" not in properties:
        print(f"ERROR: No products available for event {event_id}")
        return
        
    products = properties["products"]
    if "shakemap" not in products:
        print(f"ERROR: No ShakeMap available for event {event_id}")
        return
        
    shakemap = products["shakemap"][0]
    contents = shakemap["contents"]
    
    # Download and process ShakeMap raster data
    if "download/raster.zip" not in contents:
        print(f"ERROR: No ShakeMap raster.zip file available for event {event_id}")
        return
        
    file_url = contents["download/raster.zip"]["url"]
    file_name = os.path.join(output_path['ShakeMap'], f"{event_id}_raster.zip")
    
    # Download the raster file
    # print(f"Downloading ShakeMap data for event {event_id}...")
    file_response = requests.get(file_url)
    if file_response.status_code != 200:
        print(f"ERROR: Failed to download ShakeMap raster.zip for event {event_id}")
        return
        
    # Save and extract the zip file
    with open(file_name, "wb") as file:
        file.write(file_response.content)
    
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(output_path['ShakeMap'])
    
    # Process the .flt files to create PGA.tif
    process_flt_files_to_tif(output_path)
    # print(f"✓ PGA data processed successfully.")


def process_flt_files_to_tif(output_path):
    """
    Process USGS ShakeMap .flt file to create GeoTIFF format PGA raster.
    
    The function reads the pga_mean.flt file, applies exponential transformation
    to convert from log space to PGA values, and saves as GeoTIFF.
    
    Args:
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None
    """
    flt_file = "pga_mean.flt"
    flt_path = os.path.join(output_path['ShakeMap'], flt_file)
    
    if not os.path.exists(flt_path):
        print(f"ERROR: File {flt_path} not found")
        return
    
    try:
        # Read .flt file using GDAL
        dataset = gdal.Open(flt_path)
        if dataset is None:
            print(f"ERROR: Could not open {flt_path}")
            return
            
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # Apply exponential function to convert from log space to actual PGA values
        exp_data = np.exp(data)
        
        # Create new GeoTIFF file
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(
            output_path['PGAFile'], 
            dataset.RasterXSize, 
            dataset.RasterYSize, 
            1, 
            gdal.GDT_Float32
        )
        
        # Set geotransform and projection from source file
        out_dataset.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset.SetProjection(dataset.GetProjection())
        
        # Write transformed data to new file
        out_dataset.GetRasterBand(1).WriteArray(exp_data)
        
        # Clean up and save
        out_dataset.FlushCache()
        del out_dataset, dataset
        
        # print(f"✓ PGA raster created: {output_path['PGAFile']}")
        
    except Exception as e:
        print(f"ERROR: Failed to process PGA raster: {e}")

