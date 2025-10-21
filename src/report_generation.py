"""
Report Generation Module for QuakeSlide System.

This module generates a one-page report for landslide impacts.

Author: Shihao Xiao
Email: sxiaoai@connect.ust.hk
Date: October 2025
"""  

# Standard library imports
import json
import os
from datetime import datetime
from textwrap import fill, shorten

# Third-party scientific computing imports
import numpy as np
import pandas as pd
import geopandas as gpd

# Geospatial and mapping imports
import contextily as ctx
import rasterio
import rasterio.plot
import rasterio.shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from osgeo import gdal, ogr

# Matplotlib visualization imports
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Network and data access
import requests

# QuakeSlide local imports
from src.plot_utils import (
    generate_hillshade, 
    get_cities, 
    plot_raster_map, 
    plot_cities, 
    plot_contours, 
    plot_blockage_map
)
from src.utils import clip_features_raster_files


def report_generation(event_id, input_path, output_path):
    """
    Generate one-page report for landslide impacts.
    
    Args:
        event_id (str): USGS earthquake event ID
        input_path (dict): Dictionary containing input file paths
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None: Saves report as PDF file
    """
    print('\nGenerating one-page report...')

    # Download PGA contours and earthquake event information
    extracted_event_info = download_pga_contours_and_event_info(
        event_id, 
        output_path['plot_components']
    )

    # Convert GeoJSON contours to shapefile format
    convert_geojson_to_shapefile(
        output_path['PGAContourJsonFile'], 
        output_path['PGAContourShpFile']
    )

    # Set up page dimensions and layout (A4 format)
    mm_to_inch = 0.0393701  # Conversion factor
    page_margins = 5 * mm_to_inch
    page_width = 170 * mm_to_inch
    
    # Define row heights for different sections
    section_heights = np.array([6, 30, 7, 25, 7, 60, 7, 60, 10]) * mm_to_inch
    page_height = section_heights.sum() + 2 * page_margins
    
    # Calculate relative margins
    horizontal_margin = page_margins / page_width
    vertical_margin = page_margins / page_height

    # Create figure with high resolution for print quality
    fig = plt.figure(figsize=(page_width, page_height), dpi=600, constrained_layout=False)

    # Set up grid layout with height ratios
    height_ratios = section_heights / (page_height - 2 * page_margins)

    # Create grid specification for report layout (9 rows total)
    gs = gridspec.GridSpec(
        len(section_heights), 1, 
        figure=fig, 
        height_ratios=height_ratios, 
        wspace=0.0, 
        hspace=0.0,
        left=horizontal_margin, 
        right=1 - horizontal_margin, 
        top=1 - vertical_margin,
        bottom=vertical_margin
    )


    def plot_report_title(ax, fontsize=13):
        title = 'QuakeSlide: Prompt Earthquake-Triggered Landslide Impact Assessment'
        # title = 'Prompt Landslide Impact Assessment'
        ax.text(0, 1, title, ha='left', va='top', weight="bold", fontsize=fontsize)
        # ax.add_patch(Rectangle(xy=(0, 0.2), width=0.005, height=0.6, facecolor='k', edgecolor='k'))
        ax.axis('off')

    # The 1st row contains the logo, the report title and the report information
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    ax00 = fig.add_subplot(gs0[0, 0])
    plot_report_title(ax00, fontsize=11.5)

    # Generate current UTC timestamp
    nowString_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Generate data from extracted_event_info dictionary
    data = [
        ["Report created", nowString_utc],
        ["ShakeMap version", extracted_event_info.get("Version")],
        # ["Tool Web", "www.XXXX"],
        ["Place", extracted_event_info.get("Place")],
        ["Magnitude", extracted_event_info.get("Magnitude")],
        ["Time", datetime.strptime(extracted_event_info.get("Time"), "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Epicenter", f"{abs(float(extracted_event_info.get('Latitude', 0))):.2f}°{'N' if float(extracted_event_info.get('Latitude', 0)) >= 0 else 'S'}, "
                    f"{abs(float(extracted_event_info.get('Longitude', 0))):.2f}°{'E' if float(extracted_event_info.get('Longitude', 0)) >= 0 else 'W'}"],
        ["Depth", extracted_event_info.get("Depth")]
    ]

    def plot_earthquake_info(data, ax, fontsize):
        # Calculate max label width
        max_label_length = max(len(label) for label, _ in data)

        # Add black border accent
        ax.add_patch(Rectangle(xy=(0.02, 0.05), width=0.006, height=0.9, facecolor='k', edgecolor='k'))

        # Set smaller vertical spacing
        num_lines = len(data)
        y_spacing = 0.82 / (num_lines - 1) if num_lines > 1 else 0.70  # Original was 0.92

        for i, (label, value) in enumerate(data):
            y_position = 0.95 - i * y_spacing
            ax.text(0.04, y_position, f'{label}:', ha='left', va='top', fontsize=fontsize)
            ax.text(0.04 + max_label_length * 0.02 - 0.02, y_position, value, ha='left', va='top', fontsize=fontsize)

        # Hide coordinate axes
        ax.axis('off')

    # Earthquake longitude and latitude information
    lat = extracted_event_info.get('Latitude')
    lon = extracted_event_info.get('Longitude')

    # The 2nd row contains the earthquake information and the global map showing earthquake location
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.6, 0.4])
    ax10 = fig.add_subplot(gs1[0, 0])
    plot_earthquake_info(data, ax10, fontsize = 8)
    ax11 = fig.add_subplot(gs1[0, 1])
    plot_global_map(lat, lon, ax11)



    # ---------------Impact summary

    def plot_subtitle(ax, text, ha='left', fontsize=10, offset=0.02):
        if ha == 'left':
            ax.text(0+offset, 0.1, text, ha='left', va='bottom', weight="bold", fontsize=fontsize)
        else:
            ax.text(0.5+offset, 0.1, text, ha='center', va='bottom', weight="bold", fontsize=fontsize)
        ax.axis('off')


    # The 3th row contains the landslide title
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2])
    ax20 = fig.add_subplot(gs2[0, 0])
    # plot_subtitle(ax20, 'Estimated Landslide Impacts')
    plot_subtitle(ax20, 'Total Severity Metrics', fontsize=9, offset=0.02)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'

    # -----------------read in the statistics
    total_counts = np.load(output_path['TotalCountsFile'])
    total_areas = np.load(output_path['TotalAreasFile'])
    total_areas = total_areas / 1e6  # Convert to km²
    total_population_exposure = np.load(output_path['TotalPopulationExposureFile']) # unit: persons
    total_road_blockages = np.load(output_path['TotalRoadBlockagesFile']) # unit: count

    # ----------------------------
    # The 4th row contains the landslide table
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[3],width_ratios=[0, 1, 1, 1, 1, 1, 0],wspace=0.5)
    ax30 = fig.add_subplot(gs3[0, 1])
    plot_distribution(total_counts, ax30, "Total count", fontsize=7)
    ax31 = fig.add_subplot(gs3[0, 2])
    plot_distribution(total_areas, ax31, "Total area (km²)", fontsize=7)
    ax32 = fig.add_subplot(gs3[0, 3])
    plot_distribution(total_population_exposure, ax32, "Total population exposure", fontsize=7)
    ax33 = fig.add_subplot(gs3[0, 4])
    plot_distribution(total_road_blockages, ax33, "Total road interruption", fontsize=7)

    if os.path.exists(output_path['RiverBlockages_NonDataFile']):
        print("No river network found in the earthquake-affected area.")
        total_river_blockages = np.array([0])
    else:
        total_river_blockages = np.load(output_path['TotalRiverBlockagesFile']) # unit: count
        ax34 = fig.add_subplot(gs3[0, 5])
        plot_distribution(total_river_blockages, ax34, "Total river interruption", fontsize=7)

    # ----------------------map extent

    polygon = gpd.read_file(output_path['LandslideAffectedArea'])
    polygon = polygon.to_crs(epsg=4326)

    # Get boundary extent
    polygon_bounds = polygon.total_bounds


    # Get counts bounds with projection
    counts_file_bounds = get_raster_bounds_with_projection(output_path['MeanCountsDensityRasterFile'])
    exposure_file_bounds = get_raster_bounds_with_projection(output_path['MeanPopulationExposureDensityRasterFile'])

    # Calculate overlap area
    minx, miny, maxx, maxy = calculate_overlap(counts_file_bounds, exposure_file_bounds, polygon_bounds)

    minx, miny, maxx, maxy = adjust_bounds_to_ratio(minx, miny, maxx, maxy, ratio=1)


    # ------------------prepare hillshade
    from shapely.geometry import box

    # create a polygon using minx, miny, maxx, maxy
    polygon = box(minx, miny, maxx, maxy)
    # polygon to geodataframe
    polygon = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")

    # polygon is already made
    clip_features_raster_files(["Elevation"], [input_path['GlobalDEMFile']], polygon, output_path['plot_components'])

    generate_hillshade(output_path['ClippedElevationFile'],output_path['ClippedHillshadeFile'])

    # City locations
    max_n_cities = 3

    # Download and save as Shapefile
    cities = get_cities(input_path['GlobalCityFile'], max_n_cities, minx, miny, maxx, maxy)

    # Print results
    if cities is not None:
        print(cities.head())


    ocean_boundary_gdf = gpd.read_file(input_path['GlobalOceanBoundaryFile'])
 
    # Ensure the CRS of the ocean boundary matches the polygon's CRS
    if ocean_boundary_gdf.crs != "EPSG:4326":
        ocean_boundary_gdf = ocean_boundary_gdf.to_crs("EPSG:4326")

    # Ocean boundary and polygon intersect
    intersection = gpd.overlay(ocean_boundary_gdf, polygon, how='intersection')

    # The 5th row contains the title: Estimated landslide counts and Estimated population exposure
    # gs4 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4], width_ratios=[0, 0.5, 0.5, 0])

    # Set first interval transparent for areas with NaN values (water, low PGA/slope)

    # ax40 = fig.add_subplot(gs4[0, 1])

    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], width_ratios=[0.5, 0.5])
    ax40 = fig.add_subplot(gs4[0, 0]) 

    plot_subtitle(ax40, 'Mean Landslide Areal Percentage (100%)', fontsize=9, offset=0.06)
    # ax41 = fig.add_subplot(gs4[0, 2])
    ax41 = fig.add_subplot(gs4[0, 1])
    plot_subtitle(ax41, 'Mean Population Exposure (per km²)', fontsize=9, offset=0.06)


    # The 6th row contains two maps: Estimated landslide counts and Estimated population exposure
    # gs5 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[5], width_ratios=[0, 0.5, 0.5, 0])
    # ax50 = fig.add_subplot(gs5[0, 1])
    gs5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5], width_ratios=[0.5, 0.5])
    ax50 = fig.add_subplot(gs5[0, 0])
   
    # plot_raster_map(output_path['MeanCountsDensityRasterFile'], ax50, output_path['ClippedHillshadeFile'], intersection,plot = "counts density colorbar", is_ticks=True) # counts density ( ?? km2)
    plot_raster_map(output_path['MeanArealCoveragesRasterFile'], ax50, output_path['ClippedHillshadeFile'], intersection,plot = "areal percentage colorbar", is_ticks=True) # areal percentage (unit: 100%)
    
    plot_cities(cities, ax50, is_label=True)
    # Overlay contour lines
    plot_contours(output_path['PGAContourShpFile'], ax50, target_crs="EPSG:4326", pga_threshold=0.1)  # ??PGA  0.1g 
    ax50.set_xlim(minx, maxx)
    ax50.set_ylim(miny, maxy)
    # This is crucial!
    ax50.set_aspect('equal') 

    # ax51 = fig.add_subplot(gs5[0, 2])
    ax51 = fig.add_subplot(gs5[0, 1])
    plot_raster_map(output_path['MeanPopulationExposureDensityRasterFile'], ax51, output_path['ClippedHillshadeFile'], intersection, plot = "exposure density colorbar", is_ticks=True)
    # plot_cities(cities, ax51, is_label=True)
    # plot_contours(shapefile_path, ax51, target_crs="EPSG:4326")
    ax51.set_xlim(minx, maxx)
    ax51.set_ylim(miny, maxy)
    # This is crucial!
    ax51.set_aspect('equal') 


    # The 7th row contains the title: Potential Road Blockages and Potential River Damming 
    # gs6 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[6], width_ratios=[0, 0.5, 0.5, 0])
    # ax60 = fig.add_subplot(gs6[0, 1])
    # plot_subtitle(ax60, 'Mean Road Interruption (per km)', fontsize=9, offset=0.05)

    gs6 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6], width_ratios=[0.5, 0.5])
    ax60 = fig.add_subplot(gs6[0, 0])
    plot_subtitle(ax60, 'Mean Road Interruption (per km)', fontsize=9, offset=0.06)


    # The 8th row contains two maps: Potential Road Blockages and Potential River Damming 
    # gs7 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[7], width_ratios=[0, 0.5, 0.5, 0]) # , wspace=0.05
    # ax70 = fig.add_subplot(gs7[0, 1])

    gs7 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[7], width_ratios=[0.5, 0.5]) # , wspace=0.05
    ax70 = fig.add_subplot(gs7[0, 0])

    # plotRoadBlockagesMap(road_file_path, ax70,convex_hull_plot=False, is_ticks=True)
    plot_blockage_map(output_path['MeanRoadBlockagesDensityFile'], ax70, output_path['ClippedHillshadeFile'], intersection, plot_column="mean_d", plot="road", fontsize=8, is_cbar=True, is_ticks=True)
    ax70.set_xlim(minx, maxx)
    ax70.set_ylim(miny, maxy)

    if not os.path.exists(output_path['RiverBlockages_NonDataFile']):
        # ax61 = fig.add_subplot(gs6[0, 2])
        ax61 = fig.add_subplot(gs6[0, 1])
        plot_subtitle(ax61, 'Mean River Interruption (per km)', fontsize=9, offset=0.06)
        # ax71 = fig.add_subplot(gs7[0, 2])
        ax71 = fig.add_subplot(gs7[0, 1])
        # plotRiverDammingMap(river_file_path, ax71,convex_hull_plot=False, is_ticks=True)
        plot_blockage_map(output_path['MeanRiverBlockagesDensityFile'], ax71, output_path['ClippedHillshadeFile'], intersection, plot_column="mean_d", plot="river", fontsize=8, is_cbar=True, is_ticks=True)
        ax71.set_xlim(minx, maxx)
        ax71.set_ylim(miny, maxy)


    def plot_footnote(ax, fontsize=8):
        footnoteText = (
        "This report provides first-order quantitative estimates of potential earthquake-triggered landslide impacts.\n"
        # "This report will be automatically generated and will be updated as new data becomes available.\n"
        "Earthquake information and ground motion data are sourced from the USGS ShakeMap."
    )
        ax.text(0.02, 0, footnoteText, ha='left', va='center', style='italic', fontsize=fontsize)
        ax.axis('off')


    # The 9th row contains the footnote
    gs8 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[8])
    ax80 = fig.add_subplot(gs8[0, 0])
    plot_footnote(ax80)

    # ------------------plot

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"

    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
 
    output_path_PNG = os.path.join(output_path['Report'], f'{current_time}_ImpactReport.PNG')
    output_path_PDF = os.path.join(output_path['Report'], f'{current_time}_ImpactReport.PDF')
    # output_path_EPS = os.path.join(output_path['Report'], f'{current_time}_ImpactReport.EPS')

    fig.savefig(output_path_PNG, pad_inches=0)
    fig.savefig(output_path_PDF, pad_inches=0)
    # fig.savefig(output_path_EPS, pad_inches=0)



# =======================================================================
# UTILITY FUNCTIONS FOR REPORT GENERATION
# =======================================================================

import os
import requests
import json

def download_pga_contours_and_event_info(event_id, output_dir):
    """
    Downloads the ShakeMap PGA Contours JSON file for the given event ID.

    Args:
    event_id (str): The event ID of the earthquake.
    output_dir (str): The directory to save the ShakeMap files.

    Returns:
    None
    """
    # Create ShakeMap directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detail_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
    
    try:
        # Fetch event details
        response = requests.get(detail_url)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        # print(f"Error fetching event details for event ID '{event_id}': {e}")
        return
    
    try:
        event_data = response.json()
        properties = event_data["properties"]

        title = properties.get("title", "Unknown Event")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing event details for event ID '{event_id}': {e}")
        return
    

    # print("properties['products']['shakemap']:", json.dumps(properties["products"]["shakemap"], indent=2))

    shakemap = properties["products"]["shakemap"][0]

    # Extract event information
    extracted_event_info = {
        "Version": shakemap["properties"].get("version", "Unknown"),
        "Place": properties.get("place", "Unknown"),
        "Magnitude": f"Mw {properties.get('mag', 'Unknown')}" if properties.get("mag") is not None else "Unknown",
        "Time": shakemap["properties"].get("eventtime", "Unknown"),
        "Latitude": shakemap["properties"].get("latitude", "Unknown"),
        "Longitude": shakemap["properties"].get("longitude", "Unknown"),
        "Depth": f"{shakemap['properties'].get('depth', 'Unknown')} km"
    }
    # print("Extracted event information:", json.dumps(extracted_event_info, indent=2))

    if "products" in properties:
        products = properties["products"]

        if "shakemap" in products:
            shakemap = products["shakemap"][0]
            # print("shakemap contents:", json.dumps(shakemap["properties"], indent=2))

            contents = shakemap["contents"]

            
            # Check and download cont_pga.json file
            if "download/cont_pga.json" in contents:
                file_url = contents["download/cont_pga.json"]["url"]
                file_name = os.path.join(output_dir, "pga_contours.json")
                
                try:
                    # Download file
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()  # Raise exception for bad status
                    with open(file_name, "wb") as file:
                        file.write(file_response.content)
                    # print(f"Downloaded: {file_name}")
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading cont_pga.json for event ID '{event_id}': {e}")
            else:
                print("ShakeMap does not contain a 'cont_pga.json' file.")
        else:
            print("No ShakeMap data available for this event.")
    else:
        print("No products available for this event.")
    return extracted_event_info

import geopandas as gpd
import os

def convert_geojson_to_shapefile(input_json, shapefile_path):
    """
    Convert GeoJSON file to Shapefile format and save to specified path.

    Parameters
    ----------
    input_json : str
        Path to input GeoJSON file
    shapefile_path : str  
        Output path for Shapefile

    Returns
    -------
    shapefile_path : str
        Path to generated Shapefile
    """

    try:
        gdf = gpd.read_file(input_json)

        # Convert GeoDataFrame to Shapefile
        gdf.to_file(shapefile_path, driver="ESRI Shapefile")

        return shapefile_path
    except Exception as e:
        print(f"Error converting GeoJSON to Shapefile: {e}")
        return None
    

from mpl_toolkits.basemap import Basemap

def plot_global_map(latitude, longitude, ax):
    """
    Plot global overview map showing earthquake epicenter location.
    
    Args:
        latitude (float): Earthquake epicenter latitude
        longitude (float): Earthquake epicenter longitude
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        
    Returns:
        None: Plots directly to provided axes
    """
    # Create Basemap instance with orthographic projection
    m = Basemap(projection='ortho', 
                lat_0=latitude, lon_0=longitude,  # Center on earthquake epicenter
                resolution='l', ax=ax)

    # Fill ocean and continent colors
    m.fillcontinents(color='white', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue', linewidth=0)
    
    # Draw coordinate grid
    m.drawparallels(range(-90, 90, 30), linewidth=0.5, color='gray') 
    m.drawmeridians(range(0, 360, 30), linewidth=0.5, color='gray')  

    # Convert lat/lon to map coordinates
    x, y = m(longitude, latitude)

    # Plot earthquake epicenter as red star
    m.scatter(x, y, marker='*', color='red', s=200, zorder=10)

    ##########################
    # Get x and y limits
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()

    # Calculate offsets
    x_offset = 0.02 * (maxx - minx)  # 2% of x range
    y_offset = 0.02 * (maxy - miny)  # 2% of y range

    # Update x and y limits
    ax.set_xlim(minx - x_offset, maxx + x_offset)
    ax.set_ylim(miny - y_offset, maxy + y_offset)



def plot_distribution(data, ax, xlabel, fontsize=8, bottom=0.4, top=0.85):
    ax.axis('off')  

    # Get position of original ax
    pos = ax.get_position()  # [x0, y0, width, height]
    x0, y0, width, height = pos.x0, pos.y0, pos.width, pos.height

    # Create new ax
    new_y0 = y0 + height * bottom
    new_height = height * (top - bottom)

    # Add new ax to figure
    new_ax = ax.figure.add_axes([x0, new_y0, width, new_height])

    bins = np.logspace(0, 8, num=9, base=10)  # 1, 10, 100, ..., 10^8

    # Calculate percentage of data in each interval (relative to all data)
    counts, _ = np.histogram(data, bins=bins)
    total = counts.sum()
    percentages = (counts / total * 100) if total > 0 else np.zeros_like(counts, dtype=float)

    # Find the 4 intervals with the maximum sum of percentages
    max_sum = -1.0
    start_idx = 0
    for i in range(len(percentages) - 3):
        current_sum = percentages[i:i+4].sum()
        if current_sum > max_sum:
            max_sum = current_sum
            start_idx = i

    # Select boundaries and percentages for 4 intervals
    selected_bins = bins[start_idx:start_idx+5]                 # 5 boundaries for 4 intervals
    selected_percentages = percentages[start_idx:start_idx+4]   # Percentages for 4 intervals (relative to total)

    # Normalize to 100% within these 4 intervals, use largest remainder method for integer percentages (sum=100)
    sel_sum = selected_percentages.sum()
    if sel_sum > 0:
        norm_pct = selected_percentages / sel_sum * 100.0
        floors = np.floor(norm_pct).astype(int)
        remainder = 100 - floors.sum()
        if remainder != 0:
            frac = norm_pct - floors
            order = np.argsort(-frac)  # Prioritize largest remainders for allocation
            # Round-robin allocation, ensures cycling when allocating more than 4
            step = 1 if remainder > 0 else -1
            for k in range(abs(remainder)):
                idx = order[k % 4]
                floors[idx] += step
        label_values = floors  # Four integers, strict sum = 100
    else:
        label_values = np.zeros(4, dtype=int)

    # Create bar chart
    x = np.arange(4)
    bar_width = 1
    colorList = ["green", "yellow", "orange", "red"]
    countBars = new_ax.bar(x, label_values, bar_width, color=colorList, edgecolor='k')

    # Hide top and right borders
    for spine in ["top", "right", "left"]:
        new_ax.spines[spine].set_visible(False)

    new_ax.get_yaxis().set_visible(False)

    # Annotate bars with percentage values
    for bar, pct_int in zip(countBars, label_values):
        height = bar.get_height()
        new_ax.annotate(f'{pct_int}%', fontsize=fontsize,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontname='Arial')

    # Set visibility with bar chart annotation status (margins, boundaries)
    new_ax.set_xticks(np.arange(5) - 0.5)
    new_ax.set_xticklabels(
        [f"$10^{{{int(np.log10(selected_bins[i]))}}}$" for i in range(5)],
        fontsize=fontsize
    )

    # Set x-axis label
    new_ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=2, fontname='Arial')

    # Set x-axis limits
    new_ax.set_xlim(-0.5, 3.5)


# Get raster data and extract boundaries, also perform projection transformation
def get_raster_bounds_with_projection(raster_path, target_crs='EPSG:4326'):
    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        src_transform = src.transform

        dst_transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds
        )

        # Initialize target array as invalid
        dst_data = np.empty((height, width), dtype=rasterio.float32)
        dst_data.fill(np.nan)

        # Perform reprojection
        reproject(
            source=rasterio.band(src, 1),  # Source data (first band)
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

        dst_bounds = array_bounds(height, width, dst_transform)

        return dst_bounds  #  (minx, miny, maxx, maxy)
    

def calculate_overlap(bounds1, bounds2, bounds3):

    x_min1, y_min1, x_max1, y_max1 = bounds1
    x_min2, y_min2, x_max2, y_max2 = bounds2
    x_min3, y_min3, x_max3, y_max3 = bounds3

    # Calculate overlap boundary
    overlap_x_min = max(x_min1, x_min2, x_min3)
    overlap_x_max = min(x_max1, x_max2, x_max3)
    overlap_y_min = max(y_min1, y_min2, y_min3)
    overlap_y_max = min(y_max1, y_max2, y_max3)

    # Check if there is an overlap
    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
        return overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max
    else:
        return None
    

def adjust_bounds_to_ratio(minx, miny, maxx, maxy, ratio=1.0):
    """
    Adjust the bounds to ensure the x and y ranges match the specified ratio.
    The longer side determines the range of the shorter side.

    Args:
    - minx, miny, maxx, maxy: float, the original bounds.
    - ratio: float, the desired x/y ratio (default is 1.0 for square).

    Returns:
    - minx, miny, maxx, maxy: float, the adjusted bounds.
    """
    x_range = maxx - minx
    y_range = maxy - miny
    current_ratio = x_range / y_range

    if current_ratio > ratio:
        # Expand y-range to match the desired ratio
        new_y_range = x_range / ratio
        diff = (new_y_range - y_range) / 2
        miny -= diff
        maxy += diff
    elif current_ratio < ratio:
        # Expand x-range to match the desired ratio
        new_x_range = y_range * ratio
        diff = (new_x_range - x_range) / 2
        minx -= diff
        maxx += diff

    return minx, miny, maxx, maxy

