"""
Report Generation Module for QuakeSlide System.

This module generates a one-page report for landslide impacts.

Date: August 2026
"""  

# Standard library imports
import json
import os
import time
from datetime import datetime, timezone

# Third-party scientific computing imports
import numpy as np
import geopandas as gpd

from src.shakemap_product import (
    extract_event_metadata,
    fetch_event_geojson,
    format_shakemap_version,
    select_shakemap_product,
)

# Geospatial and mapping imports
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds

# Matplotlib visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString

# QuakeSlide local imports
from src.plot_utils import (
    generate_hillshade, 
    get_cities, 
    plot_raster_map, 
    plot_cities, 
    plot_contours, 
    plot_interruption_map
)
from src.utils import clip_features_raster_files


def make_sharp_star_marker(outer_radius=1.0, inner_radius=0.36):
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 11)[:-1]
    radii = np.array([outer_radius, inner_radius] * 5)
    vertices = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
    vertices = np.vstack([vertices, vertices[0]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


SHARP_STAR_MARKER = make_sharp_star_marker()


def _load_composite_metadata(output_path, supplied_metadata=None):
    """Use current-run metadata or reload it when regenerating a report."""
    if supplied_metadata:
        return supplied_metadata.get("manifest", supplied_metadata)
    manifest_path = os.path.join(output_path['ShakeMap'], "composite_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    return None


def _load_shakemap_source(output_path, supplied_metadata=None):
    """Load the provenance of the PGA/PGV rasters used by this run."""
    if supplied_metadata:
        return supplied_metadata
    metadata_path = output_path.get(
        'ShakeMapSourceFile',
        os.path.join(output_path['ShakeMap'], "shakemap_source.json"),
    )
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    return None


def _format_utc_time(value, compact=False):
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        pattern = "%Y-%m-%d %H:%M UTC" if compact else "%Y-%m-%d %H:%M:%S UTC"
        return parsed.strftime(pattern)
    except (TypeError, ValueError):
        return str(value)


def _format_composite_event(record):
    magnitude = record.get("magnitude")
    magnitude_text = f"M{magnitude:g}" if isinstance(magnitude, (int, float)) else "M unknown"
    return " | ".join([
        magnitude_text,
        str(record.get("event_id", "Unknown")),
        _format_utc_time(record.get("time_utc"), compact=True),
    ])


def report_generation(
        event_id,
        input_path,
        output_path,
        start_time,
        composite_metadata=None,
        event_data=None,
        shakemap_product=None,
        shakemap_source=None,
        processing_time_minutes=None):
    """
    Generate one-page report for landslide impacts.
    
    Args:
        event_id (str): USGS earthquake event ID
        input_path (dict): Dictionary containing input file paths
        output_path (dict): Dictionary containing output file paths
        composite_metadata (dict, optional): Composite event and ShakeMap metadata.
        event_data (dict, optional): USGS event GeoJSON fetched at run start.
        shakemap_product (dict, optional): ShakeMap product used for PGA/PGV.
        shakemap_source (dict, optional): Ground-motion provenance metadata.
        processing_time_minutes (float, optional): Preserved elapsed time when
            regenerating a report from completed assessment outputs.
        
    Returns:
        dict: Paths to the generated PNG, PDF, and estimates text files.
    """

    if event_data is None:
        event_data = fetch_event_geojson(
            event_id, include_superseded=False, timeout=60
        )

    composite_pga_file = os.path.join(output_path['ShakeMap'], "Composite_PGA.tif")
    if os.path.exists(composite_pga_file):
        composite_metadata = _load_composite_metadata(output_path, composite_metadata)
        source_description = "Composite ShakeMap"
        source_label = "Ground-motion source"
    else:
        shakemap_source = _load_shakemap_source(output_path, shakemap_source)
        if shakemap_source and shakemap_source.get("mode") == "custom":
            source_description = shakemap_source.get(
                "description", "Custom PGA/PGV rasters"
            )
            source_label = "Ground-motion source"
        else:
            if shakemap_product is None:
                shakemap_product = select_shakemap_product(
                    event_data, required_content="download/raster.zip"
                )
            source_description = format_shakemap_version(shakemap_product)
            source_label = "ShakeMap product"

    extracted_event_info = get_report_event_info(
        event_id,
        event_data=event_data,
        shakemap=shakemap_product,
        source_description=source_description,
        source_label=source_label,
    )
    # Derive contours from the exact PGA raster used by the assessment. This
    # keeps downloaded, composite, and custom ShakeMap reports consistent.
    generate_pga_contours_from_raster(
        output_path['PGAFile'],
        output_path['PGAContourShpFile'],
        output_path.get('PGAContourJsonFile'),
    )

    report_created_utc = datetime.now(timezone.utc)

    # Set up page dimensions and layout (A4 format)
    mm_to_inch = 0.0393701  # Conversion factor
    page_margins = 5 * mm_to_inch
    page_width = 170 * mm_to_inch
    
    # Define row heights for different sections
    section_heights = np.array([6, 30, 7, 28, 7, 4.5, 55, 7, 55, 10]) * mm_to_inch
    page_height = section_heights.sum() + 2 * page_margins

    # Calculate relative margins
    horizontal_margin = page_margins / page_width
    vertical_margin = page_margins / page_height

    # Create figure with high resolution for print quality
    fig = plt.figure(figsize=(page_width, page_height), dpi=600, constrained_layout=False)

    # Set up grid layout with height ratios
    height_ratios = section_heights / (page_height - 2 * page_margins)

    # Create grid specification for report layout (10 rows total)
    gs = gridspec.GridSpec(
        len(section_heights), 1, 
        figure=fig, 
        height_ratios=height_ratios, 
        wspace=0.0, 
        hspace=0.0,
        left=horizontal_margin/3*2, 
        right=1 - horizontal_margin,
        top=1 - vertical_margin,
        bottom=vertical_margin
    )


    def plot_report_title(ax, fontsize=13):
        title = 'QuakeSlide: Prompt Earthquake-Triggered Landslide Impact Assessment'
        # title = 'Prompt Landslide Impact Assessment'
        ax.text(0.02, 1, title, ha='left', va='top', weight="bold", fontsize=fontsize)
        # ax.add_patch(Rectangle(xy=(0, 0.2), width=0.005, height=0.6, facecolor='k', edgecolor='k'))
        ax.axis('off')

    # The 1st row contains the logo, the report title and the report information
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    ax00 = fig.add_subplot(gs0[0, 0])
    plot_report_title(ax00, fontsize=11.5)

    # Generate current UTC timestamp
    nowString_utc = report_created_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    epicenter_text = (
        f"{abs(float(extracted_event_info.get('Latitude', 0))):.3f}°"
        f"{'N' if float(extracted_event_info.get('Latitude', 0)) >= 0 else 'S'}, "
        f"{abs(float(extracted_event_info.get('Longitude', 0))):.3f}°"
        f"{'E' if float(extracted_event_info.get('Longitude', 0)) >= 0 else 'W'}"
    )
    if composite_metadata:
        component_events = composite_metadata.get("component_events", [])
        mainshock = next(
            (record for record in component_events if record.get("role") == "mainshock"),
            None,
        )
        additional_events = [
            record for record in component_events if record.get("role") != "mainshock"
        ]
        data = [
            ["Report created", nowString_utc],
            ["Assessment", f"Composite ShakeMap ({len(component_events)} earthquakes)"],
            ["Mainshock", _format_composite_event(mainshock or {})],
        ]
        for index, record in enumerate(additional_events, start=1):
            label = "Additional event" if len(additional_events) == 1 else f"Additional event {index}"
            data.append([label, _format_composite_event(record)])
        data.extend([
            ["Combination", "Pixelwise maximum PGA and PGV"],
            ["Place", extracted_event_info.get("Place")],
            ["Epicenter", epicenter_text],
            ["Depth", extracted_event_info.get("Depth")],
        ])
    else:
        data = [
            ["Report created", nowString_utc],
            [extracted_event_info.get("SourceLabel"), extracted_event_info.get("Version")],
            ["Place", extracted_event_info.get("Place")],
            ["Magnitude", extracted_event_info.get("Magnitude")],
            ["Time", _format_utc_time(extracted_event_info.get("Time"))],
            ["Epicenter", epicenter_text],
            ["Depth", extracted_event_info.get("Depth")],
        ]

    def plot_earthquake_info(data, ax, fontsize):
        # Calculate max label width
        max_label_length = max(len(label) for label, _ in data)

        # Add black border accent
        ax.add_patch(Rectangle(xy=(0.08, 0.05), width=0.006, height=0.9, facecolor='k', edgecolor='k'))

        # Set smaller vertical spacing
        num_lines = len(data)
        y_spacing = 0.82 / (num_lines - 1) if num_lines > 1 else 0.70  # Original was 0.92

        for i, (label, value) in enumerate(data):
            y_position = 0.95 - i * y_spacing
            ax.text(0.10, y_position, f'{label}:', ha='left', va='top', fontsize=fontsize)
            ax.text(0.10 + max_label_length * 0.02 - 0.02, y_position, value, ha='left', va='top', fontsize=fontsize)

        # Hide coordinate axes
        ax.axis('off')

    # Earthquake longitude and latitude information
    lat = extracted_event_info.get('Latitude')
    lon = extracted_event_info.get('Longitude')

    # The 2nd row contains the earthquake information and the global map showing earthquake location
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.6, 0.4])
    ax10 = fig.add_subplot(gs1[0, 0])
    plot_earthquake_info(data, ax10, fontsize=7.3 if composite_metadata else 8)
    ax11 = fig.add_subplot(gs1[0, 1])
    plot_global_map(lat, lon, ax11)



    # ---------------Impact summary

    def plot_subtitle(ax, text, ha='left', fontsize=10, weight="bold", offset=0.02, y=0.1):
        if ha == 'left':
            ax.text(0+offset, y, text, ha='left', va='bottom', weight=weight, fontsize=fontsize)
        else:
            ax.text(0.5+offset, y, text, ha='center', va='bottom', weight=weight, fontsize=fontsize)
        ax.axis('off')


    # The 3th row contains the landslide title
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2])
    ax20 = fig.add_subplot(gs2[0, 0])
    # plot_subtitle(ax20, 'Estimated Landslide Impacts')
    plot_subtitle(ax20, 'Event-scale severity estimates', fontsize=9, weight="bold", offset=0.04)

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

    # ----------------------------
    # The 4th row contains the landslide table
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[3],width_ratios=[0, 1, 1, 1, 1, 1, 0],wspace=0.5)
    ax30 = fig.add_subplot(gs3[0, 1])
    plot_distribution(total_counts, ax30, "Total landslide count\n(landslides)", fontsize=7)
    ax31 = fig.add_subplot(gs3[0, 2])
    plot_distribution(total_areas, ax31, "Total landslide area\n(km\u00b2)", fontsize=7)
    ax32 = fig.add_subplot(gs3[0, 3])
    plot_distribution(total_population_exposure, ax32, "Total population exposure\n(people)", fontsize=7)


    if os.path.exists(output_path['RoadInterruptionsNoDataFile']):
        print("No road network found in the landslide-affected area.")
        total_road_interruptions = np.array([0])
    else:
        total_road_interruptions = np.load(output_path['TotalRoadInterruptionsFile']) # unit: count
        ax33 = fig.add_subplot(gs3[0, 4])
        plot_distribution(total_road_interruptions, ax33, "Total road interruptions\n(interruptions)", fontsize=7)

    if os.path.exists(output_path['RiverInterruptionsNoDataFile']):
        print("No river network found in the landslide-affected area.")
        total_river_interruptions = np.array([0])
    else:
        total_river_interruptions = np.load(output_path['TotalRiverInterruptionsFile']) # unit: count
        ax34 = fig.add_subplot(gs3[0, 5])
        plot_distribution(total_river_interruptions, ax34, "Total river interruptions\n(interruptions)", fontsize=7)

    # ----------------------map extent

    polygon = gpd.read_file(output_path['LandslideAffectedArea'])
    polygon = polygon.to_crs(epsg=4326)

    # Get boundary extent
    polygon_bounds = polygon.total_bounds


    # Get landslide areal coverage bounds with projection
    landslide_file_bounds = get_raster_bounds_with_projection(output_path['MeanArealCoveragesRasterFile'])
    exposure_file_bounds = get_raster_bounds_with_projection(output_path['MeanPopulationExposureDensityRasterFile'])

    # Calculate overlap area
    minx, miny, maxx, maxy = calculate_overlap(landslide_file_bounds, exposure_file_bounds, polygon_bounds)

    minx, miny, maxx, maxy = adjust_bounds_to_ratio(minx, miny, maxx, maxy, ratio=1.1)


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

    gs4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[4])
    ax4_title = fig.add_subplot(gs4[0, 0])
    plot_subtitle(ax4_title, 'Spatial mean estimates', fontsize=9, weight="bold", offset=0.04)

    gs5_title = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5], width_ratios=[0.5, 0.5], wspace=0.02)
    ax40 = fig.add_subplot(gs5_title[0, 0]) 

    plot_subtitle(ax40, 'Landslide areal coverage (fraction)', fontsize=8, weight="normal", offset=0.11)
    # ax41 = fig.add_subplot(gs4[0, 2])
    ax41 = fig.add_subplot(gs5_title[0, 1])
    plot_subtitle(ax41, 'Population exposure density (people/km\u00b2)', fontsize=8, weight="normal", offset=0.11)


    # The 6th row contains two maps: Estimated landslide counts and Estimated population exposure
    # gs5 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[5], width_ratios=[0, 0.5, 0.5, 0])
    # ax50 = fig.add_subplot(gs5[0, 1])
    gs5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6], width_ratios=[0.5, 0.5], wspace=0.02)
    ax50 = fig.add_subplot(gs5[0, 0])
   
    plot_raster_map(output_path['MeanArealCoveragesRasterFile'], ax50, output_path['ClippedHillshadeFile'], intersection, plot="areal percentage colorbar", is_ticks=True) # areal percentage (unit: 100%)
    ax50.set_xlim(minx, maxx)
    ax50.set_ylim(miny, maxy)
    try:
        epicenter_lat = float(extracted_event_info.get('Latitude'))
        epicenter_lon = float(extracted_event_info.get('Longitude'))
        ax50.scatter(
            epicenter_lon, epicenter_lat,
            marker=SHARP_STAR_MARKER, s=105,
            facecolors='black', edgecolors='none',
            linewidths=0, zorder=13
        )
    except (TypeError, ValueError):
        pass
    
    plot_cities(cities, ax50, is_label=True)
    # Overlay contour lines
    plot_contours(output_path['PGAContourShpFile'], ax50, target_crs="EPSG:4326", pga_threshold=0.1)  # ??PGA  0.1g 
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


    gs6 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[7], width_ratios=[0.5, 0.5], wspace=0.02)
    gs7 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[8], width_ratios=[0.5, 0.5], wspace=0.02)

    
    if not os.path.exists(output_path['RoadInterruptionsNoDataFile']):
        ax60 = fig.add_subplot(gs6[0, 0])
        plot_subtitle(ax60, 'Road interruption density (interruptions/km)', fontsize=8, weight="normal", offset=0.11)
        ax70 = fig.add_subplot(gs7[0, 0])
        plot_interruption_map(output_path['MeanRoadInterruptionsDensityFile'], ax70, output_path['ClippedHillshadeFile'], intersection, plot_column="mean_d", plot="road", fontsize=8, is_cbar=True, is_ticks=True)
        ax70.set_xlim(minx, maxx)
        ax70.set_ylim(miny, maxy)
        ax70.set_aspect('equal')
    else:
        ax60 = fig.add_subplot(gs6[0, 0]) 
        plot_subtitle(ax60, 'No road network in this area', fontsize=8, weight="normal", offset=0.11)


    if not os.path.exists(output_path['RiverInterruptionsNoDataFile']):
        ax61 = fig.add_subplot(gs6[0, 1])
        plot_subtitle(ax61, 'River interruption density (interruptions/km)', fontsize=8, weight="normal", offset=0.11)
        ax71 = fig.add_subplot(gs7[0, 1])
        plot_interruption_map(output_path['MeanRiverInterruptionsDensityFile'], ax71, output_path['ClippedHillshadeFile'], intersection, plot_column="mean_d", plot="river", fontsize=8, is_cbar=True, is_ticks=True)
        ax71.set_xlim(minx, maxx)
        ax71.set_ylim(miny, maxy)
        ax71.set_aspect('equal')
    else:
        ax61 = fig.add_subplot(gs6[0, 1])
        plot_subtitle(ax61, 'No river network in this area', fontsize=8, weight="normal", offset=0.11)


    def plot_footnote(ax, fontsize=7):
        footnoteText = (
        "This report provides first-order quantitative estimates of potential earthquake-triggered landslide hazards and impacts.\n"
        "Earthquake information and ground motion data are sourced from the USGS ShakeMap.\n"
        "Estimates can be updated when revised ShakeMap products or preliminary landslide observations become available."
    )
        ax.text(0.04, 0, footnoteText, ha='left', va='center', style='italic', fontsize=fontsize)
        ax.axis('off')


    # The 10th row contains the footnote
    gs8 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[9])
    ax80 = fig.add_subplot(gs8[0, 0])
    plot_footnote(ax80)

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"

    current_time = report_created_utc.strftime("%Y%m%d_%H%M%S")
 
    output_path_PNG = os.path.join(output_path['Report'], f'{current_time}_ImpactReport.PNG')
    output_path_PDF = os.path.join(output_path['Report'], f'{current_time}_ImpactReport.PDF')
    fig.savefig(output_path_PNG, pad_inches=0)
    fig.savefig(output_path_PDF, pad_inches=0)
    plt.close(fig)

    # Save QuakeSlide estimates to a text file.
    QUANTILES = [2.5, 10, 25, 50, 75, 90, 97.5]

    def calc_quantiles(data, quantiles=QUANTILES):
        return [np.percentile(data, q) for q in quantiles]

    # Calculate quantiles for each metric
    count_q = calc_quantiles(total_counts)
    area_q  = calc_quantiles(total_areas)
    pop_q   = calc_quantiles(total_population_exposure)
    road_q = calc_quantiles(total_road_interruptions)
    river_q = calc_quantiles(total_river_interruptions)

    # Build event metadata from extracted_event_info
    event_link      = f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}/executive"
    event_place     = extracted_event_info.get("Place", "")
    event_mag       = extracted_event_info.get("Magnitude", "")
    event_time_raw  = extracted_event_info.get("Time", "")
    event_time_str = _format_utc_time(event_time_raw).replace(" UTC", "")

    # Write to file
    output_path_QuakeSlide_estimates = os.path.join(output_path['AssessmentResults'], f'{current_time}_QuakeSlide_estimates.txt')
    with open(output_path_QuakeSlide_estimates, 'w', encoding='utf-8') as f:
        f.write("QuakeSlide Estimates\n")
        f.write("=" * 90 + "\n")

        # Event info block
        if composite_metadata:
            f.write(f"Run ID:             {composite_metadata.get('run_id', output_path.get('RunID', ''))}\n")
            f.write("Assessment:         Composite ShakeMap\n")
            f.write(f"Mainshock event ID: {event_id}\n")
            f.write(f"Mainshock name:     {event_place}\n")
            f.write(f"Mainshock magnitude:{event_mag:>6}\n")
            f.write(f"Mainshock date UTC: {event_time_str}\n")
            f.write(f"Mainshock link:     {event_link}\n")
            f.write("Combination:        Pixelwise maximum PGA and PGV\n")
            f.write("Component earthquakes:\n")
            for record in composite_metadata.get("component_events", []):
                role = "Mainshock" if record.get("role") == "mainshock" else "Additional event"
                shakemap = record.get("shakemap", {})
                f.write(
                    f"  - {role}: {_format_composite_event(record)}"
                    f" | {record.get('place', '')}"
                    f" | ({record.get('latitude')}, {record.get('longitude')})"
                    f" | depth {record.get('depth_km')} km"
                    f" | {shakemap.get('description', 'ShakeMap version unknown')}\n"
                )
        else:
            f.write(f"Event ID:   {event_id}\n")
            f.write(f"Name:       {event_place}\n")
            f.write(f"Magnitude:  {event_mag}\n")
            f.write(f"Date (UTC): {event_time_str}\n")
            f.write(f"Link:       {event_link}\n")
        f.write(f"Generated (UTC): {current_time}\n")
        f.write("=" * 90 + "\n\n")

        # Calculate total affected area
        polygon = gpd.read_file(output_path['LandslideAffectedArea'])
        polygon_utm = polygon.to_crs(polygon.estimate_utm_crs())
        total_area_km2 = polygon_utm.geometry.area.sum() / 1e6  # Convert to km²
        f.write(f"Total landslide-affected area: {int(round(total_area_km2))} km²\n\n")

        # Quantile table
        q_labels = ["2.5%", "10%", "25%", "50%", "75%", "90%", "97.5%"]
        col_w = 11  # width of each quantile column

        metrics = [
            ("Total landslide count",       count_q, None),
            ("Total landslide area (km2)",   area_q,  None),
            ("Total population exposure",    pop_q,   None),
        ]
        if not os.path.exists(output_path['RiverInterruptionsNoDataFile']):
            metrics.append(("Total river interruptions", river_q, None))
        else:
            metrics.append(("Total river interruptions", None, "No river network found in the landslide-affected area"))
        if not os.path.exists(output_path['RoadInterruptionsNoDataFile']):
            metrics.append(("Total road interruptions",  road_q,  None))
        else:
            metrics.append(("Total road interruptions",  None,    "No road network found in the landslide-affected area"))

        name_w = max(len(m[0]) for m in metrics) + 2
        header = f"{'Metric':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in q_labels)
        sep    = "-" * len(header)
        f.write(header + "\n")
        f.write(sep + "\n")
        for name, q_vals, note in metrics:
            if q_vals is None:
                f.write(f"{name:<{name_w}}{note}\n")
            else:
                vals_str = "".join(f"{v:>{col_w},.0f}" for v in q_vals)
                f.write(f"{name:<{name_w}}{vals_str}\n")
        f.write("\n")

        # Summary
        elapsed_time = (
            float(processing_time_minutes)
            if processing_time_minutes is not None
            else (time.time() - start_time) / 60
        )
        f.write(f"Total Processing Time: {elapsed_time:.2f} minutes\n")

    print(f"QuakeSlide estimates saved to: {output_path_QuakeSlide_estimates}")
    return {
        'png': output_path_PNG,
        'pdf': output_path_PDF,
        'estimates': output_path_QuakeSlide_estimates,
    }

# =======================================================================
# UTILITY FUNCTIONS FOR REPORT GENERATION
# =======================================================================

def get_report_event_info(
        event_id,
        event_data=None,
        shakemap=None,
        source_description=None,
        source_label="ShakeMap product"):
    if event_data is None:
        event_data = fetch_event_geojson(
            event_id,
            include_superseded=False,
            timeout=60,
        )
    metadata = extract_event_metadata(event_data, event_id=event_id)
    if source_description is None:
        if shakemap is None:
            shakemap = select_shakemap_product(
                event_data, required_content="download/raster.zip"
            )
        source_description = format_shakemap_version(shakemap)

    magnitude = metadata["magnitude"]
    return {
        "Version": source_description,
        "SourceLabel": source_label,
        "Place": metadata["place"],
        "Magnitude": f"Mw {magnitude:g}" if magnitude is not None else "Unknown",
        "Time": metadata["time_utc"],
        "Latitude": metadata["latitude"],
        "Longitude": metadata["longitude"],
        "Depth": f"{metadata['depth_km']:.1f} km",
    }


def _remove_shapefile_if_exists(shapefile_path):
    root, _ = os.path.splitext(shapefile_path)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        sidecar = root + ext
        if os.path.exists(sidecar):
            os.remove(sidecar)


def _chaikin_smooth_line(line, iterations=3):
    """Round a contour polyline without fitting it away from the original path."""
    if iterations <= 0 or line.is_empty:
        return line

    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 4:
        return line

    is_closed = np.allclose(coords[0], coords[-1])
    work = coords[:-1] if is_closed else coords

    for _ in range(iterations):
        if len(work) < 3:
            break

        new_coords = []
        if not is_closed:
            new_coords.append(work[0])

        pairs = zip(work, np.roll(work, -1, axis=0)) if is_closed else zip(work[:-1], work[1:])
        for p0, p1 in pairs:
            new_coords.append(0.75 * p0 + 0.25 * p1)
            new_coords.append(0.25 * p0 + 0.75 * p1)

        if is_closed:
            new_coords.append(new_coords[0])
        else:
            new_coords.append(work[-1])

        work = np.asarray(new_coords, dtype=float)

    smoothed = LineString(work)
    return smoothed if not smoothed.is_empty and smoothed.length > 0 else line


def generate_pga_contours_from_raster(
        pga_raster_path,
        shapefile_path,
        geojson_path=None,
        smooth_sigma=2.5,
        max_lines_per_level=2,
        line_smoothing_iterations=3):
    """
    Generate PGA contour lines from a PGA GeoTIFF.

    PGA raster values are expected to be in g. The output uses a `value` field
    in percent g, matching USGS cont_pga.json conventions used by plot_contours:
    value=10 corresponds to 0.1 g.
    """
    contour_levels_g = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=float)

    with rasterio.open(pga_raster_path) as src:
        pga = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            pga[pga == nodata] = np.nan
        pga[~np.isfinite(pga)] = np.nan
        pga[pga <= 0] = np.nan

        if not np.isfinite(pga).any():
            raise ValueError(f"No valid PGA values found in {pga_raster_path}")

        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter

                valid = np.isfinite(pga)
                filled = np.where(valid, pga, 0.0)
                weights = valid.astype(float)
                smooth_values = gaussian_filter(filled, sigma=smooth_sigma)
                smooth_weights = gaussian_filter(weights, sigma=smooth_sigma)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pga = smooth_values / smooth_weights
                pga[smooth_weights <= 0] = np.nan
            except Exception as e:
                print(f"WARNING: PGA contour smoothing skipped: {type(e).__name__}: {e}")

        data_min = np.nanmin(pga)
        data_max = np.nanmax(pga)
        levels = contour_levels_g[(contour_levels_g >= data_min) & (contour_levels_g <= data_max)]
        if len(levels) == 0 and data_max >= 0.1:
            levels = np.array([0.1], dtype=float)

        transform = src.transform
        crs = src.crs or "EPSG:4326"
        cols = np.arange(src.width)
        rows = np.arange(src.height)
        if transform.b == 0 and transform.d == 0:
            xs = transform.c + (cols + 0.5) * transform.a
            ys = transform.f + (rows + 0.5) * transform.e
            x_grid, y_grid = np.meshgrid(xs, ys)
        else:
            col_grid, row_grid = np.meshgrid(cols, rows)
            x_grid, y_grid = rasterio.transform.xy(transform, row_grid, col_grid, offset="center")
            x_grid = np.asarray(x_grid)
            y_grid = np.asarray(y_grid)

    geometries = []
    values = []
    if len(levels) > 0:
        fig_tmp, ax_tmp = plt.subplots()
        contour_set = ax_tmp.contour(x_grid, y_grid, pga, levels=levels)
        for level_g, collection in zip(contour_set.levels, contour_set.collections):
            level_lines = []
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) < 2:
                    continue
                line = LineString(vertices)
                if line.is_empty or line.length == 0:
                    continue
                level_lines.append(line)

            if not level_lines:
                continue

            lengths = np.array([line.length for line in level_lines], dtype=float)
            min_length = max(np.nanmax(lengths) * 0.08, 0.0)
            kept_lines = [
                line for line in level_lines
                if line.length >= min_length
            ]
            kept_lines = sorted(kept_lines, key=lambda geom: geom.length, reverse=True)
            kept_lines = kept_lines[:max_lines_per_level]

            for line in kept_lines:
                line = _chaikin_smooth_line(line, iterations=line_smoothing_iterations)
                geometries.append(line)
                values.append(float(level_g) * 100.0)
        plt.close(fig_tmp)

    contours_gdf = gpd.GeoDataFrame(
        {"value": values},
        geometry=geometries,
        crs=crs,
    )

    os.makedirs(os.path.dirname(shapefile_path), exist_ok=True)
    _remove_shapefile_if_exists(shapefile_path)
    contours_gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    if geojson_path is not None:
        contours_gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"PGA contours generated from assessment raster: {shapefile_path}")
    return shapefile_path


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
    latitude = float(latitude)
    longitude = float(longitude)
    ax.set_frame_on(False)
    ax.patch.set_alpha(0)

    # Smooth radial halo behind the globe.
    halo_size = 250
    yy, xx = np.mgrid[0:halo_size, 0:halo_size]
    x_norm = (xx + 0.5) / halo_size
    y_norm = (yy + 0.5) / halo_size
    radius = np.sqrt((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2)

    halo_peak_radius = 0.472    # Radius of maximum halo intensity.
    halo_width = 0.022          # Controls the softness of the halo.
    halo_strength = 0.10        # Maximum halo opacity.
    halo_inner_radius = 0.445   # Inner transparency cutoff.
    halo_outer_radius = 0.525   # Outer transparency cutoff.

    alpha = np.exp(-((radius - halo_peak_radius) / halo_width) ** 2) * halo_strength
    alpha[radius < halo_inner_radius] = 0.0
    alpha[radius > halo_outer_radius] = 0.0

    halo = np.zeros((halo_size, halo_size, 4), dtype=float)
    halo[..., :3] = np.array([38, 70, 83]) / 255.0
    halo[..., 3] = alpha

    ax.imshow(
        halo,
        extent=(-0.015, 1.015, -0.015, 1.015),
        transform=ax.transAxes,
        origin='lower',
        interpolation='bicubic',
        zorder=0,
        clip_on=False
    )

    # Create Basemap instance with orthographic projection
    m = Basemap(projection='ortho', 
                lat_0=latitude, lon_0=longitude,  # Center on earthquake epicenter
                resolution='l', ax=ax)

    # Fill ocean and continent colors
    m.drawmapboundary(fill_color='#b8deeb', linewidth=0, color='none')
    m.fillcontinents(color='white', lake_color='#b8deeb', zorder=2)
    # m.drawcoastlines(color='black', linewidth=0.22, zorder=3)
    
    # Draw coordinate grid
    m.drawparallels(range(-90, 90, 30), linewidth=0.20, color='white',
                    alpha=0.9, dashes=[1, 0], zorder=4)
    m.drawmeridians(range(0, 360, 30), linewidth=0.20, color='white',
                    alpha=0.9, dashes=[1, 0], zorder=4)

    # Convert lat/lon to map coordinates
    x, y = m(longitude, latitude)

    # Plot earthquake epicenter as a sharp black star
    m.scatter(x, y, marker=SHARP_STAR_MARKER, color='black', s=160, linewidths=0, zorder=10)

    # Get x and y limits
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()

    # Calculate offsets
    x_offset = 0.02 * (maxx - minx)  # 2% of x range
    y_offset = 0.02 * (maxy - miny)  # 2% of y range

    # Update x and y limits
    ax.set_xlim(minx - x_offset, maxx + x_offset)
    ax.set_ylim(miny - y_offset, maxy + y_offset)
    ax.set_axis_off()



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

