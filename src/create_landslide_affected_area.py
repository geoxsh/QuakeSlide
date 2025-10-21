"""
Landslide affected area creation module for QuakeSlide system.

This module creates polygon boundaries for earthquake-affected areas based on
PGA threshold values, with options for rectangle or convex hull boundaries.

Author: Shihao Xiao
Date: Oct 2025
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from osgeo import gdal, osr
from shapely.geometry import box, MultiPoint, mapping
import fiona
from fiona.crs import from_epsg
import geopandas as gpd
from shapely.ops import unary_union


def create_landslide_affected_area(input_path, output_path):
    """
    Create earthquake-affected area polygon based on PGA threshold values.
    
    Args:
        input_path (dict): Dictionary containing input parameters and file paths
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None
    """
    # Open and read PGA raster file
    dataset = gdal.Open(output_path['PGAFile'])
    if dataset is None:
        print("ERROR: Could not open PGA raster file")
        return
        
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # Find pixels where PGA exceeds threshold
    pga_threshold = float(input_path['PGAThreshold'])
    indices = np.where(data > pga_threshold)

    if indices[0].size == 0 or indices[1].size == 0:
        print(f"WARNING: No PGA values greater than {pga_threshold} found")
        return

    # Get geotransformation for coordinate conversion
    transform = dataset.GetGeoTransform()

    # Create boundary polygon based on user preference
    is_rectangle = input_path.get('IS_rectangle_boundary', 'False') == 'True'
    
    if is_rectangle:
        # Create rectangular boundary around affected pixels
        min_x_pixel = np.min(indices[1])
        max_x_pixel = np.max(indices[1])
        min_y_pixel = np.min(indices[0])
        max_y_pixel = np.max(indices[0])

        # Convert pixel coordinates to geographic coordinates
        min_x_geo = transform[0] + min_x_pixel * transform[1] + min_y_pixel * transform[2]
        max_x_geo = transform[0] + (max_x_pixel + 1) * transform[1] + (max_y_pixel + 1) * transform[2]
        min_y_geo = transform[3] + min_x_pixel * transform[4] + min_y_pixel * transform[5]
        max_y_geo = transform[3] + (max_x_pixel + 1) * transform[4] + (max_y_pixel + 1) * transform[5]

        polygon = box(min_x_geo, min_y_geo, max_x_geo, max_y_geo)
        
    else:
        # Create convex hull boundary around affected pixels
        points = []
        for y, x in zip(indices[0], indices[1]):
            geo_x = transform[0] + x * transform[1] + y * transform[2]
            geo_y = transform[3] + x * transform[4] + y * transform[5]
            points.append((geo_x, geo_y))
        polygon = MultiPoint(points).convex_hull

    if polygon and not polygon.is_empty:
        # Remove ocean areas from the polygon
        polygon = remove_ocean_boundary_from_polygon(polygon, input_path)
        
        if polygon and not polygon.is_empty:
            # Save polygon to shapefile
            save_polygon_to_shapefile(polygon, output_path)
            
            # Calculate and display total affected area
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
            utm_crs = polygon_gdf.estimate_utm_crs()
            polygon_utm = polygon_gdf.to_crs(utm_crs)
            total_area_km2 = polygon_utm.geometry.area.sum() / 1e6  # Convert to km²
            
            print(f"Total landslide-affected area: {int(round(total_area_km2))} km²")
        else:
            print("WARNING: No valid area remains after removing ocean boundaries")
    else:
        print(f"WARNING: No valid polygon created from PGA threshold {pga_threshold}")


def remove_ocean_boundary_from_polygon(polygon, input_path):
    """
    Remove ocean areas from the affected area polygon.
    
    Args:
        polygon (shapely.geometry): Input polygon to process
        input_path (dict): Dictionary containing input file paths
        
    Returns:
        shapely.geometry: Modified polygon with ocean areas removed
    """
    try:
        # Load global ocean boundary data
        ocean_boundary_gdf = gpd.read_file(input_path['GlobalOceanBoundaryFile'])
        
        # Ensure consistent coordinate system
        if ocean_boundary_gdf.crs != "EPSG:4326":
            ocean_boundary_gdf = ocean_boundary_gdf.to_crs("EPSG:4326")
        
        # Find ocean polygons that intersect with target area
        intersected = ocean_boundary_gdf[ocean_boundary_gdf.intersects(polygon)]
        
        if len(intersected) == 0:
            return polygon  # No ocean intersection, return original polygon
        
        # Create union of intersecting ocean areas and subtract from polygon
        ocean_boundary_union = unary_union(intersected.geometry)
        modified_polygon = polygon.difference(ocean_boundary_union)
        
        return modified_polygon
        
    except Exception as e:
        print(f"WARNING: Could not process ocean boundaries: {e}")
        return polygon  # Return original polygon if processing fails


def save_polygon_to_shapefile(polygon, output_path):
    """
    Save polygon geometry to ESRI Shapefile format.
    
    Args:
        polygon (shapely.geometry): Polygon to save
        output_path (dict): Dictionary containing output file paths
        
    Returns:
        None
    """
    shapefile_path = output_path['LandslideAffectedArea']
    
    try:
        # Define shapefile schema
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }
        
        # Get projection from PGA raster file
        dataset = gdal.Open(output_path['PGAFile'])
        srs = osr.SpatialReference(wkt=dataset.GetProjection())
        epsg_code = srs.GetAttrValue('AUTHORITY', 1)
        
        # Default to WGS84 if no EPSG code found
        if epsg_code is None:
            epsg_code = 4326
        
        # Write polygon to shapefile
        with fiona.open(
            shapefile_path, 'w', 
            driver='ESRI Shapefile', 
            schema=schema, 
            crs=from_epsg(int(epsg_code))
        ) as shp:
            shp.write({
                'geometry': mapping(polygon),
                'properties': {'id': 1},
            })
        
        # print(f"✓ Affected area polygon saved: {shapefile_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save shapefile: {e}")
