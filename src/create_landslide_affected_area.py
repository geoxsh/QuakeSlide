"""
Landslide affected area creation module for QuakeSlide system.

This module creates polygon boundaries for earthquake-affected areas based on
PGA threshold values, with options for rectangle or convex hull boundaries.

Date: August 2026
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from osgeo import gdal, osr
from shapely.geometry import box, MultiPoint, mapping
import fiona
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
    nodata = band.GetNoDataValue()
    valid_data = np.isfinite(data)
    if nodata is not None:
        valid_data &= ~np.isclose(data, nodata)

    source_crs = dataset.GetProjection()
    if not source_crs:
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)
        source_crs = source_srs.ExportToWkt()

    # Find pixels where PGA exceeds threshold
    pga_threshold = float(input_path['PGAThreshold'])
    indices = np.where(valid_data & (data > pga_threshold))

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
        polygon = remove_ocean_boundary_from_polygon(
            polygon, input_path, source_crs
        )
        
        if polygon and not polygon.is_empty:
            # Save polygon to shapefile
            save_polygon_to_shapefile(polygon, output_path)
            
            # Calculate and display total affected area
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=source_crs)
            utm_crs = polygon_gdf.estimate_utm_crs()
            polygon_utm = polygon_gdf.to_crs(utm_crs)
            total_area_km2 = polygon_utm.geometry.area.sum() / 1e6  # Convert to km²
            
            print(f"Total landslide-affected area: {int(round(total_area_km2))} km2")
        else:
            print("WARNING: No valid area remains after removing ocean boundaries")
    else:
        print(f"WARNING: No valid polygon created from PGA threshold {pga_threshold}")


def remove_ocean_boundary_from_polygon(polygon, input_path, source_crs="EPSG:4326"):
    """
    Remove ocean areas from the affected area polygon.
    
    Args:
        polygon (shapely.geometry): Input polygon to process
        input_path (dict): Dictionary containing input file paths
        
    Returns:
        shapely.geometry: Modified polygon with ocean areas removed
    """
    try:
        polygon_wgs84 = gpd.GeoSeries(
            [polygon], crs=source_crs
        ).to_crs("EPSG:4326").iloc[0]

        # Load global ocean boundary data
        ocean_boundary_gdf = gpd.read_file(input_path['GlobalOceanBoundaryFile'])
        if ocean_boundary_gdf.crs is None:
            raise ValueError("Ocean boundary dataset has no CRS.")

        # Ensure consistent coordinate system
        if ocean_boundary_gdf.crs != "EPSG:4326":
            ocean_boundary_gdf = ocean_boundary_gdf.to_crs("EPSG:4326")
        
        # Find ocean polygons that intersect with target area
        intersected = ocean_boundary_gdf[
            ocean_boundary_gdf.intersects(polygon_wgs84)
        ]
        
        if len(intersected) == 0:
            return polygon  # No ocean intersection, return original polygon
        
        # Create union of intersecting ocean areas and subtract from polygon
        ocean_boundary_union = unary_union(intersected.geometry)
        modified_wgs84 = polygon_wgs84.difference(ocean_boundary_union)
        return gpd.GeoSeries(
            [modified_wgs84], crs="EPSG:4326"
        ).to_crs(source_crs).iloc[0]
        
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
        
        # Preserve the PGA raster CRS exactly, including non-EPSG custom CRSs.
        dataset = gdal.Open(output_path['PGAFile'])
        projection = dataset.GetProjection()
        if not projection:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            projection = srs.ExportToWkt()
        
        # Write polygon to shapefile
        with fiona.open(
            shapefile_path, 'w', 
            driver='ESRI Shapefile', 
            schema=schema, 
            crs_wkt=projection,
        ) as shp:
            shp.write({
                'geometry': mapping(polygon),
                'properties': {'id': 1},
            })
        
        # print(f"✓ Affected area polygon saved: {shapefile_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save shapefile: {e}")
