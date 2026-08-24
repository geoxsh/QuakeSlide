"""
Plot Utilities Module for QuakeSlide System.

This module provides comprehensive plotting and visualization functions.

Date: August 2026
"""

# Third-party imports
import earthpy.spatial as es
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point, LineString, MultiLineString


def generate_hillshade(dem_file, hillshade_file):
    """
    Generate hillshade from Digital Elevation Model (DEM).
    
    Args:
        dem_file (str): Path to input DEM raster file
        hillshade_file (str): Path to output hillshade raster file
        
    Returns:
        None: Hillshade is saved to specified output file
    """
    # Read DEM data and metadata
    with rasterio.open(dem_file) as src:
        dem = src.read(1).astype(np.float64)
        raster_meta = src.meta
        nodata_mask = dem == float(src.nodata)
        dem[nodata_mask] = np.nan
    
    # Update metadata for hillshade output
    raster_meta.update(driver='GTiff', nodata=0, dtype=rasterio.uint8)
    
    # Generate hillshade with standard illumination parameters
    hillshade = es.hillshade(dem, azimuth=315, altitude=45)
    hillshade[nodata_mask] = 0

    # Save hillshade to file
    with rasterio.open(hillshade_file, 'w', **raster_meta) as dst:
        dst.write(hillshade.astype(rasterio.uint8), 1)


def get_cities(global_cities_csv_path, max_n_cities, minx, miny, maxx, maxy):
    """
    Extract cities within specified geographic bounds from global cities dataset.
    
    Args:
        global_cities_csv_path (str): Path to global cities CSV file (worldcities.csv)
        max_n_cities (int): Maximum number of cities to return
        minx, miny, maxx, maxy (float): Bounding box coordinates
        
    Returns:
        GeoDataFrame: Cities within bounds, limited to top N by population
        None: If no cities found in the specified area
    """
    # Load global cities data
    cities_df = pd.read_csv(global_cities_csv_path)
    
    # Filter cities within geographic bounds
    cities_in_bounds = cities_df[
        (cities_df['lat'] >= miny) & (cities_df['lat'] <= maxy) & 
        (cities_df['lng'] >= minx) & (cities_df['lng'] <= maxx)
    ]
    
    if cities_in_bounds.empty:
        print("No cities found within the specified geographic bounds.")
        return None
    
    # Convert to GeoDataFrame with point geometries
    cities_gdf = gpd.GeoDataFrame(
        cities_in_bounds, 
        geometry=[Point(lng, lat) for lng, lat in zip(cities_in_bounds['lng'], cities_in_bounds['lat'])], 
        crs="EPSG:4326"
    )
    
    # Filter out minor cities and select relevant columns
    cities_filtered = cities_gdf[cities_gdf['capital'] != 'minor']
    cities_clean = cities_filtered[['geometry', 'city_ascii', 'population', 'capital']].rename(
        columns={'city_ascii': 'name'}
    )
    
    # Limit to top N cities by population if necessary
    if len(cities_clean) > max_n_cities:
        original_count = len(cities_clean)
        cities_clean = cities_clean.sort_values('population', ascending=False).head(max_n_cities)
        print(f"Found {original_count} cities, filtered to top {max_n_cities} by population.")
    
    return cities_clean


def plot_raster_map(raster_path, ax, hillshade_file, intersection, plot="counts density colorbar", is_ticks=False, fontsize=8):
    """
    Plot raster data on map with hillshade background and optional colorbar.
    
    Projects raster to WGS84 (EPSG:4326) if necessary for consistent visualization.
    
    Args:
        raster_path (str): Path to input raster file
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting
        hillshade_file (str): Path to hillshade raster for background
        intersection (dict): Geographic bounds for plotting
        plot (str): Plot type - "counts", "density", "colorbar", etc.
        is_ticks (bool): Whether to show coordinate ticks
        fontsize (int): Font size for labels and text
        
    Returns:
        None: Plots directly to provided axes
    """
    # Plot hillshade as background layer
    hillshade_norm = plt.Normalize(-100, 255)
    with rasterio.open(hillshade_file) as hillshade_src:
        show(hillshade_src, cmap='Greys_r', ax=ax, norm=hillshade_norm, alpha=0.2)

    # Read and process raster data
    with rasterio.open(raster_path) as src:
        # Get original CRS and transform
        src_crs = src.crs
        src_transform = src.transform

        # Validate CRS definition
        if src_crs is None:
            raise ValueError("Input raster data does not have a CRS defined.")

        # Target projection system (WGS84)
        target_crs = "EPSG:4326"

        if src_crs.to_string() == target_crs:
            # Raster is already in WGS84, use directly
            raster_data = src.read(1)
            final_transform = src_transform
        else:
            # Reproject raster to WGS84
            print("Reprojecting raster data to WGS 1984 (EPSG:4326).")
            final_transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )

            # Create target array and initialize
            raster_data = np.empty((height, width), dtype=rasterio.float32)
            raster_data.fill(np.nan)

            # Perform reprojection
            reproject(
                source=rasterio.band(src, 1),
                destination=raster_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=final_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
            )

    # Handle invalid values
    raster_data[np.isnan(raster_data)] = 0

    # Calculate extent for plotting
    extent = (
        final_transform.c,
        final_transform.c + final_transform.a * raster_data.shape[1],
        final_transform.f + final_transform.e * raster_data.shape[0],
        final_transform.f,
    )



    # Reserve a separate axis for the colorbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    
    # make a color map of fixed colors
    if plot == "counts density colorbar":
        # cmap = ListedColormap(plt.get_cmap('Greys')([0, 0.25, 0.5, 0.75, 1]))  #Blues
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(plt.get_cmap('Blues')([0, 0.35, 0.7, 1]))  #Blues;Greys
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1])))
        bounds = [0, 0.1, 1, 5, 10]
    elif plot == "areal percentage colorbar":
        # cmap = ListedColormap(plt.get_cmap('Reds')([0, 0.25, 0.5, 0.75, 1]))
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.25, 0.5, 0.75, 1])))
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.35, 0.7, 1])))
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1])))
        bounds = [0, 0.01, 0.05, 0.1, 0.2]
    elif plot == "exposure density colorbar":
        # cmap = ListedColormap(plt.get_cmap('Reds')([0, 0.25, 0.5, 0.75, 1]))
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.25, 0.5, 0.75, 1])))
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.35, 0.7, 1])))
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1])))
        bounds = [0, 0.1, 1, 5, 10]



    norm = BoundaryNorm(bounds, cmap.N)

    # Adjust alpha for the colormap
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[0, -1] = 0  # Set alpha of the first color (white) to 0
    cmap_colors[1:, -1] = 0.8  # Set alpha of the remaining colors to 0.8
    cmap = ListedColormap(cmap_colors)

    # Start plotting
    img = ax.imshow(raster_data, origin="upper", extent=extent, cmap=cmap, norm=norm)
    # img = ax.imshow(dst_data, origin="upper", extent=extent, cmap=cmap, norm=norm)

    # Add a colorbar using the custom colormap and normalization.
    cbar = plt.colorbar(img, cax=cax, orientation="vertical", ticks=bounds, extend="max")

    cbar.ax.tick_params(labelsize=fontsize-1, pad=1)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([str(x) for x in bounds])

    # Draw the intersecting ocean area in blue.
    if not intersection.empty:
        intersection.plot(ax=ax, color='#ADD8E6', edgecolor='none', label='Ocean', alpha=1, zorder=1)

    if not is_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.tick_params(axis='x', labelsize=fontsize-3, direction='in', pad=1, top=False, bottom=True)
        ax.tick_params(axis='y', labelsize=fontsize-3, direction='in', pad=1, labelrotation=90, right=False, left=True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        # ax.tick_params(axis='y', which='both', direction='in', labelsize=fontsize-3, pad=-15)
        # ax.xaxis.set_tick_params(which='both', direction='in', labelsize=fontsize-3)
        # ax.yaxis.set_tick_params(which='both', direction='in', labelsize=fontsize-3, labelrotation=90)

        # Format longitude and latitude tick labels.
        def format_lon(x, pos):
            return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

        def format_lat(y, pos):
            return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))


def smooth_contour_line(line, iterations=2):
    """Round a contour polyline for display while keeping it close to the source."""
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


def _axis_data_ranges(ax, xlim=None, ylim=None):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    return (
        xlim,
        ylim,
        max(abs(xlim[1] - xlim[0]), 1e-12),
        max(abs(ylim[1] - ylim[0]), 1e-12),
    )


def _text_box_size(ax, label, fontsize=7, xlim=None, ylim=None):
    """Estimate text size in data units for overlap checks."""
    _, _, x_range, y_range = _axis_data_ranges(ax, xlim, ylim)
    bbox = ax.get_window_extent()
    ax_width_in = max(bbox.width / ax.figure.dpi, 1e-6)
    ax_height_in = max(bbox.height / ax.figure.dpi, 1e-6)
    width_in = max(len(str(label)), 1) * fontsize * 0.52 / 72.0
    height_in = fontsize * 1.25 / 72.0
    return width_in / ax_width_in * x_range, height_in / ax_height_in * y_range


def _label_box(ax, x, y, label, fontsize=7, ha="center", va="center", xlim=None, ylim=None):
    width, height = _text_box_size(ax, label, fontsize=fontsize, xlim=xlim, ylim=ylim)

    if ha == "left":
        xmin, xmax = x, x + width
    elif ha == "right":
        xmin, xmax = x - width, x
    else:
        xmin, xmax = x - width / 2.0, x + width / 2.0

    if va == "bottom":
        ymin, ymax = y, y + height
    elif va == "top":
        ymin, ymax = y - height, y
    else:
        ymin, ymax = y - height / 2.0, y + height / 2.0

    return (xmin, ymin, xmax, ymax)


def _pad_box(box, x_pad, y_pad):
    xmin, ymin, xmax, ymax = box
    return (xmin - x_pad, ymin - y_pad, xmax + x_pad, ymax + y_pad)


def _box_inside_limits(box, xlim, ylim, x_pad=0.0, y_pad=0.0):
    xmin, ymin, xmax, ymax = _pad_box(box, x_pad, y_pad)
    return xmin >= xlim[0] and xmax <= xlim[1] and ymin >= ylim[0] and ymax <= ylim[1]


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _collect_text_boxes(ax, xlim, ylim):
    boxes = []
    for text in ax.texts:
        try:
            tx, ty = text.get_position()
            if not (xlim[0] <= tx <= xlim[1] and ylim[0] <= ty <= ylim[1]):
                continue
            boxes.append(
                _label_box(
                    ax, tx, ty, text.get_text(),
                    fontsize=text.get_fontsize(),
                    ha=text.get_ha(),
                    va=text.get_va(),
                    xlim=xlim,
                    ylim=ylim,
                )
            )
        except Exception:
            continue
    return boxes


def _collect_marker_points(ax, xlim, ylim):
    """Collect scatter marker centers already drawn on the map."""
    points = []
    for collection in ax.collections:
        try:
            offsets = collection.get_offsets()
        except Exception:
            continue
        if offsets is None or len(offsets) == 0:
            continue
        try:
            offsets = np.asarray(offsets, dtype=float)
        except Exception:
            continue
        if offsets.ndim != 2 or offsets.shape[1] < 2:
            continue
        finite = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
        visible = (
            finite
            & (offsets[:, 0] >= xlim[0]) & (offsets[:, 0] <= xlim[1])
            & (offsets[:, 1] >= ylim[0]) & (offsets[:, 1] <= ylim[1])
        )
        points.extend(map(tuple, offsets[visible, :2]))
    return points


def _point_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _label_candidate_score(ax, box, text_boxes, marker_points, xlim, ylim):
    _, _, x_range, y_range = _axis_data_ranges(ax, xlim, ylim)
    x_pad = x_range * 0.006
    y_pad = y_range * 0.006
    padded_box = _pad_box(box, x_pad, y_pad)

    if not _box_inside_limits(box, xlim, ylim, x_pad=x_pad, y_pad=y_pad):
        return -np.inf
    if any(_boxes_overlap(padded_box, _pad_box(other, x_pad, y_pad)) for other in text_boxes):
        return -np.inf
    if any(_point_in_box(pt, padded_box) for pt in marker_points):
        return -np.inf

    cx, cy = _box_center(box)
    edge_distance = min(
        (box[0] - xlim[0]) / x_range,
        (xlim[1] - box[2]) / x_range,
        (box[1] - ylim[0]) / y_range,
        (ylim[1] - box[3]) / y_range,
    )
    marker_distance = 1.0
    if marker_points:
        marker_distance = min(
            np.hypot((cx - mx) / x_range, (cy - my) / y_range)
            for mx, my in marker_points
        )
    text_distance = 1.0
    if text_boxes:
        text_distance = min(
            np.hypot((cx - ox) / x_range, (cy - oy) / y_range)
            for ox, oy in map(_box_center, text_boxes)
        )
    return marker_distance + text_distance + 0.5 * edge_distance


def plot_contours(shapefile_path, ax, target_crs="EPSG:4326", pga_threshold=0.1, smooth_iterations=2):
    """
    Plot contour lines from shapefile with color-coded PGA values.

    Labels are placed once per PGA level and are positioned to avoid existing
    text labels, such as city names, as much as possible.
    """
    contours_gdf = gpd.read_file(shapefile_path)

    if 'value' not in contours_gdf.columns:
        raise ValueError("Shapefile missing required 'value' column")

    if contours_gdf.crs != target_crs:
        contours_gdf = contours_gdf.to_crs(target_crs)

    contours_gdf = contours_gdf[contours_gdf['value'] / 100 >= pga_threshold]
    if contours_gdf.empty:
        return

    norm = Normalize(vmin=contours_gdf['value'].min(), vmax=contours_gdf['value'].max())
    cmap = cm.autumn

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lines_by_value = {}

    for _, row in contours_gdf.iterrows():
        geom = row.geometry
        value = row['value']
        color = cmap(norm(value))

        if isinstance(geom, MultiLineString):
            line_geoms = geom.geoms
        elif isinstance(geom, LineString):
            line_geoms = [geom]
        else:
            continue

        for line in line_geoms:
            if not isinstance(line, LineString) or line.is_empty or line.length == 0:
                continue
            line = smooth_contour_line(line, iterations=smooth_iterations)
            ax.plot(*line.xy, color=color, linewidth=1.5, zorder=10)
            lines_by_value.setdefault(value, []).append((line, color))

    for value, line_items in lines_by_value.items():
        visible_lines = [
            (line, color) for line, color in line_items
            if line.bounds[2] >= xlim[0] and line.bounds[0] <= xlim[1]
            and line.bounds[3] >= ylim[0] and line.bounds[1] <= ylim[1]
        ]
        if not visible_lines:
            continue
        line, color = max(visible_lines, key=lambda item: item[0].length)
        plot_line_label(ax, line, value, color, xlim, ylim)


def plot_line_label(ax, line, value, color, xlim, ylim):
    """Place one label on a contour line, avoiding existing map text."""
    label = f"{value / 100:g}g"
    candidate = choose_contour_label_position(ax, line, xlim, ylim, label)
    if candidate is None:
        return None
    label_x, label_y = candidate

    text = ax.text(
        label_x, label_y, label, fontsize=7, color="k",
        ha="center", va="center", zorder=11, bbox=None
    )
    text.set_clip_on(True)
    text.set_clip_path(ax.patch)
    return text


def choose_contour_label_position(ax, line, xlim, ylim, label):
    """Choose a label point on a contour line, avoiding existing text labels."""
    _, _, x_range, y_range = _axis_data_ranges(ax, xlim, ylim)
    text_boxes = _collect_text_boxes(ax, xlim, ylim)
    marker_points = _collect_marker_points(ax, xlim, ylim)

    best_point = None
    best_score = -np.inf
    for frac in np.linspace(0.12, 0.88, 25):
        pt = line.interpolate(frac, normalized=True)
        px, py = pt.x, pt.y

        box = _label_box(ax, px, py, label, fontsize=7, ha="center", va="center", xlim=xlim, ylim=ylim)
        score = _label_candidate_score(ax, box, text_boxes, marker_points, xlim, ylim)
        if score > best_score:
            best_score = score
            best_point = (px, py)

    if not np.isfinite(best_score):
        return None
    return best_point

def plot_cities(cities_gdf, ax, is_label=True):
    """
    Plot cities on map with markers and optional labels.
    
    Visualizes cities with different markers for capitals vs. regular cities,
    and optionally displays city names as labels.
    
    Args:
        cities_gdf (GeoDataFrame): Cities geodataframe with geometry and attributes
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        is_label (bool): Whether to display city name labels
        
    Returns:
        None: Plots directly to provided axes
    """
    """
    Plot cities on map with markers and optional labels.
    
    Visualizes cities with different markers for capitals vs. regular cities,
    and optionally displays city names as labels.
    
    Args:
        cities_gdf (GeoDataFrame): Cities geodataframe with geometry and attributes
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        is_label (bool): Whether to display city name labels
        
    Returns:
        None: Plots directly to provided axes
    """
    # Handle None input
    if cities_gdf is None:
        return

    # Validate required columns
    if 'name' not in cities_gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'name' column for labeling points.")
    
    # Ensure only Point geometries
    if not all(cities_gdf.geometry.geom_type == 'Point'):
        print(f"GeoDataFrame contains non-Point geometries: {cities_gdf.geometry.geom_type.unique()}")
        print("Filtering out non-Point geometries")
        cities_gdf = cities_gdf[cities_gdf.geometry.geom_type == 'Point']

    # Validate and convert CRS if needed
    if cities_gdf.crs is None:
        raise ValueError("GeoDataFrame must have a valid CRS.")
    if not cities_gdf.crs.is_geographic:
        print("Converting data to WGS 1984 (EPSG:4326).")
        cities_gdf = cities_gdf.to_crs("EPSG:4326")

    # Plot city points
    x_coords = cities_gdf.geometry.x
    y_coords = cities_gdf.geometry.y

    # Use scatter to plot points
    ax.scatter(x_coords, y_coords, color='k', s=15, zorder=11, label='Locations')

    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if is_label:
        # Place labels from several candidate offsets to avoid points, text, and frame edges.
        _, _, lon_range, lat_range = _axis_data_ranges(ax, xlim, ylim)
        offset_y = lat_range * 0.024
        offset_x = lon_range * 0.018
        text_boxes = _collect_text_boxes(ax, xlim, ylim)
        marker_points = _collect_marker_points(ax, xlim, ylim)
        candidates = [
            (0.0, offset_y, "center", "bottom"),
            (offset_x, 0.0, "left", "center"),
            (-offset_x, 0.0, "right", "center"),
            (0.0, -offset_y, "center", "top"),
            (offset_x, offset_y, "left", "bottom"),
            (-offset_x, offset_y, "right", "bottom"),
            (offset_x, -offset_y, "left", "top"),
            (-offset_x, -offset_y, "right", "top"),
        ]

        for x, y, name in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf['name']):
            best = None
            best_score = -np.inf
            for dx, dy, ha, va in candidates:
                label_x = x + dx
                label_y = y + dy
                box = _label_box(ax, label_x, label_y, name, fontsize=7, ha=ha, va=va, xlim=xlim, ylim=ylim)
                score = _label_candidate_score(ax, box, text_boxes, marker_points, xlim, ylim)
                if score > best_score:
                    best_score = score
                    best = (label_x, label_y, ha, va, box)

            if best is None or not np.isfinite(best_score):
                continue

            label_x, label_y, ha, va, box = best
            text = ax.text(
                label_x, label_y,
                name,
                fontsize=7,
                color='k',
                ha=ha,
                va=va,
                zorder=12
            )
            text.set_clip_on(True)
            text.set_clip_path(ax.patch)
            text_boxes.append(box)

    # ax.set_clip_on(True)


def plot_interruption_map(shapefile_filepath, ax, hillshade_file, intersection, plot_column, plot="road", fontsize=8, is_cbar=False, is_ticks=False):
    
    # plot hillshade as background
    norm = plt.Normalize(-100, 255)
    with rasterio.open(hillshade_file) as src:
        show(src, cmap='Greys_r', ax=ax, norm=norm, alpha=0.2)


    edges_polyline = gpd.read_file(shapefile_filepath)
    # edges_polyline = edges_polyline.to_crs(epsg=4326)
    if edges_polyline.crs != "EPSG:4326":
        edges_polyline = edges_polyline.to_crs(epsg=4326)

    if plot == "road":
        # Scale line widths continuously from the lanes attribute.
        edges_polyline['linewidth'] = edges_polyline['lanes'].apply(lambda v: scale_width(v, edges_polyline['lanes']))
    elif plot == "river":
        # Scale line widths continuously from the river width attribute.
        edges_polyline['linewidth'] = edges_polyline['width'].apply(lambda v: scale_width(v, edges_polyline['width']))


    # make a color map of fixed colors
    if plot == "road":
        cmap = ListedColormap(plt.get_cmap('Reds')([0.15, 0.3, 0.65, 1]))  # Blues;Greys
        # bounds = [0, 0.001, 0.005, 0.03]
        bounds = [0, 0.01, 0.1, 0.5, 5]
    elif plot == "river":
        cmap = ListedColormap(plt.get_cmap('Blues')([0.15, 0.3, 0.65, 1]))
        bounds = [0, 0.01, 0.1, 0.5, 5] #[0, 0.1, 1, 5, 10]

    norm = BoundaryNorm(bounds, cmap.N)

    # Replace missing values so every polyline remains visible.
    edges_polyline[[plot_column]] = edges_polyline[[plot_column]].fillna(0)

    # Color infrastructure polylines by the requested attribute.
    img = edges_polyline.plot(column=plot_column, cmap=cmap, norm=norm, legend=False,
                              linewidth=edges_polyline['linewidth'], ax=ax, edgecolor=None, zorder=3)


    # Reserve a separate axis for the colorbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    if is_cbar:
        cbar = plt.colorbar(img.collections[0], cax=cax, orientation="vertical", ticks=bounds, extend="max")
        cbar.ax.tick_params(labelsize=fontsize-1, pad=1)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels([str(x) for x in bounds])
    else:
        cax.set_visible(False)

    if not is_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        # ax.tick_params(axis='x', labelsize=fontsize - 2, direction='in', pad=2)
        # ax.tick_params(axis='y', labelsize=fontsize - 2, direction='in', pad=2)
        ax.tick_params(axis='x', labelsize=fontsize-3, direction='in', pad=1, top=False, bottom=True)
        ax.tick_params(axis='y', labelsize=fontsize-3, direction='in', pad=1, labelrotation=90, right=False, left=True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        # Format longitude and latitude tick labels.
        def format_lon(x, pos):
            return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

        def format_lat(y, pos):
            return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))


    # Draw the intersecting ocean area in blue.
    if not intersection.empty:
        intersection.plot(ax=ax, color='#ADD8E6', edgecolor='none', label='Ocean', alpha=1, zorder=1)

    # Preserve equal axis scaling so vector and raster extents align visually.
    ax.set_aspect('equal') 


def scale_width(value, series, vmin=0.8, vmax=1.2):
    ser = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    smin = ser.min()
    smax = ser.max()
    val = pd.to_numeric(value, errors='coerce')

    if pd.isna(val) or pd.isna(smin) or pd.isna(smax) or smax == smin:
        return (vmin + vmax) / 2.0

    scaled = vmin + (val - smin) * (vmax - vmin) / (smax - smin)
    return max(vmin, min(vmax, float(scaled)))

