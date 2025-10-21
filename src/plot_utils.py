"""
Plot Utilities Module for QuakeSlide System.

This module provides comprehensive plotting and visualization functions.

Author: Shihao Xiao
Date: Oct 2025
"""

# Standard library imports
import os

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



    # 创建分隔器以添加色标
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    
    # make a color map of fixed colors
    if plot == "counts density colorbar":
        # cmap = ListedColormap(plt.get_cmap('Greys')([0, 0.25, 0.5, 0.75, 1]))  #Blues
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(plt.get_cmap('Blues')([0, 0.35, 0.7, 1]))  #Blues;Greys
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1]))) # 颜色用none比较好
        bounds = [0, 1, 5, 10, 20]
    elif plot == "areal percentage colorbar":
        # cmap = ListedColormap(plt.get_cmap('Reds')([0, 0.25, 0.5, 0.75, 1]))
        # # 将第一个颜色设置为纯白
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.25, 0.5, 0.75, 1])))
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.35, 0.7, 1])))
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1])))
        bounds = [0, 0.01, 0.05, 0.1, 0.2]
    elif plot == "exposure density colorbar":
        # cmap = ListedColormap(plt.get_cmap('Reds')([0, 0.25, 0.5, 0.75, 1]))
        # # 将第一个颜色设置为纯白
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.25, 0.5, 0.75, 1])))
        # bounds = [0, 0.2, 1, 5, 10, 20]
        # cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.35, 0.7, 1])))
        cmap = ListedColormap(['white'] + list(plt.get_cmap('Reds')([0.3, 0.65, 1])))
        bounds = [0, 0.1, 1, 5, 20]



    ticks = [str(x) for x in bounds]
    norm = BoundaryNorm(bounds, cmap.N)

    # Adjust alpha for the colormap
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[0, -1] = 0  # Set alpha of the first color (white) to 0
    cmap_colors[1:, -1] = 0.8  # Set alpha of the remaining colors to 0.8
    cmap = ListedColormap(cmap_colors)

    # Start plotting
    img = ax.imshow(raster_data, origin="upper", extent=extent, cmap=cmap, norm=norm)
    # img = ax.imshow(dst_data, origin="upper", extent=extent, cmap=cmap, norm=norm)

    # 添加色标，使用自定义的 cmap 和 norm
    cbar = plt.colorbar(img, cax=cax, orientation="vertical", ticks=bounds, extend="max")

    cbar.ax.tick_params(labelsize=fontsize-1, pad=1)  # 设置色标刻度字体大小和刻度间距
    cbar.set_ticks(bounds)  # 设置色标刻度
    cbar.set_ticklabels([str(x) for x in bounds])  # 自定义色标刻度标签

    ####################### 添加ocean的蓝色
    # 如果有相交部分，绘制相交的部分
    if not intersection.empty:
        intersection.plot(ax=ax, color='#ADD8E6', edgecolor='none', label='Ocean', alpha=1)

    if not is_ticks:
        # 隐藏经纬度刻度和标签
        ax.set_xticks([])  # 隐藏 x 轴刻度
        ax.set_yticks([])  # 隐藏 y 轴刻度
    else:
        # 设置 x 和 y 轴最大显示 3 个刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))  # x 轴最多 3 个刻度
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))  # y 轴最多 3 个刻度

        # 设置刻度标签的字体大小
        ax.tick_params(axis='x', labelsize=fontsize-3, direction='in', pad=1, top=False, bottom=True)
        # 让 y 轴刻度标签旋转 90 度并居中对齐
        ax.tick_params(axis='y', labelsize=fontsize-3, direction='in', pad=1, labelrotation=90, right=False, left=True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        # ax.tick_params(axis='y', which='both', direction='in', labelsize=fontsize-3, pad=-15)
        # 把左边和下边的label置于框内
        # ax.xaxis.set_tick_params(which='both', direction='in', labelsize=fontsize-3)
        # ax.yaxis.set_tick_params(which='both', direction='in', labelsize=fontsize-3, labelrotation=90)

        # 修改经纬度刻度标签格式
        def format_lon(x, pos):
            return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

        def format_lat(y, pos):
            return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))


def plot_contours(shapefile_path, ax, target_crs="EPSG:4326", pga_threshold=0.1):
    """
    Plot contour lines from shapefile with color-coded PGA values.
    
    Overlays contour lines on existing raster plots, with colors and annotations
    based on Peak Ground Acceleration (PGA) values. Projects data to target CRS.
    
    Args:
        shapefile_path (str): Path to contour shapefile
        ax (matplotlib.axes.Axes): Matplotlib axes for plotting
        target_crs (str): Target coordinate reference system
        pga_threshold (float): Minimum PGA threshold for display
        
    Returns:
        None: Plots directly to provided axes
    """
    # Read shapefile data
    contours_gdf = gpd.read_file(shapefile_path)

    # Check for required 'value' column
    if 'value' not in contours_gdf.columns:
        raise ValueError("Shapefile missing required 'value' column")

    # Project to target coordinate system
    if contours_gdf.crs != target_crs:
        contours_gdf = contours_gdf.to_crs(target_crs)

    # Filter contours by PGA threshold
    contours_gdf = contours_gdf[contours_gdf['value']/100 >= pga_threshold]

    # Set up color mapping and normalization
    norm = Normalize(vmin=contours_gdf['value'].min(), vmax=contours_gdf['value'].max())
    cmap = cm.autumn  # Use autumn colormap

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot each contour line
    for _, row in contours_gdf.iterrows():
        geom = row.geometry
        value = row['value']
        color = cmap(norm(value))  # 根据 value 的值获取颜色

        # 如果几何是 MultiLineString，需要分别绘制每条线
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                if not isinstance(line, LineString):
                    continue
                ax.plot(*line.xy, color=color, linewidth=1.5, zorder=10)
                plot_line_label(ax, line, value, color, xlim, ylim)
        elif isinstance(geom, LineString):
            ax.plot(*geom.xy, color=color, linewidth=1.5, zorder=10)
            plot_line_label(ax, geom, value, color, xlim, ylim)
        else:
            continue


def plot_line_label(ax, line, value, color, xlim, ylim):
    """
    在指定的线条上绘制标签，并确保标签位于 ax 的范围内。

    参数:
    - ax: matplotlib 的轴对象
    - line: Shapely 的 LineString 几何对象
    - value: 等值线的值
    - color: 标签颜色
    - xlim: x 轴显示范围
    - ylim: y 轴显示范围
    """
    # 计算线条的插值点（0.5 表示中点）
    label_point = line.interpolate(0.5, normalized=True)
    label_x, label_y = label_point.x, label_point.y

    # 如果中点不在范围内，尝试从0.1到0.9插值，找到第一个在范围内的点
    if not (xlim[0] <= label_x <= xlim[1] and ylim[0] <= label_y <= ylim[1]):
        found = False
        for frac in np.linspace(0.1, 0.9, 9):
            pt = line.interpolate(frac, normalized=True)
            px, py = pt.x, pt.y
            if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                label_x, label_y = px, py
                found = True
                break
        # 如果还是没找到，取线的第一个点
        if not found:
            coords = list(line.coords)
            for px, py in coords:
                if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                    label_x, label_y = px, py
                    break

    # 检查标签是否在 ax 范围内
    if xlim[0] <= label_x <= xlim[1] and ylim[0] <= label_y <= ylim[1]:
        # label = f"{value / 100:.1f}g"  # 格式化注释 #这里别把0.05g显示为0.1g了！
        label = f"{value / 100:g}g"  # 使用:g自动去除不必要的0
        text = ax.text(
            label_x, label_y, label, fontsize=7, color="k",
            ha="center", va="center", zorder=11, bbox=None
        )
        # 启用裁剪（确保文字不会溢出到框外）
        text.set_clip_on(True)



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
        # Dynamically adjust offset based on latitude range
        lat_range = abs(ylim[1] - ylim[0])
        offset = lat_range * 0.02  # Offset is 2% of latitude range

        # Add labels above points, ensuring labels are within axis bounds
        for x, y, name in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf['name']):
            # 根据 CRS 偏移 y 坐标
            label_x, label_y = x, y + offset

            # 检查标签是否在 ax 范围内
            if xlim[0] <= label_x <= xlim[1] and ylim[0] <= label_y <= ylim[1]:
                text = ax.text(
                    label_x, label_y,
                    name,
                    fontsize=7,
                    color='k',  # 设置文字颜色为黑色
                    ha='center',  # 水平居中对齐
                    va='bottom',  # 垂直对齐方式为底部对齐
                    zorder=12      # 确保标注在点上方
                )
                # 启用裁剪（确保文字不会溢出到框外）
                text.set_clip_on(True)

    # # 确保点也不溢出
    # ax.set_clip_on(True)


def plot_blockage_map(shapefile_filepath, ax, hillshade_file, intersection, plot_column, plot="road", fontsize=8, is_cbar=False, is_ticks=False):
    
    # plot hillshade as background
    norm = plt.Normalize(-100, 255)
    with rasterio.open(hillshade_file) as src:
        show(src, cmap='Greys_r', ax=ax, norm=norm, alpha=0.2)


    edges_polyline = gpd.read_file(shapefile_filepath)
    # edges_polyline = edges_polyline.to_crs(epsg=4326)
    if edges_polyline.crs != "EPSG:4326":
        edges_polyline = edges_polyline.to_crs(epsg=4326)

    if plot == "road":
        # 连续缩放：按 lanes 列线性缩放到 0.8–1.2
        edges_polyline['linewidth'] = edges_polyline['lanes'].apply(lambda v: scale_width(v, edges_polyline['lanes']))
    elif plot == "river":
        # 连续缩放：按 width 列线性缩放到 0.8–1.2
        edges_polyline['linewidth'] = edges_polyline['width'].apply(lambda v: scale_width(v, edges_polyline['width']))


    # make a color map of fixed colors
    if plot == "road":
        cmap = ListedColormap(plt.get_cmap('Reds')([0.15, 0.3, 0.65, 1]))  # Blues;Greys
        # bounds = [0, 0.001, 0.005, 0.03]
        bounds = [0, 0.01, 0.1, 1, 10]
        legend_label = 'Road interruption (/km)'
    elif plot == "river":
        cmap = ListedColormap(plt.get_cmap('Blues')([0.15, 0.3, 0.65, 1]))
        bounds = [0, 0.01, 0.1, 1, 10] #[0, 0.1, 1, 5, 10]
        legend_label = 'River interruption (/km)'

    ticks = [str(x) for x in bounds]
    norm = BoundaryNorm(bounds, cmap.N)

    # 这是为了确保画图的这一列数据不为nan，用0替代，如果值是nan，则对应的polyline画不出来
    edges_polyline[[plot_column]] = edges_polyline[[plot_column]].fillna(0)

    # 绘制 edges_polyline，使用 plot_column 列作为颜色
    img = edges_polyline.plot(column=plot_column, cmap=cmap, norm=norm, legend=False,
                              linewidth=edges_polyline['linewidth'], ax=ax, edgecolor=None, zorder=1)


    # 创建分隔器以添加色标
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    if is_cbar:
        cbar = plt.colorbar(img.collections[0], cax=cax, orientation="vertical", ticks=bounds, extend="max")
        cbar.ax.tick_params(labelsize=fontsize-1, pad=1)  # 设置色标刻度字体大小和刻度间距
        cbar.set_ticks(bounds)
        cbar.set_ticklabels([str(x) for x in bounds])
        # cbar.set_label(legend_label, fontsize=fontsize + 2)  # 设置连续色条的标题
    else:
        cax.set_visible(False)  # 隐藏分隔轴的可见性

    if not is_ticks:
        # 隐藏经纬度刻度和标签
        ax.set_xticks([])  # 隐藏 x 轴刻度
        ax.set_yticks([])  # 隐藏 y 轴刻度
    else:
        # 设置 x 和 y 轴最大显示 3 个刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))  # x 轴最多 3 个刻度
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))  # y 轴最多 3 个刻度

        # 设置刻度标签的字体大小
        # ax.tick_params(axis='x', labelsize=fontsize - 2, direction='in', pad=2)
        # ax.tick_params(axis='y', labelsize=fontsize - 2, direction='in', pad=2)
        ax.tick_params(axis='x', labelsize=fontsize-3, direction='in', pad=1, top=False, bottom=True)
        # 让 y 轴刻度标签旋转 90 度并居中对齐
        ax.tick_params(axis='y', labelsize=fontsize-3, direction='in', pad=1, labelrotation=90, right=False, left=True)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        # 修改经纬度刻度标签格式
        def format_lon(x, pos):
            return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

        def format_lat(y, pos):
            return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))


    ####################### 添加ocean的蓝色
    # 如果有相交部分，绘制相交的部分
    if not intersection.empty:
        intersection.plot(ax=ax, color='#ADD8E6', edgecolor='none', label='Ocean', alpha=1)

    # 这个很重要！！！！！！不要画出来的shapefile图比raster图要窄一点，虽然坐标系和范围都一致
    ax.set_aspect('equal') 


# 定义线性缩放函数（传入一个值和对应整列 Series；线宽范围缩放到 [0.8, 1.2]）
def scale_width(value, series, vmin=0.8, vmax=1.2):
    ser = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    smin = ser.min()
    smax = ser.max()
    val = pd.to_numeric(value, errors='coerce')

    if pd.isna(val) or pd.isna(smin) or pd.isna(smax) or smax == smin:
        return (vmin + vmax) / 2.0

    scaled = vmin + (val - smin) * (vmax - vmin) / (smax - smin)
    return max(vmin, min(vmax, float(scaled)))


def plot_footnote(ax, fontsize=8):
    """
    Add standardized footnote text to plot axes.
    
    Displays disclaimer and data source information for QuakeSlide reports.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for footnote text
        fontsize (int): Font size for footnote text
        
    Returns:
        None: Adds text directly to provided axes
    """
    footnote_text = (
        "This report provides first-order quantitative estimates of potential earthquake-triggered landslide impacts.\n"
        "This report is automatically generated and will be updated as new data becomes available.\n"
        "Earthquake information and ground motion data are sourced from the USGS ShakeMap."
    )
    ax.text(0, 0.3, footnote_text, ha='left', va='center', style='italic', fontsize=fontsize)
    ax.axis('off')