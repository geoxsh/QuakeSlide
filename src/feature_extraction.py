"""
Feature extraction module for QuakeSlide HBM system.

Extracts current HBM predictors per grid-cell mapping unit:
  PGA, PGV, Slope, LC100, Lithology

Date: August 2026
"""

import csv
import os
import warnings

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.crs import CRS as RasterioCRS
from rasterio.warp import reproject
from shapely.geometry import Point

from src.utils import clip_features_raster_files

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
LAND_COVER_MAPPING_FILE = os.path.join(MODEL_DIR, 'land_cover_mapping.csv')
LITHOLOGY_MAPPING_FILE = os.path.join(MODEL_DIR, 'lithology_mapping.csv')

LC100_LEVELS = ['forest', 'shrubs', 'grassland', 'cropland', 'urban',
                'bare_land', 'wetland', 'other']
LITHOLOGY_LEVELS = ['su', 'ss', 'sm', 'sc', 'Volcanic', 'Plutonic', 'Other']


# =============================================================================
# Public entry point
# =============================================================================

def _load_category_mapping(csv_path):
    """Load a raw_value -> mapped_value category mapping from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Category mapping file not found: {csv_path}")

    mapping = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'raw_value', 'mapped_value'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Mapping file must contain columns {required}: {csv_path}")
        for row in reader:
            raw = str(row['raw_value']).strip()
            mapped = str(row['mapped_value']).strip()
            if raw and mapped:
                mapping[raw] = mapped
    return mapping


def _reclassify_categories(mapping_unit):
    """Reclassify LC100 and GLiM lithology using the training-time mappings."""
    lc100_map = _load_category_mapping(LAND_COVER_MAPPING_FILE)
    lithology_map = _load_category_mapping(LITHOLOGY_MAPPING_FILE)

    mapping_unit['LC100_raw'] = mapping_unit['LC100']
    mapping_unit['Lithology_raw'] = mapping_unit['Lithology']

    lc_raw = mapping_unit['LC100_raw'].astype('Int64').astype(str)
    lith_raw = mapping_unit['Lithology_raw'].astype(str)

    mapping_unit['LC100'] = lc_raw.map(lc100_map)
    mapping_unit['Lithology'] = lith_raw.map(lithology_map)

    bad_lc = sorted(set(lc_raw[mapping_unit['LC100'].isna()].dropna()))
    bad_lith = sorted(set(lith_raw[mapping_unit['Lithology'].isna()].dropna()))

    allowed_unmapped_lc = {'80', '200', '<NA>'}
    allowed_unmapped_lith = {'wb', 'nan', 'None'}
    unexpected_lc = [v for v in bad_lc if v not in allowed_unmapped_lc]
    unexpected_lith = [v for v in bad_lith if v not in allowed_unmapped_lith]

    if unexpected_lc:
        raise ValueError(
            "Unknown LC100 raw classes not present in land_cover_mapping.csv: "
            + ", ".join(unexpected_lc)
        )
    if unexpected_lith:
        raise ValueError(
            "Unknown Lithology raw classes not present in lithology_mapping.csv: "
            + ", ".join(unexpected_lith)
        )

    return mapping_unit


def _write_category_raster(mapping_unit, ref_raster_path, values, out_path):
    """Write a per-cell categorical diagnostic raster as float32 codes."""
    with rasterio.open(ref_raster_path) as src:
        H, W = src.height, src.width
        meta = src.meta.copy()
    meta.update(dtype='float32', count=1, nodata=-9999.0)

    arr = np.full((H, W), -9999.0, dtype=np.float32)
    valid = np.isfinite(values)
    pix_row = mapping_unit.loc[valid, 'pix_row'].values
    pix_col = mapping_unit.loc[valid, 'pix_col'].values
    arr[pix_row, pix_col] = values[valid].astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr, 1)


def _save_legend(rows, out_path):
    """Save code-to-category legend for diagnostic rasters."""
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['code', 'value', 'label'])
        writer.writeheader()
        writer.writerows(rows)


def _save_category_diagnostics(mapping_unit, ref_raster_path, output_dir):
    """Save raw and mapped categorical predictors before cleaning/filtering."""
    diag_cols = [
        'pix_row', 'pix_col', 'PGA', 'PGV', 'Slope',
        'LC100_raw', 'LC100', 'Lithology_raw', 'Lithology'
    ]
    diag = mapping_unit[[c for c in diag_cols if c in mapping_unit.columns]].copy()
    diag.rename(columns={
        'LC100': 'LC100_mapped',
        'Lithology': 'Lithology_mapped'
    }, inplace=True)
    diag['x'] = mapping_unit.geometry.x.values
    diag['y'] = mapping_unit.geometry.y.values
    diag.to_csv(os.path.join(output_dir, 'category_diagnostics.csv'),
                index=False, encoding='utf-8-sig')

    # LC100 raw is numeric in the source product, so save it directly.
    lc_raw_num = mapping_unit['LC100_raw'].astype('float64').to_numpy()
    _write_category_raster(
        mapping_unit, ref_raster_path, lc_raw_num,
        os.path.join(output_dir, 'LC100_raw.tif')
    )

    lc_legend = [
        {'code': i + 1, 'value': v, 'label': v}
        for i, v in enumerate(LC100_LEVELS)
    ]
    lc_code = {row['value']: row['code'] for row in lc_legend}
    lc_mapped_num = mapping_unit['LC100'].map(lc_code).astype('float64').to_numpy()
    _write_category_raster(
        mapping_unit, ref_raster_path, lc_mapped_num,
        os.path.join(output_dir, 'LC100_mapped.tif')
    )
    _save_legend(lc_legend, os.path.join(output_dir, 'LC100_mapped_legend.csv'))

    # GLiM lithology raw classes are strings. Encode them for raster diagnostics.
    lith_raw_values = sorted(
        v for v in mapping_unit['Lithology_raw'].astype(str).unique()
        if v not in {'nan', 'None', '<NA>'}
    )
    lith_raw_legend = [
        {'code': i + 1, 'value': v, 'label': v}
        for i, v in enumerate(lith_raw_values)
    ]
    lith_raw_code = {row['value']: row['code'] for row in lith_raw_legend}
    lith_raw_num = (
        mapping_unit['Lithology_raw'].astype(str).map(lith_raw_code)
        .astype('float64').to_numpy()
    )
    _write_category_raster(
        mapping_unit, ref_raster_path, lith_raw_num,
        os.path.join(output_dir, 'Lithology_raw.tif')
    )
    _save_legend(lith_raw_legend, os.path.join(output_dir, 'Lithology_raw_legend.csv'))

    lith_mapped_legend = [
        {'code': i + 1, 'value': v, 'label': v}
        for i, v in enumerate(LITHOLOGY_LEVELS)
    ]
    lith_mapped_code = {row['value']: row['code'] for row in lith_mapped_legend}
    lith_mapped_num = (
        mapping_unit['Lithology'].map(lith_mapped_code)
        .astype('float64').to_numpy()
    )
    _write_category_raster(
        mapping_unit, ref_raster_path, lith_mapped_num,
        os.path.join(output_dir, 'Lithology_mapped.tif')
    )
    _save_legend(lith_mapped_legend, os.path.join(output_dir, 'Lithology_mapped_legend.csv'))


def feature_extraction(lithology_data, input_path, output_path):
    """
    Extract all predictive features and return a cleaned mapping-unit GeoDataFrame.

    Args:
        lithology_data (GeoDataFrame): Global lithology polygons (GLiM)
        input_path (dict): Parsed Input.txt parameters
        output_path (dict): Parsed output paths

    Returns:
        tuple: (mapping_unit GeoDataFrame, grid_area_m2 float)
    """
    # ------------------------------------------------------------------
    # Step 1: Clip Slope raster to affected area → generate mapping units
    # ------------------------------------------------------------------
    polygon = gpd.read_file(output_path['LandslideAffectedArea']).to_crs(epsg=4326)

    clip_features_raster_files(
        ["Slope"],
        [input_path['GlobalSlopeFile']],
        polygon,
        output_path['feature_maps']
    )

    mapping_unit = _generate_point_mapping_unit(output_path['ClippedSlopeFile'])

    # ------------------------------------------------------------------
    # Step 2: Compute grid area (m²) from the Slope raster in UTM
    # ------------------------------------------------------------------
    grid_area_m2 = _compute_grid_area_m2(mapping_unit, output_path['ClippedSlopeFile'])

    # ------------------------------------------------------------------
    # Step 3: Align ShakeMap PGA/PGV to the Slope reference grid
    # ------------------------------------------------------------------
    print("  Aligning PGA and PGV to the Slope reference grid...", flush=True)
    resampled_pga = _resample_raster_to_reference(
        output_path['PGAFile'],
        output_path['ClippedSlopeFile'],
        resampling=Resampling.bilinear
    )
    resampled_pgv = _resample_raster_to_reference(
        output_path['PGVFile'],
        output_path['ClippedSlopeFile'],
        resampling=Resampling.bilinear
    )

    mapping_unit = _extract_reference_array_to_points(
        mapping_unit,
        resampled_pga,
        'PGA'
    )
    mapping_unit = _extract_reference_array_to_points(
        mapping_unit,
        resampled_pgv,
        'PGV'
    )

    # ------------------------------------------------------------------
    # Step 4: Extract Slope value (÷100 converts tenths-of-degree→degree for MERIT)
    # Note: dtm_slope_merit stores slope in % (0–100 range for 0°–45° approx.)
    # The value already sits in mapping_unit['Slope'] from Step 1 generation;
    # we divide by 100 to match model training units.
    # ------------------------------------------------------------------
    print("  Extracting Slope...", flush=True)
    mapping_unit['Slope'] = mapping_unit['Slope'] / 100.0

    # ------------------------------------------------------------------
    # Step 5: Extract Lithology via spatial join
    # ------------------------------------------------------------------
    print("  Extracting Lithology...", flush=True)
    mapping_unit = _extract_lithology(mapping_unit, lithology_data)

    # ------------------------------------------------------------------
    # Step 6: Extract LC100 (Copernicus land cover, integer codes)
    # ------------------------------------------------------------------
    print("  Extracting LC100...", flush=True)
    mapping_unit = _extract_lc100(mapping_unit, input_path['GlobalLC100File'])

    # Reclassify categorical predictors using the same mappings as model training.
    print("  Reclassifying LC100 and Lithology...", flush=True)
    mapping_unit = _reclassify_categories(mapping_unit)
    _save_category_diagnostics(
        mapping_unit,
        output_path['ClippedSlopeFile'],
        output_path['feature_maps']
    )

    # ------------------------------------------------------------------
    # Step 7: Data cleaning
    # ------------------------------------------------------------------
    # Drop rows missing any core feature for the current HBM models.
    core_cols = ['PGA', 'PGV', 'Slope', 'LC100', 'Lithology']
    mapping_unit = mapping_unit.dropna(subset=core_cols)

    # Filter by PGA threshold
    pga_threshold = float(input_path['PGAThreshold'])
    mapping_unit = mapping_unit[mapping_unit['PGA'] >= pga_threshold]

    # Filter by Slope threshold (degrees)
    slope_threshold = float(input_path['SlopeThreshold'])
    mapping_unit = mapping_unit[mapping_unit['Slope'] >= slope_threshold]

    # Set category dtypes using the training-time levels.
    mapping_unit['Lithology'] = (
        mapping_unit['Lithology'].astype('category').cat.set_categories(LITHOLOGY_LEVELS)
    )
    mapping_unit['LC100'] = (
        mapping_unit['LC100'].astype('category').cat.set_categories(LC100_LEVELS)
    )

    # Optional save
    if input_path.get('IS_save_PointMappingUnit', 'False') == 'True':
        os.makedirs(os.path.dirname(output_path['save_PointMappingUnitFile']), exist_ok=True)
        mapping_unit.to_file(output_path['save_PointMappingUnitFile'], driver='ESRI Shapefile')
        print(f"  Mapping unit saved: {output_path['save_PointMappingUnitFile']}")

    # Save processed feature maps as GeoTIFF rasters
    _save_feature_maps(
        mapping_unit,
        output_path['ClippedSlopeFile'],
        output_path['feature_maps'],
        pgv_output_path=output_path['ResampledPGVFile'],
    )
    print(f"  Resampled PGV feature map saved: {output_path['ResampledPGVFile']}")

    print(f"Feature extraction complete: {len(mapping_unit)} valid grid cells")
    return mapping_unit, grid_area_m2


# =============================================================================
# Internal helpers
# =============================================================================

def _generate_point_mapping_unit(slope_raster_path):
    """Create one point per valid Slope pixel."""
    with rasterio.open(slope_raster_path) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    valid = np.isfinite(data)
    if nodata is not None:
        valid &= ~np.isclose(data, nodata)
    rows_idx, cols_idx = np.where(valid)
    xs, ys = rasterio.transform.xy(transform, rows_idx, cols_idx, offset='center')
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    slope_vals = data[rows_idx, cols_idx].astype(float)

    # Store pixel row/col for later raster reconstruction
    gdf = gpd.GeoDataFrame(
        {
            'Slope':    slope_vals,
            'pix_row':  rows_idx,
            'pix_col':  cols_idx,
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs
    )
    return gdf


def _compute_grid_area_m2(mapping_unit, raster_path):
    """Compute grid cell area in m² by projecting one pixel to local UTM."""
    with rasterio.open(raster_path) as src:
        tr  = src.transform
        raster_crs = src.crs
    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    px_w = abs(tr.a)   # pixel width  in raster CRS units
    px_h = abs(tr.e)   # pixel height in raster CRS units

    if not raster_crs.is_geographic:
        # Already in metres (projected CRS)
        return px_w * px_h

    # Geographic CRS (degrees) → project one pixel to local UTM for accuracy
    cx = float(mapping_unit.geometry.x.mean())
    cy = float(mapping_unit.geometry.y.mean())
    utm_crs = _get_utm_crs(cx, cy)

    transformer = Transformer.from_crs(raster_crs, utm_crs, always_xy=True)
    # Bottom-left and top-right corners of the centre pixel
    x1, y1 = transformer.transform(cx - px_w / 2, cy - px_h / 2)
    x2, y2 = transformer.transform(cx + px_w / 2, cy + px_h / 2)

    return abs(x2 - x1) * abs(y2 - y1)


def _resample_raster_to_reference(src_raster_path, ref_raster_path,
                                  resampling=Resampling.bilinear):
    """Resample a raster to exactly match a reference raster grid and return an array."""
    dst_nodata = -9999.0

    with rasterio.open(ref_raster_path) as ref_src:
        dst_shape = (ref_src.height, ref_src.width)
        dst_transform = ref_src.transform
        dst_crs = ref_src.crs
        if dst_crs is None:
            raise ValueError(f"Reference raster has no CRS: {ref_raster_path}")

    with rasterio.open(src_raster_path) as src:
        src_data = src.read(1)
        src_crs = src.crs
        if src_crs is None:
            # Existing ShakeMap PGV.tif files made from .flt may lack CRS metadata.
            # ShakeMap raster grids are geographic WGS84, so use EPSG:4326 here.
            src_crs = RasterioCRS.from_epsg(4326)
            print(f"  PGV source CRS missing; assuming EPSG:4326 for {src_raster_path}")
        dst_data = np.full(dst_shape, dst_nodata, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return dst_data


def _extract_reference_array_to_points(mapping_unit, data, col_name, nodata=-9999.0):
    """Extract values from an array already aligned to mapping_unit pix_row/pix_col."""
    rows = mapping_unit['pix_row'].values
    cols = mapping_unit['pix_col'].values

    vals = data[rows, cols].astype(float)
    vals[np.isclose(vals, nodata)] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    mapping_unit[col_name] = vals
    return mapping_unit


def _extract_lithology(mapping_unit, lithology_data):
    """Spatial join to extract GLiM lithology code."""
    if mapping_unit.crs != lithology_data.crs:
        lithology_data = lithology_data.to_crs(mapping_unit.crs)

    # Pre-clip global lithology to study-area bounding box before sjoin
    minx, miny, maxx, maxy = mapping_unit.total_bounds
    lithology_data = lithology_data.cx[minx:maxx, miny:maxy]

    joined = gpd.sjoin(
        mapping_unit,
        lithology_data[['geometry', 'xx']],
        how='left',
        predicate='intersects'
    )
    joined.rename(columns={'xx': 'Lithology'}, inplace=True)
    joined.drop(columns='index_right', errors='ignore', inplace=True)
    return joined


def _extract_lc100(mapping_unit, lc100_path):
    """Extract Copernicus LC100 integer codes via rasterio.sample."""
    with rasterio.open(lc100_path) as src:
        if src.crs is None:
            raise ValueError(f"LC100 raster has no CRS: {lc100_path}")
        sample_points = (
            mapping_unit
            if mapping_unit.crs == src.crs
            else mapping_unit.to_crs(src.crs)
        )
        coords = list(zip(sample_points.geometry.x, sample_points.geometry.y))
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    mapping_unit['LC100'] = vals
    return mapping_unit


def _get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return CRS.from_epsg(32600 + zone)
    else:
        return CRS.from_epsg(32700 + zone)


def _save_feature_maps(mapping_unit, ref_raster_path, output_dir, pgv_output_path=None):
    """Save all extracted feature columns back as GeoTIFF rasters (float32, nodata=-9999)."""
    with rasterio.open(ref_raster_path) as src:
        H, W = src.height, src.width
        meta = src.meta.copy()
    meta.update(dtype='float32', count=1, nodata=-9999.0)

    pix_row = mapping_unit['pix_row'].values
    pix_col = mapping_unit['pix_col'].values

    numeric_cols = [
        ('PGA',      os.path.join(output_dir, 'PGA.tif')),
        ('PGV',      pgv_output_path or os.path.join(output_dir, 'Resampled_PGV.tif')),
        ('Slope',    os.path.join(output_dir, 'Slope.tif')),
        ('LC100_raw' if 'LC100_raw' in mapping_unit.columns else 'LC100',
                    os.path.join(output_dir, 'LC100.tif')),
    ]
    for col, out_path in numeric_cols:
        if col not in mapping_unit.columns:
            continue
        vals = mapping_unit[col].values.astype(float)
        if np.isnan(vals).all():
            continue
        arr = np.full((H, W), -9999.0, dtype=np.float32)
        arr[pix_row, pix_col] = vals.astype(np.float32)
        arr[arr == -9999.0] = -9999.0  # ensure fill stays intact after assignment
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(arr, 1)

    print(f"  Feature maps saved: {output_dir}")
