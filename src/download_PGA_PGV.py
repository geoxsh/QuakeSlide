"""
ShakeMap download module for QuakeSlide HBM system.
Downloads PGA and PGV rasters from USGS ShakeMap.

Date: August 2026
"""

import json
import os
import requests
import shutil
import zipfile
from datetime import datetime, timezone
from osgeo import gdal, osr
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from src.shakemap_product import (
    describe_shakemap_product,
    fetch_event_geojson,
    format_shakemap_version,
    select_shakemap_product,
    shakemap_update_time,
)


def download_shakemap_rasters(
        event_id, output_path, event_data=None, shakemap=None):
    """
    Download USGS ShakeMap raster.zip and extract both PGA.tif and PGV.tif.

    Args:
        event_id (str): USGS earthquake event ID
        output_path (dict): Dictionary containing output file paths
    """
    _remove_composite_shakemap_markers(output_path)

    # Match USGS groundfailure's default ShakeMap source='preferred' logic.
    # References are documented in src/shakemap_product.py.
    if event_data is None:
        event_data = fetch_event_geojson(
            event_id, include_superseded=False, timeout=60
        )
    if shakemap is None:
        shakemap = select_shakemap_product(
            event_data,
            required_content="download/raster.zip",
        )
    print(f"Using ShakeMap product: {describe_shakemap_product(shakemap)}")

    contents = shakemap["contents"]
    file_url  = contents["download/raster.zip"]["url"]
    zip_path  = os.path.join(output_path['ShakeMap'], f"{event_id}_raster.zip")

    file_response = requests.get(file_url, timeout=120)
    if file_response.status_code != 200:
        raise RuntimeError(f"Failed to download raster.zip (HTTP {file_response.status_code})")

    with open(zip_path, "wb") as fh:
        fh.write(file_response.content)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_path['ShakeMap'])

    _process_flt_to_tif("pga_mean.flt", output_path['ShakeMap'], output_path['PGAFile'])
    _process_flt_to_tif("pgv_mean.flt", output_path['ShakeMap'], output_path['PGVFile'])
    source_metadata = _write_ground_motion_source(
        output_path,
        {
            "mode": "usgs",
            "description": format_shakemap_version(shakemap),
            "event_id": event_id,
            "source": shakemap.get("source"),
            "code": shakemap.get("code"),
            "update_time": shakemap_update_time(shakemap),
            "raster_url": file_url,
        },
    )
    print(f"ShakeMap rasters downloaded: PGA={output_path['PGAFile']}, PGV={output_path['PGVFile']}")
    return source_metadata


def download_composite_shakemap_rasters(
        event_ids,
        output_path,
        resampling=Resampling.bilinear,
        event_data_by_id=None,
        composite_context=None):
    """
    Download ShakeMap rasters for multiple events and build a pixelwise-max composite.

    The output composite rasters are saved as Composite_PGA.tif and Composite_PGV.tif
    in the ShakeMap folder. They are also copied to output_path['PGAFile'] and
    output_path['PGVFile'] so downstream QuakeSlide steps use the composite
    ground motions.
    """
    if len(event_ids) < 2:
        raise ValueError("At least two event_ids are required to build a composite ShakeMap.")

    composite_dir = os.path.join(output_path['ShakeMap'], "composite_sources")
    os.makedirs(composite_dir, exist_ok=True)

    pga_files = []
    pgv_files = []
    selected_products = {}
    for event_id in event_ids:
        event_dir = os.path.join(composite_dir, event_id)
        os.makedirs(event_dir, exist_ok=True)

        event_pga = os.path.join(event_dir, "PGA.tif")
        event_pgv = os.path.join(event_dir, "PGV.tif")
        event_data = (
            event_data_by_id.get(event_id)
            if event_data_by_id is not None
            else None
        )
        if event_data is None:
            event_data = fetch_event_geojson(
                event_id, include_superseded=False, timeout=60
            )
        selected_products[event_id] = select_shakemap_product(
            event_data, required_content="download/raster.zip"
        )
        product = selected_products[event_id]
        cache_is_current = _composite_source_is_current(
            event_dir, event_pga, event_pgv, product
        )
        if not cache_is_current:
            print(f"  Downloading ShakeMap rasters for composite event: {event_id}")
            _download_event_raster_zip(
                event_id, event_dir, event_data=event_data, shakemap=product
            )
            _process_flt_to_tif("pga_mean.flt", event_dir, event_pga)
            _process_flt_to_tif("pgv_mean.flt", event_dir, event_pgv)
        else:
            print(f"  Reusing existing composite source rasters: {event_id}")

        pga_files.append(event_pga)
        pgv_files.append(event_pgv)

    composite_pga = os.path.join(output_path['ShakeMap'], "Composite_PGA.tif")
    composite_pgv = os.path.join(output_path['ShakeMap'], "Composite_PGV.tif")
    _write_pixelwise_max_composite(pga_files, composite_pga, resampling=resampling)
    _write_pixelwise_max_composite(pgv_files, composite_pgv, resampling=resampling)

    shutil.copy2(composite_pga, output_path['PGAFile'])
    shutil.copy2(composite_pgv, output_path['PGVFile'])
    manifest = _write_composite_manifest(
        event_ids=event_ids,
        output_path=output_path,
        composite_context=composite_context,
        selected_products=selected_products,
        pga_files=pga_files,
        pgv_files=pgv_files,
        composite_pga=composite_pga,
        composite_pgv=composite_pgv,
    )
    print(f"Composite ShakeMap rasters saved: PGA={composite_pga}, PGV={composite_pgv}")
    if composite_context is None:
        composite_context = {}
    composite_context["manifest"] = manifest
    return composite_context


def use_custom_shakemap_rasters(pga_file, pgv_file, output_path):
    """
    Use externally prepared PGA and PGV rasters instead of downloading USGS raster.zip.

    This is intended for externally prepared ShakeMaps. The event ID remains the
    source of earthquake metadata for the report.
    """
    _remove_composite_shakemap_markers(output_path)

    for src_file, dst_file, label in [
        (pga_file, output_path['PGAFile'], 'PGA'),
        (pgv_file, output_path['PGVFile'], 'PGV'),
    ]:
        if src_file is None:
            raise ValueError(f"Custom {label} raster path is required.")
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Custom {label} raster not found: {src_file}")

        src_abs = os.path.abspath(src_file)
        dst_abs = os.path.abspath(dst_file)
        if src_abs != dst_abs:
            shutil.copy2(src_abs, dst_abs)

    source_metadata = _write_ground_motion_source(
        output_path,
        {
            "mode": "custom",
            "description": "Custom PGA/PGV rasters",
            "source_pga": os.path.abspath(pga_file),
            "source_pgv": os.path.abspath(pgv_file),
        },
    )
    print(f"Custom ShakeMap rasters copied: PGA={output_path['PGAFile']}, PGV={output_path['PGVFile']}")
    return source_metadata


def _write_ground_motion_source(output_path, metadata):
    """Record the exact ground-motion source used by downstream processing."""
    metadata_path = output_path.get(
        'ShakeMapSourceFile',
        os.path.join(output_path['ShakeMap'], "shakemap_source.json"),
    )
    payload = dict(metadata)
    payload["recorded_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
    return payload


def _remove_composite_shakemap_markers(output_path):
    """Remove old composite raster markers before a non-composite ShakeMap run."""
    for filename in ["Composite_PGA.tif", "Composite_PGV.tif"]:
        path = os.path.join(output_path['ShakeMap'], filename)
        if os.path.exists(path):
            os.remove(path)


def _download_event_raster_zip(event_id, event_dir, event_data=None, shakemap=None):
    """Download and extract one USGS ShakeMap raster.zip into event_dir."""
    if event_data is None:
        event_data = fetch_event_geojson(event_id, include_superseded=False, timeout=60)
    if shakemap is None:
        shakemap = select_shakemap_product(
            event_data,
            required_content="download/raster.zip",
        )
    print(f"    Using ShakeMap product: {describe_shakemap_product(shakemap)}")

    file_url = shakemap["contents"]["download/raster.zip"]["url"]
    zip_path = os.path.join(event_dir, f"{event_id}_raster.zip")

    file_response = requests.get(file_url, timeout=120)
    if file_response.status_code != 200:
        raise RuntimeError(f"Failed to download raster.zip for {event_id} (HTTP {file_response.status_code})")

    with open(zip_path, "wb") as fh:
        fh.write(file_response.content)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(event_dir)

    metadata_path = os.path.join(event_dir, "shakemap_source.json")
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump({
            "event_id": event_id,
            "source": shakemap.get("source"),
            "code": shakemap.get("code"),
            "update_time": shakemap_update_time(shakemap),
            "description": format_shakemap_version(shakemap),
        }, file_handle, indent=2)


def _composite_source_is_current(event_dir, pga_file, pgv_file, shakemap):
    """Reuse a component cache only when it matches the selected ShakeMap."""
    metadata_path = os.path.join(event_dir, "shakemap_source.json")
    if not (os.path.exists(pga_file) and os.path.exists(pgv_file) and os.path.exists(metadata_path)):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        return (
            metadata.get("source") == shakemap.get("source")
            and metadata.get("code") == shakemap.get("code")
            and metadata.get("update_time") == shakemap_update_time(shakemap)
        )
    except (OSError, ValueError, TypeError):
        return False


def _relative_output_path(path, output_path):
    return os.path.relpath(path, output_path['BaseDir']).replace(os.sep, "/")


def _write_composite_manifest(
        event_ids,
        output_path,
        composite_context,
        selected_products,
        pga_files,
        pgv_files,
        composite_pga,
        composite_pgv):
    """Write traceable metadata for one composite ShakeMap assessment."""
    context = composite_context or {}
    records_by_id = {
        record["event_id"]: record for record in context.get("events", [])
    }
    mainshock_id = context.get("mainshock_event_id", event_ids[0])
    component_events = []
    for index, event_id in enumerate(event_ids):
        record = records_by_id.get(event_id, {})
        product = selected_products[event_id]
        component_events.append({
            "role": "mainshock" if event_id == mainshock_id else "additional_event",
            "event_id": event_id,
            "magnitude": record.get("magnitude"),
            "time_utc": record.get("time_utc"),
            "place": record.get("place", ""),
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
            "depth_km": record.get("depth_km"),
            "shakemap": {
                "description": format_shakemap_version(product),
                "source": product.get("source"),
                "code": product.get("code"),
                "update_time": shakemap_update_time(product),
            },
            "source_pga": _relative_output_path(pga_files[index], output_path),
            "source_pgv": _relative_output_path(pgv_files[index], output_path),
        })

    manifest = {
        "schema_version": 1,
        "assessment_type": "composite_shakemap",
        "run_id": context.get("run_id", output_path.get("RunID")),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mainshock_event_id": mainshock_id,
        "component_event_ids": list(event_ids),
        "combination": "pixelwise maximum PGA and PGV",
        "component_events": component_events,
        "outputs": {
            "composite_pga": _relative_output_path(composite_pga, output_path),
            "composite_pgv": _relative_output_path(composite_pgv, output_path),
            "workflow_pga": _relative_output_path(output_path['PGAFile'], output_path),
            "workflow_pgv": _relative_output_path(output_path['PGVFile'], output_path),
        },
    }
    manifest_path = os.path.join(output_path['ShakeMap'], "composite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, ensure_ascii=False)
    return manifest


def _raster_crs(src):
    # USGS ShakeMap rasters are geographic WGS84; use it if CRS metadata is absent.
    return src.crs if src.crs is not None else rasterio.crs.CRS.from_epsg(4326)


def _aligned_union_grid(raster_files):
    """Return a union grid aligned to the first raster's resolution and origin."""
    with rasterio.open(raster_files[0]) as ref:
        dst_crs = _raster_crs(ref)
        xres = abs(ref.transform.a)
        yres = abs(ref.transform.e)
        ref_left = ref.transform.c
        ref_top = ref.transform.f

    bounds_list = []
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            src_crs = _raster_crs(src)
            bounds = src.bounds
            if src_crs != dst_crs:
                bounds = transform_bounds(
                    src_crs,
                    dst_crs,
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                    densify_pts=21,
                )
            else:
                bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            bounds_list.append(bounds)

    left = min(bounds[0] for bounds in bounds_list)
    bottom = min(bounds[1] for bounds in bounds_list)
    right = max(bounds[2] for bounds in bounds_list)
    top = max(bounds[3] for bounds in bounds_list)

    left = ref_left + np.floor((left - ref_left) / xres) * xres
    top = ref_top + np.ceil((top - ref_top) / yres) * yres
    width = int(np.ceil((right - left) / xres))
    height = int(np.ceil((top - bottom) / yres))
    transform = from_origin(left, top, xres, yres)
    return dst_crs, transform, width, height


def _write_pixelwise_max_composite(raster_files, out_tif_path, resampling=Resampling.bilinear):
    """Align rasters to a common union grid and write their pixelwise maximum."""
    dst_crs, dst_transform, width, height = _aligned_union_grid(raster_files)
    composite = np.full((height, width), np.nan, dtype=np.float32)

    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            src_crs = _raster_crs(src)
            src_data = src.read(1).astype(np.float32)
            warped = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=src_data,
                destination=warped,
                src_transform=src.transform,
                src_crs=src_crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            valid = np.isfinite(warped)
            empty = ~np.isfinite(composite)
            composite[valid & empty] = warped[valid & empty]
            composite[valid & ~empty] = np.maximum(
                composite[valid & ~empty],
                warped[valid & ~empty],
            )

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(out_tif_path, "w", **meta) as dst:
        dst.write(composite, 1)


def _process_flt_to_tif(flt_filename, shakemap_dir, out_tif_path):
    """
    Read a ShakeMap .flt file (log scale), apply exp(), save as GeoTIFF.

    Args:
        flt_filename (str): Name of the .flt file inside shakemap_dir
        shakemap_dir (str): Directory containing extracted ShakeMap files
        out_tif_path (str): Full path for the output GeoTIFF
    """
    flt_path = os.path.join(shakemap_dir, flt_filename)
    if not os.path.exists(flt_path):
        raise FileNotFoundError(f"Expected ShakeMap file not found: {flt_path}")

    dataset = gdal.Open(flt_path)
    if dataset is None:
        raise RuntimeError(f"GDAL could not open {flt_path}")

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= ~np.isclose(data, nodata)
    exp_data = np.full(data.shape, np.nan, dtype=np.float32)
    exp_data[valid] = np.exp(data[valid]).astype(np.float32)

    temp_tif_path = f"{out_tif_path}.{os.getpid()}.tmp.tif"
    if os.path.exists(temp_tif_path):
        os.remove(temp_tif_path)

    driver      = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(
        temp_tif_path,
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())

    projection = dataset.GetProjection()
    if not projection:
        # USGS ShakeMap .flt grids are geographic WGS84, but some .flt files
        # do not expose their CRS through GDAL. Write it explicitly so later
        # rasterio.warp.reproject calls have a valid source CRS.
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        projection = srs.ExportToWkt()
    out_dataset.SetProjection(projection)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(exp_data)
    out_band.SetNoDataValue(np.nan)
    if flt_filename.startswith("pga_"):
        out_band.SetUnitType("g")
    elif flt_filename.startswith("pgv_"):
        out_band.SetUnitType("cm/s")
    out_dataset.FlushCache()
    del out_dataset, dataset
    os.replace(temp_tif_path, out_tif_path)
