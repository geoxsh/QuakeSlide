"""
QuakeSlide Main Module (HBM version).

8-step workflow for earthquake-triggered landslide impact assessment
using Bayesian hierarchical models (LSCount + LSCoverage).

Date: August 2026
"""

import os
import time
import hashlib

from src.define_file_path import define_file_path
from src.composite_event import prepare_composite_events
from src.download_PGA_PGV import (
    download_composite_shakemap_rasters,
    download_shakemap_rasters,
    use_custom_shakemap_rasters,
)
from src.create_landslide_affected_area import create_landslide_affected_area
from src.preload_data import preload_data
from src.feature_extraction import feature_extraction
from src.hazard_module import cleanup_prediction_stage_files, predict_hbm
from src.impacts_module import uncertainty_propagation_impacts
from src.report_generation import report_generation
from src.run_logging import run_log, workflow_step
from src.shakemap_product import (
    extract_event_metadata,
    fetch_event_geojson,
    select_shakemap_product,
    shakemap_update_time as get_shakemap_update_time,
)


def _display_report_in_notebook(report_png):
    """Display a saved report when running inside a Jupyter kernel."""
    if not report_png or not os.path.exists(report_png):
        return False

    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return False

    shell = get_ipython()
    if shell is None or not hasattr(shell, 'kernel'):
        return False

    display(Image(filename=os.path.abspath(report_png)))
    return True


def QuakeSlide(event_id=None,
               lithology_data=None,
               road_network_data=None,
               custom_pga_file=None,
               custom_pgv_file=None,
               composite_event_ids=None,
               input_file_path="Input.txt"):
    """
    Main QuakeSlide HBM workflow.

    Parameters:
        event_id (str, optional): USGS earthquake event ID for a single-event run.
            It is not required when composite_event_ids is supplied.
        lithology_data (GeoDataFrame, optional): In-memory lithology cache; auto-loaded
            from GeoParquet on first use if None.
        road_network_data: Cached road network (NetworkX graph); re-downloaded from OSM
            when event_id changes.
        custom_pga_file, custom_pgv_file (str, optional): Externally prepared
            PGA/PGV GeoTIFFs. When both are supplied, Step 2 uses these rasters
            instead of downloading USGS raster.zip. The event_id remains the
            source of earthquake metadata for this mode.
        composite_event_ids (list[str], optional): Event IDs whose ShakeMap PGA/PGV
            rasters should be combined by pixelwise maximum before prediction. The
            largest-magnitude event is selected automatically as the mainshock;
            an earlier origin time resolves an exact magnitude tie.

    Returns:
        tuple: (lithology_data, road_network_data) for reuse in subsequent calls
    """
    use_composite_shakemap = composite_event_ids is not None
    use_custom_shakemap = custom_pga_file is not None or custom_pgv_file is not None
    selected_shakemap = None
    shakemap_source = None
    if use_composite_shakemap and use_custom_shakemap:
        raise ValueError("Use either composite_event_ids or custom_pga_file/custom_pgv_file, not both.")
    if use_custom_shakemap and not (custom_pga_file and custom_pgv_file):
        raise ValueError("Both custom_pga_file and custom_pgv_file must be supplied together.")
    composite_context = None
    if use_composite_shakemap:
        composite_context = prepare_composite_events(composite_event_ids)
        composite_event_ids = composite_context["component_event_ids"]
        if event_id is not None:
            supplied_event_id = str(event_id).strip()
            if supplied_event_id not in composite_event_ids:
                raise ValueError(
                    "When event_id is supplied with composite_event_ids, it must also "
                    "appear in the composite event list."
                )
            run_log(
                "NOTICE: event_id is redundant in composite mode; "
                "the mainshock is selected automatically from USGS metadata."
            )
        event_id = composite_context["mainshock_event_id"]
        run_id = composite_context["run_id"]
        resp_json = composite_context["event_data_by_id"][event_id]
    else:
        if event_id is None or not str(event_id).strip():
            raise ValueError("event_id is required for a single-event or custom ShakeMap run.")
        event_id = str(event_id).strip()
        run_id = event_id
        resp_json = fetch_event_geojson(
            event_id,
            include_superseded=False,
            timeout=30,
        )

    event_metadata = extract_event_metadata(resp_json, event_id=event_id)
    if use_composite_shakemap:
        # Use the component ShakeMap update times as the cache token so the
        # road-network extent is reused for the same composite ShakeMap, but
        # invalidated if any component ShakeMap product changes.
        composite_parts = []
        for component_event_id in sorted(composite_event_ids):
            component_geojson = composite_context["event_data_by_id"][component_event_id]
            component_shakemap = select_shakemap_product(
                component_geojson, required_content="download/raster.zip"
            )
            component_update_time = get_shakemap_update_time(component_shakemap)
            composite_parts.append(f"{component_event_id}:{component_update_time}")
        composite_key = "|".join(composite_parts)
        shakemap_update_time = int(
            hashlib.sha1(composite_key.encode("utf-8")).hexdigest()[:12],
            16,
        )
    elif use_custom_shakemap:
        # The road-network cache extent depends on the ShakeMap raster footprint.
        # Use the composite raster mtimes as the cache token so a previous
        # single-event run of the same event_id is not accidentally reused.
        shakemap_update_time = max(
            os.path.getmtime(custom_pga_file),
            os.path.getmtime(custom_pgv_file),
        )
        shakemap_update_time = int(shakemap_update_time)
    else:
        # Extract the selected ShakeMap updateTime for road-network cache invalidation.
        # This matches USGS groundfailure's default ShakeMap source='preferred'
        # logic; references are documented in src/shakemap_product.py.
        selected_shakemap = select_shakemap_product(
            resp_json, required_content="download/raster.zip"
        )
        shakemap_update_time = get_shakemap_update_time(selected_shakemap)

    magnitude = event_metadata["magnitude"]
    magnitude_text = f"{float(magnitude):g}" if magnitude is not None else "unknown"
    component_text = ", ".join(composite_event_ids) if use_composite_shakemap else event_id
    run_log(f"QuakeSlide run started: {run_id}")
    run_log(f"Components: {component_text}")
    run_log(
        f"Primary event: {event_id} (Mw {magnitude_text}; "
        f"origin time {event_metadata['time_utc']})"
    )
    start_time = time.time()

    with workflow_step(1, 8, "Configuration: loading Input.txt"):
        input_path, output_path = define_file_path(input_file_path, run_id)
        if input_path is None:
            raise FileNotFoundError("Input.txt not found or failed to parse.")

    with workflow_step(2, 8, "Preparing ShakeMap"):
        if use_composite_shakemap:
            composite_context = download_composite_shakemap_rasters(
                composite_event_ids,
                output_path,
                event_data_by_id=composite_context["event_data_by_id"],
                composite_context=composite_context,
            )
        elif use_custom_shakemap:
            print(f"PGA source: {custom_pga_file}")
            print(f"PGV source: {custom_pgv_file}")
            shakemap_source = use_custom_shakemap_rasters(
                custom_pga_file, custom_pgv_file, output_path
            )
        else:
            shakemap_source = download_shakemap_rasters(
                event_id,
                output_path,
                event_data=resp_json,
                shakemap=selected_shakemap,
            )

    with workflow_step(3, 8, "Delineating landslide affected area"):
        create_landslide_affected_area(input_path, output_path)

    with workflow_step(4, 8, "Predictors: preparing features, rivers, and roads"):
        lithology_data, river_network_data, road_network_data = preload_data(
            event_id, input_path, output_path,
            lithology_data=lithology_data,
            road_network_data=road_network_data,
            shakemap_update_time=shakemap_update_time,
        )

    with workflow_step(5, 8, "Extracting features"):
        mapping_unit, grid_area_m2 = feature_extraction(
            lithology_data, input_path, output_path
        )

    with workflow_step(6, 8, "Hazard: running prediction"):
        simulated_counts_store, simulated_area_coverages_store, grid_area_m2 = predict_hbm(
            mapping_unit, grid_area_m2, input_path, output_path
        )

    with workflow_step(7, 8, "Impacts: propagating uncertainty"):
        uncertainty_propagation_impacts(
            grid_area_m2,
            simulated_counts_store,
            simulated_area_coverages_store,
            input_path, output_path,
            river_network_data, road_network_data,
        )

    with workflow_step(8, 8, "Generating one-page report"):
        report_files = report_generation(
            event_id,
            input_path,
            output_path,
            start_time,
            composite_metadata=composite_context,
            event_data=resp_json,
            shakemap_product=selected_shakemap,
            shakemap_source=shakemap_source,
        )

    removed_stage_files = cleanup_prediction_stage_files(output_path)
    if removed_stage_files:
        run_log(f"Cleaned {len(removed_stage_files)} prediction staging files")

    elapsed = (time.time() - start_time) / 60
    run_log(f"QuakeSlide completed in {elapsed:.1f} min")
    run_log(f"Results saved to: {output_path['BaseDir']}/")
    _display_report_in_notebook(report_files.get('png'))

    return lithology_data, road_network_data
