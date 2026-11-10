"""
Data preloading module for QuakeSlide HBM system.
Loads and caches global datasets: lithology, river network, road network.

Date: August 2026
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from contextlib import contextmanager
from datetime import datetime
import pathlib
import sys
import threading
import time
import geopandas as gpd
import pyogrio
import osmnx as ox
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout


# Public, read-only global Overpass instance currently used by QuakeSlide.
OVERPASS_ENDPOINT = "https://maps.mail.ru/osm/tools/overpass/api"
ROAD_REQUEST_MAX_ATTEMPTS = 5
ROAD_REQUEST_RETRY_DELAYS = (5, 15, 30, 60)


@contextmanager
def _road_download_progress(label="Downloading road network from OpenStreetMap"):
    """Display progress for the actual number of OSMnx Overpass requests."""
    stop_event = threading.Event()
    started = time.perf_counter()
    stream = getattr(sys.stdout, 'stream', sys.stdout)
    bar_width = 24
    last_width = 0
    lock = threading.Lock()
    state = {'completed': 0, 'total': None, 'note': ''}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_total(total):
        with lock:
            state['total'] = max(1, int(total))

    def set_completed(completed):
        with lock:
            state['completed'] = max(0, int(completed))

    def set_note(note):
        with lock:
            state['note'] = str(note)

    def render(status=None):
        nonlocal last_width
        with lock:
            completed = state['completed']
            total = state['total']
            note = state['note']
        if total is None:
            return
        completed = min(completed, total)
        elapsed = time.perf_counter() - started
        fraction = completed / total
        filled = round(bar_width * fraction)
        bar = '#' * filled + '-' * (bar_width - filled)
        percentage = round(fraction * 100)
        status_text = f", {status}" if status else (f", {note}" if note else "")
        line = (
            f"{timestamp} |       {label} [{bar}] "
            f"{percentage:3d}% ({completed}/{total}) {elapsed:6.1f} s{status_text}"
        )
        padding = ' ' * max(0, last_width - len(line))
        stream.write(f"\r{line}{padding}")
        stream.flush()
        last_width = len(line)

    def update():
        while not stop_event.wait(0.2):
            render()

    worker = threading.Thread(target=update, daemon=True)
    worker.start()
    status = 'done'
    try:
        yield set_total, set_completed, set_note
    except Exception:
        status = 'failed'
        raise
    finally:
        stop_event.set()
        worker.join(timeout=1)
        with lock:
            if state['total'] is None:
                state['total'] = 1
            if status == 'done':
                state['completed'] = state['total']
        render(status)
        stream.write("\n")
        stream.flush()


def _find_windows_socket_error(error):
    """Return a nested Windows socket error number without rendering its locale text."""
    pending = [error]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))

        winerror = getattr(current, 'winerror', None)
        if winerror is not None:
            return winerror

        for item in getattr(current, 'args', ()):
            if isinstance(item, BaseException):
                pending.append(item)
        cause = getattr(current, '__cause__', None)
        context = getattr(current, '__context__', None)
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return None


def _format_road_download_error(error):
    """Format network failures without OS-localized exception messages."""
    winerror = _find_windows_socket_error(error)
    if isinstance(error, RequestsTimeout):
        return "Timeout: Overpass API request timed out"
    if isinstance(error, RequestsConnectionError):
        if winerror == 10054:
            return "ConnectionError: remote host reset the connection (WinError 10054)"
        if winerror is not None:
            return f"ConnectionError: unable to reach Overpass API (WinError {winerror})"
        return "ConnectionError: unable to reach Overpass API"
    return f"{type(error).__name__}: {error}"


def _apply_osmnx_overpass_status_fix():
    """Work around OSMnx 1.9.x masking a status-endpoint connection error.

    In OSMnx 1.9.x, ``_get_overpass_pause`` can reference ``response`` before
    it has been assigned when the Overpass ``/status`` request raises a
    ConnectionError.  Python 3.8 cannot install OSMnx 2.x, so retain the
    package's normal rate-limit behaviour but fall back to its default pause
    when this specific upstream bug occurs.
    """
    if not str(getattr(ox, '__version__', '')).startswith('1.9.'):
        return

    from osmnx import _overpass

    original = _overpass._get_overpass_pause
    if getattr(original, '_quakeslide_safe_status', False):
        return

    def safe_get_overpass_pause(base_endpoint, recursive_delay=5, default_duration=60):
        try:
            return original(base_endpoint, recursive_delay, default_duration)
        except UnboundLocalError as exc:
            if "local variable 'response'" not in str(exc):
                raise
            print(
                "  Overpass status endpoint was temporarily unreachable; "
                "continuing with the data request..."
            )
            return 0

    safe_get_overpass_pause._quakeslide_safe_status = True
    _overpass._get_overpass_pause = safe_get_overpass_pause


def _download_road_network_from_overpass(
    polygon,
    custom_filter,
    endpoint=OVERPASS_ENDPOINT,
):
    """Download a road graph from the configured global Overpass endpoint."""
    from osmnx import _overpass

    original_endpoint = ox.settings.overpass_url
    original_pause_checker = _overpass._get_overpass_pause
    original_downloader = _overpass._download_overpass_network
    ox.settings.overpass_url = endpoint
    _overpass._get_overpass_pause = lambda *args, **kwargs: 0
    print(f"  Querying road data from {endpoint}")
    try:
        with _road_download_progress() as (set_total, set_completed, set_note):
            def monitored_downloader(query_polygon, network_type, query_filter):
                polygon_strings = _overpass._make_overpass_polygon_coord_strs(
                    query_polygon
                )
                set_total(len(polygon_strings))
                osm_filter = (
                    query_filter
                    if query_filter is not None
                    else _overpass._get_osm_filter(network_type)
                )
                overpass_settings = _overpass._make_overpass_settings()

                for request_number, polygon_string in enumerate(
                    polygon_strings, start=1
                ):
                    query = (
                        f"{overpass_settings};"
                        f"(way{osm_filter}(poly:{polygon_string!r});>;);out;"
                    )
                    for attempt in range(1, ROAD_REQUEST_MAX_ATTEMPTS + 1):
                        try:
                            response = _overpass._overpass_request(
                                data={'data': query}
                            )
                            set_note('')
                            break
                        except (RequestsConnectionError, RequestsTimeout):
                            if attempt == ROAD_REQUEST_MAX_ATTEMPTS:
                                raise
                            delay = ROAD_REQUEST_RETRY_DELAYS[attempt - 1]
                            set_note(
                                f"request {request_number}/{len(polygon_strings)} "
                                f"retry {attempt + 1}/{ROAD_REQUEST_MAX_ATTEMPTS} "
                                f"in {delay} s"
                            )
                            time.sleep(delay)
                    set_completed(request_number)
                    yield response

            _overpass._download_overpass_network = monitored_downloader
            return ox.graph_from_polygon(
                polygon,
                custom_filter=custom_filter,
            )
    finally:
        ox.settings.overpass_url = original_endpoint
        _overpass._get_overpass_pause = original_pause_checker
        _overpass._download_overpass_network = original_downloader


def preload_data(event_id, input_path, output_path,
                 lithology_data=None,
                 road_network_data=None,
                 shakemap_update_time=0):
    """
    Load and cache global geographic datasets needed for feature extraction and impact assessment.

    Lithology is automatically cached as GeoParquet next to the source file so that
    subsequent calls (even with lithology_data=None) load in seconds rather than minutes.

    River network is always reloaded for the affected-area bbox.

    Args:
        event_id (str): USGS earthquake event ID
        input_path (dict): Input parameters and file paths
        output_path (dict): Output file paths
        lithology_data (GeoDataFrame, optional): In-memory lithology cache
        road_network_data: Cached road network (NetworkX graph); reused when same event_id
            and ShakeMap has not been updated
        shakemap_update_time (int): USGS ShakeMap update_time timestamp; used to detect
            ShakeMap updates that would change the road network extent
    Returns:
        tuple: (lithology_data, river_network_data, road_network_data)
    """
    # ------------------------------------------------------------------
    # Lithology (Global Lithological Map – GLiM)
    # Auto-cached as GeoParquet on first load for fast re-reads.
    # ------------------------------------------------------------------
    lith_source = pathlib.Path(input_path['GlobalLithologyFile'])
    lith_parquet = (lith_source if lith_source.suffix == '.parquet'
                    else lith_source.parent / 'lithology_cached.parquet')

    if lithology_data is None:
        if lith_parquet.exists():
            print("Loading lithology data (parquet cache)...")
            t0 = time.time()
            lithology_data = gpd.read_parquet(lith_parquet)
            print(f"  Lithology loaded in {(time.time()-t0)/60:.2f} min")
        else:
            print("Loading lithology data (first time - will save parquet cache)...")
            t0 = time.time()
            lithology_data = pyogrio.read_dataframe(
                input_path['GlobalLithologyFile'],
                layer='GLiM_export'
            )
            lithology_data = gpd.GeoDataFrame(lithology_data, geometry=lithology_data.geometry)
            lithology_data.to_parquet(lith_parquet)
            print(f"  Lithology loaded in {(time.time()-t0)/60:.2f} min  (parquet cache saved)")
    else:
        print("Skipping lithology data loading (in-memory)")

    # ------------------------------------------------------------------
    # River network (clipped to affected-area bbox) – fast, always reload
    # ------------------------------------------------------------------
    print("Loading river network data (clipped to affected area)...")
    t0 = time.time()
    polygon = gpd.read_file(output_path['LandslideAffectedArea']).to_crs(epsg=4326)
    bbox = tuple(polygon.total_bounds)   # (minx, miny, maxx, maxy)
    river_network_data = gpd.read_file(input_path['GlobalRiverNetwork'], bbox=bbox)
    print(f"  River network loaded in {(time.time()-t0)/60:.2f} min  "
          f"({len(river_network_data)} features)")

    # ------------------------------------------------------------------
    # Road network (event-specific – downloaded from OSM)
    # Cached as GraphML in intermediate/cache/; filename encodes event_id +
    # shakemap_update_time so the cache is reused on reruns and
    # automatically invalidated when the ShakeMap is updated.
    # ------------------------------------------------------------------
    cache_dir = pathlib.Path(output_path['CacheDir'])
    road_graphml = cache_dir / f'road_network_{event_id}_{shakemap_update_time}.graphml'

    in_memory_valid = (
        road_network_data is not None
        and getattr(road_network_data, 'event_id', None) == event_id
        and getattr(road_network_data, 'shakemap_update_time', None) == shakemap_update_time
    )

    if in_memory_valid:
        print("Skipping road network download (in-memory cache)")
    elif road_graphml.exists():
        print(f"Loading road network from cache ({road_graphml.name})...")
        t0 = time.time()
        road_network_data = ox.load_graphml(str(road_graphml))
        road_network_data.event_id = event_id
        road_network_data.shakemap_update_time = shakemap_update_time
        print(f"  Road network loaded from cache in {(time.time()-t0)/60:.2f} min")
    else:
        print(f"Downloading road network from OpenStreetMap (cache miss: {road_graphml.name})...")
        t0 = time.time()
        polygon = gpd.read_file(output_path['LandslideAffectedArea'])
        polygon = polygon.to_crs(epsg=4326)
        polygon_union = polygon.geometry.unary_union

        major_road_filter = '["highway"~"secondary|tertiary|primary|motorway|trunk"]'
        _apply_osmnx_overpass_status_fix()
        try:
            road_network_data = _download_road_network_from_overpass(
                polygon_union,
                major_road_filter,
            )

            road_network_data.event_id = event_id
            road_network_data.shakemap_update_time = shakemap_update_time
            print(f"  Road network downloaded in {(time.time()-t0)/60:.2f} min")
            ox.save_graphml(road_network_data, str(road_graphml))
            print(f"  Road network saved to cache ({road_graphml.name})")
        except Exception as e:
            print(f"  Road network download failed: {_format_road_download_error(e)}")
            road_network_data = None

    return lithology_data, river_network_data, road_network_data

