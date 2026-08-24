"""Composite-earthquake metadata and stable run naming.

Date: August 2026
"""

import re

from src.shakemap_product import extract_event_metadata, fetch_event_geojson


_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_composite_event_ids(event_ids):
    """Validate and de-duplicate component event IDs while preserving input order."""
    if isinstance(event_ids, (str, bytes)) or event_ids is None:
        raise ValueError("composite_event_ids must be a list or tuple of USGS event IDs.")

    normalized = []
    for raw_event_id in event_ids:
        event_id = str(raw_event_id).strip()
        if not event_id or not _EVENT_ID_PATTERN.fullmatch(event_id):
            raise ValueError(f"Invalid USGS event ID in composite_event_ids: {raw_event_id!r}")
        if event_id not in normalized:
            normalized.append(event_id)

    if len(normalized) < 2:
        raise ValueError("At least two distinct event IDs are required for a composite ShakeMap.")
    return normalized


def _event_record(event_id, event_data):
    metadata = extract_event_metadata(event_data, event_id=event_id)
    return {
        "event_id": event_id,
        "magnitude": metadata["magnitude"],
        "time_ms": metadata["time_ms"],
        "time_utc": metadata["time_utc"],
        "place": metadata["place"],
        "longitude": metadata["longitude"],
        "latitude": metadata["latitude"],
        "depth_km": metadata["depth_km"],
    }


def prepare_composite_events(event_ids, fetcher=fetch_event_geojson):
    """Fetch component metadata and select the largest-magnitude mainshock."""
    normalized_ids = normalize_composite_event_ids(event_ids)
    event_data_by_id = {}
    records = []
    for event_id in normalized_ids:
        event_data = fetcher(event_id, include_superseded=False, timeout=30)
        event_data_by_id[event_id] = event_data
        records.append(_event_record(event_id, event_data))

    candidates = [record for record in records if record["magnitude"] is not None]
    if not candidates:
        raise ValueError(
            "USGS returned no valid magnitude for any composite event; "
            "the mainshock cannot be selected automatically."
        )

    # Largest magnitude wins; an earlier origin time resolves an exact tie.
    mainshock = max(
        candidates,
        key=lambda record: (record["magnitude"], -record["time_ms"]),
    )
    mainshock_id = mainshock["event_id"]
    ordered_ids = [mainshock_id] + sorted(
        event_id for event_id in normalized_ids if event_id != mainshock_id
    )
    record_by_id = {record["event_id"]: record for record in records}
    ordered_records = [record_by_id[event_id] for event_id in ordered_ids]
    for record in ordered_records:
        record["role"] = "mainshock" if record["event_id"] == mainshock_id else "additional_event"

    return {
        "mainshock_event_id": mainshock_id,
        "component_event_ids": ordered_ids,
        "events": ordered_records,
        "event_data_by_id": event_data_by_id,
        "run_id": build_composite_run_id(mainshock_id, ordered_ids),
    }


def build_composite_run_id(mainshock_event_id, component_event_ids):
    """Return a stable, readable composite output identifier."""
    other_ids = sorted(
        event_id for event_id in component_event_ids if event_id != mainshock_event_id
    )
    return "composite_" + "_".join([mainshock_event_id, *other_ids])
