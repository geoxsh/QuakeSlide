"""Utilities for selecting a consistent USGS ShakeMap product.

Reference sources:
- USGS groundfailure default ShakeMap retrieval uses
  getShakefiles(..., source='preferred'):
  https://usgs.github.io/groundfailure/gfail.gfailrun.html
- USGS ComCat defines the preferred product by preferredWeight, with ties
  resolved by updateTime:
  https://earthquake.usgs.gov/data/comcat/

Date: August 2026
"""

from datetime import datetime, timezone
import math

import requests


# USGS ComCat event detail GeoJSON endpoint.
# Reference: https://earthquake.usgs.gov/fdsnws/event/1/query
USGS_EVENT_DETAIL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_event_geojson(event_id, include_superseded=False, timeout=60):
    """Fetch the USGS event detail GeoJSON for one event."""
    params = {"eventid": event_id, "format": "geojson"}
    if include_superseded:
        params["includesuperseded"] = "true"

    response = requests.get(USGS_EVENT_DETAIL_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_event_metadata(event_data, event_id=None):
    """Extract the current preferred event solution from USGS event GeoJSON.

    Event origin metadata belongs to the top-level GeoJSON feature. ShakeMap
    product properties can describe a different origin and must not be used as
    the authoritative event time, hypocenter, or magnitude.
    """
    properties = event_data.get("properties", {})
    coordinates = (event_data.get("geometry") or {}).get("coordinates") or []
    resolved_event_id = event_data.get("id") or event_id or "Unknown"

    event_time_ms = _finite_number(properties.get("time"), "origin time", resolved_event_id)
    event_time_ms = int(event_time_ms)
    event_time_utc = datetime.fromtimestamp(
        event_time_ms / 1000, tz=timezone.utc
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if len(coordinates) < 3:
        raise ValueError(
            f"USGS event {resolved_event_id!r} has no valid hypocenter coordinates."
        )

    longitude = _finite_number(coordinates[0], "longitude", resolved_event_id)
    latitude = _finite_number(coordinates[1], "latitude", resolved_event_id)
    depth_km = _finite_number(coordinates[2], "depth", resolved_event_id)

    magnitude = properties.get("mag")
    if magnitude is not None:
        magnitude = _finite_number(magnitude, "magnitude", resolved_event_id)

    updated_ms = properties.get("updated")
    try:
        updated_ms = int(updated_ms) if updated_ms is not None else None
    except (TypeError, ValueError):
        updated_ms = None

    return {
        "event_id": resolved_event_id,
        "place": properties.get("place") or "Unknown",
        "magnitude": magnitude,
        "time_ms": event_time_ms,
        "time_utc": event_time_utc,
        "longitude": longitude,
        "latitude": latitude,
        "depth_km": depth_km,
        "status": properties.get("status"),
        "updated_ms": updated_ms,
    }


def _finite_number(value, field_name, event_id):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"USGS event {event_id!r} has no valid {field_name}."
        ) from None
    if not math.isfinite(number):
        raise ValueError(
            f"USGS event {event_id!r} has no valid {field_name}."
        )
    return number


def _preferred_weight(product):
    # ComCat product preference field.
    # Reference: https://earthquake.usgs.gov/data/comcat/  ("preferredWeight")
    value = product.get("preferredWeight", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _update_time(product):
    # ComCat uses updateTime to resolve ties between equally preferred products.
    # Reference: https://earthquake.usgs.gov/data/comcat/  ("updateTime")
    value = product.get("updateTime", product.get("update_time", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _has_required_content(product, required_content):
    if required_content is None:
        return True
    return required_content in product.get("contents", {})


def select_shakemap_product(event_data, required_content=None, source="preferred"):
    """
    Select the preferred ShakeMap product from a USGS event detail GeoJSON.

    This follows the same default source logic used by USGS groundfailure:
    ``getShakefiles(..., source='preferred')``.

    References:
    - USGS groundfailure:
      https://usgs.github.io/groundfailure/gfail.gfailrun.html
    - USGS ComCat product preference:
      https://earthquake.usgs.gov/data/comcat/

    In libcomcat, the preferred source is the source of the product with the
    largest ComCat ``preferredWeight``; ties are resolved by the most recent
    ``updateTime``. The preferred product is then selected within that source
    using the same ``preferredWeight`` / ``updateTime`` ordering.
    """
    products = (
        event_data.get("properties", {})
        .get("products", {})
        .get("shakemap", [])
    )
    candidates = [
        product for product in products
        if _has_required_content(product, required_content)
    ]
    if not candidates:
        detail = f" with {required_content}" if required_content else ""
        raise RuntimeError(f"No ShakeMap product{detail} is available for this event.")

    if source == "preferred":
        preferred_product = max(
            candidates,
            key=lambda product: (_preferred_weight(product), _update_time(product)),
        )
        preferred_source = preferred_product.get("source")
        candidates = [
            product for product in candidates
            if product.get("source") == preferred_source
        ]
    elif source != "all":
        candidates = [
            product for product in candidates
            if product.get("source") == source
        ]
        if not candidates:
            detail = f" with {required_content}" if required_content else ""
            raise RuntimeError(
                f"No ShakeMap product from source '{source}'{detail} is available."
            )

    return max(
        candidates,
        key=lambda product: (_preferred_weight(product), _update_time(product)),
    )


def shakemap_update_time(product):
    """Return the selected product update time as an integer timestamp."""
    return _update_time(product)


def format_shakemap_version(product):
    """Format a ShakeMap product description for the report table.

    ComCat ``properties.version`` is the product revision. Atlas products are
    identified by their ShakeMap code generation (for example, ``Atlas v4``).
    Other products use ComCat's ``event-description`` when available (for
    example, ``USGS NEIC ShakeMap``), so the description and product revision
    remain distinct.
    """
    source = str(product.get("source", "Unknown"))
    source_lower = source.lower()
    props = product.get("properties", {})
    product_version = props.get("version", "Unknown")
    code_version = props.get("shakemap-code-version")

    if source_lower == "atlas" and code_version:
        code_major_version = str(code_version).split(".", 1)[0]
        product_name = f"ShakeMap Atlas v{code_major_version}"
    elif source_lower == "atlas":
        product_name = "ShakeMap Atlas"
    else:
        event_description = str(props.get("event-description") or "").strip()
        product_name = event_description or f"ShakeMap {source.upper()}"

    return f"{product_name} / product v{product_version}"


def describe_shakemap_product(product):
    """Return a compact human-readable description of the selected product."""
    props = product.get("properties", {})
    return (
        f"{format_shakemap_version(product)} "
        f"(code={product.get('code', 'Unknown')}, "
        f"preferredWeight={product.get('preferredWeight', 'Unknown')}, "
        f"status={props.get('map-status', 'Unknown')}, "
        f"updateTime={_update_time(product)})"
    )
