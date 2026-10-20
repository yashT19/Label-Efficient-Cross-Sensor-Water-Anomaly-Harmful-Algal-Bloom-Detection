"""Metadata-only PACE granule catalogue and offline-testable geometry helpers."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

import earthaccess

LOGGER = logging.getLogger(__name__)

CATALOGUE_COLUMNS = [
    "site", "water_type", "roi_bbox", "date_window", "granule_id", "version",
    "start_utc", "end_utc", "start_ist", "end_ist", "footprint_wkt",
    "roi_coverage_pct", "size_mb", "valid_pixels", "valid_fraction",
    "dominant_flags", "evidence_level", "enmap_scene_id", "time_gap_h", "notes",
    "rank",
]


def _nested_values(value: Any, key: str) -> Iterable[Any]:
    """Yield values for a key from nested UMM-like mappings."""
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name == key:
                yield child
            yield from _nested_values(child, key)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _nested_values(child, key)


def _as_mapping(granule: Any) -> Mapping[str, Any]:
    if isinstance(granule, Mapping):
        return granule
    for attr in ("umm_json", "_umm_metadata", "umm"):
        value = getattr(granule, attr, None)
        if isinstance(value, Mapping):
            return value
    getter = getattr(granule, "get", None)
    if callable(getter):
        value = getter("umm")
        if isinstance(value, Mapping):
            return value
    return {}


def _polygon_from_points(points: Any) -> Polygon | None:
    if not isinstance(points, Sequence) or len(points) < 3:
        return None
    coords: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            lon, lat = point.get("Longitude"), point.get("Latitude")
        elif isinstance(point, Sequence) and len(point) >= 2:
            lon, lat = point[0], point[1]
        else:
            return None
        try:
            coords.append((float(lon), float(lat)))
        except (TypeError, ValueError):
            return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    polygon = Polygon(coords)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def parse_umm_footprint(umm: Mapping[str, Any]) -> Polygon | MultiPolygon | None:
    """Parse UMM polygons or bounding rectangles, returning a Shapely geometry."""
    polygons: list[Polygon] = []
    for group in _nested_values(umm, "GPolygons"):
        groups = group if isinstance(group, Sequence) else [group]
        for item in groups:
            if not isinstance(item, Mapping):
                continue
            polygon = _polygon_from_points(item.get("Boundary", {}).get("Point", []))
            if polygon is not None:
                polygons.append(polygon)
    for rectangle in _nested_values(umm, "BoundingRectangles"):
        rectangles = rectangle if isinstance(rectangle, Sequence) else [rectangle]
        for item in rectangles:
            if not isinstance(item, Mapping):
                continue
            try:
                polygons.append(box(
                    float(item["WestBoundingCoordinate"]),
                    float(item["SouthBoundingCoordinate"]),
                    float(item["EastBoundingCoordinate"]),
                    float(item["NorthBoundingCoordinate"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    if not polygons:
        return None
    return unary_union(polygons)


def utc_to_ist(value: str | datetime, offset_hours: float = 5.5) -> str:
    """Convert an ISO UTC timestamp to an ISO timestamp with the IST offset."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    converted = parsed.astimezone(timezone(timedelta(hours=offset_hours)))
    return converted.isoformat()


def roi_coverage_pct(footprint: Polygon | MultiPolygon | None, bbox: Sequence[float]) -> float | None:
    """Return planar-degree ROI coverage; this approximation is documented in the catalogue."""
    if footprint is None:
        return None
    roi = box(*map(float, bbox))
    if roi.area == 0:
        return None
    return float(footprint.intersection(roi).area / roi.area * 100.0)


def rank_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank records by coverage descending, with missing coverage last."""
    ranked = sorted(records, key=lambda row: (
        row.get("roi_coverage_pct") is None,
        -(float(row["roi_coverage_pct"]) if row.get("roi_coverage_pct") is not None else 0.0),
    ))
    return [dict(row, rank=index) for index, row in enumerate(ranked, 1)]


def merge_catalogue_rows(
    old_rows: Iterable[Mapping[str, Any]], new_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Preserve processing backfills while replacing metadata from a fresh search."""
    preserved = ("valid_pixels", "valid_fraction", "dominant_flags", "enmap_scene_id", "time_gap_h")
    old_by_key = {(row.get("site"), row.get("granule_id")): row for row in old_rows}
    merged: list[dict[str, Any]] = []
    for current in new_rows:
        row = dict(current)
        previous = old_by_key.get((row.get("site"), row.get("granule_id")), {})
        for field in preserved:
            if not row.get(field) and previous.get(field) not in (None, ""):
                row[field] = previous[field]
        merged.append(row)
    return merged


def _granule_id(granule: Any) -> str:
    try:
        return str(granule.data_links()[0]).rsplit("/", 1)[-1]
    except (AttributeError, IndexError, TypeError):
        return str(getattr(granule, "concept_id", granule))


def _metadata_time(umm: Mapping[str, Any], key: str) -> str:
    for temporal in _nested_values(umm, "RangeDateTime"):
        if isinstance(temporal, Mapping) and temporal.get(key):
            return str(temporal[key])
    return ""


def _granule_size_mb(granule: Any) -> str:
    size = getattr(granule, "size", None)
    try:
        value = size() if callable(size) else size
        return f"{float(value) / 1e6:.3f}" if value is not None else ""
    except (TypeError, ValueError):
        return ""


def search_site(site_name: str, site: Mapping[str, Any], pace: Mapping[str, Any],
                offset_hours: float) -> list[dict[str, Any]]:
    """Search one site and build metadata rows without downloading data."""
    results = earthaccess.search_data(
        short_name=pace["short_name"], version=str(pace["version"]),
        bounding_box=tuple(site["bbox"]), temporal=tuple(site["dates"]), count=25,
    )
    rows: list[dict[str, Any]] = []
    for granule in results:
        umm = _as_mapping(granule)
        start = _metadata_time(umm, "BeginningDateTime")
        end = _metadata_time(umm, "EndingDateTime")
        footprint = parse_umm_footprint(umm)
        if footprint is None:
            LOGGER.warning("Missing or invalid footprint for %s", _granule_id(granule))
        rows.append({
            "site": site_name, "water_type": site["water_type"],
            "roi_bbox": ",".join(str(value) for value in site["bbox"]),
            "date_window": f"{site['dates'][0]}/{site['dates'][1]}",
            "granule_id": _granule_id(granule), "version": str(pace["version"]),
            "start_utc": start, "end_utc": end,
            "start_ist": utc_to_ist(start, offset_hours) if start else "",
            "end_ist": utc_to_ist(end, offset_hours) if end else "",
            "footprint_wkt": footprint.wkt if footprint is not None else "",
            "roi_coverage_pct": roi_coverage_pct(footprint, site["bbox"]),
            "size_mb": _granule_size_mb(granule), "valid_pixels": "",
            "valid_fraction": "", "dominant_flags": "",
            "evidence_level": site["evidence_level"], "enmap_scene_id": "",
            "time_gap_h": "", "notes": site.get("notes", ""),
        })
    if not rows:
        LOGGER.warning("finding: 0 granules returned for site %s", site_name)
    return rank_records(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_catalogue(path: Path) -> list[dict[str, str]]:
    """Read catalogue rows for processing."""
    return _read_rows(path)


def write_catalogue(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Merge and write the catalogue deterministically."""
    previous = _read_rows(path)
    merged = merge_catalogue_rows(previous, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOGUE_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in CATALOGUE_COLUMNS} for row in merged)


def backfill_catalogue(path: Path, qc_rows: Sequence[Mapping[str, Any]]) -> None:
    """Back-fill processing fields in existing catalogue rows."""
    rows = _read_rows(path)
    by_key = {(row.get("site"), row.get("granule_id")): row for row in qc_rows}
    for row in rows:
        stats = by_key.get((row.get("site"), row.get("granule_id")))
        if stats:
            for field in ("valid_pixels", "valid_fraction", "dominant_flags"):
                row[field] = str(stats.get(field, ""))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOGUE_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in CATALOGUE_COLUMNS} for row in rows)


def build_catalogue(config: Mapping[str, Any], site_name: str | None = None) -> list[dict[str, Any]]:
    """Search configured sites and write the metadata-only catalogue."""
    sites = config["sites"]
    selected = [site_name] if site_name else list(sites)
    rows: list[dict[str, Any]] = []
    for name in selected:
        rows.extend(search_site(name, sites[name], config["pace"], config["catalogue"]["ist_offset_hours"]))
    path = Path(config["catalogue"]["out_csv"])
    write_catalogue(path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the metadata-only PACE catalogue.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--site")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import yaml
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    build_catalogue(config, args.site)


if __name__ == "__main__":
    main()
