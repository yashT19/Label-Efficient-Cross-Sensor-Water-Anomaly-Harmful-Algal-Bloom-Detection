"""Offline tests for catalogue parsing, ranking, and idempotent merging."""

from pathlib import Path

from shapely.geometry import box

from src.catalogue import (
    merge_catalogue_rows,
    parse_umm_footprint,
    rank_records,
    roi_coverage_pct,
    utc_to_ist,
    write_catalogue,
)


def test_parse_polygon_and_bounding_rectangle() -> None:
    polygon = parse_umm_footprint({
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "GPolygons": [{
                        "Boundary": {"Point": [
                            {"Longitude": 0, "Latitude": 0},
                            {"Longitude": 2, "Latitude": 0},
                            {"Longitude": 2, "Latitude": 2},
                            {"Longitude": 0, "Latitude": 2},
                        ]}
                    }]
                }
            }
        }
    })
    rectangle = parse_umm_footprint({
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "BoundingRectangles": [{
                        "WestBoundingCoordinate": 0, "SouthBoundingCoordinate": 0,
                        "EastBoundingCoordinate": 1, "NorthBoundingCoordinate": 1,
                    }]
                }
            }
        }
    })
    assert polygon is not None and polygon.area == 4
    assert rectangle is not None and rectangle.area == 1


def test_missing_footprint_and_ist_conversion() -> None:
    assert parse_umm_footprint({"NoSpatialExtent": True}) is None
    assert utc_to_ist("2024-08-10T18:03:43Z") == "2024-08-10T23:33:43+05:30"


def test_coverage_and_ranking_put_missing_last() -> None:
    assert roi_coverage_pct(box(0, 0, 2, 2), [0, 0, 4, 4]) == 25.0
    ranked = rank_records([
        {"granule_id": "missing", "roi_coverage_pct": None},
        {"granule_id": "low", "roi_coverage_pct": 10},
        {"granule_id": "high", "roi_coverage_pct": 80},
    ])
    assert [row["granule_id"] for row in ranked] == ["high", "low", "missing"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_merge_preserves_backfilled_fields(tmp_path: Path) -> None:
    old = [{"site": "x", "granule_id": "g", "valid_pixels": "12",
            "valid_fraction": "0.5", "dominant_flags": "LAND"}]
    new = [{"site": "x", "granule_id": "g", "valid_pixels": "",
            "valid_fraction": "", "dominant_flags": ""}]
    merged = merge_catalogue_rows(old, new)
    assert merged[0]["valid_pixels"] == "12"
    path = tmp_path / "catalogue.csv"
    write_catalogue(path, merged)
    assert path.read_text(encoding="utf-8").count("12") == 1
