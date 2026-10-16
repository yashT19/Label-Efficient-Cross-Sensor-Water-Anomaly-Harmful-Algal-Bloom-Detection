"""Regression checks for the Lake Erie PACE baseline."""

from pathlib import Path

import pytest

from src.masking import load_masked_granule

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pace_oci_l2_aop"
BBOX = [-83.70, 41.20, -82.30, 42.10]
DATES = ["2024-08-10", "2024-08-19"]
FLAGS = [
    "ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "STRAYLIGHT", "CLDICE",
    "COCCOLITH", "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "MAXAERITER",
    "CHLWARN", "ATMWARN", "NAVFAIL", "PRODFAIL",
]


@pytest.mark.parametrize(
    ("granule_id", "expected_valid"),
    [("20240810T180343", 29), ("20240811T165956", 0), ("20240811T183815", 246)],
)
def test_lake_erie_baseline(granule_id: str, expected_valid: int) -> None:
    """The refactored mask preserves the old band and valid-pixel counts."""
    nc_path = next(RAW_DIR.glob(f"*{granule_id}*.nc"))
    ds = load_masked_granule(nc_path, BBOX, DATES, "PACE_OCI_L2_AOP", FLAGS)
    try:
        assert ds.sizes["wavelength"] == 172
        assert int(ds["valid_water"].sum()) == expected_valid
    finally:
        ds.close()
