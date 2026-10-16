"""PACE OCI Earthdata and NetCDF I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import earthaccess
import xarray as xr

LOGGER = logging.getLogger(__name__)


def login() -> None:
    """Authenticate with Earthdata and persist the local token."""
    auth = earthaccess.login(persist=True)
    if not auth.authenticated:
        raise RuntimeError(
            "Earthdata login failed. Create an account at "
            "https://urs.earthdata.nasa.gov and re-run."
        )
    LOGGER.info("Earthdata login: OK")


def granule_label(granule: Any) -> str:
    """Return a useful file-like label for an Earthdata granule."""
    try:
        return granule.data_links()[0].rsplit("/", 1)[-1]
    except (AttributeError, IndexError):
        return str(granule)


def search_granules(
    short_name: str,
    version: str,
    bbox: Sequence[float],
    dates: Sequence[str],
    count: int,
) -> list[Any]:
    """Search and select the first requested PACE granules."""
    LOGGER.info("Searching %s bbox=%s temporal=%s", short_name, bbox, dates)
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        bounding_box=tuple(bbox),
        temporal=tuple(dates),
        count=25,
    )
    if not results:
        raise RuntimeError("No granules found. Widen the configured dates or bbox.")
    selected = list(results[:count])
    LOGGER.info("Search returned %d granule(s); keeping first %d", len(results), count)
    for index, granule in enumerate(selected, 1):
        LOGGER.info("  [%d] %s", index, granule_label(granule))
    return selected


def download_granules(granules: Sequence[Any], raw_dir: Path) -> list[Path]:
    """Download granules into the configured raw-data directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(granules, local_path=str(raw_dir))
    files = [Path(path) for path in paths]
    for path in files:
        LOGGER.info("  downloaded: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return files


def open_pace_groups(nc_path: Path) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open the root, geophysical, navigation, and band NetCDF groups."""
    return (
        xr.open_dataset(nc_path),
        xr.open_dataset(nc_path, group="geophysical_data", mask_and_scale=True),
        xr.open_dataset(nc_path, group="navigation_data"),
        xr.open_dataset(nc_path, group="sensor_band_parameters"),
    )
