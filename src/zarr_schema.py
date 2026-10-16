"""Writer for the existing PACE OCI Zarr layout."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import xarray as xr

LOGGER = logging.getLogger(__name__)


def write_zarr(ds: xr.Dataset, zarr_path: Path) -> None:
    """Write one granule using the original chunking and Zarr layout."""
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.chunk({
        "number_of_lines": min(256, ds.sizes["number_of_lines"]),
        "pixels_per_line": min(256, ds.sizes["pixels_per_line"]),
        "wavelength": ds.sizes["wavelength"],
    }).to_zarr(zarr_path, mode="w", consolidated=True)
    LOGGER.info("  wrote %s", zarr_path)
