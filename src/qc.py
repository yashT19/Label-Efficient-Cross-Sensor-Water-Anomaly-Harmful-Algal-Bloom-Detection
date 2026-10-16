"""Per-granule quality checks and regression values."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def validate_zarr(
    zarr_path: Path, rrs_range: Sequence[float]
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Re-open a Zarr store, log checks, and return its QC row."""
    ds = xr.open_zarr(zarr_path, consolidated=True).load()
    n_bands = ds.sizes["wavelength"]
    waves = ds["wavelength"].values
    valid = ds["valid_water"]
    rrs = ds["Rrs"]
    n_valid = int(valid.sum())
    n_pix = int(np.prod([ds.sizes["number_of_lines"], ds.sizes["pixels_per_line"]]))
    finite_invalid = int(np.isfinite(rrs.where(~valid).values).sum())
    finite_valid = rrs.values[valid.values]
    finite_valid = finite_valid[np.isfinite(finite_valid)]
    in_range = (
        float(((finite_valid >= rrs_range[0]) & (finite_valid <= rrs_range[1])).mean())
        if finite_valid.size else 0.0
    )
    LOGGER.info(
        "  %s: shape=(%d, %d, %d), wavelengths=%.1f-%.1f nm, valid_water=%d/%d, "
        "finite_invalid=%d, in_range=%.3f",
        zarr_path.name, ds.sizes["number_of_lines"], ds.sizes["pixels_per_line"],
        n_bands, float(waves.min()), float(waves.max()), n_valid, n_pix,
        finite_invalid, in_range,
    )
    row = {
        "file_name": zarr_path.name,
        "band_count": n_bands,
        "valid_water_pixels": n_valid,
        "pixel_count": n_pix,
        "finite_rrs_on_invalid": finite_invalid,
        "in_range_fraction": in_range,
    }
    return ds, row
