"""Per-granule quality checks and regression values."""

from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)
REVIEW_FLAGS = (
    "COASTZ", "TURBIDW", "PRODWARN", "MODGLINT", "ABSAER", "SEAICE",
    "HIPOL", "FILTER", "BOWTIEDEL",
)


def summarize_granule(
    ds: xr.Dataset,
    site: str,
    granule_id: str,
    invalid_flags: Sequence[str],
    rrs_range: Sequence[float],
) -> dict[str, Any]:
    """Calculate catalogue/QC statistics without changing the mask."""
    from .masking import L2_FLAG_BITS

    flags = ds["l2_flags"].astype("int64")
    n_pixels = int(flags.size)
    valid_pixels = int(ds["valid_water"].sum())
    values = ds["Rrs"].values
    finite = np.isfinite(values)
    out_of_range = finite & ((values < rrs_range[0]) | (values > rrs_range[1]))
    row: dict[str, Any] = {
        "site": site,
        "granule_id": granule_id,
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_pixels / n_pixels if n_pixels else 0.0,
        "out_of_range_rrs_fraction": int(out_of_range.sum()) / int(finite.sum())
        if int(finite.sum()) else 0.0,
    }
    removal_masks: dict[str, xr.DataArray] = {}
    for name in [*invalid_flags, *REVIEW_FLAGS]:
        if name not in L2_FLAG_BITS:
            raise ValueError(f"Unknown PACE L2 flag name: {name}")
        mask = (flags & (1 << L2_FLAG_BITS[name])) != 0
        removal_masks[name] = mask
        prefix = "removed" if name in invalid_flags else "would_remove"
        row[f"{prefix}_{name}"] = int(mask.sum())
    invalid_union = xr.zeros_like(flags, dtype=bool)
    for name in invalid_flags:
        invalid_union = invalid_union | removal_masks[name]
    row["removed_total_union"] = int(invalid_union.sum())
    ranked = sorted(
        ((name, int(removal_masks[name].sum())) for name in invalid_flags),
        key=lambda item: (-item[1], item[0]),
    )
    row["dominant_flags"] = ",".join(name for name, count in ranked[:3] if count)
    if valid_pixels == 0:
        LOGGER.warning("finding: no valid water pixels at %s / %s", site, granule_id)
    return row


def write_qc_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the additive per-granule QC summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "site", "granule_id", "valid_pixels", "valid_fraction",
        "out_of_range_rrs_fraction",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
