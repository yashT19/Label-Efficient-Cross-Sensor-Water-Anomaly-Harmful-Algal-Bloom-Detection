"""PACE OCI quality-flag masking and ROI cropping."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

L2_FLAG_BITS: Mapping[str, int] = {
    "ATMFAIL": 0, "LAND": 1, "PRODWARN": 2, "HIGLINT": 3, "HILT": 4,
    "HISATZEN": 5, "COASTZ": 6, "STRAYLIGHT": 8, "CLDICE": 9,
    "COCCOLITH": 10, "TURBIDW": 11, "HISOLZEN": 12, "LOWLW": 14,
    "CHLFAIL": 15, "NAVWARN": 16, "ABSAER": 17, "MAXAERITER": 19,
    "MODGLINT": 20, "CHLWARN": 21, "ATMWARN": 22, "SEAICE": 24,
    "NAVFAIL": 25, "FILTER": 26, "BOWTIEDEL": 28, "HIPOL": 29,
    "PRODFAIL": 30,
}


def invalid_flag_mask(flag_names: Sequence[str]) -> int:
    """Build the integer mask from configured flag names."""
    unknown = [name for name in flag_names if name not in L2_FLAG_BITS]
    if unknown:
        raise ValueError(f"Unknown PACE L2 flag name(s): {', '.join(unknown)}")
    return sum(1 << L2_FLAG_BITS[name] for name in flag_names)


def invalid_from_flags(
    l2_flags: xr.DataArray, flag_names: Sequence[str]
) -> xr.DataArray:
    """Return true where any configured quality flag bit is set."""
    flags = l2_flags.astype("int64")
    return (flags & invalid_flag_mask(flag_names)) != 0


def crop_to_roi(ds: xr.Dataset, bbox: Sequence[float]) -> xr.Dataset:
    """Keep an index window covering the geographic ROI on the swath."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat = ds["latitude"]
    lon = ds["longitude"]
    in_roi = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if int(in_roi.sum()) == 0:
        raise ValueError("Granule does not intersect the configured ROI after load.")
    lines, pixels = np.where(in_roi.values)
    cropped = ds.isel(
        number_of_lines=slice(int(lines.min()), int(lines.max()) + 1),
        pixels_per_line=slice(int(pixels.min()), int(pixels.max()) + 1),
    )
    lat_c = cropped["latitude"]
    lon_c = cropped["longitude"]
    still_in = (
        (lat_c >= lat_min) & (lat_c <= lat_max)
        & (lon_c >= lon_min) & (lon_c <= lon_max)
    )
    cropped["valid_water"] = cropped["valid_water"] & still_in
    cropped["Rrs"] = cropped["Rrs"].where(cropped["valid_water"])
    return cropped


def load_masked_granule(
    nc_path: Path,
    bbox: Sequence[float],
    dates: Sequence[str],
    short_name: str,
    flag_names: Sequence[str],
) -> xr.Dataset:
    """Load, flag-mask, and crop one PACE OCI granule."""
    from .pace_io import open_pace_groups

    root, geo, nav, bands = open_pace_groups(nc_path)
    try:
        if "Rrs" not in geo:
            raise KeyError(f"{nc_path.name} has no geophysical_data/Rrs")
        rrs = geo["Rrs"]
        if "wavelength_3d" in rrs.dims and "wavelength" not in rrs.dims:
            rrs = rrs.rename({"wavelength_3d": "wavelength"})
        if "wavelength" not in rrs.dims:
            raise KeyError(
                f"{nc_path.name} Rrs dims={rrs.dims}; expected "
                "'wavelength' or 'wavelength_3d'"
            )
        if "wavelength" not in rrs.coords:
            if "wavelength_3d" not in bands:
                raise KeyError(f"{nc_path.name} has no wavelength coordinate to attach")
            rrs = rrs.assign_coords(
                wavelength=("wavelength", bands["wavelength_3d"].astype("float32").values)
            )
        if (
            "wavelength_3d" in bands
            and bands["wavelength_3d"].sizes.get("wavelength_3d", 0)
            == rrs.sizes["wavelength"]
        ):
            rrs = rrs.assign_coords(
                wavelength_3d_nm=("wavelength", bands["wavelength_3d"].values)
            )

        flags = geo["l2_flags"]
        valid_water = ~invalid_from_flags(flags, flag_names)
        rrs_masked = rrs.where(valid_water)
        ds = xr.Dataset(
            data_vars={"Rrs": rrs_masked, "l2_flags": flags, "valid_water": valid_water},
            coords={
                "latitude": nav["latitude"],
                "longitude": nav["longitude"],
                "wavelength": rrs_masked["wavelength"],
            },
            attrs={
                "title": "PACE OCI L2 AOP Rrs, Module-1 validation extract",
                "source_file": nc_path.name,
                "short_name": short_name,
                "roi_bbox_west_south_east_north": ",".join(str(v) for v in bbox),
                "temporal_window": f"{dates[0]}/{dates[1]}",
                "invalid_flag_names": ",".join(flag_names),
                "invalid_flag_mask": int(invalid_flag_mask(flag_names)),
                "flag_source": "https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/",
                "rrs_units": "sr^-1",
                "wavelength_units": "nm",
                "product_name": str(root.attrs.get("product_name", "")),
                "processing_version": str(root.attrs.get("processing_version", "")),
                "time_coverage_start": str(root.attrs.get("time_coverage_start", "")),
                "time_coverage_end": str(root.attrs.get("time_coverage_end", "")),
                "platform": str(root.attrs.get("platform", "PACE")),
                "instrument": str(root.attrs.get("instrument", "OCI")),
                "note": (
                    "Rrs is L2 remote-sensing reflectance after NASA atmospheric "
                    "correction (l2gen). Further SRF alignment to EnMAP is out of scope."
                ),
            },
        )
        ds["Rrs"].attrs.update({
            "long_name": "Remote sensing reflectance (flag-masked)",
            "units": "sr^-1",
            "standard_name": geo["Rrs"].attrs.get(
                "standard_name",
                "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_"
                "downwelling_radiative_flux_in_air",
            ),
        })
        ds["wavelength"].attrs.update({"long_name": "band center wavelength", "units": "nm"})
        ds["valid_water"].attrs["description"] = (
            "True where none of INVALID_FLAG_NAMES bits are set on l2_flags"
        )
        return crop_to_roi(ds, bbox)
    finally:
        root.close()
        geo.close()
        nav.close()
        bands.close()
