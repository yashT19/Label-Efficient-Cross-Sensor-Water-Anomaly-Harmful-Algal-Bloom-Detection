"""
Module 1 (first pass) — Acquire, flag-mask, and write PACE OCI L2 AOP Rrs to Zarr.

Scope
-----
PACE/OCI only. Small-scale end-to-end check before scaling up or adding HysIS.

Run
---
    python module1_pace_oci_validation.py
"""

from __future__ import annotations

from pathlib import Path

import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROI_BBOX = (-83.70, 41.20, -82.30, 42.10)
TEMPORAL = ("2024-08-10", "2024-08-19")
SHORT_NAME = "PACE_OCI_L2_AOP"  
N_GRANULES = 3

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "pace_oci_l2_aop"
ZARR_DIR = ROOT / "data" / "processed" / "pace_oci_l2_aop_zarr"
FIG_DIR = ROOT / "figures"
RNG_SEED = 42
N_SPECTRA = 20


L2_FLAG_BITS = {
    "ATMFAIL": 0,
    "LAND": 1,
    "PRODWARN": 2,
    "HIGLINT": 3,
    "HILT": 4,
    "HISATZEN": 5,
    "COASTZ": 6,
    "STRAYLIGHT": 8,
    "CLDICE": 9,
    "COCCOLITH": 10,
    "TURBIDW": 11,
    "HISOLZEN": 12,
    "LOWLW": 14,
    "CHLFAIL": 15,
    "NAVWARN": 16,
    "ABSAER": 17,
    "MAXAERITER": 19,
    "MODGLINT": 20,
    "CHLWARN": 21,
    "ATMWARN": 22,
    "SEAICE": 24,
    "NAVFAIL": 25,
    "FILTER": 26,
    "BOWTIEDEL": 28,
    "HIPOL": 29,
    "PRODFAIL": 30,
}

INVALID_FLAG_NAMES = (
    "ATMFAIL",
    "LAND",
    "HIGLINT",
    "HILT",
    "HISATZEN",
    "STRAYLIGHT",
    "CLDICE",
    "COCCOLITH",
    "HISOLZEN",
    "LOWLW",
    "CHLFAIL",
    "NAVWARN",
    "MAXAERITER",
    "CHLWARN",
    "ATMWARN",
    "NAVFAIL",
    "PRODFAIL",
)
INVALID_FLAG_MASK = sum(1 << L2_FLAG_BITS[name] for name in INVALID_FLAG_NAMES)


RRS_PHYSICAL_MIN = -0.01
RRS_PHYSICAL_MAX = 0.10


REF_WAVELENGTH_NM = 443.0


def login() -> None:
    """Authenticate to NASA Earthdata; persist token/netrc on this machine only."""
    auth = earthaccess.login(persist=True)
    if not auth.authenticated:
        raise SystemExit(
            "Earthdata login failed. Create an account at "
            "https://urs.earthdata.nasa.gov and re-run this script."
        )
    print("Earthdata login: OK")


def search_granules():
    print(f"Searching {SHORT_NAME}  bbox={ROI_BBOX}  temporal={TEMPORAL}")
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        bounding_box=ROI_BBOX,
        temporal=TEMPORAL,
        count=25,
    )
    if not results:
        raise SystemExit(
            "No granules found. Widen TEMPORAL or ROI_BBOX and retry."
        )
    print(f"Search returned {len(results)} granule(s); keeping first {N_GRANULES}.")
    selected = results[:N_GRANULES]
    for i, g in enumerate(selected, 1):
        print(f"  [{i}] {granule_label(g)}")
    return selected


def granule_label(granule) -> str:
    try:
        return granule.data_links()[0].rsplit("/", 1)[-1]
    except Exception:
        return str(granule)


def download_granules(granules) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(granules, local_path=str(RAW_DIR))
    files = [Path(p) for p in paths]
    for p in files:
        print(f"  downloaded: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    return files


def invalid_from_flags(l2_flags: xr.DataArray) -> xr.DataArray:
    """True where any selected quality/land/cloud bit is set."""
    flags = l2_flags.astype("int64")
    return (flags & INVALID_FLAG_MASK) != 0


def crop_to_roi(ds: xr.Dataset) -> xr.Dataset:
    """Keep a rectangular index window covering the ROI (swath is not a grid)."""
    lat = ds["latitude"]
    lon = ds["longitude"]
    lon_min, lat_min, lon_max, lat_max = ROI_BBOX
    in_roi = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )
    if int(in_roi.sum()) == 0:
        raise ValueError("Granule does not intersect ROI_BBOX after load.")
    lines, pixels = np.where(in_roi.values)
    ds_c = ds.isel(
        number_of_lines=slice(int(lines.min()), int(lines.max()) + 1),
        pixels_per_line=slice(int(pixels.min()), int(pixels.max()) + 1),
    )
    
    lat_c = ds_c["latitude"]
    lon_c = ds_c["longitude"]
    still_in = (
        (lat_c >= lat_min)
        & (lat_c <= lat_max)
        & (lon_c >= lon_min)
        & (lon_c <= lon_max)
    )
    ds_c["valid_water"] = ds_c["valid_water"] & still_in
    ds_c["Rrs"] = ds_c["Rrs"].where(ds_c["valid_water"])
    return ds_c


def load_masked_granule(nc_path: Path) -> xr.Dataset:
    """Load Rrs + flags + wavelengths, mask invalid pixels, crop to ROI."""
    root = xr.open_dataset(nc_path)
    geo = xr.open_dataset(nc_path, group="geophysical_data", mask_and_scale=True)
    nav = xr.open_dataset(nc_path, group="navigation_data")
    bands = xr.open_dataset(nc_path, group="sensor_band_parameters")

    if "Rrs" not in geo:
        raise KeyError(f"{nc_path.name} has no geophysical_data/Rrs")

    rrs = geo["Rrs"]
   
    if "wavelength_3d" in rrs.dims and "wavelength" not in rrs.dims:
        rrs = rrs.rename({"wavelength_3d": "wavelength"})
    if "wavelength" not in rrs.dims:
        raise KeyError(
            f"{nc_path.name} Rrs dims={rrs.dims}; expected 'wavelength' or 'wavelength_3d'"
        )
    if "wavelength" not in rrs.coords:
        if "wavelength_3d" not in bands:
            raise KeyError(f"{nc_path.name} has no wavelength coordinate to attach")
        rrs = rrs.assign_coords(
            wavelength=("wavelength", bands["wavelength_3d"].astype("float32").values)
        )
    if "wavelength_3d" in bands and bands["wavelength_3d"].sizes.get(
        "wavelength_3d", 0
    ) == rrs.sizes["wavelength"]:
        rrs = rrs.assign_coords(
            wavelength_3d_nm=("wavelength", bands["wavelength_3d"].values)
        )

    flags = geo["l2_flags"]
    invalid = invalid_from_flags(flags)
    valid_water = ~invalid

    rrs_masked = rrs.where(valid_water)

    ds = xr.Dataset(
        data_vars={
            "Rrs": rrs_masked,
            "l2_flags": flags,
            "valid_water": valid_water,
        },
        coords={
            "latitude": nav["latitude"],
            "longitude": nav["longitude"],
            "wavelength": rrs_masked["wavelength"],
        },
        attrs={
            "title": "PACE OCI L2 AOP Rrs, Module-1 validation extract",
            "source_file": nc_path.name,
            "short_name": SHORT_NAME,
            "roi_bbox_west_south_east_north": ",".join(str(v) for v in ROI_BBOX),
            "temporal_window": f"{TEMPORAL[0]}/{TEMPORAL[1]}",
            "invalid_flag_names": ",".join(INVALID_FLAG_NAMES),
            "invalid_flag_mask": int(INVALID_FLAG_MASK),
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
                "correction (l2gen). Further SRF alignment to HysIS is out of scope."
            ),
        },
    )
    ds["Rrs"].attrs.update(
        {
            "long_name": "Remote sensing reflectance (flag-masked)",
            "units": "sr^-1",
            "standard_name": geo["Rrs"].attrs.get(
                "standard_name",
                "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
            ),
        }
    )
    ds["wavelength"].attrs.update({"long_name": "band center wavelength", "units": "nm"})
    ds["valid_water"].attrs["description"] = (
        "True where none of INVALID_FLAG_NAMES bits are set on l2_flags"
    )

    root.close()
    geo.close()
    nav.close()
    bands.close()

    return crop_to_roi(ds)


def write_zarr(ds: xr.Dataset, zarr_path: Path) -> None:
    if zarr_path.exists():
        # zarr stores are directories; replace the previous validation write.
        import shutil

        shutil.rmtree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.chunk(
        {
            "number_of_lines": min(256, ds.sizes["number_of_lines"]),
            "pixels_per_line": min(256, ds.sizes["pixels_per_line"]),
            "wavelength": ds.sizes["wavelength"],
        }
    ).to_zarr(zarr_path, mode="w", consolidated=True)
    print(f"  wrote {zarr_path}")


def validate_zarr(zarr_path: Path) -> xr.Dataset:
    ds = xr.open_zarr(zarr_path, consolidated=True).load()
    n_bands = ds.sizes["wavelength"]
    waves = ds["wavelength"].values
    rrs = ds["Rrs"]
    valid = ds["valid_water"]
    n_valid = int(valid.sum())
    n_pix = int(np.prod([ds.sizes["number_of_lines"], ds.sizes["pixels_per_line"]]))

    # Where valid_water is False, Rrs should be NaN (we used .where).
    n_finite_on_invalid = int(np.isfinite(rrs.where(~valid).values).sum())

    valid_vals = rrs.values[valid.values]
    finite_valid = valid_vals[np.isfinite(valid_vals)]

    print(f"  re-read {zarr_path.name}")
    print(f"    shape (lines, pixels, bands) = "
          f"({ds.sizes['number_of_lines']}, {ds.sizes['pixels_per_line']}, {n_bands})")
    print(f"    band count: {n_bands}  (expect 172 for OCI AOP Rrs)")
    print(f"    wavelength range: {float(waves.min()):.1f}-{float(waves.max()):.1f} nm")
    print(f"    valid_water pixels: {n_valid} / {n_pix} "
          f"({100.0 * n_valid / max(n_pix, 1):.2f}%)")
    print(f"    finite Rrs values on INVALID pixels (expect 0): {n_finite_on_invalid}")
    if finite_valid.size:
        print(
            "    Rrs on valid pixels: "
            f"min={finite_valid.min():.5f}  p1={np.nanpercentile(finite_valid, 1):.5f}  "
            f"median={np.nanmedian(finite_valid):.5f}  "
            f"p99={np.nanpercentile(finite_valid, 99):.5f}  "
            f"max={finite_valid.max():.5f}  sr^-1"
        )
        in_range = (
            (finite_valid >= RRS_PHYSICAL_MIN) & (finite_valid <= RRS_PHYSICAL_MAX)
        ).mean()
        print(
            f"    fraction of finite valid Rrs in "
            f"[{RRS_PHYSICAL_MIN}, {RRS_PHYSICAL_MAX}] sr^-1: {in_range:.3f}"
        )
    else:
        print("    WARNING: no finite Rrs on valid_water pixels (cloud/land-heavy granule?)")

    checks = {
        "band_count_172": bool(n_bands == 172),
        "wavelength_covers_chl_pc": bool(waves.min() < 400 and waves.max() > 680),
        "no_finite_rrs_on_invalid": bool(n_finite_on_invalid == 0),
        "has_any_valid_water": bool(n_valid > 0),
    }
    for k, ok in checks.items():
        print(f"    check {k}: {'PASS' if ok else 'FAIL'}")
    return ds


def plot_sample_spectra(ds: xr.Dataset, fig_path: Path) -> None:
    """Plot ~20 random valid-water spectra plus their mean."""
    wave = ds["wavelength"].values
    rrs = ds["Rrs"]
    # Pixel is usable if it is valid_water AND has a finite reference-band Rrs.
    i_ref = int(np.argmin(np.abs(wave - REF_WAVELENGTH_NM)))
    ref = rrs.isel(wavelength=i_ref).values
    usable = ds["valid_water"].values & np.isfinite(ref)
    ys, xs = np.where(usable)
    n = ys.size
    if n == 0:
        print("  skip plot: no usable water pixels")
        return

    rng = np.random.default_rng(RNG_SEED)
    k = min(N_SPECTRA, n)
    pick = rng.choice(n, size=k, replace=False)
    spectra = np.stack(
        [rrs.values[ys[i], xs[i], :] for i in pick],
        axis=0,
    )
    mean_spec = np.nanmean(spectra, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for spec in spectra:
        ax.plot(wave, spec, color="0.75", lw=0.8, alpha=0.85)
    ax.plot(wave, mean_spec, color="C0", lw=2.2, label=f"mean (n={k})")
    for pigment, color in ((443, "C2"), (615, "C3"), (670, "C1")):
        ax.axvline(pigment, color=color, ls="--", lw=0.9, alpha=0.7)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Rrs (sr$^{-1}$)")
    ax.set_title(
        "PACE OCI L2 AOP — sample valid-water spectra\n"
        f"{ds.attrs.get('source_file', '')}"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(wave.min(), wave.max())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}  (sampled {k} of {n} usable pixels)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ZARR_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(RAW_DIR.glob("PACE_OCI.*.L2.OC_AOP*.nc"))
    if len(existing) >= N_GRANULES:
        nc_files = existing[:N_GRANULES]
        print("Using existing NetCDF downloads (skip Earthdata search):")
        for p in nc_files:
            print(f"  {p.name}")
    else:
        login()
        granules = search_granules()
        nc_files = download_granules(granules)

    for nc_path in nc_files:
        print(f"\nProcessing {nc_path.name}")
        ds = load_masked_granule(nc_path)
        zarr_path = ZARR_DIR / (nc_path.stem + ".zarr")
        write_zarr(ds, zarr_path)
        ds.close()

        ds_r = validate_zarr(zarr_path)
        fig_path = FIG_DIR / (nc_path.stem + "_sample_spectra.png")
        plot_sample_spectra(ds_r, fig_path)
        ds_r.close()

    print("\nDone. Review figures/ for water-like spectra (blue peak, red absorption).")
    print("If a granule is all cloud, its valid_water count will be ~0 — skip it.")


if __name__ == "__main__":
    main()
