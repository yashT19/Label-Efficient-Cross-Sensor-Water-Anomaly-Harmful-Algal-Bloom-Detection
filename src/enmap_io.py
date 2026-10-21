"""Read an EnMAP L2A water-product scene (the .ZIP that DLR delivers) into a SpectraSet.

What the delivered files are, as checked on real scenes:
- SPECTRAL_IMAGE.TIF: 224 bands, 16-bit integers; reflectance = value * Gain + Offset
  (Gain 0.0001, Offset 0 in the metadata); nodata is -32768.
- QL_QUALITY_CLASSES.TIF: 1 = land, 2 = water (its share of land+water matches the
  metadata water cover); QL_QUALITY_CLOUD / CLOUDSHADOW / HAZE / CIRRUS / SNOW are 0
  where clear.
- METADATA.XML: band centres and widths, acquisition time, cloud and sunglint percentages.

Not settled: the metadata calls the water product "Normalized_Rrs". Whether the stored values
equal Rrs (as in PACE) or pi times Rrs is still to be confirmed from DLR's documentation.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from .spectra_set import SpectraSet

NODATA = -32768
WATER_CLASS = 2
# Same window as PACE's Rrs product (346-719 nm) so both sensors are compared on common bands.
DEFAULT_WINDOW_NM = (400.0, 720.0)


def _numbers(xml: str, tag: str) -> np.ndarray:
    return np.array([float(v) for v in re.findall(rf"<{tag}[^>]*>([-\d.eE+]+)</{tag}>", xml)])


def _one(xml: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}[^>]*>([^<]*)</{tag}>", xml)
    return m.group(1).strip() if m else None


def _inner(names: list[str], suffix: str) -> str:
    hits = [n for n in names if n.endswith(suffix)]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *{suffix} in the zip, found {len(hits)}")
    return hits[0]


def read_band_table(zip_path: Path) -> dict[str, np.ndarray]:
    """Band centres, widths and scaling for all 224 bands, taken from the scene's metadata."""
    with zipfile.ZipFile(zip_path) as z:
        xml = z.read(_inner(z.namelist(), "-METADATA.XML")).decode("utf-8")
    table = {k: _numbers(xml, t) for k, t in
             [("wavelength_nm", "wavelengthCenterOfBand"), ("fwhm_nm", "FWHMOfBand"),
              ("gain", "GainOfBand"), ("offset", "OffsetOfBand")]}
    if not all(len(v) == 224 for v in table.values()):
        raise ValueError(f"{zip_path.name}: expected 224 band entries, got "
                         f"{ {k: len(v) for k, v in table.items()} }")
    return table


def band_grid(zip_path: Path, window_nm=DEFAULT_WINDOW_NM) -> tuple[np.ndarray, np.ndarray]:
    """(centres, widths) of the EnMAP bands inside the window: the real target band layout."""
    t = read_band_table(zip_path)
    keep = (t["wavelength_nm"] >= window_nm[0]) & (t["wavelength_nm"] <= window_nm[1])
    return t["wavelength_nm"][keep], t["fwhm_nm"][keep]


def read_enmap_scene(
    zip_path: Path,
    site: str = "unknown",
    window_nm=DEFAULT_WINDOW_NM,
    max_pixels: int = 100_000,
    seed: int = 0,
) -> tuple[SpectraSet, dict]:
    """Clear-water pixels of one scene as a SpectraSet, plus a dict of scene facts."""
    zip_path = Path(zip_path)
    table = read_band_table(zip_path)
    wl = table["wavelength_nm"]
    sel = np.where((wl >= window_nm[0]) & (wl <= window_nm[1]))[0]
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        xml = z.read(_inner(names, "-METADATA.XML")).decode("utf-8")
    scene = Path(_inner(names, "-METADATA.XML")).name.removesuffix("-METADATA.XML")
    vsi = lambda suffix: "/vsizip/" + zip_path.resolve().as_posix() + "/" + _inner(names, suffix)

    def layer(q: str) -> np.ndarray:
        with rasterio.open(vsi(f"-QL_QUALITY_{q}.TIF")) as ds:
            return ds.read(1)

    clear = layer("CLASSES") == WATER_CLASS
    for q in ("CLOUD", "CLOUDSHADOW", "HAZE", "CIRRUS", "SNOW"):
        clear &= layer(q) == 0
    n_clear = int(clear.sum())

    with rasterio.open(vsi("-SPECTRAL_IMAGE.TIF")) as ds:
        cube = ds.read([int(i) + 1 for i in sel])            # (bands, rows, cols), int16
        transform, crs = ds.transform, ds.crs
    ok = clear & (cube != NODATA).all(0)
    rows, cols = np.nonzero(ok)
    if len(rows) > max_pixels:
        pick = np.random.default_rng(seed).choice(len(rows), max_pixels, replace=False)
        rows, cols = rows[pick], cols[pick]
    dn = cube[:, rows, cols].T.astype("float64")             # (n, bands)
    refl = (dn * table["gain"][sel] + table["offset"][sel]).astype("float32")
    xs, ys = xy(transform, rows, cols)
    lon, lat = warp_transform(crs, "EPSG:4326", xs, ys)

    info = {
        "scene": scene,
        "start_utc": _one(xml, "startTime"),
        "water_product": _one(xml, "waterReflectanceProduct"),
        "cloud_cover_pct": _one(xml, "cloudCover"),
        "water_cover_pct": _one(xml, "waterCover"),
        "sunglint_pct": _one(xml, "sceneSunglint"),
        "clear_water_pixels": n_clear,
        "pixels_returned": int(len(rows)),
        "bands_returned": int(len(sel)),
    }
    s = SpectraSet(
        refl=refl, wavelength_nm=wl[sel].astype("float32"), fwhm_nm=table["fwhm_nm"][sel].astype("float32"),
        band_valid=np.ones(len(sel), bool), lat=np.asarray(lat, "float32"), lon=np.asarray(lon, "float32"),
        granule=np.zeros(len(rows), int), granule_names=np.array([scene]), site=np.array([site]), sensor="EnMAP",
    )
    return s, info
