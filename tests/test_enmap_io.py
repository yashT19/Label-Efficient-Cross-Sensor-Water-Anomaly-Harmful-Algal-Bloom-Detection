"""EnMAP reader check on a tiny made-up scene in the same layout DLR delivers (no real data needed)."""

import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.enmap_io import NODATA, band_grid, read_enmap_scene

N, H, W = 224, 6, 8
CENTRES = np.linspace(418.0, 2445.0, N)


def _tif(path, arr, dtype):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=len(arr), dtype=dtype,
                       crs="EPSG:32636", transform=from_origin(500000, 9700000, 30, 30)) as ds:
        ds.write(arr)


def make_scene(tmp_path: Path) -> Path:
    name = "ENMAP01-____L2A-DT0000000001_20240101T080000Z_001_V010506_20260101T000000Z"
    cube = np.full((N, H, W), 100, "int16")                     # reflectance 0.01 everywhere
    cube[:, 0, 0] = NODATA                                      # one pixel without data
    classes = np.full((H, W), 2, "uint8"); classes[1, :] = 1    # row 1 is land
    cloud = np.zeros((H, W), "uint8"); cloud[2, 3] = 1          # one cloudy pixel
    files = {"SPECTRAL_IMAGE": (cube, "int16"), "QL_QUALITY_CLASSES": (classes, "uint8"),
             "QL_QUALITY_CLOUD": (cloud, "uint8")}
    for q in ("CLOUDSHADOW", "HAZE", "CIRRUS", "SNOW"):
        files[f"QL_QUALITY_{q}"] = (np.zeros((H, W), "uint8"), "uint8")
    bands = "".join(f"<bandID number=\"{i+1}\"><wavelengthCenterOfBand>{c}</wavelengthCenterOfBand>"
                    f"<FWHMOfBand>7.0</FWHMOfBand><GainOfBand>0.0001</GainOfBand><OffsetOfBand>0</OffsetOfBand></bandID>"
                    for i, c in enumerate(CENTRES))
    xml = (f"<metadata><startTime>2024-01-01T08:00:00Z</startTime><waterReflectanceProduct>Normalized_Rrs"
           f"</waterReflectanceProduct><cloudCover>3</cloudCover><waterCover>70</waterCover>"
           f"<sceneSunglint>0</sceneSunglint><bandCharacterisation>{bands}</bandCharacterisation></metadata>")
    zp = tmp_path / f"{name}.ZIP"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(f"{name}/{name}-METADATA.XML", xml)
        for key, (arr, dt) in files.items():
            tif = tmp_path / f"{key}.TIF"; _tif(tif, arr, dt); z.write(tif, f"{name}/{name}-{key}.TIF")
    return zp


def test_reader_keeps_only_clear_water_with_data(tmp_path):
    s, info = read_enmap_scene(make_scene(tmp_path), site="test")
    # 48 pixels - 8 land - 1 nodata - 1 cloud = 38
    assert len(s.refl) == 38 and info["clear_water_pixels"] == 39
    assert np.allclose(s.refl, 0.01, atol=1e-6)                 # value * 0.0001 gain
    assert s.sensor == "EnMAP" and s.band_valid.all()
    assert s.wavelength_nm.min() >= 400 and s.wavelength_nm.max() <= 720   # common window only
    assert len(s.wavelength_nm) == len(s.fwhm_nm) == s.refl.shape[1]
    assert 32.0 < s.lon.min() < 34.0 and -3.0 < s.lat.max() < 90        # coordinates were converted from UTM


def test_band_grid_reads_real_centres_and_widths(tmp_path):
    c, f = band_grid(make_scene(tmp_path))
    assert np.allclose(f, 7.0) and c[0] >= 400 and c[-1] <= 720 and len(c) == len(f)
