"""Turn delivered EnMAP scenes into data/processed/enmap_spectra.npz and enmap_bands.npz.

Run:  python build_enmap_dataset.py ../enmap_data
(the folder holding the unpacked DLR orders; every ENMAP.HSI.L2A/*.ZIP below it is read)
"""
import sys
from pathlib import Path

import numpy as np

from src.enmap_io import band_grid, read_enmap_scene
from src.spectra_set import SpectraSet

MIN_PIXELS = 1000          # skip scenes that are almost all cloud or land
# rough place names for the scenes we have; anything else is labelled "unknown"
PLACES = {"lake_victoria": (-4, 0, 30, 35), "gulf_khambhat": (20, 22, 72, 74),
          "lake_erie": (41, 42.5, -84, -82), "lake_winnipeg": (50, 54, -99, -96)}


def place(lat: float, lon: float) -> str:
    return next((n for n, (a, b, c, d) in PLACES.items() if a <= lat <= b and c <= lon <= d), "unknown")


root = Path(sys.argv[1] if len(sys.argv) > 1 else "../enmap_data")
zips = sorted(root.glob("*/ENMAP.HSI.L2A/*.ZIP"))
parts, names, sites, kept = [], [], [], []
centres = None
for z in zips:
    c, f = band_grid(z)
    if centres is None:
        centres, fwhm = c, f
    elif len(c) != len(centres) or not np.allclose(c, centres, atol=0.1):
        print("SKIP (band centres differ from the first scene):", z.name); continue
    s, info = read_enmap_scene(z, site="tmp")
    where = place(float(np.median(s.lat)), float(np.median(s.lon))) if len(s.lat) else "unknown"
    print(f"{info['start_utc'][:16]}Z {where:14s} clear water px {info['clear_water_pixels']:>8,}  "
          f"cloud {info['cloud_cover_pct']}%  sunglint {info['sunglint_pct']}%  kept {info['pixels_returned']:,}")
    if info["pixels_returned"] < MIN_PIXELS:
        print("   -> too few clear water pixels, skipped"); continue
    parts.append(s); names.append(info["scene"]); sites.append(where); kept.append(info)
S = SpectraSet(
    refl=np.concatenate([p.refl for p in parts]), wavelength_nm=centres.astype("float32"),
    fwhm_nm=fwhm.astype("float32"), band_valid=np.ones(len(centres), bool),
    lat=np.concatenate([p.lat for p in parts]), lon=np.concatenate([p.lon for p in parts]),
    granule=np.concatenate([np.full(len(p.refl), i) for i, p in enumerate(parts)]),
    granule_names=np.array(names), site=np.array(sites), sensor="EnMAP")
out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
S.save(out / "enmap_spectra.npz")
np.savez(out / "enmap_bands.npz", wavelength_nm=centres, fwhm_nm=fwhm)
print(f"\nsaved {len(S.refl):,} spectra from {len(parts)} scenes, {len(centres)} bands "
      f"({centres[0]:.1f}-{centres[-1]:.1f} nm) -> {out}")
