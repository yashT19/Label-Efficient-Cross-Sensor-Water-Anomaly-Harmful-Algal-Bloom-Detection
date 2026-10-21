"""Check spectral alignment on real data.

1. PACE spectra are resampled onto the REAL EnMAP band layout (centres and widths read from
   EnMAP metadata by build_enmap_dataset.py) and onto a plain 10 nm grid; we check which bands
   are filled and whether the 670 nm chlorophyll dip survives.
2. Real EnMAP spectra are plotted next to PACE spectra resampled to the same bands.
Run build_enmap_dataset.py first (it writes data/processed/enmap_bands.npz).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.align import align_spectra
from src.preprocess import clean_spectra, shape_and_brightness
from src.spectra_set import SpectraSet

pace = SpectraSet.load(Path("data/processed/pace_spectra.npz"))
eb = np.load("data/processed/enmap_bands.npz")
grids = {"10 nm grid": (np.arange(400, 720, 10.0), np.full(32, 10.0)),
         "real EnMAP bands": (eb["wavelength_nm"].astype(float), eb["fwhm_nm"].astype(float))}
wl = pace.wavelength_nm


def dip(spec, w, c=670, d=12):
    a, b, m = (np.interp(v, w, spec) for v in (c - d, c + d, c))
    return (a + b) / 2 - m


def site_spectra(s: SpectraSet, site: str) -> np.ndarray:
    m = np.isin(s.granule, [i for i, t in enumerate(s.site) if t == site])
    return s.refl[m]


for site in ("lake_erie", "n_arabian_sea"):
    x, _ = clean_spectra(site_spectra(pace, site), wl)
    x = x[np.random.default_rng(0).choice(len(x), min(len(x), 20000), replace=False)]
    for name, (tgt, fw) in grids.items():
        out, ok = align_spectra(x, wl, tgt, fw)
        print(f"{site} -> {name}: {tgt.size} bands, {int(ok.sum())} filled; empty at {tgt[~ok].round(1).tolist()}")
        print(f"   670 nm dip depth: PACE {np.median([dip(s, wl) for s in x[:2000]]):.5f} -> "
              f"aligned {np.median([dip(s[ok], tgt[ok]) for s in out[:2000]]):.5f}")

# real EnMAP vs PACE resampled to the same bands, shapes only (each spectrum divided by its own mean)
en = SpectraSet.load(Path("data/processed/enmap_spectra.npz"))
tgt, fw = grids["real EnMAP bands"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for site, col in (("lake_victoria", "#2E9A57"), ("lake_winnipeg", "#2F5FD0"), ("gulf_khambhat", "#D24545"), ("lake_erie", "#E0861A")):
    x, _ = clean_spectra(site_spectra(en, site), en.wavelength_nm)
    if len(x):
        sh, _ = shape_and_brightness(x, en.wavelength_nm)
        ax.plot(en.wavelength_nm, np.median(sh, 0), color=col, label=f"EnMAP {site}")
for site, col in (("lake_erie", "#E0861A"), ("n_arabian_sea", "#0B6B7A")):
    x, _ = clean_spectra(site_spectra(pace, site), wl)
    out, ok = align_spectra(x[:20000], wl, tgt, fw)
    sh, _ = shape_and_brightness(np.nan_to_num(out), tgt, ok)
    ax.plot(tgt[ok], np.median(sh[:, ok], 0), "--", color=col, label=f"PACE {site} (resampled)")
ax.set(xlabel="wavelength (nm)", ylabel="spectrum / its own mean", title="Real EnMAP vs PACE resampled to the same bands (shapes)")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/inspect/enmap_vs_pace_shape.png", dpi=130)
print("saved figures/inspect/enmap_vs_pace_shape.png")
