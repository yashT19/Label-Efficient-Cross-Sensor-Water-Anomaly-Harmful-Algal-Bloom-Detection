"""Inspect valid-water PACE spectra: per-band stats, gaps, negatives, site comparison."""
import glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

groups = {"lake_erie": "data/processed/pace_oci_l2_aop_zarr/*.zarr"}
for s in ["n_arabian_sea", "vembanad", "pulicat", "chilika"]:
    groups[s] = f"data/processed/sites/{s}/*.zarr"
spec, wl = {}, None
for site, pat in groups.items():
    chunks = []
    for z in sorted(glob.glob(pat)):
        ds = xr.open_zarr(z, consolidated=True)
        v = ds["valid_water"].values
        if v.sum() == 0: continue
        chunks.append(ds["Rrs"].values[v])          # (n_valid, bands)
        wl = ds["wavelength"].values
    if chunks: spec[site] = np.concatenate(chunks)
for s, x in spec.items():
    fin = np.isfinite(x)
    print(f"{s}: {x.shape[0]} px | all-NaN bands {int((~fin.any(0)).sum())} | "
          f"px with any NaN {np.mean(~fin.all(1)):.1%} | negatives {np.mean(x[fin]<0):.2%} | "
          f"min {np.nanmin(x):.4f} max {np.nanmax(x):.4f}")
x = spec["n_arabian_sea"]
nan_band = np.isnan(x).mean(0)
bad = wl[nan_band > 0.2]
print("bands with >20% NaN (Arabian Sea):", len(bad), "range", (bad.min(), bad.max()) if len(bad) else None)
print("band spacing nm: min %.2f median %.2f max %.2f" % (np.diff(wl).min(), np.median(np.diff(wl)), np.diff(wl).max()))
for nm in (443, 550, 615, 670):
    i = int(np.argmin(abs(wl-nm)))
    print(f"  {nm}nm band {wl[i]:.1f}: median {np.nanmedian(x[:,i]):.4f} p1 {np.nanpercentile(x[:,i],1):.4f} p99 {np.nanpercentile(x[:,i],99):.4f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for s, v in spec.items():
    ax[0].plot(wl, np.nanmedian(v, 0), label=f"{s} (n={len(v)})")
ax[0].set(title="Median Rrs per site", xlabel="nm", ylabel="Rrs (sr^-1)"); ax[0].legend(fontsize=7)
ax[1].plot(wl, nan_band*100); ax[1].set(title="% NaN per band (Arabian Sea)", xlabel="nm", ylabel="%")
fig.tight_layout(); fig.savefig("figures/inspect/site_spectra_and_gaps.png", dpi=120)
# normalisation candidates: per-spectrum L2 (shape only) vs per-band z-score
ok = np.isfinite(x).all(1); xc = x[ok]
l2 = xc / np.linalg.norm(xc, axis=1, keepdims=True)
print("complete-spectrum px:", ok.sum(), "| brightness (L2 norm) p5/p50/p95:", np.percentile(np.linalg.norm(xc,axis=1),[5,50,95]).round(3))
print("after L2: per-band std range %.4f-%.4f" % (l2.std(0).min(), l2.std(0).max()))
