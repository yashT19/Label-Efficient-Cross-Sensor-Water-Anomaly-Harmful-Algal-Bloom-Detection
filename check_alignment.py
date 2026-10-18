"""Align real PACE spectra to an EnMAP-LIKE grid and check bloom features survive.
EnMAP-like grid below is a PLACEHOLDER (TODO[VERIFY] against real EnMAP XML band list)."""
import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, xarray as xr
from src.align import align_spectra
from src.preprocess import clean_spectra

def load(pat):
    xs = []
    for z in sorted(glob.glob(pat)):
        ds = xr.open_zarr(z, consolidated=True); v = ds["valid_water"].values
        if v.sum(): xs.append(ds["Rrs"].values[v]); wl = ds["wavelength"].values
    return np.concatenate(xs), wl
grids = {"10nm grid (plan)": (np.arange(400, 720, 10.0), 10.0),
         "EnMAP-like 6.5nm (placeholder)": (np.arange(420, 720, 6.5), 8.0)}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for k, (name, pat) in enumerate({"Lake Erie": "data/processed/pace_oci_l2_aop_zarr/*.zarr",
                                 "Arabian Sea": "data/processed/sites/n_arabian_sea/*.zarr"}.items()):
    x, wl = load(pat); x, _ = clean_spectra(x, wl)
    x = x[np.random.default_rng(0).choice(len(x), min(len(x), 20000), replace=False)]
    for gname, (tgt, fw) in grids.items():
        out, ok = align_spectra(x, wl, tgt, np.full(tgt.shape, fw))
        print(f"{name} | {gname}: {tgt.size} target bands, valid {int(ok.sum())}, "
              f"invalid centres nm: {tgt[~ok].round(0).tolist()}")
        # feature check: 670 dip depth = mean(660,680) - 670 ; before vs after
        def dip(spec, w, c, d=12):
            a = np.interp(c - d, w, spec); b = np.interp(c + d, w, spec); m = np.interp(c, w, spec)
            return (a + b) / 2 - m
        i_pace = np.median([dip(s, wl, 670) for s in x[:2000]])
        sel = ok
        i_al = np.median([dip(s[sel], tgt[sel], 670) for s in out[:2000]])
        print(f"   670 nm dip depth median: PACE {i_pace:.5f} -> aligned {i_al:.5f}")
        if gname.startswith("10"):
            ax[k].plot(wl, np.median(x, 0), label="PACE median"); ax[k].plot(tgt[ok], np.nanmedian(out, 0)[ok], "o-", ms=3, label=gname)
    ax[k].set(title=name, xlabel="nm", ylabel="Rrs"); ax[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/inspect/alignment_before_after.png", dpi=120)
