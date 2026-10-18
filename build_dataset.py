"""Collect cleaned valid-water PACE spectra from all Zarr stores into one SpectraSet."""
import glob
from pathlib import Path
import numpy as np, xarray as xr
from src.preprocess import clean_spectra
from src.spectra_set import SpectraSet

stores = {"lake_erie": "data/processed/pace_oci_l2_aop_zarr/*.zarr"}
for s in ["n_arabian_sea", "vembanad", "pulicat", "chilika"]:
    stores[s] = f"data/processed/sites/{s}/*.zarr"
refl, lat, lon, gran, names, sites, wl = [], [], [], [], [], [], None
for site, pat in stores.items():
    for z in sorted(glob.glob(pat)):
        ds = xr.open_zarr(z, consolidated=True); v = ds["valid_water"].values
        if not v.sum(): continue
        x, keep = clean_spectra(ds["Rrs"].values[v], ds["wavelength"].values)
        if not len(x): continue
        wl = ds["wavelength"].values
        refl.append(x.astype("float32"))
        lat.append(ds["latitude"].values[v][keep]); lon.append(ds["longitude"].values[v][keep])
        gran.append(np.full(len(x), len(names))); names.append(Path(z).stem); sites.append(site)
S = SpectraSet(np.concatenate(refl), wl.astype("float32"), np.full(wl.shape, np.nan, "float32"),
               np.ones(wl.shape, bool), np.concatenate(lat).astype("float32"),
               np.concatenate(lon).astype("float32"), np.concatenate(gran), np.array(names),
               np.array(sites), "PACE_OCI")
S.save(Path("data/processed/pace_spectra.npz"))
print("pixels", len(S.refl), "granules", len(names))
for s in dict.fromkeys(sites):
    m = np.isin(S.granule, [i for i, t in enumerate(sites) if t == s]); print(" ", s, int(m.sum()))
