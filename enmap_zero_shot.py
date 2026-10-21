"""Run REAL EnMAP spectra through the PACE-trained model (never trained on EnMAP) and see which
PACE training pixels they resemble most, per EnMAP site.
Run after train_ssl.py and build_enmap_dataset.py:  python enmap_zero_shot.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model import Encoder, features
from src.preprocess import clean_spectra
from src.spectra_set import SpectraSet

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
enc = Encoder().to(dev); enc.load_state_dict(torch.load("data/processed/encoder_pace_ssl.pt", map_location=dev)); enc.eval()
P = SpectraSet.load(Path("data/processed/pace_spectra.npz")); E = SpectraSet.load(Path("data/processed/enmap_spectra.npz"))


@torch.no_grad()
def embed(refl, wavelength, batch=1024):
    wl = torch.tensor(wavelength, device=dev); out = []
    for i in range(0, len(refl), batch):
        x = torch.tensor(refl[i:i + batch], dtype=torch.float32, device=dev)
        ok = torch.ones(x.shape, dtype=torch.bool, device=dev)
        s, lb = features(x, wl, ok)
        out.append(F.normalize(enc(s, wl, ok, lb), dim=1))
    return torch.cat(out)


# a spread of PACE pixels: equal numbers per site, so the big Arabian Sea set does not drown the rest
px, psite = [], []
for site in dict.fromkeys(P.site):
    idx = np.where(np.isin(P.granule, [i for i, t in enumerate(P.site) if t == site]))[0]
    idx = rng.choice(idx, min(15000, len(idx)), replace=False); px.append(idx); psite += [site] * len(idx)
px = np.concatenate(px); psite = np.array(psite)
pr, _ = clean_spectra(P.refl[px], P.wavelength_nm); zp = embed(pr, P.wavelength_nm)
psite = psite[clean_spectra(P.refl[px], P.wavelength_nm)[1]]
print(f"PACE reference pixels: { {s: int((psite == s).sum()) for s in dict.fromkeys(psite)} }")

er, keep = clean_spectra(E.refl, E.wavelength_nm); esite = E.site[E.granule[keep]]
print("\nreal EnMAP pixels, zero-shot: which PACE site is the nearest neighbour?")
for s in dict.fromkeys(esite):
    idx = np.where(esite == s)[0]; idx = rng.choice(idx, min(5000, len(idx)), replace=False)
    z = embed(er[idx], E.wavelength_nm); best, nn = (z @ zp.T).max(1)
    frac = {k: f"{(psite[nn.cpu().numpy()] == k).mean():.0%}" for k in dict.fromkeys(psite)}
    print(f"  EnMAP {s:14s} n={len(idx):5d}  finite={bool(torch.isfinite(z).all())}  mean best cosine {best.mean():.3f}  nearest PACE site: {frac}")
