"""E6 classical baseline: NDCI = (R708-R665)/(R708+R665), a standard chlorophyll index. Computed per site; no labels, so no AUPRC."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.spectra_set import SpectraSet
S = SpectraSet.load(Path("data/processed/pace_spectra.npz")); wl = S.wavelength_nm
i708, i665 = int(np.argmin(abs(wl-708))), int(np.argmin(abs(wl-665)))
print("bands used:", wl[i708], wl[i665])
a, b = S.refl[:, i708], S.refl[:, i665]
ndci = (a - b) / np.maximum(a + b, 1e-6)
fig, ax = plt.subplots(figsize=(7, 4))
for s in ["lake_erie", "n_arabian_sea"]:
    gi = [i for i, t in enumerate(S.site) if t == s]; m = np.isin(S.granule, gi)
    v = ndci[m]; print(f"{s}: n={m.sum()} NDCI median {np.median(v):.3f} p10 {np.percentile(v,10):.3f} p90 {np.percentile(v,90):.3f}")
    ax.hist(v, bins=80, range=(-1, 1), density=True, alpha=.6, label=f"{s} (n={m.sum()})")
ax.set(xlabel="NDCI", ylabel="density", title="Classical chlorophyll index (baseline)"); ax.legend()
fig.tight_layout(); fig.savefig("figures/inspect/ndci_baseline.png", dpi=130)
