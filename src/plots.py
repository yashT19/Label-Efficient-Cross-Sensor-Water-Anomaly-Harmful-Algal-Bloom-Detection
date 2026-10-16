"""Sample PACE OCI spectrum plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def plot_sample_spectra(
    ds: xr.Dataset,
    fig_path: Path,
    n_spectra: int,
    seed: int,
    reference_wavelength_nm: float,
    marker_wavelengths_nm: Sequence[float],
) -> None:
    """Plot random valid spectra and their mean, matching the original script."""
    wave = ds["wavelength"].values
    rrs = ds["Rrs"]
    i_ref = int(np.argmin(np.abs(wave - reference_wavelength_nm)))
    usable = ds["valid_water"].values & np.isfinite(rrs.isel(wavelength=i_ref).values)
    ys, xs = np.where(usable)
    if ys.size == 0:
        LOGGER.info("  skip plot: no usable water pixels")
        return
    rng = np.random.default_rng(seed)
    k = min(n_spectra, ys.size)
    pick = rng.choice(ys.size, size=k, replace=False)
    spectra = np.stack([rrs.values[ys[i], xs[i], :] for i in pick], axis=0)
    mean_spec = np.nanmean(spectra, axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for spec in spectra:
        ax.plot(wave, spec, color="0.75", lw=0.8, alpha=0.85)
    ax.plot(wave, mean_spec, color="C0", lw=2.2, label=f"mean (n={k})")
    for pigment in marker_wavelengths_nm:
        ax.axvline(pigment, color="C2", ls="--", lw=0.9, alpha=0.7)
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
    LOGGER.info("  wrote %s (sampled %d of %d usable pixels)", fig_path, k, ys.size)
