"""Spectral alignment: resample a spectrum onto another sensor's bands (Gaussian SRF).

Target band centres/FWHM come from the target sensor's metadata (EnMAP XML), never
hard-coded counts. Source bands may be unevenly spaced (PACE has a gap at 588-613 nm),
so each source band is weighted by its local spacing.
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def srf_weights(
    src_nm: np.ndarray, tgt_nm: np.ndarray, tgt_fwhm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (W[tgt, src] spacing-weighted Gaussian weights, coverage[tgt] in 0..1)."""
    src_nm, tgt_nm = np.asarray(src_nm, float), np.asarray(tgt_nm, float)
    sigma = np.asarray(tgt_fwhm, float)[:, None] * FWHM_TO_SIGMA
    d = np.gradient(src_nm)[None, :]
    w = np.exp(-0.5 * ((src_nm[None, :] - tgt_nm[:, None]) / sigma) ** 2) * d
    ideal = np.sqrt(2 * np.pi) * sigma                    # integral of the Gaussian
    return w, w.sum(1) / ideal[:, 0]


def align_spectra(
    rrs: np.ndarray,
    src_nm: np.ndarray,
    tgt_nm: np.ndarray,
    tgt_fwhm: np.ndarray,
    src_valid: np.ndarray | None = None,
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample (n, src_bands) spectra to target bands. Returns (values, band_valid).

    A target band is valid only if its centre lies inside the source range and the
    source bands supply >= min_coverage of its response (so gaps/edges are masked).
    """
    rrs = np.asarray(rrs, float)
    w, _ = srf_weights(src_nm, tgt_nm, tgt_fwhm)
    if src_valid is not None:
        w = w * np.asarray(src_valid, float)[None, :]
    sigma = np.asarray(tgt_fwhm, float) * FWHM_TO_SIGMA
    coverage = w.sum(1) / (np.sqrt(2 * np.pi) * sigma)
    inside = (tgt_nm >= np.min(src_nm)) & (tgt_nm <= np.max(src_nm))
    valid = inside & (coverage >= min_coverage)
    norm = np.where(w.sum(1, keepdims=True) > 0, w.sum(1, keepdims=True), 1.0)
    out = rrs @ (w / norm).T
    out[:, ~valid] = np.nan
    return out, valid
