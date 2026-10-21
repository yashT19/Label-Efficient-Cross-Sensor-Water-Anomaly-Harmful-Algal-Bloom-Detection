"""Sensor-independent spectrum cleaning and normalisation (fit nothing on data)."""

from __future__ import annotations

import numpy as np

NEG_DROP = -0.001   # drop pixel if any band >= MIN_NM is below this (failed atm. correction)
MIN_NM = 400.0
MAX_NM = 720.0      # PACE's Rrs stops at 719 nm; both sensors are judged on the same window
MIN_MEAN = 1e-4     # drop near-zero-signal pixels: shape is undefined there


def clean_spectra(
    rrs: np.ndarray, wavelength: np.ndarray, band_valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned spectra, keep mask). rrs is (n_pixels, n_bands)."""
    rrs = np.asarray(rrs, dtype="float64")
    use = (wavelength >= MIN_NM) & (wavelength <= MAX_NM)
    if band_valid is not None:
        use &= band_valid
    sub = rrs[:, use]
    keep = np.isfinite(sub).all(1) & (sub.min(1) >= NEG_DROP) & (sub.mean(1) >= MIN_MEAN)
    return np.clip(rrs[keep], 0.0, None), keep


def shape_and_brightness(
    rrs: np.ndarray, wavelength: np.ndarray, band_valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split spectra into shape (divided by own mean) and log-brightness.

    Shape is sensor-independent; brightness is kept as a separate feature.
    The mean runs over the same bands for every pixel (>= MIN_NM, valid bands).
    """
    use = (wavelength >= MIN_NM) & (wavelength <= MAX_NM)
    if band_valid is not None:
        use &= band_valid
    mean = rrs[:, use].mean(1, keepdims=True)
    return rrs / mean, np.log(mean[:, 0])
