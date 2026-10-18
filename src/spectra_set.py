"""Sensor-agnostic container for pixel spectra. Any sensor reader (PACE, EnMAP) returns this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpectraSet:
    refl: np.ndarray            # (n_pixels, n_bands) reflectance (Rrs, sr^-1), cleaned
    wavelength_nm: np.ndarray   # (n_bands,) band centres from file metadata
    fwhm_nm: np.ndarray         # (n_bands,) band widths; NaN where unknown
    band_valid: np.ndarray      # (n_bands,) bool
    lat: np.ndarray             # (n_pixels,)
    lon: np.ndarray             # (n_pixels,)
    granule: np.ndarray         # (n_pixels,) int index into granule_names
    granule_names: np.ndarray   # (n_granules,) str: provenance, also used for leak-free splits
    site: np.ndarray            # (n_granules,) str, site of each granule
    sensor: str

    def save(self, path: Path) -> None:
        np.savez_compressed(path, sensor=self.sensor, **{
            k: v for k, v in self.__dict__.items() if k != "sensor"})

    @classmethod
    def load(cls, path: Path) -> "SpectraSet":
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["sensor"] = str(d["sensor"])
        return cls(**d)
