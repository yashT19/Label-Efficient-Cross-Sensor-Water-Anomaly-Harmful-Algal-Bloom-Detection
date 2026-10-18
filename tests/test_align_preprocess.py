"""Self-checks for alignment and preprocessing (synthetic data, no downloads)."""

import numpy as np

from src.align import align_spectra
from src.preprocess import clean_spectra, shape_and_brightness

# PACE-like grid: 2.5 nm, with a gap 588-613 nm
SRC = np.concatenate([np.arange(346, 589, 2.5), np.arange(613, 720, 1.25)])
TGT = np.arange(420, 700, 10.0)          # 10 nm target grid
FWHM = np.full(TGT.shape, 10.0)


def test_constant_spectrum_is_preserved():
    out, ok = align_spectra(np.full((3, SRC.size), 0.01), SRC, TGT, FWHM)
    assert ok[(TGT < 570) | (TGT > 630)].all()
    assert np.allclose(out[:, ok], 0.01)


def test_linear_spectrum_is_preserved_away_from_gap():
    out, ok = align_spectra((SRC * 1e-5)[None, :], SRC, TGT, FWHM)
    far = ok & ((TGT < 570) | (TGT > 630))
    assert np.allclose(out[0, far], TGT[far] * 1e-5, atol=2e-5)


def test_edges_and_gap_are_masked():
    tgt = np.array([300.0, 600.0, 800.0])
    _, ok = align_spectra(np.ones((1, SRC.size)), SRC, tgt, np.full(3, 4.0))
    assert not ok[0] and not ok[2]       # outside source range
    assert not ok[1]                     # centre inside the 588-613 nm gap


def test_clean_and_normalise():
    wl = np.arange(400.0, 700.0, 10.0)
    x = np.vstack([np.full(wl.size, 0.01), np.full(wl.size, -0.01), np.full(wl.size, 0.0)])
    x[0, 0] = -0.0005                    # small negative noise is clipped, kept
    c, keep = clean_spectra(x, wl)
    assert keep.tolist() == [True, False, False] and c.min() >= 0
    shape, bright = shape_and_brightness(c, wl)
    assert np.allclose(shape.mean(1), 1.0) and np.isfinite(bright).all()
