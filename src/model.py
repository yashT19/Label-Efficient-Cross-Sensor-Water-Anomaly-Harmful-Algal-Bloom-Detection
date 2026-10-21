"""Wavelength-conditioned spectrum encoder + Barlow Twins loss + on-GPU spectral alignment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

MIN_NM = 400.0
MAX_NM = 720.0   # same window as src/preprocess.py: PACE and EnMAP are averaged over the same bands
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def align_t(x, src, tgt, fwhm, min_cov=0.5):
    """Torch twin of src.align.align_spectra (Gaussian SRF, spacing-weighted)."""
    sigma = (fwhm * FWHM_TO_SIGMA)[:, None]
    d = torch.gradient(src)[0][None]
    w = torch.exp(-0.5 * ((src[None] - tgt[:, None]) / sigma) ** 2) * d
    cov = w.sum(1) / (math.sqrt(2 * math.pi) * sigma[:, 0])
    valid = (tgt >= src.min()) & (tgt <= src.max()) & (cov >= min_cov)
    out = x @ (w / w.sum(1, keepdim=True).clamp_min(1e-12)).T
    return out, valid


def features(x, wl, valid):
    """Spectrum -> (shape = x / own mean over valid bands >= 400 nm, log-brightness)."""
    use = (valid & (wl >= MIN_NM) & (wl <= MAX_NM)).float()
    mean = ((x * use).sum(-1) / use.sum(-1).clamp_min(1)).clamp_min(1e-5)
    shape = torch.where(valid, x / mean[:, None], torch.zeros_like(x))
    return shape, (torch.log(mean) + 6.0) / 2.0   # fixed constants, not fit on any sensor


class Encoder(nn.Module):
    """Each band is a token (value + wavelength embedding); invalid bands are masked out."""

    def __init__(self, d=64, layers=2, heads=4, out=64):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(0, 6).float())
        self.val = nn.Linear(1, d)
        self.wl = nn.Linear(12, d)
        self.norm = nn.LayerNorm(d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d + 1, out)

    def wl_emb(self, wl):
        t = ((wl - 346.0) / 373.0)[:, None] * self.freqs[None] * math.pi
        return self.wl(torch.cat([t.sin(), t.cos()], -1))

    def forward(self, shape, wl, valid, logb):
        h = self.norm(self.val(shape[..., None]) + self.wl_emb(wl)[None])
        h = self.tf(h, src_key_padding_mask=~valid)
        m = valid[..., None].float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return self.head(torch.cat([pooled, logb[:, None]], -1))


class BarlowTwins(nn.Module):
    def __init__(self, enc, out=64, proj=256, lam=5e-3):
        super().__init__()
        self.enc, self.lam = enc, lam
        self.proj = nn.Sequential(nn.Linear(out, proj), nn.BatchNorm1d(proj), nn.ReLU(), nn.Linear(proj, proj))

    def loss(self, za, zb):
        a, b = self.proj(za), self.proj(zb)
        a = (a - a.mean(0)) / (a.std(0) + 1e-5); b = (b - b.mean(0)) / (b.std(0) + 1e-5)
        c = a.T @ b / a.shape[0]
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
        return on + self.lam * off
