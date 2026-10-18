"""Self-supervised pretraining on PACE spectra. View A = native PACE bands; view B = the same
pixel resampled to a random EnMAP-LIKE band grid (placeholder grids, TODO[VERIFY] vs real EnMAP)."""
import sys, time
from pathlib import Path
import numpy as np, torch
from src.model import BarlowTwins, Encoder, align_t, features
from src.spectra_set import SpectraSet

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
dev = "cuda"; torch.manual_seed(0); rng = np.random.default_rng(0)
S = SpectraSet.load(Path("data/processed/pace_spectra.npz"))
# leak-free split: hold out whole granules (2 Arabian Sea + 1 Lake Erie)
names, site = S.granule_names, S.site
hold = [np.where(site == "n_arabian_sea")[0][i] for i in (1, 5)] + [np.where(site == "lake_erie")[0][0]]
is_val = np.isin(S.granule, hold)
X = torch.tensor(S.refl, device=dev); wl = torch.tensor(S.wavelength_nm, device=dev)
tr, va = np.where(~is_val)[0], np.where(is_val)[0]
cnt = np.bincount(S.granule, minlength=len(names)).astype(float)
p = 1 / np.sqrt(cnt[S.granule[tr]]); p /= p.sum()    # soften dominance of the big granules
print("train px", len(tr), "val px", len(va), "val granules", [str(names[i]) for i in hold])
allv = torch.ones_like(wl, dtype=torch.bool)

def grid(r):
    sp = float(r.choice([5.0, 6.5, 8.0, 10.0])); st = float(r.uniform(400, 430))
    tgt = torch.arange(st, 720.0, sp, device=dev)
    return tgt, tgt.new_full(tgt.shape, sp * float(r.uniform(1.0, 1.4)))

def view_b(x, r, noise=True):
    tgt, fw = grid(r)
    out, v = align_t(x, wl, tgt, fw)
    if noise:
        out = (out + torch.randn_like(out) * float(r.uniform(0, 1e-4))) * torch.exp(0.1 * torch.randn(len(x), 1, device=dev))
        v = v[None] & (torch.rand(len(x), len(tgt), device=dev) > float(r.uniform(0, 0.2)))
    else:
        v = v[None].expand(len(x), -1)
    return features(out.clamp_min(0), tgt, v) + (tgt, v)

def view_a(x, noise=True):
    if noise:
        x = (x + torch.randn_like(x) * 5e-5) * torch.exp(0.05 * torch.randn(len(x), 1, device=dev))
    s, lb = features(x.clamp_min(0), wl, allv[None].expand(len(x), -1)); return s, lb

def embed(enc, x, b_grid=(6.5, 8.0)):
    sa, la = view_a(x, False)
    tgt = torch.arange(420.0, 720.0, b_grid[0], device=dev)
    out, v = align_t(x, wl, tgt, tgt.new_full(tgt.shape, b_grid[1])); v = v[None].expand(len(x), -1)
    sb, lb = features(out, tgt, v)
    return enc(sa, wl, allv[None].expand(len(x), -1), la), enc(sb, tgt, v, lb)

@torch.no_grad()
def retrieval(enc, idx, n=2000):
    enc.eval(); sel = torch.tensor(np.random.default_rng(1).choice(idx, n, replace=False), device=dev)
    za, zb = embed(enc, X[sel]); za, zb = torch.nn.functional.normalize(za, dim=1), torch.nn.functional.normalize(zb, dim=1)
    acc = (torch.argmax(za @ zb.T, 1) == torch.arange(n, device=dev)).float().mean().item()
    cos = (za * zb).sum(1).mean().item(); enc.train(); return acc, cos

enc = Encoder().to(dev); model = BarlowTwins(enc).to(dev)
print("val retrieval BEFORE training (random encoder): top1 %.3f cos %.3f" % retrieval(enc, va))
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
t0 = time.time()
for it in range(1, steps + 1):
    idx = torch.tensor(rng.choice(tr, 1024, p=p), device=dev); x = X[idx]
    sa, la = view_a(x); sb, lb, tgt, v = view_b(x, rng)
    loss = model.loss(enc(sa, wl, allv[None].expand(len(x), -1), la), enc(sb, tgt, v, lb))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 250 == 0 or it == 1:
        a, c = retrieval(enc, va); print(f"step {it} loss {loss.item():.1f} | val top1 {a:.3f} cos {c:.3f} | {time.time()-t0:.0f}s", flush=True)
torch.save(enc.state_dict(), "data/processed/encoder_pace_ssl.pt")
print("saved encoder")
