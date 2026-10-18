"""Download and process N spread-out PACE granules per site (metadata search first)."""
import csv, logging, sys
from pathlib import Path

import earthaccess
import numpy as np
import yaml

from src.masking import load_masked_granule
from src.pace_io import download_granules
from src.qc import validate_zarr
from src.zarr_schema import write_zarr

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch")
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
sites = sys.argv[2:] or [s for s in cfg["sites"] if s != "lake_erie"]
earthaccess.login(strategy="netrc")
pace, rows = cfg["pace"], []
for name in sites:
    site = cfg["sites"][name]
    res = earthaccess.search_data(short_name=pace["short_name"], version=pace["version"],
        bounding_box=tuple(site["bbox"]), temporal=tuple(site["dates"]), count=2000)
    # ponytail: evenly spaced over the window, not ranked by ROI coverage/cloud (metadata lacks it)
    pick = [res[i] for i in np.linspace(0, len(res) - 1, min(N, len(res))).astype(int)]
    log.info("%s: %d granules in window, taking %d", name, len(res), len(pick))
    raw = ROOT / "data/raw/sites" / name
    zdir = ROOT / "data/processed/sites" / name
    zdir.mkdir(parents=True, exist_ok=True)
    for g in pick:
        try:
            nc = download_granules([g], raw)[0]
            ds = load_masked_granule(nc, site["bbox"], site["dates"], pace["short_name"], pace["flags"]["invalid"])
            z = zdir / f"{nc.stem}.zarr"
            write_zarr(ds, z); ds.close()
            ds2, row = validate_zarr(z, cfg["qc"]["rrs_range"]); ds2.close()
            rows.append({"site": name, "granule": nc.stem, **row})
        except Exception as e:
            log.error("%s failed: %r", g, e)
out = ROOT / "data/processed/qc_summary_sites.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])) if rows else None
    if w: w.writeheader(); w.writerows(rows)
log.info("wrote %s (%d rows)", out, len(rows))
