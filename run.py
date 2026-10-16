"""Command-line entry point for the config-driven PACE pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.masking import load_masked_granule
from src.pace_io import download_granules, login, search_granules
from src.plots import plot_sample_spectra
from src.qc import validate_zarr
from src.zarr_schema import write_zarr

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML pipeline configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def manifest_path(zarr_dir: Path, nc_path: Path) -> Path:
    """Return the per-granule manifest path."""
    return zarr_dir / f"{nc_path.stem}.json"


def write_manifest(
    path: Path, nc_path: Path, version: str, bbox: list[float], flags: list[str]
) -> None:
    """Write the small reproducibility manifest for one granule."""
    payload = {
        "file_name": nc_path.name,
        "version": version,
        "roi": bbox,
        "flags_used": flags,
        "processing_date": date.today().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the requested sensor pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--sensor", choices=["pace"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.site not in config["sites"]:
        raise ValueError(f"Unknown site: {args.site}")

    site = config["sites"][args.site]
    pace = config["pace"]
    paths = config["paths"]
    raw_dir = ROOT / paths["raw"]
    zarr_dir = ROOT / paths["zarr"]
    figure_dir = ROOT / paths["figures"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob("PACE_OCI.*.L2.OC_AOP*.nc"))
    if len(existing) >= pace["n_granules"]:
        nc_files = existing[:pace["n_granules"]]
        LOGGER.info("Using existing NetCDF downloads (skip Earthdata search)")
    else:
        login()
        granules = search_granules(
            pace["short_name"], pace["version"], site["bbox"], site["dates"],
            pace["n_granules"],
        )
        nc_files = download_granules(granules, raw_dir)

    for nc_path in nc_files:
        LOGGER.info("Processing %s", nc_path.name)
        ds = load_masked_granule(
            nc_path, site["bbox"], site["dates"], pace["short_name"],
            pace["flags"]["invalid"],
        )
        zarr_path = zarr_dir / f"{nc_path.stem}.zarr"
        write_zarr(ds, zarr_path)
        ds.close()
        ds_r, _ = validate_zarr(zarr_path, config["qc"]["rrs_range"])
        plot = config["plots"]
        plot_sample_spectra(
            ds_r, figure_dir / f"{nc_path.stem}_sample_spectra.png",
            plot["n_spectra"], plot["seed"], plot["reference_wavelength_nm"],
            plot["marker_wavelengths_nm"],
        )
        ds_r.close()
        write_manifest(
            manifest_path(zarr_dir, nc_path), nc_path, pace["version"],
            site["bbox"], pace["flags"]["invalid"],
        )
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
