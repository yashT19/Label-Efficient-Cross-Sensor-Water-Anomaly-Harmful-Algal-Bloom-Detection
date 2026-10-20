"""Command-line entry point for the config-driven PACE pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import earthaccess
import yaml

from src.catalogue import backfill_catalogue, build_catalogue, read_catalogue
from src.masking import load_masked_granule
from src.pace_io import download_granules, granule_label, login
from src.plots import plot_sample_spectra
from src.qc import summarize_granule, validate_zarr, write_qc_summary
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


def _configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _granule_id(nc_path: Path, catalogue_rows: list[dict[str, str]]) -> str:
    for row in catalogue_rows:
        if row["granule_id"] in nc_path.name:
            return row["granule_id"]
    return nc_path.stem


def _catalogue_targets(
    config: dict[str, Any], site_name: str, raw_dir: Path, catalogue_path: Path
) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve ranked catalogue rows to local files, downloading only selected rows."""
    rows = read_catalogue(catalogue_path)
    site_rows = sorted(
        [row for row in rows if row["site"] == site_name],
        key=lambda row: int(row.get("rank") or 999999),
    )
    limit = int(config["sites"][site_name].get(
        "n_granules", config["pace"]["n_granules_process"]
    ))
    site_rows = site_rows[:limit]
    existing: dict[str, Path] = {
        row["granule_id"]: path for row in raw_dir.glob("*.nc")
        for row in site_rows
        if row["granule_id"] in path.name
        and (
            not row.get("size_mb")
            or path.stat().st_size >= float(row["size_mb"]) * 1e6 * 0.99
        )
    }
    missing = [row for row in site_rows if row["granule_id"] not in existing]
    if missing:
        login()
        results = earthaccess.search_data(
            short_name=config["pace"]["short_name"],
            version=str(config["pace"]["version"]),
            bounding_box=tuple(config["sites"][site_name]["bbox"]),
            temporal=tuple(config["sites"][site_name]["dates"]),
            count=25,
        )
        selected = [
            granule for granule in results
            if any(granule_label(granule) == row["granule_id"] for row in missing)
        ]
        if len(selected) != len(missing):
            found = {granule_label(granule) for granule in selected}
            LOGGER.warning("Could not resolve catalogue granules: %s",
                           ", ".join(row["granule_id"] for row in missing
                                     if row["granule_id"] not in found))
        for path in download_granules(selected, raw_dir):
            for row in missing:
                if row["granule_id"] in path.name:
                    existing[row["granule_id"]] = path
    return [existing[row["granule_id"]] for row in site_rows if row["granule_id"] in existing], site_rows


def main() -> None:
    """Run the requested sensor pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--sensor", choices=["pace"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.site not in config["sites"]:
        raise ValueError(f"Unknown site: {args.site}")

    site = config["sites"][args.site]
    pace = config["pace"]
    paths = config["paths"]
    raw_dir = _configured_path(paths["raw"])
    zarr_dir = _configured_path(paths["zarr"])
    figure_dir = _configured_path(paths["figures"])
    catalogue_path = _configured_path(config["catalogue"]["out_csv"])
    qc_path = _configured_path(config["qc"]["summary_csv"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob("PACE_OCI.*.L2.OC_AOP*.nc"))
    lake_erie_baseline = args.site == "lake_erie" and len(existing) >= int(
        site.get("n_granules", pace["n_granules"])
    )
    if lake_erie_baseline:
        nc_files = existing[:int(site.get("n_granules", pace["n_granules"]))]
        catalogue_rows = read_catalogue(catalogue_path) if catalogue_path.exists() else []
        LOGGER.info("Using existing NetCDF downloads (skip Earthdata search)")
    else:
        if not catalogue_path.exists():
            build_catalogue(config, args.site)
        nc_files, catalogue_rows = _catalogue_targets(
            config, args.site, raw_dir, catalogue_path
        )

    qc_rows: list[dict[str, Any]] = []
    for nc_path in nc_files:
        LOGGER.info("Processing %s", nc_path.name)
        zarr_path = zarr_dir / f"{nc_path.stem}.zarr"
        manifest = manifest_path(zarr_dir, nc_path)
        granule_id = _granule_id(nc_path, catalogue_rows)
        if not args.force and zarr_path.exists() and manifest.exists():
            LOGGER.info("Skipping complete processed granule %s", nc_path.name)
            ds_r, _ = validate_zarr(zarr_path, config["qc"]["rrs_range"])
            qc_rows.append(summarize_granule(
                ds_r, args.site, granule_id, pace["flags"]["invalid"],
                config["qc"]["rrs_range"],
            ))
            ds_r.close()
            continue
        ds = load_masked_granule(
            nc_path, site["bbox"], site["dates"], pace["short_name"],
            pace["flags"]["invalid"],
        )
        qc_rows.append(summarize_granule(
            ds, args.site, granule_id, pace["flags"]["invalid"],
            config["qc"]["rrs_range"],
        ))
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
            manifest, nc_path, pace["version"],
            site["bbox"], pace["flags"]["invalid"],
        )
    write_qc_summary(qc_path, qc_rows)
    if catalogue_path.exists() and qc_rows:
        backfill_catalogue(catalogue_path, qc_rows)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
