# Cross-sensor hyperspectral SSL for HAB detection

Final-year BTP: **cross-sensor hyperspectral self-supervised learning for harmful algal bloom (HAB) detection**.

- **Source sensor (data-rich):** NASA PACE OCI
- **Target sensor (data-scarce):** ISRO HysIS VNIR

High-level goal: learn representations on abundant PACE OCI spectra and transfer them to HysIS, where labelled HAB data are scarce.

The proposal’s Module 1 is **Acquire, Calibrate & Spectrally Align**. This repo currently implements only the first slice of that: **PACE OCI, small scale**, to confirm search → mask → Zarr → plot before adding HysIS or scaling.

Proposal PDF (not in this repo): `c:\Users\tippa\Desktop\BTP material\BTP_Proposal_HAB_Hyperspectral_v2.pdf`

---

## Status

**Done:** Module 1 first pass for PACE OCI. Three `PACE_OCI_L2_AOP` granules are on disk, flag-masked, cropped to a western Lake Erie ROI, written to Zarr (with wavelength metadata), sanity-checked, and plotted.

**Not started:** HysIS, SRF / spectral alignment, SSL training, Indian target sites.

---

## Setup and run

Needs a free [NASA Earthdata](https://urs.earthdata.nasa.gov) account. Use the **username**, not email, unless they are the same. Credentials are stored locally via `earthaccess.login(persist=True)` (`.netrc` / `_netrc`), never in this repo.

```powershell
cd "c:\Users\tippa\BTP Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python module1_pace_oci_validation.py
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If at least three matching NetCDFs already exist under `data/raw/pace_oci_l2_aop/`, the script **skips Earthdata** and reprocesses those files.

Zarr is a **local** chunked-array format (a folder of files). No Zarr account.

---

## Layout

| Path | Contents |
|---|---|
| `module1_pace_oci_validation.py` | PACE OCI acquire / mask / Zarr / plot script |
| `requirements.txt` | Python dependencies |
| `data/raw/pace_oci_l2_aop/` | Downloaded L2 AOP NetCDFs |
| `data/processed/pace_oci_l2_aop_zarr/` | One Zarr store per granule |
| `figures/` | Sample valid-water spectra PNGs |
| `.venv/` | Local virtualenv (do not commit) |

---

## Pipeline

`python module1_pace_oci_validation.py`

1. Login to Earthdata only if fewer than three NetCDFs are already downloaded.
2. Search `PACE_OCI_L2_AOP` over the ROI and scene window; keep the first three granules.
3. Open NetCDF groups: `geophysical_data` (`Rrs`, `l2_flags`), `navigation_data` (lat/lon), `sensor_band_parameters`.
4. `mask_and_scale=True` so packed `Rrs` becomes physical sr⁻¹.
5. Mask pixels where any selected flag bit is set; crop to the ROI.
6. Write Zarr with `Rrs`, `l2_flags`, `valid_water`, lat/lon, wavelengths, and granule attrs.
7. Re-open Zarr and check shape, 172 bands, no finite `Rrs` on invalid pixels, value-range fraction, and whether any valid water remains.
8. Save sample-spectra PNGs under `figures/`.

### Dependencies

| Package | Role |
|---|---|
| `earthaccess` | NASA Earthdata search and download |
| `xarray` | Open NetCDF groups, mask, write/read Zarr |
| `zarr` | On-disk chunked array stores |
| `netCDF4` | Primary NetCDF backend |
| `h5netcdf` | Alternate NetCDF-4/HDF5 backend |
| `matplotlib` | Spectrum plots |
| `numpy` | Flag bitmasks, sampling, stats |
| `dask` | Chunked write/read with xarray |

---

## Scene and processing results

**Scene:** western Lake Erie, `2024-08-10`–`2024-08-19`, bbox `(-83.70, 41.20, -82.30, 42.10)`. Product version on disk is **V3.2**.

| Granule | Valid water pixels | Plot |
|---|---|---|
| `PACE_OCI.20240810T180343.L2.OC_AOP.V3_2` | 29 | yes |
| `PACE_OCI.20240811T165956.L2.OC_AOP.V3_2` | 0 (cloudy / fully flagged) | no |
| `PACE_OCI.20240811T183815.L2.OC_AOP.V3_2` | 246 | yes |

On the two usable granules, spectra look like water: green peak near ~550 nm, drop toward red, `Rrs` roughly **0.002–0.02 sr⁻¹**. Band count **172**, wavelength ~**346–719 nm** (L2 AOP Rrs, not the full OCI L1 range). Zarr exists for all three granules.

---

## Decisions

| Topic | Choice | Why |
|---|---|---|
| First scene | Western **Lake Erie**, bbox above | Inland optically complex water; NOAA NCCOS 2024 CyanoHAB peak 10-day window (severity index 6.6); PACE OCI already used on 2024 Lake Erie blooms. |
| Role of this scene | **Source-domain validation**, not an Indian HysIS target | Indian sites come later in Module 1. |
| Scene window | `2024-08-10` to `2024-08-19` | Matches that NOAA peak window. |
| Product | `PACE_OCI_L2_AOP` (science, not NRT) | Standard L2 AOP; `Rrs` after NASA atmospheric correction (`l2gen`). |
| Processing version | **Not pinned** in search | Earthdata served V3.2. Pin `version="3.2"` later if results must be frozen. |
| Granule count | **3** | Small end-to-end check. |
| Granule ranking | **First N search hits**, not cloud-cover ranked | PACE L2 UMM cloud-cover is often missing. |
| Code shape | **One commented Python script** | First validation pass, not a production pipeline. |
| Environment | Project **`.venv`** | Keep the NASA/xarray stack off system Python. |
| Credentials | `earthaccess.login(persist=True)` on this machine only | Never store passwords in the repo. |
| Output | **One Zarr store per granule** | Chunked arrays for later ML; keep band-center wavelengths. |
| Crop | Bounding-box **index window** on the L2 swath; pixels outside the geographic ROI marked invalid | L2 is a swath, not a regular lat/lon grid. |
| Invalid L2 flags | NASA **L2 Mask Default = ON** bits **plus `PRODFAIL`** | Product failure should not enter train/val pixels. Stricter than some AOP examples (`ATMFAIL,LAND,CLDICE,HILT`). Source: [OBPG L2 flags](https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/). |
| Flags **not** masked | `COASTZ`, `TURBIDW`, plus `PRODWARN`, `MODGLINT`, `ABSAER`, `SEAICE`, `HIPOL`, `FILTER`, `BOWTIEDEL` | Shallow/turbid flags are often true in HAB / western Lake Erie. Review before scaling. |
| `Rrs` range check | **[-0.01, 0.10] sr⁻¹** | From packed valid_min/max. Typical water ~0–0.05; bloom/turbid can be higher in green; small negatives from noise/AC. |
| Spectral axis (V3.2) | Use `Rrs` **`wavelength` in nm**. Keep integer-nm `wavelength_3d` as `wavelength_3d_nm` when lengths match. | Older CDL used dim `wavelength_3d` on `Rrs`; V3.2 files already have `wavelength`. |
| Band-count check | **172** bands, covers chl/PC (`min < 400` nm and `max > 680` nm) | L2 AOP Rrs is ~346–719 nm, not full OCI L1 (~350–890 nm). |
| Plot | ~20 random valid-water spectra + mean; seed **42**; usable if finite at **443 nm**; lines at 443, 615, 670 nm | Visual water vs noise. |
| Re-download | Skip if ≥3 NetCDFs already in `data/raw/pace_oci_l2_aop/` | Avoid hitting Earthdata again. |

### Invalid flag list

`ATMFAIL`, `LAND`, `HIGLINT`, `HILT`, `HISATZEN`, `STRAYLIGHT`, `CLDICE`, `COCCOLITH`, `HISOLZEN`, `LOWLW`, `CHLFAIL`, `NAVWARN`, `MAXAERITER`, `CHLWARN`, `ATMWARN`, `NAVFAIL`, `PRODFAIL`.

---

## Out of scope (for now)

- HysIS ingest, calibration, or SRF / wavelength alignment to PACE
- Indian HAB sites as target-domain scenes
- Ranking granules by cloud cover or bloom intensity
- Pinning processing version in `earthaccess.search_data`
- SSL / embedding model, labels, or toxin ground truth (Lake Erie is **event-level** evidence, not toxin GT)
- Scaling beyond three granules or a production pipeline

## Before scaling

- Re-check the invalid-flag list against current PACE / `l2gen` docs (especially `COASTZ` / `TURBIDW`).
- Keep or replace the low-yield / fully flagged granules if more valid water is needed.
- Pin `processing_version` when Module 1 results must be reproducible for the thesis.
- Add Indian water bodies when Module 1 moves past PACE-only validation.
