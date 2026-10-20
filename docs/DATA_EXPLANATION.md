# Dataset Explanation and Sharing Guide

## 1. Short explanation for a reviewer

This project is building a cross-sensor hyperspectral learning pipeline for
harmful algal bloom (HAB) detection. The current dataset is **not yet the final
HAB classification dataset**. It is the first, source-domain validation
dataset used to prove that the project can:

1. find NASA PACE OCI Level-2 aquatic optical property data;
2. download and read the NetCDF product correctly;
3. select a geographic and date window;
4. remove pixels with selected quality flags;
5. preserve the spectral bands and geolocation; and
6. write a reusable, analysis-ready Zarr store.

At present, the repository contains three PACE OCI `PACE_OCI_L2_AOP` granules
from western Lake Erie, collected between 10 and 19 August 2024. They provide
water-leaving remote-sensing reflectance (`Rrs`) across 172 spectral bands.
This is a deliberately small engineering and quality-control dataset, not
enough by itself to train or evaluate the final model.

The eventual dataset will add EnMAP observations, Indian water-body sites,
cross-sensor match-ups, spectral alignment, and carefully defined HAB labels or
proxy evidence. Those later components do not exist in the repository yet.

## 2. What is available now

### 2.1 Observation source

| Property | Current value |
|---|---|
| Sensor | NASA PACE OCI |
| Product | `PACE_OCI_L2_AOP` |
| Product version | `3.2` |
| Measurement | Water-leaving remote-sensing reflectance, `Rrs` |
| Reflectance units | `sr^-1` |
| Spectral coverage | Approximately 346--719 nm |
| Spectral bands | 172 bands |
| Approximate native pixel size | 1 km |
| Site | Western Lake Erie |
| ROI bbox (W, S, E, N) | `(-83.70, 41.20, -82.30, 42.10)` |
| Acquisition window | 2024-08-10 through 2024-08-19 |
| Processing state | Flag-masked, ROI-cropped, sanity-checked, and stored as Zarr |

The PACE data are the **source-domain data** for the planned transfer-learning
workflow. They are not EnMAP observations and are not being presented as
30-metre target-domain data.

### 2.2 Granules currently on disk

| Granule | Acquisition timestamp (UTC) | Valid water pixels | Valid fraction | Interpretation |
|---|---:|---:|---:|---|
| `PACE_OCI.20240810T180343.L2.OC_AOP.V3_2` | 2024-08-10 18:03:43 | 29 | 0.002343 | Very small usable yield |
| `PACE_OCI.20240811T165956.L2.OC_AOP.V3_2` | 2024-08-11 16:59:56 | 0 | 0.000000 | Fully flagged/cloudy after the current mask |
| `PACE_OCI.20240811T183815.L2.OC_AOP.V3_2` | 2024-08-11 18:38:15 | 246 | 0.051250 | Usable validation granule |

The valid-pixel count is a **quality-controlled pixel count**, not the number
of all pixels in the original satellite granule. The valid fraction is
calculated after the ROI crop and the configured flag mask. The zero-yield
granule is retained because it is useful evidence that the pipeline handles
cloudy or fully invalid observations without silently treating them as usable
training data.

### 2.3 Quality masking

The current mask invalidates a pixel if any of these PACE L2 flags is set:

`ATMFAIL`, `LAND`, `HIGLINT`, `HILT`, `HISATZEN`, `STRAYLIGHT`, `CLDICE`,
`COCCOLITH`, `HISOLZEN`, `LOWLW`, `CHLFAIL`, `NAVWARN`, `MAXAERITER`,
`CHLWARN`, `ATMWARN`, `NAVFAIL`, and `PRODFAIL`.

The current configuration deliberately does **not** mask `COASTZ` or
`TURBIDW`. These flags can be relevant in shallow, turbid water and HAB
scenes, so they are retained pending a documented decision. The QC output
records their potential removal as `would_remove_*` statistics rather than
removing those pixels.

The configured `Rrs` sanity-check range is `[-0.01, 0.10] sr^-1`. In the
current three-granule QC output, the out-of-range fraction is `0.0` for every
granule. This is a range check, not a HAB label.

### 2.4 What each stored output contains

The current data tree is:

```text
data/
  raw/
    pace_oci_l2_aop/
      *.nc                         # Original PACE OCI L2 AOP granules
  processed/
    pace_oci_l2_aop_zarr/
      *.zarr/                      # One processed store per granule
      *.json                       # Processing manifest per granule
    qc_summary.csv                 # Per-granule QC and flag statistics
```

Each Zarr store contains the processed spectral and geospatial arrays needed
for later work, including:

- `Rrs` values;
- the PACE quality flags;
- `valid_water`;
- latitude and longitude;
- wavelength metadata; and
- granule and processing attributes.

The exact schema will be migrated to the planned sensor-independent schema in
a later stage. The intended shared schema includes `refl(y, x, band)`,
`wavelength_nm(band)`, `fwhm_nm(band)` when available, `valid_water(y, x)`,
`band_valid(band)`, `lat`, and `lon`.

The JSON manifest records the source file name, product version, ROI, the
flags used, and the processing date. The CSV records valid-pixel counts,
valid fractions, out-of-range fractions, per-flag removal counts, and
would-remove counts for flags that are currently not masked.

## 3. What this dataset does and does not mean

### It does provide

- Real satellite hyperspectral water observations.
- A reproducible PACE OCI product/version, location, and date window.
- Pixel-level spectra with geolocation and quality information.
- A tested first step for later cross-sensor processing.
- Evidence that the pipeline preserves 172 PACE bands and handles a
  zero-valid-pixel granule explicitly.

### It does not provide yet

- EnMAP scenes or EnMAP spectral metadata.
- Indian sites such as Chilika, Pulicat, or Vembanad.
- A PACE--EnMAP spatial/temporal match-up table.
- Spectral-response-function alignment between sensors.
- Field measurements, toxin measurements, or pixel-level HAB labels.
- A balanced positive/negative classification dataset.
- A self-supervised model, embeddings, or model evaluation split.

The Lake Erie time window is evidence that the project selected a relevant
HAB-related event window, but it is **event-level context, not ground-truth
toxin or pixel-level bloom annotation**. A reviewer should therefore not
interpret every valid-water pixel as a HAB pixel.

## 4. What will be added in later stages

The following describes the planned dataset, not data currently available.
Items marked `TODO[VERIFY]` require confirmation from actual EnMAP products
or their documentation before they are treated as facts.

### Stage 2: More PACE coverage and catalogue/QC

- A metadata-only catalogue of PACE granules for Lake Erie and Indian site
  presets.
- More granules selected by ROI coverage rather than arbitrary search order.
- Additional sites: Chilika, Pulicat, Vembanad, and northern Arabian Sea.
- Per-granule QC summaries and catalogue fields such as coverage,
  valid-pixel yield, and dominant flags.
- Resumable processing for approximately 5--10 granules per site, subject to
  the configured limits and observed data quality.

### Stage 3: EnMAP inventory and spectral specifications

- PACE band-centre and width specifications generated from actual metadata.
- EnMAP band centres and FWHM parsed from a real EnMAP XML product.
- EnMAP scene discovery through the DLR STAC service.
- Scene footprints, ROI overlap, acquisition times, available modes, and
  quality properties read from the actual STAC item properties.
- The specific EnMAP STAC collection, product format, nodata value, and scale
  factor remain `TODO[VERIFY]` until confirmed from the supplied products and
  metadata.

The STAC land-mode L2A inventory is planned for discovery and metadata. The
spectra used for training are expected to come from EOWEB water/combined
orders, subject to product availability and verification.

### Stage 4: Common schema and match-ups

- EnMAP water/combined-order scenes read into the shared Zarr schema.
- EnMAP water, cloud, cirrus, haze, shadow, and sun-glint masks, using the
  masks actually supplied by the product.
- A sensor-independent `refl` array: PACE `Rrs` on the PACE side and
  converted EnMAP water reflectance on the EnMAP side.
- PACE--EnMAP match-up records containing scene IDs, acquisition time gap,
  overlap area, and PACE valid-pixel yield.
- The EnMAP reflectance-to-`Rrs` conversion, including the possible
  `rho_w / pi` relationship, must be verified against the EnMAP water ATBD
  before being used.

### Stage 5: Spectral alignment and model-ready data

- Spectral-response-function (SRF) alignment from PACE bands to EnMAP bands.
- A configurable common spectral grid, with the default grid to be confirmed
  by configuration.
- Band-validity masks for wavelengths outside the measured overlap.
- Before/after alignment error metrics and diagnostic plots.
- Larger multi-site, multi-date arrays suitable for self-supervised
  representation learning.
- Carefully documented labels or weak/institutional/event-level evidence for
  downstream HAB evaluation, if obtained.

The eventual training set should retain provenance for every sample: source
sensor, scene/granule ID, acquisition time, coordinates, wavelength grid,
masking rules, processing version, and any label/evidence source. This is
important because aligned spectra from different sensors must not be treated
as independent observations without accounting for their shared scene or
match-up.

## 5. How to explain the project in a review meeting

Use the following concise explanation:

> “At this stage, our dataset is a validated PACE OCI source-domain dataset,
> not the final HAB-labelled dataset. We have three Level-2 aquatic optical
> property granules over western Lake Erie from August 2024. Each has 172
> `Rrs` bands, geolocation, quality flags, and a documented ROI/masking
> process. Two granules contain usable water pixels, with 29 and 246 valid
> pixels; one is fully flagged and yields zero valid pixels. We store both the
> original NetCDFs and processed per-granule Zarr stores, plus QC summaries and
> manifests. The next stages add more PACE sites, EnMAP scenes, cross-sensor
> match-ups, spectral alignment, and eventually labels or proxy evidence for
> HAB evaluation. We are explicitly not claiming that the current Lake Erie
> pixels are toxin ground truth.”

If asked why the current set is small, explain that it is an end-to-end
validation slice. It verifies the acquisition, masking, spectral metadata,
geolocation, storage, and QC decisions before scaling to more sites and
before introducing the additional uncertainty of EnMAP calibration and
cross-sensor alignment.

## 6. How to share the data with a reviewer

### Recommended approach

Share the data package separately from the source-code repository:

1. Share the repository link and this document.
2. Provide the three original NetCDF files and the processed Zarr directory
   through an institution-approved file service or research data repository.
3. Include `data/processed/qc_summary.csv` and all three JSON manifests.
4. Include the two generated sample-spectrum PNGs if the reviewer wants a
   quick visual check.
5. Provide a checksum file (for example, SHA-256) for the archive so the
   reviewer can verify that the download was not changed.
6. Tell the reviewer to reproduce the processing with:

   ```powershell
   cd "C:\Users\tippa\BTP Project"
   python run.py --site lake_erie --sensor pace
   ```

The raw NetCDFs and generated Zarr stores are intentionally ignored by Git
because they are large and reproducible. A reviewer who has NASA Earthdata
access can regenerate the raw files using the documented product, version,
ROI, and date window. Do not send Earthdata credentials, `.netrc` files,
cookies, or any other local authentication files.

### Suggested archive layout

Create an archive outside the Git working tree with this structure:

```text
pace_lake_erie_validation/
  README.md                         # Optional handout for the reviewer
  data/
    raw/pace_oci_l2_aop/*.nc
    processed/pace_oci_l2_aop_zarr/*.zarr/**
    processed/pace_oci_l2_aop_zarr/*.json
    processed/qc_summary.csv
  figures/
    *_sample_spectra.png
  SHA256SUMS.txt
```

Zarr is a directory-based format, so the archive must include the complete
Zarr directory trees rather than only a single file. If the file-sharing
service supports folders, uploading the Zarr directory directly is also
acceptable.

### Data access and provenance

The raw observations originate from NASA's Earthdata/OB.DAAC distribution of
PACE OCI products. The reviewer should check the applicable NASA data-use
terms and cite the product documentation. This repository should contain
processing code and metadata, not credentials or private account material.

When later EnMAP data are added, their provider-specific access conditions,
licence, attribution requirements, and redistribution permissions must be
checked before sharing the actual scenes. If redistribution is not permitted,
share the scene identifiers, metadata, processing code, and instructions for
authorized reviewers to obtain the data themselves.

## 7. Reproducibility checklist for reviewers

- [ ] Product is `PACE_OCI_L2_AOP`, version `3.2`.
- [ ] Site is western Lake Erie with bbox
      `(-83.70, 41.20, -82.30, 42.10)`.
- [ ] Date window is 2024-08-10 through 2024-08-19.
- [ ] The 17 configured invalid flags are documented above.
- [ ] `COASTZ` and `TURBIDW` are not currently masked.
- [ ] The expected PACE band count is 172.
- [ ] The expected valid-pixel counts are 29, 0, and 246 in granule order.
- [ ] QC summary and JSON manifests accompany the processed data.
- [ ] No EnMAP, Indian-site, match-up, or ground-truth label claims are made
      for the current package.
- [ ] Checksums match the supplied archive.
