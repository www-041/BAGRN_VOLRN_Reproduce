# Spectralmatch R1 compatibility review (read-only)

Review date: 2026-10-11. Evidence is limited to the official public GitHub README/`pyproject.toml`, release page, and JOSS record; no clone, install, import, or algorithm run was performed.

## Project and method

The official README describes a Python library/CLI/QGIS plugin for relative radiometric normalization of mosaics and time series, supporting arbitrarily many images and bands. Its workflow is joint georeferencing/coregistration, `Match.global_regression`, `Match.local_block_adjustment`, then seamline/mask/merge. It assumes known rough geotransforms but says strict pixel co-registration is not required; its methods seek scale/offset harmonization and use overlap consistency/PIF options. The published JOSS article is Lindiwe et al. (2026), JOSS 11(117), 8974, DOI 10.21105/joss.08974.

The README presents a sensor/unit-agnostic workflow and multi-band support. This is a promising candidate for a full-pipeline R2 comparison, but not evidence that its assumptions hold for these 13 scenes. In particular, its assumption of a consistent spectral profile across overlap may be violated by real vegetation/seasonal change. R1's independently labeled `REAL_CHANGE` regions are therefore required before making a change-preservation claim. Exact per-band raster input behavior, chosen model-image semantics, output-grid/resampling, mask handling, local-block scale, and version-specific output contracts remain to be checked in an isolated R2 environment.

## Version, license, dependencies, fit with current environment

| Item | Official evidence / current R1 environment | Assessment |
|---|---|---|
| Current repository metadata | `main` `pyproject.toml` says package `spectralmatch` 1.6.0; Python `>=3.10`; MIT license | Main metadata is ahead of the latest tagged release observed below. Pin a commit/tag before R2. |
| Latest tagged release observed | GitHub Releases lists v1.4.1 (released 2026-08-13); no code fetched | Prefer a pinned released tag for first reproducible R2 integration, then separately evaluate newer main if desired. |
| Software license | Project repository and pyproject identify MIT | Candidate software license is permissive; transitive dependency/data licenses still require review. JOSS article itself is CC BY 4.0, a separate work. |
| Python | Current `.venv-registration`: 3.11.6 | Within README's 3.10–3.12 range. |
| Python dependencies at `main` | NumPy ≥2.2.3, SciPy ≥1.15.2, Fiona ≥1.10.1, `omnicloudmask` ≥1.0.11, GeoPandas ≥1.0.1, Fire ≥0.7.0, Matplotlib ≥3.8, OpenCV headless ≥4.10 | Current environment has NumPy 2.4.6/SciPy 1.17.1, but GeoPandas, Fiona, omnicloudmask, Fire, and OpenCV headless are not installed. Do not add them to the existing environment. |
| System GIS requirements | README installation instructions call for PROJ ≥9.3 and GDAL ≥3.11 | Current Rasterio is linked to GDAL 3.10.3; `pyproj`/PROJ and standalone GDAL Python bindings are unavailable in this environment. Not ready for installation/run here. |
| Training | Public API/readme presents image-statistics, regression, PIF, block adjustment and masking workflow; no trained weights or model-training stage is documented | Appears training-free from published API; confirm exact pinned-source execution path in isolated R2 compatibility tests. |
| Local/global operation | Global regression and local block adjustment are documented; joint coregistration can use a global correction and optional smooth local displacement mesh | Not directly interchangeable with a fixed-geometry local-module ablation. Separate full-pipeline comparison; record geometric preprocessing and resampling. |
| Multiband/time series | README explicitly describes arbitrary image/band scale and time-series workflows | Potentially suitable after per-band input and output-grid behavior is verified. Do not infer physical reflectance compatibility from “unit-agnostic.” |

## R1 decision

**Priority: secondary full-pipeline candidate for R2, not an assumed final method and not ready to run now.** Do not install into `.venv-registration`. Before R2: create a fresh isolated environment; pin an immutable source tag/commit; record dependency lock and license inventory; verify a tiny synthetic smoke, then a bounded sample that proves input band mapping, masks, output transform, nodata, local adjustment and timing. Keep the project’s geometry/resampling path visible as part of the full-pipeline comparison; do not call it a pure radiometric-only ablation if its joint coregistration changes geometry.

## Official sources

- [Official GitHub README](https://github.com/spectralmatch/spectralmatch)
- [Official main `pyproject.toml`](https://github.com/spectralmatch/spectralmatch/blob/main/pyproject.toml)
- [Official GitHub releases](https://github.com/spectralmatch/spectralmatch/releases)
- [JOSS paper record](https://joss.theoj.org/papers/10.21105/joss.08974)
