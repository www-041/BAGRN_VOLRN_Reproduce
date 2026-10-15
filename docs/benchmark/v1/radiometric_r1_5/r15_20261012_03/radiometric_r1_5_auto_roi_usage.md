# R1.5 automatic ROI review — usage

Run ID: `r15_20261012_03`. The candidate GeoJSON/CSV files are immutable and separate from the R1 review store.
All suggestions begin `UNVERIFIED`; human review snapshot and JSONL history are stored under
`data/output/benchmark_v1_radiometric_r1_5/reviews/`. Units remain UNKNOWN.

Generate the first pair (already completed for this run):

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\generate_radiometric_r1_5_auto_rois.py --pair 00_01 --max-candidates 15 --run-id r15_20261012_03
```

Append the remaining pairs with the same config and thresholds:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\generate_radiometric_r1_5_auto_rois.py --pairs 02_03 05_07 08_10 --max-candidates 15 --run-id r15_20261012_03
```

Open a pair for first human review (replace the reviewer placeholder with a real person):

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 00_01 --reviewer "真实审核人" --auto-candidates data\output\benchmark_v1_radiometric_r1_5\auto_roi_candidates\r15_20261012_03\00_01.geojson
```

Independent TEST review uses a different real person and the explicit second-review mode:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 00_01 --reviewer "第二审核人" --second-review --auto-candidates data\output\benchmark_v1_radiometric_r1_5\auto_roi_candidates\r15_20261012_03\00_01.geojson
```

Review steps: inspect both original-resolution image windows; select one candidate; use zoom controls or the band selector; optionally edit EPSG:32650 WKT and apply (this clears prior approval and recalculates per-scene/per-band mask support and frozen role); choose a human label/status; enter visual evidence; apply. An approval is never inferred from the score. TEST decisions require a distinct reviewer; disagreements remain unresolved. Save events are append-only and can be resumed by reopening the same candidate file.

The 32×32 window and strong-edge screen can under-sample narrow/boundary-rich, fragmented, low-texture, shadowed or low-contrast terrain. Do not weaken thresholds to fill a quota. Empty/short candidate sets are valid outputs.
