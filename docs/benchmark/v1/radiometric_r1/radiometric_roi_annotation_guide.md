# Radiometric R1 ROI annotation guide

## Safety and scope

This tool is isolated from Task15/Task16 and does not edit source GeoTIFFs. Candidate polygons are navigation/review proposals only. Every proposal starts as `label=UNKNOWN` and `review_status=UNVERIFIED`; no matching score, metadata, date, or mask fraction approves a stable/changed label. Display percentile stretching is transient and never changes source samples.

The current ten-pair shortlist is a draft, not a final scientific selection. It contains the required 00_01, 00_10, 02_10, and 04_11 pairs. Three selected pairs without a pre-existing polygon use a bounding-box-only proposal (`BBOX_INTERSECTION_UNVERIFIED`). It is not evidence of valid pixels. The GUI calculates the valid fractions from both raster masks when an ROI is selected.

Some existing GeoJSON candidate records declare EPSG:32650 while their coordinates numerically match the scenes' geographic extents. The R1 builder records this discrepancy and only interprets them as EPSG:4326 when the polygon intersects both scene geographic metadata extents. This is recorded as `source_crs_resolution=EPSG:4326_CONFIRMED_BY_SCENE_EXTENTS`; unsupported conflicts fail closed.

The broad candidate proposal can be `UNASSIGNED` under the frozen Phase3 rule because it crosses grid margins. This does not mean all small ROIs in it are unassignable. Draw a smaller polygon/rectangle around a reviewable region; the GUI applies the unchanged global EPSG:32650 10 km grid, origin `(0,0)`, 500 m safety margin, and role formula. It never changes the formula. Cross-pair overlap audit remains `PENDING_PAIRWISE_AUDIT` until separately audited, so a human-approved label is not automatically formal TEST eligible.

## Start in PowerShell

Run from the repository root; paths are script-rooted, so the current directory is not used to locate the input data:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\prepare_radiometric_r1_rois.py
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --check-env
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --validate-pair --pair 00_01
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 00_01 --reviewer "你的真实审核身份"
```

Use a real identity, not a generated username. Supported current shortlist IDs are `00_01`, `10_11`, `04_06`, `06_08`, `02_03`, `02_08`, `00_10`, `02_10`, `04_11`, and `06_11`.

## GUI workflow

1. Select an unverified candidate on the right. The two B9 windows center on its geospatial location and use the same projected viewport. Zoom/pan controls affect both views.
2. Use the proposal as a navigation aid, then draw a smaller review ROI: click **New polygon**, click vertices on either panel, and click **Finish polygon**; or click **New rectangle (drag)** and drag on either panel. A polygon/rectangle uses the shared map grid, not equal native row/column indices.
3. Click **Save ROI** to save the geometry as `UNKNOWN/UNVERIFIED`. To revise an existing saved ROI, select it, click **Edit selected geometry**, then use New polygon or New rectangle and Save ROI. Geometry changes invalidate prior approvals, retain the old snapshot in the JSONL audit log, and require fresh review.
4. Choose the visual class (`STABLE`, `REAL_CHANGE`, `CLOUD`, `SHADOW`, or `UNKNOWN`), purpose, and review state. `UNKNOWN` can remain unreviewed. Before `APPROVED`, provide evidence notes and confirm the human-only prompt. Matching score or valid fraction is not label evidence by itself.
5. **Apply human review** commits label/status/purpose and reviewer/time. It never assigns TEST eligibility by itself. **Delete saved ROI** requires confirmation and appends a delete event; original candidate GeoJSON remains unchanged.
6. For independent TEST review, another real person launches the same pair with `--second-review`. The first class/status/evidence are hidden in the review controls. The second reviewer records a fresh class, evidence, and result. Different reviewer identity alone is not proof of independent human work. A class disagreement cannot be approved; record `DOUBTFUL` and resolve manually.

## Saved files

- Candidate proposals and their provenance: `docs/benchmark/v1/radiometric_r1/radiometric_roi_candidates.geojson` (read-only during annotation).
- Blank schema/example rows: `docs/benchmark/v1/radiometric_r1/radiometric_roi_review_template.csv` (not the live store).
- Live human annotations: `data/output/benchmark_v1_radiometric_r1/roi_reviews/<PAIR>_review.csv`.
- Append-only before/after review history: `data/output/benchmark_v1_radiometric_r1/roi_reviews/<PAIR>_review_events.jsonl`.

Each saved ROI includes source file SHA256s from the existing B9 manifest, source pair/quality/time metadata, projected geometry, spatial role/status, valid-mask fractions, reviewer/time, and current decision. Candidate-level matching quality is not copied as a precision estimate. The output directory is new and does not overwrite prior benchmark records.
