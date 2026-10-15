# R1.5 multiband visual review

This opt-in GUI adds visual aids and a separate schema-v2 label log. It does not
edit source rasters, frozen candidate geometry/roles, legacy R1/R1.5 review
records, or radiometric outputs. Candidate scores are not accuracy, and opening
the tool does not approve any candidate.

## Open a pair

From the repository root, use the existing registration environment:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 05_07 --reviewer "REAL_REVIEWER_NAME" --auto-candidates data\output\benchmark_v1_radiometric_r1_5\auto_roi_candidates\r15_20261012_04\05_07.geojson --multiband-review
```

Replace the reviewer placeholder with the real human reviewer. Repeat with the
pair and matching GeoJSON path for `00_01`, `02_03`, or `08_10`. Do not run the
candidate generator to use the GUI. Header/inventory preflight, without opening
the GUI, is:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 05_07 --validate-pair --auto-candidates data\output\benchmark_v1_radiometric_r1_5\auto_roi_candidates\r15_20261012_04\05_07.geojson --multiband-review
```

The new independent records are written under
`data/output/benchmark_v1_radiometric_r1_5/multiband_reviews/` as
`<PAIR>_snapshot_v2.json` and append-only `<PAIR>_events_v2.jsonl`. The legacy
`reviews/` directory is read-only in this mode. A second reviewer must be a
different real person; launch with `--second-review` for that review.

## Display and interpretation

| View | Channel mapping / behavior |
|---|---|
| `GRAY` | Select one band from the shared available-band list. |
| `TRUE_COLOR` | B7/B4/B3 → R/G/B (633.5/530.5/490 nm per inventory). |
| `NIR_FALSE_COLOR` | B14/B7/B4 → R/G/B (850.5/633.5/530.5 nm per inventory). |
| Stretch `INDEPENDENT` | Each date/channel uses its own valid-window 2nd–98th percentile range. |
| Stretch `SHARED` | Both dates use a pooled valid-window 2nd–98th percentile range per channel. |
| `SIDE_BY_SIDE` | Separate date windows. |
| `BLINK` | Alternates the two dates in the viewer. |
| `DIFFERENCE` | Signed raw `scene_i - scene_j` values on the common-valid mask; red is positive, blue negative. |

The display and difference map scene-j to the scene-i EPSG:32650 affine grid with
display-only bilinear resampling. Differences use the intersection of both
valid-data masks. This is a visual aid, not calibration or change detection:
cross-scene geometry has not been independently established, and apparent
differences may arise from misregistration, view/illumination conditions, or
unknown-unit DN values. The interface retains this warning. No suitable DEM was
found in the checked project data, so terrain is visual interpretation only;
there is no slope/elevation validation.

ROI statistics show per-date valid fraction, median, MAD, and standard deviation
for B9 and the selected band, using bounded raster windows and raw source values.
They are not physical reflectance statistics.

## New labels and review safeguards

Schema v2 fields are separate from the legacy `human_label` and review events:

- `landcover_type`: `VEGETATION/BARE_GROUND/WATER/BUILT/MIXED/OTHER/UNKNOWN`
- `terrain_type`: `RIDGE/SLOPE/VALLEY/FLAT/OTHER/UNKNOWN`
- `temporal_status`: `NO_CHANGE/POSSIBLE_CHANGE/CONFIRMED_CHANGE/UNCERTAIN/UNKNOWN`
- `interference_type`: `NONE/CLOUD/CLOUD_SHADOW/TERRAIN_SHADOW/HAZE/SATURATION/OTHER/UNKNOWN`
- independent review decision: `UNVERIFIED/APPROVED/DOUBTFUL/REJECTED`

Legacy labels/history are displayed read-only and are never mapped to schema-v2
approval. `NO_CHANGE` means only that no definite change was observed at the
available native review scale; it is not formal `STABLE` truth. `CONFIRMED_CHANGE`
requires auditable evidence including `source:` and `reference:`. A radiometric
difference alone is insufficient. `APPROVED` also requires reviewer evidence;
the software never infers approval from visualization or statistics. TEST still
requires an independent second real reviewer and does not become score-eligible
from this GUI alone.

After editing geometry, the tool recomputes valid support and spatial/leakage
checks. It retains the original candidate identity and frozen role; an unsafe
edit becomes ineligible and is not reassigned. Geometry edits clear earlier
eligibility and approval state.

Only the separate schema-v2 snapshot/event files are written by this mode. Do not
manually edit candidate GeoJSON/CSV or legacy review files to make a candidate
eligible.
