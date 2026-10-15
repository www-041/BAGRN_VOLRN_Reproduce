# R1.5 machine label suggestions

Run `r15m_20261012_01` contains suggestions for the existing immutable candidate
run `r15_20261012_04` only. The machine artifacts are separate from both the
candidate GeoJSON/CSV and schema-v2 human review snapshots/events. They contain
no reviewer identity, human timestamp, approval event, or score-ready truth.

## What this run found

All 34 candidates have complete per-band bounded-window evidence for the
available B1–B16 inputs, with raw values explicitly marked `UNKNOWN_RAW_DN`.
No semantic class could be supported safely with the current evidence. Therefore
all four fields remain `UNKNOWN` for all candidates:

- Land cover confidence: `LOW` — no validated classifier/reference, unknown DN
  units; no reflectance indices were calculated.
- Terrain confidence: `MEDIUM` for withholding a class — no DEM, and local
  texture cannot validate ridge/slope/valley.
- Temporal status confidence: `MEDIUM` for withholding a class — raw-DN
  differences are recorded but cannot establish change; geometry is not
  independently verified and no independent change reference exists.
- Interference confidence: `MEDIUM` for withholding a class — no quality/cloud
  mask; shadow cannot safely be distinguished from water or dark vegetation.

These confidence values describe the conservative recommendation/decision to
withhold a semantic class. They are **not** class probabilities or measured
classification accuracy. This run intentionally did not use an external model,
Task15/16 outputs, BAGRN, Ours, VOLRN, Spectralmatch, or classifier-training
labels.

The report includes a deterministic 12-ROI human inspection sample balanced by
pair and frozen role (where available); it does not use candidate score or
algorithm performance. Since all suggestions are UNKNOWN, it cannot be balanced
by predicted semantic class. The existing fixed-window/strong-edge candidate
screen may under-sample low-texture, narrow-boundary, homogeneous, or shadowed
regions. No preview images were generated; the existing GUI displays the
underlying windows on demand.

## Files

- `data/output/benchmark_v1_radiometric_r1_5/machine_label_suggestions/r15m_20261012_01/`: per-pair suggestion GeoJSON/CSV and provenance manifest.
- `docs/benchmark/v1/radiometric_r1_5/machine_suggestions/r15m_20261012_01/machine_suggestion_quality_report.md`: label/confidence distributions and limitations.
- `docs/benchmark/v1/radiometric_r1_5/machine_suggestions/r15m_20261012_01/human_review_sample.csv`: pair/role-balanced manual sample.

Each machine record has its own label, LOW/MEDIUM confidence, evidence list,
uncertainty reasons, missing auxiliary data, measured per-band raw-DN statistics,
common-valid support, normalized within-band gradient texture, and the median
scene-i minus scene-j raw-DN difference. Scene-j is display/evidence-mapped to
scene-i's B9 EPSG:32650 grid with bilinear value resampling and nearest-neighbor
validity masks; this does not verify geometric registration.

## Run the batch generator

The completed batch command was:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\suggest_radiometric_r1_5_labels.py --candidate-run-id r15_20261012_04 --run-id r15m_20261012_01
```

Output paths are immutable: the tool refuses to reuse an existing machine run
ID. To open a pair with machine recommendations loaded as editable GUI drafts:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 05_07 --reviewer "REAL_REVIEWER_NAME" --auto-candidates data\output\benchmark_v1_radiometric_r1_5\auto_roi_candidates\r15_20261012_04\05_07.geojson --multiband-review --machine-suggestions data\output\benchmark_v1_radiometric_r1_5\machine_label_suggestions\r15m_20261012_01\05_07.machine_suggestions.geojson
```

The GUI checks exact candidate identities and source-candidate SHA256 before
loading. Machine labels prefill only still-UNKNOWN, unreviewed fields. They can
be changed to `UNKNOWN` or another allowed value. The review decision remains
`UNVERIFIED`; loading suggestions writes no human snapshot/event. An explicit
human review action is required to create a human record. TEST still requires a
different real second reviewer.

## Agreement summary after human inspection

The current run has no machine-vs-human agreement result because no eligible
human labels were created by this task. Later, after review, query it with:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\summarize_radiometric_r1_5_machine_agreement.py --machine-run-id r15m_20261012_01
```

Only explicitly approved human labels are compared; TEST labels also need an
approved independent second review with no dispute. The output is descriptive
agreement, **not** classifier accuracy or validated ground truth.

Do not expand to the ten R1 radiation pairs yet. First manually inspect the
sample and decide whether further data (calibrated units, DEM, quality masks,
independent land-cover/change reference, or verified geometry) is available.
