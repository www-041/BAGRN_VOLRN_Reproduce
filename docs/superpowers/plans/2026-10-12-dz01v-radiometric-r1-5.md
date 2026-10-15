# DZ01V Radiometric Benchmark R1.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate auditable, spatially distributed automatic ROI proposals for four DZ01V V-sensor pairs and review them through an opt-in human GUI without modifying frozen R1 or earlier artifacts.

**Architecture:** Isolate generation, spatial audit, and review state in `src/benchmark_v1_radiometric_r15/`; reuse R1 geospatial helpers where safe. Add an opt-in branch to the existing annotation CLI while keeping existing R1 inputs/outputs unchanged. Generate and validate `00_01` first; then process the remaining three pairs with identical frozen parameters.

**Tech Stack:** Python 3.11, Rasterio/GDAL, NumPy, SciPy, Shapely, Tkinter/Pillow, pytest; no heavyweight dependencies or models.

**Spec:** `docs/superpowers/specs/2026-10-12-dz01v-radiometric-r1-5-design.md`

## Global Constraints

- Only `00_01`, `02_03`, `05_07`, `08_10`; these are tool-development pairs, not a replacement for R1's ten-pair shortlist.
- Read inventory-identified B1–B16 files; B9 is mandatory and defines the initial geometry/reference browsing grid.
- Keep imagery, Phase3–5, Task15/16, R1 candidate/review data and all existing formal outputs read-only.
- Read bounded windows only; never load full scenes or run radiometric algorithms/formal evaluation.
- Every automatic suggestion starts `UNVERIFIED`; no generated approval, reviewer, review timestamp, or human precision estimate.
- Units remain `UNKNOWN`; no MTL calibration or reflectance claim.
- Reuse the exact frozen EPSG:32650 origin `(0,0)`, 10 km cells, strict 500 m margin, and `DEVELOPMENT iff (ix + 2*iy) % 3 == 0` rule.
- Candidate GeoJSON stays immutable. Edits/decisions go to a new R1.5 review path.
- Any human geometry edit invalidates inherited role, mask support, and pairwise-leakage validity. Recompute all three from the edited geometry; persist prior states only in audit history. A changed geometry has pairwise leakage status `PENDING` until the full four-pair audit is rerun.
- Report selection bias from the fixed 32×32 windows and edge filter, including accepted/rejected counts by spatial subregion and texture/edge strata. Do not relax thresholds for difficult areas or to reach candidate quotas.
- Without independent evidence that cross-scene geometry and auxiliary-band co-registration are adequate, do not suggest `REAL_CHANGE`; return `UNKNOWN` with a geometry/co-registration uncertainty reason.
- Preserve unrelated pre-existing untracked worktree files; Git metadata is read-only; do not commit.
- Confidence is an evidence tier, not correctness probability; `HIGH` is prohibited with current evidence. Landcover/terrain remain `UNKNOWN` absent verified external data.

## Review Focus

1. Wrong pair, missing/duplicate band, or wrong product-to-scene join must fail before pixel reads (Task 1 tests).
2. No overlap, differing origins, nodata edges, or incompatible transforms must map by georeferencing and exclude unsupported windows with reasons (Tasks 1–2 tests).
3. Unknown units/wavelengths must remain unknown and suppress unsupported semantic labels (Task 2 tests).
4. Cell/buffer crossings and cross-pair role conflicts must remain `UNASSIGNED`/audited, never cropped or reassigned (Task 3 tests).
5. GUI edits must not mutate candidates or old R1 reviews; all review changes must survive reload in append-only history (Task 4 tests).

---

## File map and shared interfaces

- `src/benchmark_v1_radiometric_r15/inputs.py`: immutable band/pair records and inventory/header validation.
- `src/benchmark_v1_radiometric_r15/proposals.py`: bounded-window features, deterministic candidate selection, conservative suggestion cues.
- `src/benchmark_v1_radiometric_r15/spatial.py`: frozen role calculation and overlap/leakage audit.
- `src/benchmark_v1_radiometric_r15/storage.py`: immutable GeoJSON/CSV proposals and safe serialization.
- `src/benchmark_v1_radiometric_r15/review_store.py`: separate human review CSV/JSONL history.
- `src/benchmark_v1_radiometric_r15/gui.py`: opt-in candidate queue and paired-window reviewer.
- `scripts/benchmark/generate_radiometric_r1_5_auto_rois.py`: allowlisted generator CLI.
- Modify `scripts/benchmark/annotate_radiometric_rois.py`: add optional `--auto-candidates PATH`; without it preserve current behavior.
- `configs/radiometric_r1_5_auto_roi.yaml`: frozen proposal settings and pair allowlist.
- Tests in `tests/radiometric_r1_5/`.
- Candidate outputs only in `data/output/benchmark_v1_radiometric_r1/auto_roi_candidates/<run_id>/`; review outputs only in `.../auto_roi_reviews/<run_id>/`; reports/maps/docs only in `docs/benchmark/v1/radiometric_r1/auto_roi_r1_5/<run_id>/`.

Exact shared types:

```python
@dataclass(frozen=True)
class BandInput:
    band: int
    path_i: Path | None
    path_j: Path | None
    wavelength_center_nm: tuple[float | None, float | None]
    unit: tuple[str, str]
    dtype: tuple[str, str]
    nodata: tuple[float | None, float | None]
    sha256: tuple[str | None, str | None]
    source_fingerprint: tuple[str, str]
    grid_check: str
    usable: bool
    exclusion_reason: str

@dataclass(frozen=True)
class PairInputs:
    pair_id: str
    scene_i: str
    scene_j: str
    days_apart: float
    quality_i: str
    quality_j: str
    bands: tuple[BandInput, ...]
    reference_crs: str
    reference_transform: object
    reference_width: int
    reference_height: int
    pair_metadata: dict[str, str]

def load_pair_inputs(pair_id: str, repo_root: Path) -> PairInputs: ...

@dataclass(frozen=True)
class ProposalConfig:
    window_px: int = 32
    min_b9_valid_fraction: float = 0.95
    max_strong_edge_fraction: float = 0.10
    edge_buffer_px: int = 2
    target_min: int = 15
    target_max: int = 30
    seed: int = 20261012

def generate_pair_proposals(inputs: PairInputs, config: ProposalConfig) -> list[dict]: ...
```

For source bands without a frozen SHA256, record resolved path, byte size, `mtime_ns`, header fingerprint and `sha256_status=NOT_FROZEN`. Do not claim this fingerprint is a hash; do not hash whole files in this phase. Reuse existing SHA256 when present.

### Task 1: Pair and band input registry

**Files:** create `inputs.py`, package `__init__.py`, `tests/radiometric_r1_5/test_inputs.py`.

**Produces:** `BandInput`, `PairInputs`, `load_pair_inputs(pair_id, repo_root) -> PairInputs`.

- [ ] Write failing tests for four-pair allowlist, product-ID join, 16 band records, missing/duplicate inputs, missing B9, invalid/no-overlap pair, unknown units, missing wavelengths, and header fingerprints.
- [ ] Run `.\.venv-registration\Scripts\python.exe -m pytest tests\radiometric_r1_5\test_inputs.py -q -p no:cacheprovider`; confirm absent API failure.
- [ ] Implement header-only validation from scene, pair, band inventories and B9 manifest. Require positive frozen B9 overlap; B9 missing in either scene is fatal. Exclude unusable auxiliary bands with explicit reasons.
- [ ] Re-run tests; expected: pass with no source pixel reads.

### Task 2: Proposal features and cautious suggestions

**Files:** create `proposals.py`, `configs/radiometric_r1_5_auto_roi.yaml`, `tests/radiometric_r1_5/test_proposals.py`.

**Consumes:** Task 1 `PairInputs`.
**Produces:** `ProposalConfig`; `generate_pair_proposals(inputs, config) -> list[dict]`.

- [ ] Write synthetic tests for no overlap, partial nodata, unequal grid origins, mapping, edge contamination, low texture, missing bands/unknown units, brightness-only stable false positives, dark-water/shadow ambiguity, and cloud/bright-bare-ground ambiguity.
- [ ] Run tests and confirm expected absent-function failures.
- [ ] Implement bounded 32×32-native-B9-pixel windows (448 m at 14 m), common B9 valid fraction ≥0.95, strong-edge fraction ≤0.10, 2-pixel edge buffer. Map scene-j using georeferencing; record bilinear value and nearest-neighbor mask resampling when required. Never read full-resolution scenes.
- [ ] Select deterministically across spatial subregions and texture strata; deduplicate intersecting windows; target 15–30, cap 30, allow fewer. Report each exclusion reason. Do not rank using corrected imagery, seams, matchers, registration residuals, or algorithm outcomes.
- [ ] Generate conservative cues: STABLE needs independent structural and multispectral-shape evidence, never brightness alone; REAL_CHANGE means suspected coherent structural/spectral-shape change, not scalar brightness difference, and is suppressed to UNKNOWN unless independent cross-scene geometry and relevant-band co-registration are verified; CLOUD/SHADOW use conservative multi-band/context cues, otherwise UNKNOWN. Confidence LOW/MEDIUM only. Landcover and terrain UNKNOWN. No human fields are generated.
- [ ] Verify unique IDs, valid geometries, support/edge thresholds, brightness-only similarity never yielding STABLE, ambiguous dark/bright cases remaining UNKNOWN, and geometry/co-registration uncertainty suppressing REAL_CHANGE.

### Task 3: Frozen roles, leakage audit, and immutable candidate store

**Files:** create `spatial.py`, `storage.py`, `tests/radiometric_r1_5/test_spatial_storage.py`.

**Consumes:** Tasks 1–2.
**Produces:** `assign_frozen_role(geometry) -> tuple[str,str,str]`, `audit_candidate_sets(collections) -> dict`, `write_candidate_artifacts(...)`.

- [ ] Write tests for frozen origin/cell/formula, strict 500 m edge, crossing cells, duplicate and overlapping pairs, opposing-role proximity, and GeoJSON/CSV round-trip.
- [ ] Run tests and confirm missing API failures.
- [ ] Reuse/test against `src/benchmark_v1_radiometric_r1/roi.py::classify_roi_role`; do not duplicate adjustable parameters. Unsafe geometry is `UNASSIGNED`; never crop or reassign.
- [ ] Add review-edit API/tests proving that a geometry edit recomputes both-scene and per-band valid support, frozen role, and cross-pair leakage; clears prior eligibility/approval on the current snapshot while preserving the complete prior state in the append-only event log.
- [ ] Report duplicate/intersection and opposite-role proximity conflicts by candidate IDs; never repair them.
- [ ] Write candidate files atomically, reload/validate, preserve suggestions and `UNVERIFIED` status. Candidate source is immutable after generation.
- [ ] Run focused tests; expected: exact round-trip and leakage is not hidden by serialization.

### Task 4: Separate review store and opt-in GUI mode

**Files:** create `review_store.py`, `gui.py`, `tests/radiometric_r1_5/test_review_gui.py`; modify only the auto-mode branch of `scripts/benchmark/annotate_radiometric_rois.py`.

**Consumes:** Tasks 1–3.
**Produces:** `load_candidate_collection(path, expected_pair_id)`, `AutoReviewStore.load/save/record_event`, `launch_auto_roi_reviewer(...)`.

- [ ] Write temp-store/Tk tests for filters, previous/next, suggestion display, band switching, geometry edit and recomputation of role/support/leakage, approval confirmation/evidence, doubt/reject, second-review identity/disagreement, and reload.
- [ ] Run tests and confirm the new API is absent.
- [ ] Implement separate R1.5 review snapshot and append-only JSONL keyed by `(pair_id,candidate_id)`, preserving candidate ID, before/after geometry, suggestion, human label/status, real reviewer and timestamp. On every geometry edit, recompute geometry role, both-scene/per-band support, and the cross-pair leakage audit; set current pairwise audit to `PENDING` until recomputed for all four pairs, clear prior eligibility/approval, and preserve prior status only in history. Require evidence for approval; no batch approval. TEST second review requires a distinct real identity and explicit independent class/status; disagreement remains unresolved, never automatically preferred.
- [ ] Implement previous/next queue navigation, pending/reviewed/all filters, progress counts, synchronized paired view, zoom/pan, geospatial geometry overlay/editing, suggestion/evidence/uncertainty/support/role display, and B1–B16 selector restricted to validated bands. Editing changes review-side records only.
- [ ] Add `--auto-candidates PATH`; default CLI branch and old `roi_reviews` destination remain unchanged.
- [ ] Run R1.5 GUI/store tests and all `tests/radiometric_r1`; expected: prior R1 behavior green and candidate artifact unchanged after test review.

### Task 5: First-pair generation and smoke validation (`00_01`)

**Files:** create `scripts/benchmark/generate_radiometric_r1_5_auto_rois.py`, `tests/radiometric_r1_5/test_cli_smoke.py`; create first-pair outputs only under versioned R1.5 candidate/docs paths.

**Produces:** `--pair 00_01 --max-candidates N`, candidate manifest/report/map.

- [ ] Write CLI tests for invalid pair, missing input, existing-output refusal, bounded run and deterministic manifest; verify pre-implementation failures.
- [ ] Implement script-rooted CLI and output checks. Require `--run-id`; reject protected R1/Task15/16/Phase3–5/source paths. Create a run directory only once; before candidate writes, record hashes for small existing manifests/review files plus path/size/header identities for the read-only inputs. Later invocations may append a previously absent pair only when its input/config fingerprint matches the run manifest, and must refuse to overwrite any existing pair artifact.
- [ ] Generate up to 15 initial `00_01` candidates using the frozen config and bounded windows; fewer is valid if quality gates reject windows, with explicit reasons. The GUI smoke-review uses a handful from this collection.
- [ ] Verify load/display, role/support, temporary geometry/review edit/save/reload, and bounded Tk GUI smoke. Synthetic tests must never approve real candidates. If GUI display is unavailable, report it and do not claim visual validation.
- [ ] Run first-pair tests; expected: reproducible manifest and no writes to protected paths.

### Task 6: Remaining pairs, aggregate report, map and instructions

**Files:** add `02_03`, `05_07`, `08_10` outputs; create run-scoped `candidate_quality_report.md`, `candidate_spatial_distribution.png`, `radiometric_r1_5_auto_roi_usage.md`, `tests/radiometric_r1_5/test_four_pair_outputs.py`.

**Consumes:** validated Task 5 config/schema. **Produces:** all four pair collections and aggregate audit/report/map/commands.

- [ ] Test four IDs, consistent run config, reconciled counts, role map, missing metadata and cross-pair audit.
- [ ] Run tests and confirm remaining-artifact failures.
- [ ] Generate remaining pairs with the exact Task 5 scoring thresholds, window size, selection order, and output cap; do not alter them after viewing first-pair output. Each pair may yield fewer than 15 if quality gates require it.
- [ ] Report counts by pair/label/confidence, valid support, usable/missing bands, exclusion reasons, cells/roles, duplicates/overlaps and leakage. Quantify fixed-window/edge-filter selection bias by spatial subregion and texture/edge strata, including accepted/rejected support and difficult/low-confidence areas; explicitly report where no high-confidence candidate exists. Map DEVELOPMENT/TEST/UNASSIGNED and frozen safety margins; it is not label evidence.
- [ ] Document PowerShell generation/review/save/resume commands; state units UNKNOWN and all automatic records UNVERIFIED.
- [ ] Verify exactly four pair outputs, no protected-path writes, and zero automatic approvals. Task 6 appends the three not-yet-generated pair outputs to the locked Task 5 run; it does not overwrite the run manifest or `00_01` artifacts.

### Task 7: Full regression and protection audit

**Files:** tests all R1.5 plus R1; update only R1.5 report/docs if correcting findings.

- [ ] Run `.\.venv-registration\Scripts\python.exe -m pytest tests\radiometric_r1_5 tests\radiometric_r1 -q -p no:cacheprovider`.
- [ ] Run `.\.venv-registration\Scripts\python.exe -m compileall -q src\benchmark_v1_radiometric_r15 scripts\benchmark tests\radiometric_r1_5`.
- [ ] Run `git diff --check` and scoped trailing-whitespace scan.
- [ ] Compare the run-manifest preflight against final R1 candidate/review files, Phase3–5 inputs, Task15/16 artifacts, and source path/header identities; confirm no protected writes and no changed small-file hashes.
- [ ] Report actual counts/coverage, exclusions, test/warning results, GUI validation type, all labels still unverified, and R2 not started.

## Generation/verification commands

From repository-root PowerShell:

```powershell
.\.venv-registration\Scripts\python.exe scripts\benchmark\generate_radiometric_r1_5_auto_rois.py --pair 00_01 --max-candidates 15 --run-id r15_20261012_01
.\.venv-registration\Scripts\python.exe scripts\benchmark\annotate_radiometric_rois.py --pair 00_01 --reviewer reviewer_real_person --auto-candidates data\output\benchmark_v1_radiometric_r1\auto_roi_candidates\r15_20261012_01\00_01.geojson
.\.venv-registration\Scripts\python.exe scripts\benchmark\generate_radiometric_r1_5_auto_rois.py --pairs 02_03 05_07 08_10 --max-candidates 15 --run-id r15_20261012_01
.\.venv-registration\Scripts\python.exe -m pytest tests\radiometric_r1_5 tests\radiometric_r1 -q -p no:cacheprovider
git diff --check
```

The implementation must document a concrete run ID and ensure output creation refuses to overwrite an existing run.
