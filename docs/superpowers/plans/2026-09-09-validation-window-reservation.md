# Validation-Window-First Reservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reserve exact independent validation windows from the common geographic overlap before registration training and reuse those windows during final validation.

**Architecture:** A pair overlap context remains the single coordinate system. A geometry-only reservation routine enumerates complete candidate windows, selects a spatially stratified set from the largest viable block size, and creates the training mask. Final validation consumes those exact windows and fails before registration when the configured minimum cannot be reserved.

**Tech Stack:** Python, NumPy, rasterio Affine/window geometry, SciPy-backed existing registration helpers, pytest.

**Spec:** User request dated 2026-09-09: “validation-window-first reservation”.

## Global Constraints

- Do not change quality thresholds, `final_min_blocks`, BAGRN, VOLRN, RBF CV thresholds, or local displacement limits.
- Candidate selection may use only overlap geometry and common-valid masks; never residuals, confidence, RMSE, or P95.
- Do not use a minimum-shape top-left crop to align scenes.
- Do not run real DZ01V N=2/N=4/N=6 registration.

---

### Task 1: Geometry-only validation-window reservation

**Files:**
- Modify: `src/coregistration.py`
- Test: `tests/test_validation_block_selection.py`

**Interfaces:**
- Produce `enumerate_validation_windows(...)`, `reserve_validation_windows(...)`, and a reservation result containing exact windows, masks, counts, centers, and failure reason.

- [ ] Add failing tests for geometry-only selection, narrow diagonal masks, and accounting for unused candidate cells.
- [ ] Run the focused tests and confirm failure before implementation.
- [ ] Implement candidate enumeration over complete windows for `[384, 256, 192]`, largest-first selection, spatial stratification, optional buffer, and exact train/holdout masks.
- [ ] Run focused tests and confirm the reservation contract.

### Task 2: Early failure and shared pipeline context

**Files:**
- Modify: `src/multiband_pipeline.py`
- Test: `tests/test_registration_spatial_holdout.py`, `tests/test_dz01_registration_adaptation.py`

**Interfaces:**
- Extend the pair holdout context with `validation_reservation` and exact reserved windows.
- Stop `register_scenes()` before matching/refinement when an enabled required edge cannot reserve `final_min_blocks` windows.

- [ ] Add failing tests proving reservation happens before matching and impossible geometry fails early.
- [ ] Run the focused tests and confirm failure.
- [ ] Build reservation immediately after the common overlap context and pass its train mask to all training matching paths.
- [ ] Return structured `insufficient independent validation geometry` diagnostics without entering global/local registration.
- [ ] Run focused tests and confirm the early-failure behavior.

### Task 3: Exact-window final validation

**Files:**
- Modify: `src/coregistration.py`, `src/multiband_pipeline.py`
- Test: `tests/test_final_holdout_validation.py`

**Interfaces:**
- Add `reserved_validation_windows` to final validation and evaluate exactly those windows on the shared overlap grid.

- [ ] Add failing tests for exact-window reuse and no regeneration of a second grid.
- [ ] Run the focused tests and confirm failure.
- [ ] Translate exact overlap-local windows to full-scene arrays only through the shared pair context.
- [ ] Remove final-time geometry re-selection when reserved windows are available.
- [ ] Run focused tests and confirm all reserved windows are evaluated.

### Task 4: Diagnostics, regression suite, and delivery

**Files:**
- Modify: `scripts/diagnose_registration_pair.py`, `docs/REGISTRATION_QUALITY_GATE.md`
- Test: `tests/test_validation_block_selection.py`, `tests/test_registration_spatial_holdout.py`, `tests/test_final_holdout_validation.py`

- [ ] Add `validation_reservation` fields for selected size, candidate count, exact windows, centers, train/holdout counts, unused cells, and buffer.
- [ ] Run focused pytest.
- [ ] Run full `pytest -q`.
- [ ] Review that no forbidden thresholds or registration parameters changed.
- [ ] Commit the implementation and attempt to push the current repair branch.
