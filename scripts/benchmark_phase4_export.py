"""Strict formal TEST truth gate; no matcher or optimization is run here."""

from __future__ import annotations

import csv
from contextlib import ExitStack
import json
from pathlib import Path

import rasterio
from rasterio.windows import Window

from benchmark_phase4_freeze import FrozenTrial, load_frozen_trial, verify_b9_identity
from benchmark_phase4_review import (ImageGeometry, _spatial_check, load_review_events,
                                     review_state, validate_review_geometry_history)


def _attestation_valid(snapshot: FrozenTrial, attestation: dict | None) -> bool:
    if not isinstance(attestation, dict):
        return False
    test_ids = {key[1] for key, row in snapshot.assignments.items() if row["split"] == "TEST"}
    return (attestation.get("dataset_version") == snapshot.assignment_sha256
            and bool(str(attestation.get("responsible_person", "")).strip())
            and bool(str(attestation.get("signature", "")).strip())
            and isinstance(attestation.get("run_ids"), list) and bool(attestation["run_ids"])
            and set(attestation.get("excluded_test_ids", [])) == test_ids
            and attestation.get("declared_no_test_use") is True)


def validate_test_candidates(snapshot: FrozenTrial, events_by_pair: dict[str, list[dict]],
                             nonuse_attestation: dict | None,
                             image_headers: dict[str, ImageGeometry]) -> tuple[list[dict], list[dict]]:
    """Return eligible truth and explicit exclusions; every frozen TEST appears once."""
    all_events = [event for pair_events in events_by_pair.values() for event in pair_events]
    validate_review_geometry_history(snapshot, all_events, image_headers)
    states = review_state(snapshot, all_events)
    attested = _attestation_valid(snapshot, nonuse_attestation)
    truth, excluded = [], []
    for key in sorted(states):
        frozen, source, state = snapshot.assignments[key], snapshot.candidates[key], states[key]
        event = state["latest_event"]
        reasons = []
        if source.get("human_review_status") != "APPROVED" or not (source.get("primary_reviewer") or source.get("reviewer")):
            reasons.append("INVALID_FIRST_REVIEW")
        if event is None:
            reasons.append("MISSING_SECOND_REVIEW")
        else:
            if (event.get("phase3_assignment_sha256") != snapshot.assignment_sha256
                    or event.get("source_candidate_sha256") != snapshot.candidate_sha256.get(key[0])
                    or event.get("frozen_grid_cell_id") != frozen["grid_cell_id"]):
                reasons.append("SECOND_REVIEW_FROZEN_IDENTITY_MISMATCH")
            if (not event.get("reviewer") or event["reviewer"] in
                    {source.get("primary_reviewer"), source.get("reviewer")}):
                reasons.append("REVIEWER_NOT_INDEPENDENT")
            inspections = event.get("inspections", {})
            if (inspections.get("left_original_resolution") is not True
                    or inspections.get("right_original_resolution") is not True
                    or not event.get("human_uncertainty", "").strip()):
                reasons.append("INCOMPLETE_HUMAN_INSPECTION")
            if event.get("human_outcome") != "APPROVED":
                reasons.append(f"SECOND_REVIEW_{event.get('human_outcome', 'INVALID')}")
            if state["spatial_status"] == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT":
                reasons.append("NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT")
            if state["dispute_pending"]:
                reasons.append("DISPUTE_PENDING")
            if (frozen["scene_i"] not in image_headers or frozen["scene_j"] not in image_headers):
                reasons.append("MISSING_IMAGE_GEOMETRY")
            else:
                try:
                    reasons.extend(_spatial_check(frozen, event["after"], image_headers, snapshot))
                except (KeyError, ValueError, TypeError):
                    reasons.append("INVALID_SECOND_REVIEW_COORDINATES")
            if event.get("spatial_status") != "VALID":
                reasons.append("SECOND_REVIEW_SPATIAL_STATUS_INVALID")
        if not attested:
            reasons.append("MISSING_NONUSE_ATTESTATION")
        if reasons:
            excluded.append({"pair_id": key[0], "candidate_id": key[1],
                             "frozen_role": "TEST", "reasons": sorted(set(reasons))})
            continue
        after = event["after"]
        truth.append({
            "point_id": key[1], "pair_id": key[0], "scene_i": frozen["scene_i"],
            "scene_j": frozen["scene_j"], "band": "B9", "split": "TEST",
            **{f"{field}_{suffix}": after[side][field]
               for suffix, side in (("i", "left"), ("j", "right"))
               for field in ("row", "col", "x", "y")},
            "first_reviewer": source.get("primary_reviewer") or source.get("reviewer"),
            "second_reviewer": event["reviewer"], "second_review_time_utc": event["reviewed_at_utc"],
            "human_uncertainty": event["human_uncertainty"],
            "source_candidate_sha256": snapshot.candidate_sha256[key[0]],
            "phase3_assignment_sha256": snapshot.assignment_sha256,
            "second_review_event_id": event["event_id"],
        })
    return truth, excluded


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ensure_new_export_dir(path: Path, *, root: Path | None = None) -> Path:
    """Never leave a stale formal truth file when a later export has zero truth."""
    path = Path(path).resolve()
    if root is not None:
        allowed = Path(root).resolve() / "docs/benchmark/v1/phase4"
        if not path.is_relative_to(allowed):
            raise ValueError(f"Formal TEST export must stay within Phase4: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Formal TEST output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_formal_test(root: Path, event_dir: Path, attestation_path: Path,
                       output_dir: Path) -> dict:
    """Validate eight B9 versions and all frozen identities before writing anything."""
    snapshot = load_frozen_trial(root)
    verify_b9_identity(snapshot)
    attestation = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    events = {pair: load_review_events(Path(event_dir) / f"{pair}.jsonl", snapshot)
              for pair in snapshot.trial_pairs}
    with ExitStack() as stack:
        geometry = {}
        for record in snapshot.audit["b9_verified_inputs"]:
            dataset = stack.enter_context(rasterio.open(record["path"]))
            if dataset.crs.to_epsg() != 32650:
                raise ValueError("Formal TEST B9 CRS mismatch")
            def valid_at(row, col, ds=dataset):
                r, c = int(round(row)), int(round(col))
                return (0 <= r < ds.height and 0 <= c < ds.width
                        and bool(ds.read_masks(1, window=Window(c, r, 1, 1))[0, 0]))
            geometry[record["scene_id"]] = ImageGeometry(dataset.transform, dataset.height,
                                                           dataset.width, valid_at)
        truth, excluded = validate_test_candidates(snapshot, events, attestation, geometry)
    summary = {"frozen_test_candidates": 84, "eligible": len(truth), "excluded": len(excluded),
               "status": "READY_TO_SCORE" if truth and not excluded else "INCOMPLETE_SECOND_REVIEW"}
    output_dir = ensure_new_export_dir(output_dir, root=root)
    if truth:
        _write_csv(output_dir / "formal_test_truth.csv", truth, list(truth[0]))
    exclusions = [{**item, "reasons": "|".join(item["reasons"])} for item in excluded]
    _write_csv(output_dir / "formal_test_exclusions.csv", exclusions,
               ["pair_id", "candidate_id", "frozen_role", "reasons"])
    (output_dir / "formal_test_status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
