"""Independent append-only human second review for frozen TEST checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from affine import Affine

from benchmark_phase2_core import pixel_to_projected
from benchmark_phase4_freeze import FrozenTrial


class ReviewError(ValueError):
    """Invalid second-review input or damaged event history."""


@dataclass(frozen=True)
class ImageGeometry:
    transform: Affine
    height: int
    width: int
    valid_at: Callable[[float, float], bool]


def validate_event_destination(path: Path, snapshot: FrozenTrial) -> Path:
    """Confine repository writes to Phase4 second_reviews; allow tmp test logs."""
    path = Path(path).resolve()
    root = snapshot.root.resolve()
    if path.suffix.lower() != ".jsonl":
        raise ReviewError("Second-review event destination must be .jsonl; source files are read-only")
    if path.is_relative_to(root):
        allowed = root / "docs/benchmark/v1/phase4/second_reviews"
        if not path.is_relative_to(allowed):
            raise ReviewError("Phase2/Phase3 and all other repository files are read-only in second-review mode")
    return path


def _identity_and_source(snapshot: FrozenTrial, key: tuple[str, str]) -> tuple[dict, dict]:
    try:
        frozen, source = snapshot.assignments[key], snapshot.candidates[key]
    except KeyError as exc:
        raise ReviewError(f"Unknown frozen candidate: {key}") from exc
    if frozen["split"] != "TEST" or source.get("human_review_status") != "APPROVED":
        raise ReviewError("Only first-approved frozen TEST candidates can be second reviewed")
    return frozen, source


def _position(geometry: ImageGeometry, rowcol: tuple[float, float]) -> dict:
    row, col = map(float, rowcol)
    if not all(math.isfinite(v) for v in (row, col)):
        raise ReviewError("Nonfinite pixel coordinate")
    x, y = pixel_to_projected(geometry.transform, row, col)
    return {"row": row, "col": col, "x": x, "y": y}


def _event_bytes(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def event_sha256(event: dict) -> str:
    return hashlib.sha256(_event_bytes(event)).hexdigest()


def _spatial_check(frozen: dict, after: dict, geometry: dict, snapshot: FrozenTrial) -> list[str]:
    reasons = []
    ix, iy = (int(v) for v in frozen["grid_cell_id"].lstrip("g").split("_"))
    expected_cell = (ix, iy)
    for suffix, scene in (("left", frozen["scene_i"]), ("right", frozen["scene_j"])):
        point = after[suffix]
        x, y = point["x"], point["y"]
        cell = (math.floor(x / 10000), math.floor(y / 10000))
        if cell != expected_cell:
            reasons.append("DIFFERENT_CELLS")
        if not (500 < x - ix * 10000 < 9500 and 500 < y - iy * 10000 < 9500):
            reasons.append("BUFFER_ZONE")
        geo = geometry[scene]
        row, col = point["row"], point["col"]
        if (not 0 <= row < geo.height or not 0 <= col < geo.width
                or not geo.valid_at(row, col)):
            reasons.append("INVALID_IMAGE_SUPPORT")
        for other in snapshot.assignments.values():
            if other["split"] != "DEVELOPMENT":
                continue
            for other_suffix in ("i", "j"):
                distance = math.hypot(x - float(other[f"effective_x_{other_suffix}"]),
                                      y - float(other[f"effective_y_{other_suffix}"]))
                if distance < 1000:
                    reasons.append("DEVELOPMENT_TEST_ISOLATION")
                    break
            if "DEVELOPMENT_TEST_ISOLATION" in reasons:
                break
    return sorted(set(reasons))


def validate_review_geometry_history(snapshot: FrozenTrial, events: list[dict],
                                     geometry: dict[str, ImageGeometry]) -> None:
    """Recompute every attempted edit from verified native grids, including sticky failures."""
    sticky: set[tuple[str, str]] = set()
    for event in events:
        key = (event["pair_id"], event["candidate_id"])
        frozen, source = _identity_and_source(snapshot, key)
        prefix = "adjusted_" if str(source.get("human_adjusted", "")).lower() == "true" else ""
        try:
            before = {}
            after = {}
            for side, scene, suffix in (("left", frozen["scene_i"], "i"),
                                        ("right", frozen["scene_j"], "j")):
                geo = geometry[scene]
                before[side] = _position(geo, (float(source[prefix + "row_" + suffix]),
                                                float(source[prefix + "col_" + suffix])))
                logged = event["after"][side]
                after[side] = _position(geo, (logged["row"], logged["col"]))
                for label, computed in (("before", before[side]), ("after", after[side])):
                    supplied = event[label][side]
                    for field in ("row", "col", "x", "y"):
                        if not math.isclose(float(supplied[field]), computed[field],
                                            rel_tol=0, abs_tol=1e-6):
                            raise ReviewError(f"Second-review {label} projected coordinate/pixel mismatch: {key}")
            adjusted = any(after[side][axis] != before[side][axis]
                           for side in ("left", "right") for axis in ("row", "col"))
            if event.get("adjusted") is not adjusted:
                raise ReviewError(f"Second-review adjusted flag mismatch: {key}")
            if adjusted and not str(event.get("adjustment_reason", "")).strip():
                raise ReviewError(f"Second-review adjustment lacks reason: {key}")
            reasons = _spatial_check(frozen, after, geometry, snapshot)
            if adjusted and reasons:
                sticky.add(key)
            status = ("NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT" if key in sticky else
                      "INVALID_SPATIAL" if reasons else "VALID")
            if event.get("spatial_status") != status or event.get("spatial_reasons") != reasons:
                raise ReviewError(f"Second-review spatial status/reasons mismatch: {key}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ReviewError):
                raise
            raise ReviewError(f"Invalid second-review geometry event: {key}: {exc}") from exc


def propose_review(snapshot: FrozenTrial, key: tuple[str, str], reviewer: str, outcome: str,
                   left_rowcol: tuple[float, float], right_rowcol: tuple[float, float],
                   inspections: dict, uncertainty: str, adjustment_reason: str,
                   disagreement: str, disagreement_note: str, image_headers: dict[str, ImageGeometry],
                   previous_events: list[dict]) -> dict:
    frozen, source = _identity_and_source(snapshot, key)
    first_names = {source.get("primary_reviewer", "").strip(), source.get("reviewer", "").strip()}
    if not reviewer.strip() or reviewer.strip() in first_names:
        raise ReviewError("Second reviewer must be distinct from first reviewer")
    if outcome not in {"APPROVED", "REJECTED", "DOUBTFUL"}:
        raise ReviewError("Invalid second-review outcome")
    for side in ("left", "right"):
        if inspections.get(f"{side}_original_resolution") is not True:
            raise ReviewError(f"Explicit {side} original-resolution inspection required")
    if outcome == "APPROVED" and not uncertainty.strip():
        raise ReviewError("APPROVED needs a new human localization uncertainty statement")
    if disagreement not in {"YES", "NO"} or (disagreement == "YES" and not disagreement_note.strip()):
        raise ReviewError("Material disagreement requires YES/NO and explanation")
    try:
        left_geo, right_geo = image_headers[frozen["scene_i"]], image_headers[frozen["scene_j"]]
    except KeyError as exc:
        raise ReviewError("Missing native image geometry") from exc
    prefix = "adjusted_" if str(source.get("human_adjusted", "")).lower() == "true" else ""
    before = {
        "left": _position(left_geo, (float(source[prefix + "row_i"]), float(source[prefix + "col_i"]))),
        "right": _position(right_geo, (float(source[prefix + "row_j"]), float(source[prefix + "col_j"]))),
    }
    after = {"left": _position(left_geo, left_rowcol), "right": _position(right_geo, right_rowcol)}
    adjusted = any(after[side][name] != before[side][name]
                   for side in ("left", "right") for name in ("row", "col"))
    if adjusted and not adjustment_reason.strip():
        raise ReviewError("Coordinate adjustment requires a human reason")
    reasons = _spatial_check(frozen, after, image_headers, snapshot)
    previous = [event for event in previous_events if (event.get("pair_id"), event.get("candidate_id")) == key]
    sticky = any(event.get("spatial_status") == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT"
                 for event in previous)
    status = ("NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT" if sticky or (adjusted and reasons)
              else "INVALID_SPATIAL" if reasons else "VALID")
    delta = {side: math.hypot(after[side]["row"] - before[side]["row"],
                              after[side]["col"] - before[side]["col"]) for side in ("left", "right")}
    return {
        "schema_version": "phase4_second_review_v1", "event_id": str(uuid4()),
        "pair_id": key[0], "candidate_id": key[1], "frozen_role": "TEST",
        "frozen_grid_cell_id": frozen["grid_cell_id"],
        "phase3_assignment_sha256": snapshot.assignment_sha256,
        "source_candidate_sha256": snapshot.candidate_sha256[key[0]],
        "previous_event_sha256": event_sha256(previous_events[-1]) if previous_events else "",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer.strip(), "first_reviewer": source.get("primary_reviewer", "") or source.get("reviewer", ""),
        "first_review_status": source["human_review_status"], "first_confidence": source.get("confidence", ""),
        "inspections": dict(inspections), "human_outcome": outcome, "human_uncertainty": uncertainty.strip(),
        "adjusted": adjusted, "adjustment_reason": adjustment_reason.strip(),
        "before": before, "after": after, "delta_pixels": delta,
        "material_disagreement": disagreement, "disagreement_note": disagreement_note.strip(),
        "spatial_reasons": reasons, "spatial_status": status,
    }


def append_review_event(path: Path, event: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ReviewError("Concurrent second-review writer lock exists") from exc
    try:
        os.close(descriptor)
        previous_hash = ""
        if path.exists():
            lines = path.read_bytes().splitlines(keepends=True)
            if lines and (not lines[-1].endswith(b"\n")):
                raise ReviewError("JSONL last event is incomplete")
            previous_hash = hashlib.sha256(lines[-1]).hexdigest() if lines else ""
        if event.get("previous_event_sha256") != previous_hash:
            raise ReviewError("Stale second-review event chain")
        payload = _event_bytes(event)
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(payload).hexdigest()
    finally:
        lock.unlink(missing_ok=True)


def load_review_events(path: Path, snapshot: FrozenTrial) -> list[dict]:
    if not Path(path).exists():
        return []
    lines = Path(path).read_bytes().splitlines(keepends=True)
    events, previous_hash, ids = [], "", set()
    for line in lines:
        if not line.endswith(b"\n"):
            raise ReviewError("Incomplete second-review JSONL event")
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ReviewError(f"Malformed second-review JSONL: {exc}") from exc
        key = (event.get("pair_id"), event.get("candidate_id"))
        if (key not in snapshot.assignments or event.get("frozen_role") != "TEST"
                or event.get("phase3_assignment_sha256") != snapshot.assignment_sha256
                or event.get("source_candidate_sha256") != snapshot.candidate_sha256.get(key[0])):
            raise ReviewError(f"Second-review frozen identity/SHA mismatch: {key}")
        if event.get("schema_version") != "phase4_second_review_v1":
            raise ReviewError("Unknown second-review event schema")
        frozen, source = snapshot.assignments[key], snapshot.candidates[key]
        if event.get("frozen_grid_cell_id") != frozen["grid_cell_id"]:
            raise ReviewError("Second-review frozen cell mismatch")
        if (not str(event.get("reviewer", "")).strip()
                or event["reviewer"] in {source.get("primary_reviewer"), source.get("reviewer")}):
            raise ReviewError("Second-review identity must be distinct from first reviewer")
        if event.get("human_outcome") not in {"APPROVED", "REJECTED", "DOUBTFUL"}:
            raise ReviewError("Unknown second-review human outcome")
        inspections = event.get("inspections", {})
        if (inspections.get("left_original_resolution") is not True
                or inspections.get("right_original_resolution") is not True):
            raise ReviewError("Second-review event lacks both image inspections")
        if event["human_outcome"] == "APPROVED" and not str(event.get("human_uncertainty", "")).strip():
            raise ReviewError("Approved second-review event lacks independent uncertainty")
        try:
            stamp = datetime.fromisoformat(event["reviewed_at_utc"])
            if stamp.tzinfo is None:
                raise ValueError("timezone missing")
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError("Invalid second-review UTC timestamp") from exc
        if event.get("event_id") in ids or event.get("previous_event_sha256") != previous_hash:
            raise ReviewError("Duplicate/replayed or out-of-order second-review event")
        ids.add(event["event_id"])
        previous_hash = hashlib.sha256(line).hexdigest()
        events.append(event)
    return events


def review_state(snapshot: FrozenTrial, events: list[dict]) -> dict[tuple[str, str], dict]:
    states = {}
    for key, frozen in snapshot.assignments.items():
        if frozen["split"] == "TEST":
            states[key] = {"frozen_role": "TEST", "human_outcome": "PENDING",
                           "spatial_status": "NOT_REVIEWED", "eligible": False,
                           "dispute_pending": False, "latest_event": None}
    for event in events:
        key = (event["pair_id"], event["candidate_id"])
        if key not in states:
            raise ReviewError(f"Event targets non-TEST candidate: {key}")
        state = states[key]
        sticky = (state["spatial_status"] == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT"
                  or event["spatial_status"] == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT")
        spatial = "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT" if sticky else event["spatial_status"]
        dispute = state["dispute_pending"] or event["material_disagreement"] == "YES"
        state.update(human_outcome=event["human_outcome"], spatial_status=spatial,
                     dispute_pending=dispute, latest_event=event,
                     eligible=(event["human_outcome"] == "APPROVED" and spatial == "VALID" and not dispute))
    return states
