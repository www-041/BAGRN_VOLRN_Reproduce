"""Read-only authority for the frozen four-pair Phase3 benchmark snapshot."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


PAIRS = ("00_01", "02_03", "05_07", "08_10")
EXPECTED = {"DEVELOPMENT": 43, "TEST": 84, "UNASSIGNED": 33}
FROZEN_PHASE3_AUDIT_SHA256 = "44765f95717b742f8e0c08bc6bc14d51bb3574b5c921d8cdd444c17877c20c43"


class FrozenInputError(ValueError):
    """Phase3 or first-review bytes/identities differ from the frozen audit."""


@dataclass(frozen=True)
class FrozenTrial:
    root: Path
    assignments: dict[tuple[str, str], dict]
    candidates: dict[tuple[str, str], dict]
    audit: dict
    candidate_sha256: dict[str, str]
    assignment_sha256: str
    trial_pairs: tuple[str, ...] = PAIRS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_frozen_trial(root: Path) -> FrozenTrial:
    root = Path(root)
    phase3 = root / "docs/benchmark/v1/phase3"
    try:
        if sha256_file(phase3 / "split_leakage_audit.json") != FROZEN_PHASE3_AUDIT_SHA256:
            raise FrozenInputError("Phase3 audit SHA256 differs from the accepted four-pair freeze")
        audit = json.loads((phase3 / "split_leakage_audit.json").read_text(encoding="utf-8"))
        if audit.get("freeze_scope") != "TRIAL4_ONLY" or audit.get("leakage_pass") is not True:
            raise FrozenInputError("Phase3 audit is not a passing four-pair freeze")
        for name, expected in audit["output_sha256"].items():
            if Path(name).name != name or sha256_file(phase3 / name) != expected:
                raise FrozenInputError(f"Phase3 SHA256 mismatch: {name}")
        assignment_path = phase3 / "checkpoint_split_assignments.csv"
        assignments = {}
        for row in _rows(assignment_path):
            key = (row["pair_id"], row["candidate_id"])
            if key in assignments or key[0] not in PAIRS:
                raise FrozenInputError(f"Duplicate/unexpected Phase3 identity: {key}")
            assignments[key] = row
        if len(assignments) != 160:
            raise FrozenInputError("Expected 160 Phase3 assignment records")
        counts = {role: sum(row["split"] == role for row in assignments.values()) for role in EXPECTED}
        if counts != EXPECTED or any(row["split"] not in EXPECTED for row in assignments.values()):
            raise FrozenInputError(f"Frozen role counts mismatch: {counts}")
        candidates = {}
        hashes = {}
        for pair in PAIRS:
            path = root / "docs/benchmark/v1/phase2/auto_candidates" / f"{pair}.csv"
            hashes[pair] = sha256_file(path)
            if hashes[pair] != audit["source_candidate_sha256"][pair]:
                raise FrozenInputError(f"Candidate SHA256 mismatch: {pair}")
            rows = _rows(path)
            if len(rows) != 40:
                raise FrozenInputError(f"Expected 40 candidates: {pair}")
            for source in rows:
                key = (source["pair_id"], source["candidate_id"])
                if key in candidates or key not in assignments or key[0] != pair:
                    raise FrozenInputError(f"Candidate identity mismatch: {key}")
                frozen = assignments[key]
                for name, value in source.items():
                    frozen_name = {"split": "source_split", "grid_cell_id": "original_candidate_grid_cell_id"}.get(name, name)
                    if frozen.get(frozen_name) != value:
                        raise FrozenInputError(f"Candidate/Phase3 field mismatch: {key} {name}")
                if (frozen["scene_i"], frozen["scene_j"]) != (source["scene_i"], source["scene_j"]):
                    raise FrozenInputError(f"Scene identity mismatch: {key}")
                candidates[key] = source
        if set(candidates) != set(assignments):
            raise FrozenInputError("Candidate/Phase3 key set mismatch")
        if len(audit.get("b9_verified_inputs", [])) != 8:
            raise FrozenInputError("Frozen eight-scene B9 identity list missing")
        return FrozenTrial(root, assignments, candidates, audit, hashes,
                           audit["output_sha256"]["checkpoint_split_assignments.csv"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise FrozenInputError(f"Frozen input read/identity failure: {exc}") from exc


def verify_b9_identity(snapshot: FrozenTrial) -> None:
    for record in snapshot.audit["b9_verified_inputs"]:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise FrozenInputError(f"B9 missing/size mismatch: {record['scene_index']}")
        if sha256_file(path) != record["sha256"]:
            raise FrozenInputError(f"B9 SHA256 mismatch: {record['scene_index']}")


def quality_summary(snapshot: FrozenTrial) -> dict:
    rows = snapshot.candidates.values()
    notes = [row["coordinate_precision_note"].lower() for row in rows]
    return {
        "candidate_count": len(snapshot.candidates),
        "high_confidence_labels": sum(row["confidence"] == "HIGH" for row in snapshot.candidates.values()),
        "human_adjusted_count": sum(row["human_adjusted"].lower() == "true" for row in snapshot.candidates.values()),
        "half_pixel_like_notes": sum(bool(re.search(r"0\.5\s*pixel", note)) for note in notes),
        "one_pixel_like_notes": sum(bool(re.search(r"(?<![.\d])1\s*pixel", note)) for note in notes),
    }
