from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


LABELS = {"STABLE", "REAL_CHANGE", "CLOUD", "SHADOW", "UNKNOWN"}
REVIEW_STATES = {"UNVERIFIED", "APPROVED", "DOUBTFUL", "REJECTED"}
SPLIT_ROLES = {"DEVELOPMENT", "TEST", "UNASSIGNED"}
PURPOSES = {"CALIBRATION", "EVALUATION", "UNASSIGNED"}
ROI_FIELDS = (
    "roi_id", "pair_id", "scene_i", "scene_j", "geometry", "geometry_crs", "days_apart", "label",
    "review_status", "reviewer", "review_time_utc", "evidence_notes", "landcover", "terrain",
    "role", "grid_cell_id", "spatial_status", "purpose", "second_reviewer", "second_review_status",
    "second_review_time_utc", "valid_fraction_i", "valid_fraction_j", "display_band", "source_sha256_i",
    "source_sha256_j", "candidate_source", "spatial_overlap_audit", "candidate_geometry_source",
    "source_crs_resolution", "quality_i", "quality_j", "temporal_bin", "second_review_label", "second_review_notes",
)


def validate_review_record(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    for key in ROI_FIELDS:
        row.setdefault(key, "")
    if not row["roi_id"]:
        raise ValueError("roi_id is required")
    if row["label"] not in LABELS or row["review_status"] not in REVIEW_STATES:
        raise ValueError("invalid ROI label or review state")
    if row["role"] not in SPLIT_ROLES or row["purpose"] not in PURPOSES:
        raise ValueError("invalid split role or purpose")
    if row["second_review_status"] not in REVIEW_STATES:
        raise ValueError("invalid second-review state")
    if row["second_review_label"] not in LABELS:
        raise ValueError("invalid second-review label")
    if row["review_status"] == "APPROVED":
        if not row["reviewer"] or not row["review_time_utc"]:
            raise ValueError("human approval requires reviewer and timestamp")
        if row["label"] == "UNKNOWN":
            raise ValueError("UNKNOWN cannot be approved as a determinate label")
    if row["second_review_status"] == "APPROVED":
        if row["role"] != "TEST":
            raise ValueError("second approval is reserved for frozen TEST-role ROIs")
        if not row["second_reviewer"] or not row["second_review_time_utc"]:
            raise ValueError("second approval requires reviewer and timestamp")
        if row["second_reviewer"].strip().casefold() == row["reviewer"].strip().casefold():
            raise ValueError("second reviewer must be different from first reviewer")
        if row["review_status"] != "APPROVED":
            raise ValueError("second review cannot approve a ROI without first approval")
        if row["second_review_label"] != row["label"] or row["second_review_label"] == "UNKNOWN":
            raise ValueError("first and second reviewer labels must agree for approval")
    return row


def is_formal_test_eligible(record: dict[str, Any]) -> bool:
    try:
        row = validate_review_record(record)
    except ValueError:
        return False
    return (
        row["role"] == "TEST"
        and row["purpose"] == "EVALUATION"
        and row["spatial_status"] == "SAFE_SINGLE_CELL"
        and row["spatial_overlap_audit"] == "PASS"
        and row["label"] in {"STABLE", "REAL_CHANGE"}
        and row["review_status"] == "APPROVED"
        and row["second_review_status"] == "APPROVED"
        and row["second_reviewer"].strip().casefold() != row["reviewer"].strip().casefold()
    )


class ReviewStore:
    """Atomic CSV snapshot plus append-only JSONL audit events."""

    def __init__(self, csv_path: Path, events_path: Path):
        self.csv_path = Path(csv_path)
        self.events_path = Path(events_path)
        self.records: dict[str, dict[str, str]] = {}
        self._pending_events: list[dict[str, Any]] = []
        if self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for record in csv.DictReader(stream):
                    normalized = validate_review_record(record)
                    key = normalized["roi_id"]
                    if key in self.records:
                        raise ValueError(f"duplicate roi_id in saved annotations: {key}")
                    self.records[key] = {k: str(v) for k, v in normalized.items()}

    def upsert(self, record: dict[str, Any], *, actor: str, event_type: str = "EDIT") -> None:
        new = validate_review_record(record)
        old = self.records.get(new["roi_id"])
        self.records[new["roi_id"]] = {k: str(v) for k, v in new.items()}
        self._pending_events.append({
            "event_id": uuid4().hex,
            "event_type": event_type if old else "CREATE",
            "roi_id": new["roi_id"],
            "actor": actor,
            "event_time_utc": datetime.now(timezone.utc).isoformat(),
            "before": old,
            "after": self.records[new["roi_id"]],
        })

    def delete(self, roi_id: str, *, actor: str) -> None:
        old = self.records.pop(roi_id)
        self._pending_events.append({
            "event_id": uuid4().hex,
            "event_type": "DELETE", "roi_id": roi_id, "actor": actor,
            "event_time_utc": datetime.now(timezone.utc).isoformat(), "before": old, "after": None,
        })

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{self.csv_path.stem}_", suffix=".tmp", dir=self.csv_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ROI_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for key in sorted(self.records):
                    writer.writerow({name: self.records[key].get(name, "") for name in ROI_FIELDS})
            os.replace(tmp_name, self.csv_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        if self._pending_events:
            with self.events_path.open("a", encoding="utf-8", newline="") as stream:
                for event in self._pending_events:
                    stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            self._pending_events.clear()

