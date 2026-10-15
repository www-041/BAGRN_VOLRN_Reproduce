from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from shapely.geometry import mapping

from .spatial import revalidate_geometry_edit
from .storage import load_candidate_collection


REVIEW_STATES = {"UNVERIFIED", "APPROVED", "DOUBTFUL", "REJECTED"}
LABELS = {"STABLE", "REAL_CHANGE", "CLOUD", "SHADOW", "UNKNOWN"}


class AutoReviewStore:
    """Separate R1.5 review snapshot with append-only JSONL decision history."""

    def __init__(self, candidate_path: str | Path, snapshot_path: str | Path, events_path: str | Path,
                 *, pairwise_audit: dict[str, Any] | None = None, all_pairs_available: bool = False):
        self.candidate_path = Path(candidate_path)
        self.snapshot_path = Path(snapshot_path)
        self.events_path = Path(events_path)
        raw_collection = json.loads(self.candidate_path.read_text(encoding="utf-8"))
        pair_ids = {str(f.get("properties", {}).get("pair_id", "")) for f in raw_collection.get("features", [])}
        if len(pair_ids) != 1 or not next(iter(pair_ids), ""):
            raise ValueError("candidate artifact must contain exactly one non-empty pair_id")
        self.collection = load_candidate_collection(self.candidate_path, next(iter(pair_ids)))
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        for feature in self.collection["features"]:
            props = feature["properties"]
            if (props.get("review_status") != "UNVERIFIED" or props.get("reviewer") or
                    props.get("review_time_utc") or props.get("eligible_for_test")):
                raise ValueError("candidate source contains human review or eligibility state; refusing to load")
            row = {**feature["properties"], "geometry": feature["geometry"]}
            row.update({"human_label": "UNKNOWN", "review_status": "UNVERIFIED", "reviewer": "",
                        "review_time_utc": "", "evidence_notes": "", "second_review_status": "UNVERIFIED",
                        "second_reviewer": "", "second_review_time_utc": "", "second_review_label": "UNKNOWN",
                        "second_review_notes": "", "dispute_status": "NONE", "eligible_for_test": False})
            if all_pairs_available and pairwise_audit is not None:
                row["pairwise_leakage_audit"] = "PASS" if pairwise_audit.get("status") == "PASS" else "CONFLICTS_REPORTED"
                row["pairwise_leakage_audit_scope"] = "ALL_FOUR_PAIRS"
                row["full_pairwise_leakage_recomputed"] = True
            key = (str(row["pair_id"]), str(row["candidate_id"]))
            if key in self.records:
                raise ValueError(f"duplicate candidate key: {key}")
            self.records[key] = row
        if self.snapshot_path.is_file():
            saved = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            for row in saved.get("records", []):
                key = (str(row.get("pair_id", "")), str(row.get("candidate_id", "")))
                if key not in self.records:
                    raise ValueError(f"review snapshot references unknown candidate {key}")
                # Candidate geometry and suggestions remain sourced from immutable GeoJSON.
                immutable = self.records[key]
                if row.get("suggested_label") != immutable.get("suggested_label"):
                    raise ValueError(f"review snapshot altered frozen suggestion for {key}")
                immutable.update(row)
        if self.events_path.is_file():
            with self.events_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    key = (str(event.get("pair_id", "")), str(event.get("candidate_id", "")))
                    after = event.get("after")
                    if key not in self.records or not isinstance(after, dict):
                        raise ValueError(f"invalid R1.5 review event at line {line_number}")
                    if after.get("suggested_label") != self.records[key].get("suggested_label"):
                        raise ValueError(f"review event altered frozen suggestion for {key}")
                    self.records[key].update(after)

    def get(self, pair_id: str, candidate_id: str) -> dict[str, Any]:
        return deepcopy(self.records[(pair_id, candidate_id)])

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self.records)

    def _event(self, key, event_type, actor, before, after):
        event = {"schema_version": getattr(self, "snapshot_schema_version", 1),
                 "event_id": uuid4().hex, "event_type": event_type,
                 "pair_id": key[0], "candidate_id": key[1], "actor": actor,
                 "event_time_utc": datetime.now(timezone.utc).isoformat(),
                 "before": deepcopy(before), "after": deepcopy(after)}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.save()
        return event

    def record_review(self, pair_id: str, candidate_id: str, *, reviewer: str,
                      status: str, label: str, evidence: str,
                      second_review: bool = False) -> dict[str, Any]:
        key = (pair_id, candidate_id)
        if key not in self.records:
            raise KeyError(key)
        actor = reviewer.strip()
        if not actor:
            raise ValueError("a real reviewer identity is required")
        if status not in REVIEW_STATES - {"UNVERIFIED"} or label not in LABELS:
            raise ValueError("invalid human review status or label")
        if status == "APPROVED" and label == "UNKNOWN":
            raise ValueError("APPROVED requires a determinate human label")
        if status == "APPROVED" and not evidence.strip():
            raise ValueError("approval requires human visual evidence notes")
        old = deepcopy(self.records[key])
        row = self.records[key]
        if second_review:
            if row.get("role") != "TEST":
                raise ValueError("second review is only for frozen TEST candidates")
            if row.get("review_status") != "APPROVED":
                raise ValueError("first human review must be approved before second review")
            first = str(row.get("reviewer", "")).strip().casefold()
            if actor.casefold() == first:
                raise ValueError("second reviewer must be a different real person")
            row.update({"second_review_status": status, "second_reviewer": actor,
                        "second_review_time_utc": datetime.now(timezone.utc).isoformat(),
                        "second_review_label": label, "second_review_notes": evidence.strip()})
            agrees = status == "APPROVED" and label == row.get("human_label")
            row["dispute_status"] = "NONE" if agrees else "UNRESOLVED_DISAGREEMENT"
            try:
                valid_support = float(row.get("valid_support_fraction", 0)) >= 0.95
            except (TypeError, ValueError):
                valid_support = False
            row["eligible_for_test"] = bool(agrees and row.get("role") == "TEST"
                                            and row.get("spatial_status") == "SAFE_SINGLE_CELL"
                                            and row.get("pairwise_leakage_audit") == "PASS"
                                            and row.get("full_pairwise_leakage_recomputed") is True
                                            and valid_support)
            event_type = "SECOND_REVIEW_" + status
        else:
            row.update({"review_status": status, "reviewer": actor,
                        "review_time_utc": datetime.now(timezone.utc).isoformat(),
                        "human_label": label, "evidence_notes": evidence.strip(),
                        "eligible_for_test": False})
            if row.get("role") == "TEST":
                row["second_review_status"] = "UNVERIFIED"
                row["second_reviewer"] = ""
                row["second_review_time_utc"] = ""
            event_type = "FIRST_REVIEW_" + status
        self._event(key, event_type, actor, old, row)
        return deepcopy(row)

    def edit_geometry(self, pair_id: str, candidate_id: str, geometry, *, support_validator,
                      all_candidate_features, editor: str,
                      all_pairs_available: bool = False,
                      preserve_frozen_role: bool = False) -> dict[str, Any]:
        key = (pair_id, candidate_id)
        if key not in self.records:
            raise KeyError(key)
        if not editor.strip():
            raise ValueError("a real geometry editor identity is required")
        before = deepcopy(self.records[key])
        feature = {"type": "Feature", "geometry": before["geometry"],
                   "properties": {k: v for k, v in before.items() if k != "geometry"}}
        new_candidates = []
        for row in all_candidate_features:
            if isinstance(row, dict) and row.get("type") == "Feature":
                new_candidates.append(row)
            else:
                new_candidates.append({"type": "Feature", "geometry": row["geometry"],
                                       "properties": {k: v for k, v in row.items() if k != "geometry"}})
        updated_feature, event = revalidate_geometry_edit(feature, geometry,
            support_validator=support_validator, all_candidates=new_candidates,
            all_pairs_available=all_pairs_available, preserve_frozen_role=preserve_frozen_role)
        row = {**updated_feature["properties"], "geometry": updated_feature["geometry"]}
        row.update({"human_label": "UNKNOWN", "review_status": "UNVERIFIED", "reviewer": "",
                    "review_time_utc": "", "evidence_notes": "", "second_review_status": "UNVERIFIED",
                    "second_reviewer": "", "second_review_time_utc": "", "second_review_label": "UNKNOWN",
                    "second_review_notes": "", "dispute_status": "NONE", "eligible_for_test": False})
        self.records[key] = row
        event.update({"before": before, "after": deepcopy(row), "actor": editor.strip(),
                      "event_id": uuid4().hex, "event_time_utc": datetime.now(timezone.utc).isoformat()})
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.save()
        return deepcopy(row)

    def progress(self) -> dict[str, int]:
        rows = list(self.records.values())
        return {"pending": sum(r.get("review_status") == "UNVERIFIED" for r in rows),
                "reviewed": sum(r.get("review_status") != "UNVERIFIED" for r in rows),
                "second_pending": sum(r.get("role") == "TEST" and r.get("review_status") == "APPROVED"
                                      and r.get("second_review_status") == "UNVERIFIED" for r in rows),
                "second_approved": sum(r.get("second_review_status") == "APPROVED" and
                                        r.get("dispute_status") == "NONE" for r in rows),
                "doubtful": sum(r.get("review_status") == "DOUBTFUL" or r.get("second_review_status") == "DOUBTFUL" for r in rows),
                "rejected": sum(r.get("review_status") == "REJECTED" or r.get("second_review_status") == "REJECTED" for r in rows)}

    def save(self) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{self.snapshot_path.name}.", suffix=".tmp", dir=self.snapshot_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, "records": [self.records[k] for k in sorted(self.records)]},
                          stream, ensure_ascii=False, indent=2)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, self.snapshot_path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
