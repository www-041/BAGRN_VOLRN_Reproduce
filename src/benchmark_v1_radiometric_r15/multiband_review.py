from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .review_store import AutoReviewStore, LABELS, REVIEW_STATES


LANDCOVER_TYPES = {"VEGETATION", "BARE_GROUND", "WATER", "BUILT", "MIXED", "OTHER", "UNKNOWN"}
TERRAIN_TYPES = {"RIDGE", "SLOPE", "VALLEY", "FLAT", "OTHER", "UNKNOWN"}
TEMPORAL_STATUSES = {"NO_CHANGE", "POSSIBLE_CHANGE", "CONFIRMED_CHANGE", "UNCERTAIN", "UNKNOWN"}
INTERFERENCE_TYPES = {"NONE", "CLOUD", "CLOUD_SHADOW", "TERRAIN_SHADOW", "HAZE",
                      "SATURATION", "OTHER", "UNKNOWN"}
LABEL_FIELDS = ("landcover_type", "terrain_type", "temporal_status", "interference_type")
_ALLOWED = {"landcover_type": LANDCOVER_TYPES, "terrain_type": TERRAIN_TYPES,
            "temporal_status": TEMPORAL_STATUSES, "interference_type": INTERFERENCE_TYPES}


class MultiBandReviewStore(AutoReviewStore):
    """Schema-v2 review state stored separately from the legacy R1.5 review log."""

    def __init__(self, candidate_path: str | Path, snapshot_path: str | Path, events_path: str | Path,
                 *, pairwise_audit: dict[str, Any] | None = None, all_pairs_available: bool = False,
                 legacy_snapshot_path: str | Path | None = None,
                 legacy_events_path: str | Path | None = None):
        self.snapshot_schema_version = 2
        super().__init__(candidate_path, snapshot_path, events_path,
                         pairwise_audit=pairwise_audit, all_pairs_available=all_pairs_available)
        legacy = _read_legacy_reviews(legacy_snapshot_path, legacy_events_path)
        for key, row in self.records.items():
            legacy_row = legacy.get(key, {})
            row["legacy_human_label"] = legacy_row.get("human_label", row.get("human_label", "UNKNOWN"))
            row["legacy_review_status"] = legacy_row.get("review_status", row.get("review_status", "UNVERIFIED"))
            row["legacy_reviewer"] = legacy_row.get("reviewer", "")
            row["legacy_review_time_utc"] = legacy_row.get("review_time_utc", "")
            row["frozen_role"] = row.get("frozen_role", row.get("role", "UNASSIGNED"))
            row["frozen_grid_cell_id"] = row.get("frozen_grid_cell_id", row.get("grid_cell_id", ""))
            try:
                initial_support_ok = float(row.get("valid_support_fraction", 0)) >= 0.95
            except (TypeError, ValueError):
                initial_support_ok = False
            if row.get("spatial_status") == "SAFE_SINGLE_CELL" and initial_support_ok:
                if all_pairs_available and row.get("pairwise_leakage_audit") == "PASS":
                    row["spatial_edit_eligibility"] = "SAFE_FROZEN_ROLE"
                else:
                    row["spatial_edit_eligibility"] = "PENDING_PAIRWISE_REAUDIT"
            else:
                row["spatial_edit_eligibility"] = "INELIGIBLE_INITIAL_GEOMETRY_OR_SUPPORT"
            row.setdefault("geometry_correspondence_verified", False)
            row.setdefault("landcover_type", "UNKNOWN")
            row.setdefault("terrain_type", "UNKNOWN")
            row.setdefault("temporal_status", "UNKNOWN")
            row.setdefault("interference_type", "UNKNOWN")
            row.setdefault("multiband_review_decision", "UNVERIFIED")
            row.setdefault("multiband_reviewer", "")
            row.setdefault("multiband_review_time_utc", "")
            row.setdefault("multiband_evidence", "")
            row.setdefault("multiband_second_review_status", "UNVERIFIED")
            row.setdefault("multiband_second_reviewer", "")
            row.setdefault("multiband_second_review_time_utc", "")
            row.setdefault("multiband_second_evidence", "")
            row.setdefault("multiband_dispute_status", "NONE")
            row["formal_stable_truth_eligible"] = False
            row.setdefault("eligible_for_test", False)

    def save(self) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{self.snapshot_path.name}.", suffix=".tmp",
                                    dir=self.snapshot_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 2, "schema": "R1.5_MULTIBAND_VISUAL_REVIEW_V2",
                           "records": [self.records[k] for k in sorted(self.records)]},
                          stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.snapshot_path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def record_labels(self, pair_id: str, candidate_id: str, *, reviewer: str, decision: str,
                      landcover_type: str, terrain_type: str, temporal_status: str,
                      interference_type: str, evidence: str, second_review: bool = False) -> dict[str, Any]:
        key = (pair_id, candidate_id)
        if key not in self.records:
            raise KeyError(key)
        actor = reviewer.strip()
        if not actor:
            raise ValueError("a real reviewer identity is required")
        if decision not in REVIEW_STATES:
            raise ValueError("invalid multiband human review decision")
        values = {"landcover_type": landcover_type, "terrain_type": terrain_type,
                  "temporal_status": temporal_status, "interference_type": interference_type}
        for name, value in values.items():
            if value not in _ALLOWED[name]:
                raise ValueError(f"invalid {name}: {value}")
        if decision == "APPROVED" and not evidence.strip():
            raise ValueError("APPROVED multiband labels require human evidence notes")
        if temporal_status == "CONFIRMED_CHANGE":
            parts = {key.strip().casefold(): value.strip() for line in evidence.splitlines()
                     if ":" in line for key, value in (line.split(":", 1),)}
            if not parts.get("source") or not parts.get("reference"):
                raise ValueError("CONFIRMED_CHANGE requires checkable evidence with non-empty 'source:' and 'reference:' lines")
        before = deepcopy(self.records[key])
        row = self.records[key]
        if second_review:
            if row.get("role") != "TEST":
                raise ValueError("independent second label review is only for frozen TEST candidates")
            if row.get("multiband_review_decision") != "APPROVED":
                raise ValueError("first multiband label review must be approved before second review")
            if actor.casefold() == str(row.get("multiband_reviewer", "")).strip().casefold():
                raise ValueError("second reviewer must be a different real reviewer")
            row.update({f"second_{name}": value for name, value in values.items()})
            row.update({"multiband_second_review_status": decision, "multiband_second_reviewer": actor,
                        "multiband_second_review_time_utc": _utc_now(), "multiband_second_evidence": evidence.strip()})
            agrees = decision == "APPROVED" and all(row.get(f"second_{name}") == row.get(name) for name in LABEL_FIELDS)
            row["multiband_dispute_status"] = "NONE" if agrees else "UNRESOLVED_DISAGREEMENT"
        else:
            row.update(values)
            row.update({"multiband_review_decision": decision, "multiband_reviewer": actor,
                        "multiband_review_time_utc": _utc_now(), "multiband_evidence": evidence.strip(),
                        "multiband_second_review_status": "UNVERIFIED", "multiband_second_reviewer": "",
                        "multiband_second_review_time_utc": "", "multiband_second_evidence": "",
                        "multiband_dispute_status": "NONE", "eligible_for_test": False})
        row["temporal_status_scope"] = "VISUAL_OBSERVATION_AT_NATIVE_RESOLUTION"
        # NO_CHANGE is not a formal STABLE truth; formal eligibility remains a separate, unmet gate.
        row["formal_stable_truth_eligible"] = False
        if row.get("role") == "TEST" and second_review:
            row["eligible_for_test"] = bool(
                agrees and row.get("spatial_edit_eligibility") == "SAFE_FROZEN_ROLE"
                and row.get("pairwise_leakage_audit") == "PASS"
                and row.get("full_pairwise_leakage_recomputed") is True
                and row.get("geometry_correspondence_verified") is True
                and row.get("interference_type") == "NONE"
                and float(row.get("valid_support_fraction") or 0) >= 0.95)
        event_type = "MULTIBAND_SECOND_REVIEW_" if second_review else "MULTIBAND_REVIEW_"
        self._event(key, event_type + decision, actor, before, row)
        return deepcopy(row)

    def progress(self) -> dict[str, int]:
        rows = list(self.records.values())
        return {"label_pending": sum(r.get("multiband_review_decision") == "UNVERIFIED" for r in rows),
                "label_reviewed": sum(r.get("multiband_review_decision") != "UNVERIFIED" for r in rows),
                "test_second_pending": sum(r.get("role") == "TEST" and r.get("multiband_review_decision") == "APPROVED"
                                           and r.get("multiband_second_review_status") == "UNVERIFIED" for r in rows),
                "test_second_approved": sum(r.get("role") == "TEST" and r.get("multiband_second_review_status") == "APPROVED"
                                            and r.get("multiband_dispute_status") == "NONE" for r in rows),
                "doubtful": sum(r.get("multiband_review_decision") == "DOUBTFUL" or
                                r.get("multiband_second_review_status") == "DOUBTFUL" for r in rows),
                "rejected": sum(r.get("multiband_review_decision") == "REJECTED" or
                                r.get("multiband_second_review_status") == "REJECTED" for r in rows)}

    def edit_geometry(self, pair_id: str, candidate_id: str, geometry, *, support_validator,
                      all_candidate_features, editor: str,
                      all_pairs_available: bool = False,
                      preserve_frozen_role: bool = True) -> dict[str, Any]:
        updated = super().edit_geometry(pair_id, candidate_id, geometry,
            support_validator=support_validator, all_candidate_features=all_candidate_features,
            editor=editor, all_pairs_available=all_pairs_available, preserve_frozen_role=True)
        key = (pair_id, candidate_id)
        before = deepcopy(self.records[key])
        for name in LABEL_FIELDS:
            self.records[key][name] = "UNKNOWN"
            self.records[key][f"second_{name}"] = "UNKNOWN"
        self.records[key].update({"multiband_review_decision": "UNVERIFIED", "multiband_reviewer": "",
                                  "multiband_review_time_utc": "", "multiband_evidence": "",
                                  "multiband_second_review_status": "UNVERIFIED", "multiband_second_reviewer": "",
                                  "multiband_second_review_time_utc": "", "multiband_second_evidence": "",
                                  "multiband_dispute_status": "NONE", "eligible_for_test": False,
                                  "formal_stable_truth_eligible": False})
        self._event(key, "MULTIBAND_LABELS_INVALIDATED_BY_GEOMETRY_EDIT", editor.strip(), before, self.records[key])
        return self.get(*key)


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _read_legacy_reviews(snapshot_path: str | Path | None,
                         events_path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Read legacy R1.5 review artifacts without writing or modifying them."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if snapshot_path and Path(snapshot_path).is_file():
        saved = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        for row in saved.get("records", []):
            key = (str(row.get("pair_id", "")), str(row.get("candidate_id", "")))
            rows[key] = {name: row.get(name) for name in ("human_label", "review_status", "reviewer", "review_time_utc")}
    if events_path and Path(events_path).is_file():
        with Path(events_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                after = event.get("after") or {}
                key = (str(event.get("pair_id", "")), str(event.get("candidate_id", "")))
                if after:
                    rows.setdefault(key, {}).update({name: after.get(name) for name in
                        ("human_label", "review_status", "reviewer", "review_time_utc") if name in after})
    return rows
