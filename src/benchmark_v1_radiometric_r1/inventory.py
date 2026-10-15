from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MANDATORY_PAIRS = {"00_01", "04_11", "02_10", "00_10"}
BIN_TARGETS = {"T0": 2, "T1": 2, "T2": 2, "T3": 4}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def stratify_pair(row: dict[str, Any]) -> str:
    """Assign frozen T0/T1/T2/T3 bins; no effective support is not an algorithm fail."""
    valid = _int(row.get("valid_overlap_pixels"))
    if valid <= 0:
        return "NO_VALID_OVERLAP"
    try:
        days = float(row["days_apart"])
    except (KeyError, TypeError, ValueError):
        return "UNKNOWN_TIME"
    if not (days >= 0):
        return "UNKNOWN_TIME"
    if str(row.get("same_date", "")).lower() == "true" or days < 1e-3:
        return "T0"
    if days < 7:
        return "T1"
    if days < 30:
        return "T2"
    return "T3"


def build_pair_inventory(
    pair_candidates_csv: Path,
    temporal_inventory_csv: Path,
    scene_inventory_csv: Path,
) -> list[dict[str, Any]]:
    """Join the frozen 78-pair table to the 48-pair true-overlap table."""
    candidates = _read_csv(pair_candidates_csv)
    temporal = _read_csv(temporal_inventory_csv)
    scenes = _read_csv(scene_inventory_csv)
    if len(scenes) != 13:
        raise ValueError(f"expected 13 scene records, found {len(scenes)}")
    if len(candidates) != 78:
        raise ValueError(f"expected 78 pair combinations, found {len(candidates)}")
    by_pair = {row["pair_id"]: row for row in temporal}
    if len(by_pair) != len(temporal):
        raise ValueError("duplicate pair_id in temporal pair inventory")
    rows: list[dict[str, Any]] = []
    valid_pair_ids: set[str] = set()
    for candidate in candidates:
        pair_id = candidate["pair_id"]
        valid_pixels = _int(candidate.get("valid_overlap_pixels"))
        if valid_pixels > 0:
            valid_pair_ids.add(pair_id)
            if pair_id not in by_pair:
                raise ValueError(f"effective overlap {pair_id} missing from temporal inventory")
            temporal_row = by_pair[pair_id]
            if _int(temporal_row.get("valid_overlap_pixels")) != valid_pixels:
                raise ValueError(f"effective overlap pixel count differs for {pair_id}")
        elif pair_id in by_pair:
            raise ValueError(f"zero-support pair {pair_id} is listed as a valid temporal overlap")
        merged = {**candidate}
        if pair_id in by_pair:
            merged.update({k: v for k, v in by_pair[pair_id].items() if v not in (None, "")})
        merged["temporal_bin"] = stratify_pair(merged)
        merged["status"] = "NO_VALID_OVERLAP" if valid_pixels <= 0 else "CANDIDATE_REQUIRES_HUMAN_REVIEW"
        merged["stable_roi_review"] = "UNVERIFIED"
        merged["real_change_review"] = "UNVERIFIED"
        merged["source_scene_count"] = 2
        rows.append(merged)
    if valid_pair_ids != set(by_pair):
        missing = sorted(valid_pair_ids.symmetric_difference(by_pair))
        raise ValueError(f"candidate/temporal valid-overlap pair sets differ: {missing[:8]}")
    return rows


def _selection_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    quality = (row.get("quality_i") == "PASS") + (row.get("quality_j") == "PASS")
    return (-int(quality), -_int(row.get("valid_overlap_pixels")), str(row["pair_id"]))


def select_representative_pairs(rows: list[dict[str, Any]], target: int = 10) -> list[dict[str, Any]]:
    """Select a deterministic time-stratified draft, never consulting matcher metrics."""
    if not 8 <= target <= 12:
        raise ValueError("target shortlist size must be between 8 and 12")
    available = {r["pair_id"]: r for r in rows if r.get("temporal_bin") in BIN_TARGETS}
    selected: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(MANDATORY_PAIRS):
        if pair_id not in available:
            raise ValueError(f"mandatory radiometric pair is missing or has no valid overlap: {pair_id}")
        selected[pair_id] = available[pair_id]
    for bin_name, quota in BIN_TARGETS.items():
        bin_rows = sorted(
            (r for r in available.values() if r["temporal_bin"] == bin_name and r["pair_id"] not in selected),
            key=_selection_rank,
        )
        already = sum(r["temporal_bin"] == bin_name for r in selected.values())
        for item in bin_rows[: max(0, quota - already)]:
            if len(selected) >= target:
                break
            selected[item["pair_id"]] = item
    remaining = sorted((r for key, r in available.items() if key not in selected), key=_selection_rank)
    for item in remaining:
        if len(selected) >= target:
            break
        selected[item["pair_id"]] = item
    if len(selected) < min(target, len(available)):
        raise ValueError("not enough valid overlap pairs for requested shortlist")
    return sorted(selected.values(), key=lambda r: (r["temporal_bin"], str(r["pair_id"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty radiometric inventory")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
