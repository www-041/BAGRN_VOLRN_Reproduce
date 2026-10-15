"""Nine-scene geographic overlap network diagnostics (metadata only).

Answers: given the original 5 scenes (0-4) plus 4 new scenes (5-8), what is
the real geographic overlap graph, and which scenes neighbour scene0 / scene1?

Reuses the project's existing overlap semantics: bounding-rectangle
intersection and ``overlap_ratio_i = intersection_area / area_i`` as defined
in ``src.multiscene_sift.overlap_graph.py``.  No registration is run; this is
pure footprint/timing statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.multiscene_sift.models import Scene
from src.multiscene_sift.overlap_graph import _intersection_area

BAND_DEFAULT = "B14"

# Fixed order: original 5 scenes then 4 new scenes.  Index order is fixed and
# must not be re-sorted.
SCENE_NAMES_9 = [
    "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1",  # 0
    "DZ01V_L2_E113.6_N36.3_20260616031133_01_T1",  # 1
    "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",  # 2
    "DZ01V_L2_E114.0_N36.4_20260714030410_01_T1",  # 3
    "DZ01V_L2_E113.7_N36.6_20260616031127_01_T1",  # 4
    "DZ01V_L2_E113.1_N36.8_20260222031830_01_T1",  # 5 (new)
    "DZ01V_L2_E113.5_N36.9_20260810030926_01_T1",  # 6 (new)
    "DZ01V_L2_E113.9_N36.0_20260714030416_01_T1",  # 7 (new)
    "DZ01V_L2_E113.4_N35.9_20260616031139_01_T1",  # 8 (new)
]

ORIGINAL_INDICES = {0, 1, 2, 3, 4}
NEW_INDICES = {5, 6, 7, 8}


@dataclass
class Footprint:
    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def area(self) -> float:
        return self.width * self.height


def validate_scene_names(names: list[str]) -> bool:
    """Exactly nine, unique names (a duplicate like a re-packed scene4 is rejected)."""
    return len(names) == 9 and len(set(names)) == 9


def parse_acquisition_datetime(name: str) -> datetime:
    """Parse ``..._YYYYMMDDHHMMSS_...`` from a scene name."""
    m = re.search(r"_(\d{8})(\d{6})_", name)
    if not m:
        raise ValueError(f"Cannot parse acquisition timestamp from {name!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def scene_footprint(scene: Scene, band: str = BAND_DEFAULT) -> Footprint:
    b = scene.bounds[band]
    return Footprint(b.left, b.bottom, b.right, b.top)


def acquisition_groups(scenes: list[Scene]) -> dict:
    """Group scenes by acquisition date (from names, never hard-coded)."""
    groups: dict[str, dict] = {}
    for idx, scene in enumerate(scenes):
        dt = parse_acquisition_datetime(scene.name)
        key = dt.strftime("%Y-%m-%d")
        entry = groups.setdefault(key, {
            "date": key,
            "scene_indices": [],
            "scene_names": [],
            "timestamps": [],
        })
        entry["scene_indices"].append(idx)
        entry["scene_names"].append(scene.name)
        entry["timestamps"].append(dt.strftime("%H:%M:%S"))
    gaps = {}
    for key, entry in groups.items():
        items = sorted(zip(entry["timestamps"], entry["scene_indices"]))
        times = [datetime.strptime(t, "%H:%M:%S") for t, _ in items]
        entry["scene_indices"] = [i for _, i in items]
        entry["timestamps"] = [t for t, _ in items]
        entry["time_gap_seconds"] = [
            int((b - a).total_seconds())
            for a, b in zip(times, times[1:])
        ]
        gaps[key] = entry
        entry["note"] = (
            "acquisition-time sequence only; orbit/strip relation is NOT claimed"
        )
    return gaps


def all_pair_overlaps(scenes: list[Scene], band: str = BAND_DEFAULT) -> list[dict]:
    """All C(n,2) pairs with full geometry, including zero-overlap pairs.

    Reuses ``overlap_graph._intersection_area`` (exact production semantics).
    """
    n = len(scenes)
    rows = []
    for i in range(n):
        fp_i = scene_footprint(scenes[i], band)
        for j in range(i + 1, n):
            fp_j = scene_footprint(scenes[j], band)
            inter = _intersection_area(
                (fp_i.left, fp_i.bottom, fp_i.right, fp_i.top),
                (fp_j.left, fp_j.bottom, fp_j.right, fp_j.top),
            )
            rows.append({
                "idx_i": i, "idx_j": j,
                "scene_i": scenes[i].name, "scene_j": scenes[j].name,
                "intersection_exists": inter > 0,
                "intersection_width_m": None if inter <= 0 else (
                    min(fp_i.right, fp_j.right) - max(fp_i.left, fp_j.left)),
                "intersection_height_m": None if inter <= 0 else (
                    min(fp_i.top, fp_j.top) - max(fp_i.bottom, fp_j.bottom)),
                "intersection_area_m2": inter,
                "scene_i_area_m2": fp_i.area,
                "scene_j_area_m2": fp_j.area,
                "overlap_ratio_i": inter / fp_i.area if fp_i.area > 0 else 0.0,
                "overlap_ratio_j": inter / fp_j.area if fp_j.area > 0 else 0.0,
            })
    return rows


def overlap_matrices(
    overlaps: list[dict], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """(ratio_matrix, exists_matrix) symmetric n×n, diag ratio=1, exists=1."""
    ratio = np.zeros((n, n), dtype=float)
    exists = np.zeros((n, n), dtype=int)
    np.fill_diagonal(ratio, 1.0)
    np.fill_diagonal(exists, 1)
    for r in overlaps:
        i, j = r["idx_i"], r["idx_j"]
        ratio[i, j] = ratio[j, i] = r["overlap_ratio_i"]
        exists[i, j] = exists[j, i] = 1 if r["intersection_exists"] else 0
    return ratio, exists


def neighbors_of(
    scenes: list[Scene], overlaps: list[dict], idx: int
) -> list[dict]:
    """Neighbours of *idx* with a real intersection, sorted by ratio desc."""
    neigh = []
    for r in overlaps:
        if r["idx_i"] == idx and r["intersection_exists"]:
            other = r["idx_j"]
            ratio = r["overlap_ratio_i"]
        elif r["idx_j"] == idx and r["intersection_exists"]:
            other = r["idx_i"]
            ratio = r["overlap_ratio_j"]
        else:
            continue
        neigh.append({
            "neighbor_scene": other,
            "neighbor_name": scenes[other].name,
            "overlap_ratio": ratio,
            "overlap_area_km2": r["intersection_area_m2"] / 1e6,
            "overlap_area_m2": r["intersection_area_m2"],
            "same_acquisition_date": (
                parse_acquisition_datetime(scenes[idx].name).date()
                == parse_acquisition_datetime(scenes[other].name).date()
            ),
            "time_gap_seconds": abs(
                int((parse_acquisition_datetime(scenes[idx].name)
                     - parse_acquisition_datetime(scenes[other].name))
                    .total_seconds())),
            "existing_or_new": "existing" if other in ORIGINAL_INDICES else "new",
            "note": (
                "tiny bounds corner overlap; verify with imagery"
                if ratio < 0.01 else ""
            ),
        })
    return sorted(neigh, key=lambda x: x["overlap_ratio"], reverse=True)


def enumerate_triangles(
    overlaps: list[dict],
) -> list[tuple[int, int, int]]:
    """All 3-node cycles (a<b<c with all three geographic edges present)."""
    edges = {
        (r["idx_i"], r["idx_j"])
        for r in overlaps if r["intersection_exists"]
    }
    tri = []
    for a in range(9):
        for b in range(a + 1, 9):
            if (a, b) not in edges:
                continue
            for c in range(b + 1, 9):
                if (a, c) in edges and (b, c) in edges:
                    tri.append((a, b, c))
    return tri


def triangle_priority(
    tri: tuple[int, int, int], overlaps_by_pair: dict[tuple[int, int], float],
    overlaps: list[dict],
) -> tuple[str, float, float]:
    """Priority A/B/C and min/mean overlap for a triangle."""
    a, b, c = tri
    ratios = []
    for pair in ((a, b), (a, c), (b, c)):
        lo, hi = min(pair), max(pair)
        ratios.append(overlaps_by_pair[(lo, hi)])
    mn, mean = float(np.min(ratios)), float(np.mean(ratios))
    has0 = 0 in tri
    has1 = 1 in tri
    has_new = any(x in NEW_INDICES for x in tri)
    if has0 and has1 and has_new:
        priority = "A"
    elif (has0 or has1) and has_new:
        priority = "B"
    else:
        priority = "C"
    return priority, mn, mean


def triangle_rows(
    triangles: list[tuple[int, int, int]],
    scenes: list[Scene],
    overlaps: list[dict],
) -> list[dict]:
    """CSV-ready triangle rows with priority and overlap stats."""
    by_pair = {
        (r["idx_i"], r["idx_j"]): r["overlap_ratio_i"]
        for r in overlaps
        if r["intersection_exists"]
    }
    rows = []
    for tid, tri in enumerate(triangles):
        priority, mn, mean = triangle_priority(tri, by_pair, overlaps)
        rows.append({
            "triangle_id": tid,
            "scene_a": tri[0], "scene_b": tri[1], "scene_c": tri[2],
            "overlap_ab": round(by_pair[(min(tri[0], tri[1]), max(tri[0], tri[1]))], 4),
            "overlap_bc": round(by_pair[(min(tri[1], tri[2]), max(tri[1], tri[2]))], 4),
            "overlap_ca": round(by_pair[(min(tri[0], tri[2]), max(tri[0], tri[2]))], 4),
            "min_overlap": round(mn, 4),
            "mean_overlap": round(mean, 4),
            "contains_scene0": int(0 in tri),
            "contains_scene1": int(1 in tri),
            "contains_problem_edge_0_1": int(0 in tri and 1 in tri),
            "priority": priority,
        })
    rows.sort(key=lambda r: (r["priority"], -r["min_overlap"]))
    return rows


def select_recommended_pairs(
    scenes: list[Scene],
    overlaps: list[dict],
    triangles: list[tuple[int, int, int]],
    max_n: int = 10,
    min_overlap_ratio: float = 0.01,
) -> list[dict]:
    """Recommend the next registration pairs (no registration is run here).

    Prioritises evidence for new scene0/scene1 neighbours and cycles that
    would extend the 9-scene network.
    """
    by_pair = {
        (r["idx_i"], r["idx_j"]): r for r in overlaps
    }
    triangle_edges: dict[tuple[int, int], set[str]] = {}
    for tri in triangles:
        for pair in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
            lo, hi = min(pair), max(pair)
            triangle_edges.setdefault((lo, hi), set()).add(str(tri))

    def _score(row: dict) -> float:
        i, j = row["idx_i"], row["idx_j"]
        enables = triangle_edges.get((i, j), set())
        prob0 = i == 0 or j == 0
        prob1 = i == 1 or j == 1
        new_side = (i in NEW_INDICES) or (j in NEW_INDICES)
        score = 0.0
        if prob0 and new_side:
            score += 4.0
        if prob1 and new_side:
            score += 3.0
        if enables:
            score += 2.0
        if (prob0 or prob1) and not new_side:
            score += 1.0
        return score

    candidates = [
        r for r in overlaps
        if r["intersection_exists"]
        and r["overlap_ratio_i"] >= min_overlap_ratio
        and (
            0 in (r["idx_i"], r["idx_j"])
            or 1 in (r["idx_i"], r["idx_j"])
            or (r["idx_i"], r["idx_j"]) in triangle_edges
        )
    ]
    candidates.sort(key=lambda r: (_score(r), -r["overlap_ratio_i"]),
                    reverse=True)
    out = []
    for row in candidates[:max_n]:
        i, j = row["idx_i"], row["idx_j"]
        enables = triangle_edges.get((i, j), set())
        out.append({
            "pair": f"{i}-{j}",
            "overlap_ratio": round(row["overlap_ratio_i"], 4),
            "overlap_area_km2": round(row["intersection_area_m2"] / 1e6, 2),
            "why_important": _why(i, j, row, enables),
            "enables_cycle": sorted(enables)[:3],
            "probes_scene0": int(i == 0 or j == 0),
            "probes_scene1": int(i == 1 or j == 1),
        })
    return out


def _why(i: int, j: int, row: dict, enables: set[str]) -> str:
    reasons = []
    if 0 in (i, j):
        other = j if i == 0 else i
        kind = "existing" if other in ORIGINAL_INDICES else "new neighbor"
        reasons.append(f"probes scene0 ({kind} {other})")
    if 1 in (i, j):
        other = j if i == 1 else i
        kind = "existing" if other in ORIGINAL_INDICES else "new neighbor"
        reasons.append(f"probes scene1 ({kind} {other})")
    if enables:
        reasons.append(f"enables triangle(s) {', '.join(sorted(enables)[:2])}")
    return "; ".join(reasons) if reasons else "high-overlap pair"