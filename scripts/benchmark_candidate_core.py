"""Lightweight, independent B9 candidate matching and audited human review.

This module does not import Task15 matchers or any formal experiment outputs.
Its recommendations are never evaluation truth until separately reviewed.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from affine import Affine

from benchmark_phase2_core import TRUTH_FIELDS, pixel_to_projected


CANDIDATE_FIELDS = [
    "candidate_id", "pair_id", "scene_i", "scene_j", "row_i", "col_i", "x_i", "y_i",
    "row_j", "col_j", "x_j", "y_j", "recommendation_method", "matching_score",
    "forward_backward_error", "uniqueness_margin", "texture_strength", "grid_cell_id",
    "recommendation_status", "human_review_status", "human_adjusted",
    "adjusted_row_i", "adjusted_col_i", "adjusted_x_i", "adjusted_y_i",
    "adjusted_row_j", "adjusted_col_j", "adjusted_x_j", "adjusted_y_j",
    "landcover_type", "terrain_class", "confidence", "coordinate_precision_note", "split",
    "primary_reviewer", "second_reviewer", "reviewer", "review_time_utc", "review_notes",
]
AUTO_TRUTH_FIELDS = TRUTH_FIELDS + ["point_source", "origin_candidate_id", "primary_reviewer", "second_reviewer"]


def _stretch_u8(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(image, dtype=np.float32)[mask]
    if not valid.size:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    result = np.clip((image.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    result[~mask] = 0
    return result


def _gradient(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    visual = _stretch_u8(image, mask).astype(np.float32) / 255.0
    gx = cv2.Sobel(visual, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(visual, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _valid_square(mask: np.ndarray, row: int, col: int, radius: int) -> bool:
    if row-radius < 0 or col-radius < 0 or row+radius >= mask.shape[0] or col+radius >= mask.shape[1]:
        return False
    return bool(np.all(mask[row-radius:row+radius+1, col-radius:col+radius+1]))


def _safe_centres(mask: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    safe = cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return safe.astype(bool)


def _one_way(template_image: np.ndarray, search_image: np.ndarray, template_mask: np.ndarray,
             search_mask: np.ndarray, start: tuple[int, int], expected: tuple[int, int],
             patch_radius: int, search_radius: int,
             safe_template: np.ndarray, safe_search: np.ndarray):
    row, col = start
    erow, ecol = expected
    if not (0 <= row < safe_template.shape[0] and 0 <= col < safe_template.shape[1]
            and safe_template[row, col]):
        return None
    large = patch_radius + search_radius
    if erow-large < 0 or ecol-large < 0 or erow+large >= search_image.shape[0] or ecol+large >= search_image.shape[1]:
        return None
    template = template_image[row-patch_radius:row+patch_radius+1,
                              col-patch_radius:col+patch_radius+1]
    if float(np.std(template)) < 0.025:
        return None
    search = search_image[erow-large:erow+large+1, ecol-large:ecol+large+1]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    valid_centres = safe_search[erow-search_radius:erow+search_radius+1,
                                ecol-search_radius:ecol+search_radius+1]
    if not np.any(valid_centres):
        return None
    scores[valid_centres == 0] = -2.0
    _, best, _, loc = cv2.minMaxLoc(scores)
    if best < 0.58:
        return None
    best_x, best_y = loc
    copy = scores.copy()
    copy[max(0, best_y-2):best_y+3, max(0, best_x-2):best_x+3] = -2
    second = float(np.max(copy)) if copy.size else -2.0
    margin = float(best - second)
    if margin < 0.035:
        return None
    return (erow-search_radius+best_y, ecol-search_radius+best_x, float(best), margin,
            float(np.std(template)))


def match_bidirectional(left: np.ndarray, right: np.ndarray,
                        mask_i: np.ndarray, mask_j: np.ndarray,
                        point_i: tuple[int, int], *, search_radius: int = 10,
                        patch_radius: int = 10,
                        gradient_i: np.ndarray | None = None,
                        gradient_j: np.ndarray | None = None,
                        safe_i: np.ndarray | None = None,
                        safe_j: np.ndarray | None = None) -> dict | None:
    """Gradient-NCC forward/backward match on a common geospatial overlap grid."""
    if left.shape != right.shape or left.shape != mask_i.shape or right.shape != mask_j.shape:
        raise ValueError("Images and masks must share the georeferenced overlap grid")
    if gradient_i is None:
        gradient_i = _gradient(left, mask_i)
    if gradient_j is None:
        gradient_j = _gradient(right, mask_j)
    if safe_i is None:
        safe_i = _safe_centres(mask_i, patch_radius + 1)
    if safe_j is None:
        safe_j = _safe_centres(mask_j, patch_radius + 1)
    forward = _one_way(gradient_i, gradient_j, mask_i, mask_j, point_i, point_i,
                       patch_radius, search_radius, safe_i, safe_j)
    if forward is None:
        return None
    row_j, col_j, score, margin, texture = forward
    backward = _one_way(gradient_j, gradient_i, mask_j, mask_i, (row_j, col_j), point_i,
                        patch_radius, search_radius, safe_j, safe_i)
    if backward is None:
        return None
    error = float(np.hypot(backward[0] - point_i[0], backward[1] - point_i[1]))
    if error > 1.0:
        return None
    return {"row_i": float(point_i[0]), "col_i": float(point_i[1]),
            "row_j": float(row_j), "col_j": float(col_j),
            "matching_score": score, "forward_backward_error": error,
            "uniqueness_margin": margin, "texture_strength": texture}


def deduplicate_candidates(rows: list[dict], min_distance_px: float = 32) -> list[dict]:
    chosen = []
    for row in sorted(rows, key=lambda r: float(r["matching_score"]), reverse=True):
        if all(np.hypot(float(row["row_i"])-float(old["row_i"]),
                        float(row["col_i"])-float(old["col_i"])) >= min_distance_px
               for old in chosen):
            chosen.append(row)
    return chosen


def select_spatially_balanced(per_cell: list[list[dict]], max_candidates: int,
                              min_distance_px: float) -> list[dict]:
    """Choose across cells by farthest-point coverage before taking second points."""
    selected: list[dict] = []
    remaining = [bucket[:] for bucket in per_cell if bucket]
    while remaining and len(selected) < max_candidates:
        scored = []
        for index, bucket in enumerate(remaining):
            point = bucket[0]
            distances = [np.hypot(float(point["row_i"])-float(old["row_i"]),
                                  float(point["col_i"])-float(old["col_i"])) for old in selected]
            nearest = min(distances) if distances else float("inf")
            if nearest >= min_distance_px:
                scored.append((nearest, float(point["matching_score"]), -index, index))
        if not scored:
            break
        index = max(scored)[3]
        selected.append(remaining[index].pop(0))
        if not remaining[index]:
            remaining.pop(index)
    return selected


def generate_candidates_arrays(left: np.ndarray, right: np.ndarray,
                               mask_i: np.ndarray, mask_j: np.ndarray,
                               transform_i: Affine, transform_j: Affine,
                               *, pair_id: str, scene_i: str, scene_j: str,
                               max_candidates: int = 40, grid_side: int = 8,
                               search_radius: int = 10, patch_radius: int = 10,
                               min_distance_px: float = 32) -> list[dict]:
    """At most one per cell in the first pass, then fill from remaining cells."""
    if max_candidates < 1 or grid_side < 1:
        raise ValueError("max_candidates and grid_side must be positive")
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    mask_i, mask_j = np.asarray(mask_i, bool), np.asarray(mask_j, bool)
    if left.shape != right.shape or left.shape != mask_i.shape or right.shape != mask_j.shape:
        raise ValueError("Common-grid image/mask shape mismatch")
    shared = mask_i & mask_j & np.isfinite(left) & np.isfinite(right)
    if not np.any(shared):
        return []
    gi, gj = _gradient(left, shared), _gradient(right, shared)
    safe = _safe_centres(shared, patch_radius + 1)
    corners = _stretch_u8(left, shared)
    height, width = left.shape
    per_cell: list[list[dict]] = []
    for gy in range(grid_side):
        y0, y1 = round(gy*height/grid_side), round((gy+1)*height/grid_side)
        for gx in range(grid_side):
            x0, x1 = round(gx*width/grid_side), round((gx+1)*width/grid_side)
            if y1-y0 < 8 or x1-x0 < 8:
                continue
            cell_mask = shared[y0:y1, x0:x1].astype(np.uint8)
            points = cv2.goodFeaturesToTrack(corners[y0:y1, x0:x1], maxCorners=10, qualityLevel=.02,
                                             minDistance=max(8, min(y1-y0, x1-x0)//8),
                                             mask=cell_mask, blockSize=5,
                                             useHarrisDetector=False)
            bucket = []
            if points is not None:
                for pt in points.reshape(-1, 2):
                    col, row = x0 + int(round(float(pt[0]))), y0 + int(round(float(pt[1])))
                    match = match_bidirectional(left, right, shared, shared, (row, col),
                                                search_radius=search_radius,
                                                patch_radius=patch_radius,
                                                gradient_i=gi, gradient_j=gj,
                                                safe_i=safe, safe_j=safe)
                    if match is None:
                        continue
                    xi, yi = pixel_to_projected(transform_i, match["row_i"], match["col_i"])
                    xj, yj = pixel_to_projected(transform_j, match["row_j"], match["col_j"])
                    bucket.append({**match, "x_i": xi, "y_i": yi, "x_j": xj, "y_j": yj,
                                   "pair_id": pair_id, "scene_i": scene_i, "scene_j": scene_j,
                                   "grid_cell_id": f"r{gy:02d}_c{gx:02d}",
                                   "recommendation_method": "ShiTomasi_gradientNCC_bidirectional",
                                   "recommendation_status": "SUGGESTED",
                                   "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""})
            per_cell.append(sorted(bucket, key=lambda r: r["matching_score"], reverse=True))
    chosen = select_spatially_balanced(per_cell, max_candidates, min_distance_px)
    for index, row in enumerate(chosen, 1):
        row["candidate_id"] = f"{pair_id}_C{index:04d}"
    return chosen


class CandidateReviewStore:
    def __init__(self, path: Path, pair_id: str, scene_i: str, scene_j: str):
        self.path, self.pair_id, self.scene_i, self.scene_j = path, pair_id, scene_i, scene_j
        self.rows: list[dict] = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != CANDIDATE_FIELDS:
                    raise ValueError("Candidate CSV schema mismatch")
                self.rows = list(reader)
            for row in self.rows:
                if (row["pair_id"], row["scene_i"], row["scene_j"]) != (pair_id, scene_i, scene_j):
                    raise ValueError("Candidate CSV belongs to a different image pair")

    def replace_suggestions(self, suggestions: list[dict]) -> None:
        if self.rows or self.path.exists():
            raise FileExistsError("Candidate file already exists; reviewed suggestions cannot be replaced")
        self.rows = []
        for row in suggestions:
            if row.get("human_review_status", "UNVERIFIED") != "UNVERIFIED":
                raise ValueError("Generator may only create UNVERIFIED suggestions")
            if (row["pair_id"], row["scene_i"], row["scene_j"]) != (self.pair_id, self.scene_i, self.scene_j):
                raise ValueError("Candidate pair identity mismatch")
            self.rows.append({key: row.get(key, "") for key in CANDIDATE_FIELDS} |
                             {"recommendation_status": "SUGGESTED", "human_review_status": "UNVERIFIED",
                              "human_adjusted": "False", "split": "UNASSIGNED",
                              "landcover_type": "UNKNOWN", "terrain_class": "TERRAIN_UNKNOWN",
                              "confidence": "UNKNOWN"})

    def _row(self, candidate_id: str) -> dict:
        matches = [r for r in self.rows if r["candidate_id"] == candidate_id]
        if len(matches) != 1:
            raise KeyError(candidate_id)
        return matches[0]

    def adjust(self, candidate_id: str, left: tuple[float, float], right: tuple[float, float],
               transform_i: Affine, transform_j: Affine, *, reviewer: str) -> None:
        row = self._row(candidate_id)
        xi, yi = pixel_to_projected(transform_i, *left)
        xj, yj = pixel_to_projected(transform_j, *right)
        row.update(adjusted_row_i=left[0], adjusted_col_i=left[1], adjusted_x_i=xi, adjusted_y_i=yi,
                   adjusted_row_j=right[0], adjusted_col_j=right[1], adjusted_x_j=xj, adjusted_y_j=yj,
                   human_adjusted="True", human_review_status="UNVERIFIED",
                   primary_reviewer="", second_reviewer="", reviewer=reviewer,
                   review_time_utc=datetime.now(timezone.utc).isoformat())

    def review(self, candidate_id: str, status: str, reviewer: str, split: str,
               confidence: str, precision_note: str, *, second_reviewer: str = "",
               landcover: str | None = None, terrain: str | None = None, notes: str = "") -> None:
        if status not in {"APPROVED", "REJECTED", "DOUBTFUL", "UNVERIFIED", "SKIPPED"}:
            raise ValueError("Invalid human review status")
        row = self._row(candidate_id)
        if status == "APPROVED":
            if (not reviewer.strip() or split not in {"UNASSIGNED", "DEVELOPMENT", "TEST"}
                    or confidence == "UNKNOWN" or not precision_note.strip()):
                raise ValueError("Approval requires reviewer, valid split, confidence, and precision note")
            if split == "TEST":
                first = row.get("primary_reviewer", "")
                if not first:
                    row["primary_reviewer"] = reviewer
                    status = "PENDING_SECOND_REVIEW"
                elif reviewer == first or second_reviewer != reviewer:
                    raise ValueError("TEST approval requires a distinct second reviewer")
                else:
                    row["second_reviewer"] = reviewer
            else:  # UNASSIGNED is reviewed, but never exported as formal truth.
                row["primary_reviewer"] = reviewer
        elif status in {"REJECTED", "DOUBTFUL", "UNVERIFIED", "SKIPPED"}:
            row["primary_reviewer"] = ""
            row["second_reviewer"] = ""
        row.update(human_review_status=status, reviewer=reviewer,
                   review_time_utc=datetime.now(timezone.utc).isoformat(),
                   split=split, confidence=confidence, coordinate_precision_note=precision_note,
                   landcover_type=row["landcover_type"] if landcover is None else landcover,
                   terrain_class=row["terrain_class"] if terrain is None else terrain,
                   review_notes=notes)

    def approved_test_truth(self) -> list[dict]:
        result = []
        for row in self.rows:
            if (row["human_review_status"] != "APPROVED" or row["split"] != "TEST"
                    or not row["primary_reviewer"] or not row["second_reviewer"]
                    or row["primary_reviewer"] == row["second_reviewer"]):
                continue
            result.append(self._truth_row(row))
        return result

    def approved_truth(self) -> list[dict]:
        return [self._truth_row(row) for row in self.rows if row["human_review_status"] == "APPROVED"
                and (row["split"] == "DEVELOPMENT" or
                     (row["split"] == "TEST" and row["primary_reviewer"] and row["second_reviewer"]
                      and row["primary_reviewer"] != row["second_reviewer"]))]

    @staticmethod
    def _truth_row(row: dict) -> dict:
        adjusted = str(row.get("human_adjusted", "False")) == "True"
        prefix = "adjusted_" if adjusted else ""
        return {"point_id": row["candidate_id"], "pair_id": row["pair_id"],
                "scene_i": row["scene_i"], "scene_j": row["scene_j"], "band": "B9",
                **{name: row[prefix + name] for name in
                   ("row_i", "col_i", "x_i", "y_i", "row_j", "col_j", "x_j", "y_j")},
                "coordinate_precision_note": row["coordinate_precision_note"],
                "landcover_type": row["landcover_type"], "terrain_class": row["terrain_class"],
                "confidence": row["confidence"], "review_status": "APPROVED",
                "reviewer": row["reviewer"], "review_time_utc": row["review_time_utc"],
                "split": row["split"], "notes": row["review_notes"],
                "point_source": "AUTO_SUGGESTED", "origin_candidate_id": row["candidate_id"],
                "primary_reviewer": row["primary_reviewer"], "second_reviewer": row["second_reviewer"]}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.path)

    def save_approved_truth(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=AUTO_TRUTH_FIELDS)
            writer.writeheader()
            writer.writerows(self.approved_truth())
        temporary.replace(path)
