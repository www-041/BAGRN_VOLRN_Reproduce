"""Pure coordinate, annotation, and registration-evaluation helpers.

No matcher is imported or executed here. Pixel coordinates use integer-valued
centres: (row=0, col=0) denotes the centre of the upper-left source pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from affine import Affine


TRUTH_FIELDS = [
    "point_id", "pair_id", "scene_i", "scene_j", "band",
    "row_i", "col_i", "x_i", "y_i", "row_j", "col_j", "x_j", "y_j",
    "coordinate_precision_note", "landcover_type", "terrain_class", "confidence",
    "review_status", "reviewer", "review_time_utc", "split", "notes",
]
AUTO_TRUTH_EXTRA_FIELDS = ["point_source", "origin_candidate_id", "primary_reviewer", "second_reviewer"]


def load_formal_test_points(manual_path: Path | None, auto_truth_path: Path | None,
                            pair_id: str) -> list[dict]:
    """Load only separately reviewed truth files, never suggestion CSVs."""
    points = []
    for path, expected, source in ((manual_path, TRUTH_FIELDS, "MANUAL"),
                                   (auto_truth_path, TRUTH_FIELDS + AUTO_TRUTH_EXTRA_FIELDS, "AUTO_SUGGESTED")):
        if path is None or not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected:
                raise ValueError("candidate CSV is not formal truth; wrong reviewed-truth schema")
            for row in reader:
                if row["pair_id"] != pair_id:
                    raise ValueError("Truth pair identity mismatch")
                if row["review_status"] != "APPROVED" or row["split"] != "TEST":
                    continue
                if source == "AUTO_SUGGESTED":
                    if (row["point_source"] != "AUTO_SUGGESTED" or not row["origin_candidate_id"]
                            or not row["primary_reviewer"] or not row["second_reviewer"]
                            or row["primary_reviewer"] == row["second_reviewer"]):
                        raise ValueError("Auto TEST truth lacks independent second review")
                else:
                    row["point_source"] = "MANUAL"
                points.append(row)
    ids = [row["point_id"] for row in points]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate point ID across reviewed truth files")
    return points


@dataclass(frozen=True)
class Viewport:
    col0: int
    row0: int
    scale: float
    width: int
    height: int

    def native_to_screen(self, row: float, col: float) -> tuple[float, float]:
        # Screen coordinates also denote pixel centres (pixel 0 has coordinate 0).
        return ((col - self.col0 + 0.5) * self.scale - 0.5,
                (row - self.row0 + 0.5) * self.scale - 0.5)

    def screen_to_native(self, x: float, y: float) -> tuple[float, float]:
        return (self.row0 + (y + 0.5) / self.scale - 0.5,
                self.col0 + (x + 0.5) / self.scale - 0.5)

    def contains(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


def pixel_to_projected(transform: Affine, row: float, col: float) -> tuple[float, float]:
    return transform @ (col + 0.5, row + 0.5)


def projected_to_pixel(transform: Affine, x: float, y: float) -> tuple[float, float]:
    col_corner, row_corner = (~transform) @ (x, y)
    return row_corner - 0.5, col_corner - 0.5


def save_truth(path: Path, rows: list[dict]) -> None:
    """Atomic replacement of this tool's annotation CSV only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRUTH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_truth(path: Path, pair_id: str, scene_i: str, scene_j: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != TRUTH_FIELDS:
            raise ValueError("Annotation CSV schema mismatch; refusing to overwrite")
        rows = list(reader)
    for row in rows:
        if (row["pair_id"], row["scene_i"], row["scene_j"]) != (pair_id, scene_i, scene_j):
            raise ValueError("Annotation CSV belongs to another pair; refusing to merge")
    return rows


def new_row(pair_id: str, scene_i: str, scene_j: str, reviewer: str,
            point_id: str, left: tuple[float, float], right: tuple[float, float],
            transform_i: Affine, transform_j: Affine) -> dict:
    row_i, col_i = left
    row_j, col_j = right
    x_i, y_i = pixel_to_projected(transform_i, row_i, col_i)
    x_j, y_j = pixel_to_projected(transform_j, row_j, col_j)
    return dict(point_id=point_id, pair_id=pair_id, scene_i=scene_i, scene_j=scene_j,
                band="B9", row_i=row_i, col_i=col_i, x_i=x_i, y_i=y_i,
                row_j=row_j, col_j=col_j, x_j=x_j, y_j=y_j,
                coordinate_precision_note="mouse_click_subpixel_coordinate_not_accuracy_claim",
                landcover_type="UNKNOWN", terrain_class="TERRAIN_UNKNOWN", confidence="UNKNOWN",
                review_status="UNVERIFIED", reviewer=reviewer,
                review_time_utc=datetime.now(timezone.utc).isoformat(), split="UNASSIGNED", notes="")


def pair_metrics(errors_px: Iterable[float]) -> dict:
    errors = np.asarray(list(errors_px), dtype=float)
    if errors.size == 0 or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("Expected nonempty finite nonnegative point errors")
    return {"rmse_px": float(np.sqrt(np.mean(errors**2))),
            "p95_px": float(np.percentile(errors, 95)),
            "max_px": float(np.max(errors)),
            "pck_1px": float(np.mean(errors <= 1)),
            "pck_2px": float(np.mean(errors <= 2)),
            "pck_3px": float(np.mean(errors <= 3)),
            "approved_point_count": int(errors.size)}


def evaluate_pair(pair_id: str, approved_points: list[dict], transform,
                  model_status: str) -> dict:
    """Evaluate predicted reference-grid coordinates; no fitting occurs."""
    if model_status != "SUCCESS" or transform is None:
        return {"pair_id": pair_id, "status": "FAIL", "metrics": None}
    if not approved_points:
        return {"pair_id": pair_id, "status": "NO_INDEPENDENT_TRUTH", "metrics": None}
    errors = []
    for point in approved_points:
        if point.get("review_status") != "APPROVED" or point.get("split") != "TEST":
            raise ValueError("Only APPROVED TEST points may enter final evaluation")
        pred_row, pred_col = transform(float(point["row_i"]), float(point["col_i"]))
        errors.append(math.hypot(pred_row - float(point["row_j"]),
                                 pred_col - float(point["col_j"])))
    return {"pair_id": pair_id, "status": "SUCCESS", "metrics": pair_metrics(errors)}


def summarize_pairs(results: list[dict], eligible_pair_ids: Iterable[str]) -> dict:
    """Eligible IDs must already have B9 valid overlap; no-overlap is not FAIL."""
    eligible = set(eligible_pair_ids)
    by_id = {r["pair_id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) - eligible:
        raise ValueError("Duplicate or ineligible pair result")
    success = [by_id[k] for k in eligible if k in by_id and by_id[k]["status"] == "SUCCESS"]
    failed = [k for k in eligible if k in by_id and by_id[k]["status"] == "FAIL"]
    missing = [k for k in eligible if k not in by_id]
    no_truth = [k for k in eligible if k in by_id and by_id[k]["status"] == "NO_INDEPENDENT_TRUTH"]
    metric_names = ["rmse_px", "p95_px", "max_px", "pck_1px", "pck_2px", "pck_3px"]
    return {"eligible_pairs": len(eligible), "attempted_pairs": len(by_id),
            "success_pairs": len(success), "failed_pairs": len(failed),
            "not_attempted_pairs": len(missing), "no_truth_pairs": len(no_truth),
            "success_rate_attempted": len(success) / len(by_id) if by_id else None,
            "macro_mean_success_only": {name: float(np.mean([r["metrics"][name] for r in success]))
                                        if success else None for name in metric_names}}


def boxes_separated(a: tuple[float, float, float, float],
                    b: tuple[float, float, float, float], buffer_m: float) -> bool:
    """Boxes are (left,bottom,right,top) in a common metre CRS."""
    if buffer_m < 0:
        raise ValueError("buffer_m must be nonnegative")
    return (a[2] + buffer_m < b[0] or b[2] + buffer_m < a[0] or
            a[3] + buffer_m < b[1] or b[3] + buffer_m < a[1])
