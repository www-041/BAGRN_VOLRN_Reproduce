"""
通用多景多波段处理管线模块

取代四景硬编码管线，提供可配置的通用管线接口：
  - 多景加载与波段验证
  - 重叠区域自动检测
  - 基于生成树的多波段配准
  - 多种辐射归一化方法并行对比
  - 镶嵌、评价指标一站式输出

关键设计原则：
  - 不修改已有的 four_image_pipeline.py
  - 仅依赖 src/ 中已有模块的公开接口
  - 支持 dry-run / smoke-test / caching
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from collections import deque

import numpy as np

from src.experiment_config import ExperimentConfig, get_common_bands, validate_config
from src.io_utils import read_geotiff, write_geotiff, read_geotiff_band
from src.overlap import get_overlap_window, overlap_pixel_count, has_overlap
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.metrics import compute_all, compute_per_pair, compute_per_band, save_metrics_csv
from src.comparison import run_comparison
from src.mosaic import create_mosaic

# ---------------------------------------------------------------------------
# 日志设置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# 辅助函数
# ===========================================================================

def registration_quality_meets_requirement(actual, required) -> bool:
    """Return whether a final registration quality meets its configured floor."""
    quality_order = {"fail": 0, "warn": 1, "pass": 2}
    if isinstance(actual, dict):
        actual = actual.get("quality")
    if isinstance(required, dict):
        required = required.get("quality")
    actual = str(actual).lower()
    required = str(required).lower()
    if actual not in quality_order or required not in quality_order:
        return False
    return quality_order[actual] >= quality_order[required]


def _registration_status_and_failure(connected, quality, required_quality):
    """Return the canonical status/failure pair for a registration result."""
    if not connected:
        return "fail", {
            "code": "registration_connectivity_failed",
            "reason": "registration graph is not connected",
        }
    actual_quality = (
        quality.get("quality", "unknown")
        if isinstance(quality, dict)
        else quality
    )
    if (
        str(actual_quality).lower() == "fail"
        or not registration_quality_meets_requirement(quality, required_quality)
    ):
        if str(actual_quality).lower() == "fail":
            reason = "final registration quality is classified as fail"
        else:
            reason = (
                f"final registration quality {actual_quality!r} "
                f"is below required quality {required_quality!r}"
            )
        return "fail", {
            "code": "registration_quality_failed",
            "reason": reason,
        }
    return "pass", {}


def _local_spatial_group_labels(points_xy, n_groups_x=4, n_groups_y=4):
    """Assign local controls to normalized spatial grid cells."""
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.ndim != 2 or len(points_xy) == 0:
        return np.array([], dtype=int)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_bin = np.clip(
        ((x - x.min()) / max(x.max() - x.min(), 1e-10) * n_groups_x).astype(int),
        0, n_groups_x - 1,
    )
    y_bin = np.clip(
        ((y - y.min()) / max(y.max() - y.min(), 1e-10) * n_groups_y).astype(int),
        0, n_groups_y - 1,
    )
    return y_bin * n_groups_x + x_bin


def _build_local_cv_fold_plan(points_xy, params):
    """Build one deterministic spatial fold plan shared by every candidate."""
    params = params or {}
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        points = np.empty((0, 2), dtype=float)
    groups = _local_spatial_group_labels(points)
    group_ids = np.unique(groups)
    folds = []
    for group_id in group_ids.tolist():
        test_idx = np.flatnonzero(groups == group_id).astype(int)
        train_idx = np.flatnonzero(groups != group_id).astype(int)
        if len(test_idx) == 0 or len(train_idx) < 3:
            continue
        folds.append({
            "group_id": int(group_id),
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    validation_indices = (
        np.concatenate([fold["test_idx"] for fold in folds]).astype(int)
        if folds else np.array([], dtype=int)
    )
    n_validation_controls = int(len(validation_indices))
    min_folds = int(params.get("local_cv_min_folds", 3))
    min_validation_controls = int(params.get(
        "local_cv_min_validation_controls",
        min(int(params.get("local_min_controls", 12)), len(points)),
    ))
    reasons = []
    if len(folds) < min_folds:
        reasons.append(f"validation folds {len(folds)} < {min_folds}")
    if n_validation_controls < min_validation_controls:
        reasons.append(
            f"validation controls {n_validation_controls} < {min_validation_controls}"
        )
    return {
        "available": not reasons,
        "groups": groups,
        "folds": folds,
        "n_folds_attempted": int(len(group_ids)),
        "n_folds": int(len(folds)),
        "n_validation_controls": n_validation_controls,
        "validation_indices": validation_indices,
        "validation_coverage": float(n_validation_controls / len(points)) if len(points) else 0.0,
        "failure_reason": "; ".join(reasons) or None,
    }


def _evaluate_local_rbf_smoothing_candidate(controls, fold_plan, smoothing, params):
    """Evaluate one smoothing value on the fixed spatial folds."""
    from src.coregistration import fit_local_rbf

    points = np.asarray(controls.get("points_xy", []), dtype=float)
    residual_dx = np.asarray(controls.get("residual_dx", []), dtype=float)
    residual_dy = np.asarray(controls.get("residual_dy", []), dtype=float)
    base = {
        "smoothing": float(smoothing),
        "available": False,
        "candidate_rmse": None,
        "candidate_p95": None,
        "baseline_rmse": None,
        "baseline_p95": None,
        "n_folds": 0,
        "n_folds_attempted": int(fold_plan.get("n_folds_attempted", 0)),
        "n_validation_controls": 0,
        "validation_coverage": 0.0,
        "failed_group_ids": [],
        "failure_reason": None,
    }
    if not fold_plan.get("available", False):
        base["failure_reason"] = fold_plan.get(
            "failure_reason", "spatial CV fold plan unavailable"
        )
        return base
    if len(points) != len(residual_dx) or len(points) != len(residual_dy):
        base["failure_reason"] = "control and residual arrays have mismatched lengths"
        return base

    baseline_errors = []
    candidate_errors = []
    failed_group_ids = []
    for fold in fold_plan["folds"]:
        group_id = int(fold["group_id"])
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        try:
            rbf_dx, rbf_dy, coord_min, coord_max = fit_local_rbf(
                points[train_idx], residual_dx[train_idx], residual_dy[train_idx],
                smoothing=float(smoothing), neighbors=min(20, len(train_idx)),
            )
            tx = (points[test_idx, 0] - coord_min[0]) / max(
                coord_max[0] - coord_min[0], 1e-10
            )
            ty = (points[test_idx, 1] - coord_min[1]) / max(
                coord_max[1] - coord_min[1], 1e-10
            )
            normalized = np.column_stack([tx, ty])
            predicted_dx = np.asarray(rbf_dx(normalized), dtype=float).reshape(-1)
            predicted_dy = np.asarray(rbf_dy(normalized), dtype=float).reshape(-1)
            if len(predicted_dx) != len(test_idx) or len(predicted_dy) != len(test_idx):
                raise ValueError("prediction shape does not match validation controls")
            if not np.all(np.isfinite(predicted_dx)) or not np.all(np.isfinite(predicted_dy)):
                raise ValueError("non-finite local RBF predictions")
            actual_dx = residual_dx[test_idx]
            actual_dy = residual_dy[test_idx]
            if not np.all(np.isfinite(actual_dx)) or not np.all(np.isfinite(actual_dy)):
                raise ValueError("non-finite held-out residuals")
            max_component = float(params.get(
                "local_hard_max_component",
                params.get("local_max_component", 2.5),
            ))
            predicted_dx = np.clip(predicted_dx, -max_component, max_component)
            predicted_dy = np.clip(predicted_dy, -max_component, max_component)
            baseline_errors.extend(np.hypot(actual_dx, actual_dy).tolist())
            candidate_errors.extend(
                np.hypot(actual_dx - predicted_dx, actual_dy - predicted_dy).tolist()
            )
        except Exception as exc:
            failed_group_ids.append(group_id)
            base["failure_reason"] = f"fold {group_id} failed: {exc}"
            base["failed_group_ids"] = failed_group_ids
            return base

    baseline_errors = np.asarray(baseline_errors, dtype=float)
    candidate_errors = np.asarray(candidate_errors, dtype=float)
    if (len(baseline_errors) == 0 or len(baseline_errors) != len(candidate_errors)
            or not np.all(np.isfinite(baseline_errors))
            or not np.all(np.isfinite(candidate_errors))):
        base["failure_reason"] = "held-out errors are incomplete or non-finite"
        return base
    base.update({
        "available": True,
        "baseline_rmse": float(np.sqrt(np.mean(baseline_errors ** 2))),
        "candidate_rmse": float(np.sqrt(np.mean(candidate_errors ** 2))),
        "baseline_p95": float(np.percentile(baseline_errors, 95)),
        "candidate_p95": float(np.percentile(candidate_errors, 95)),
        "n_folds": int(len(fold_plan["folds"])),
        "n_validation_controls": int(len(baseline_errors)),
        "validation_coverage": float(len(baseline_errors) / len(points)) if len(points) else 0.0,
        "failure_reason": None,
    })
    return base


def _select_local_rbf_cv_candidate(candidate_results, baseline_rmse, baseline_p95, params):
    """Apply both held-out gates and select the deterministic best candidate."""
    params = params or {}
    rmse_threshold = float(params.get("local_cv_min_rmse_improvement", 0.10))
    p95_threshold = float(params.get("local_cv_min_p95_improvement", 0.15))
    normalized = []
    for raw in candidate_results:
        item = dict(raw)
        item.setdefault("available", False)
        item.setdefault("failed_group_ids", [])
        item.setdefault("failure_reason", None)
        item.setdefault("candidate_rmse", None)
        item.setdefault("candidate_p95", None)
        if item.get("available"):
            try:
                item["rmse_improvement"] = float(baseline_rmse) - float(item["candidate_rmse"])
                item["p95_improvement"] = float(baseline_p95) - float(item["candidate_p95"])
                item["passes_rmse_gate"] = bool(
                    np.isfinite(item["rmse_improvement"])
                    and item["rmse_improvement"] >= rmse_threshold
                )
                item["passes_p95_gate"] = bool(
                    np.isfinite(item["p95_improvement"])
                    and item["p95_improvement"] >= p95_threshold
                )
                item["passes_gate"] = bool(
                    item["passes_rmse_gate"] and item["passes_p95_gate"]
                )
            except (TypeError, ValueError):
                item["available"] = False
                item["failure_reason"] = "candidate metrics are not finite numbers"
        if not item.get("available"):
            item.setdefault("rmse_improvement", None)
            item.setdefault("p95_improvement", None)
            item["passes_rmse_gate"] = False
            item["passes_p95_gate"] = False
            item["passes_gate"] = False
        normalized.append(item)

    available = [
        item for item in normalized
        if item.get("available") and np.isfinite(float(item["candidate_rmse"]))
        and np.isfinite(float(item["candidate_p95"]))
    ]
    passing = [item for item in available if item.get("passes_gate")]
    best = min(
        passing or available,
        key=lambda item: (
            float(item["candidate_rmse"]),
            float(item["candidate_p95"]),
            float(item["smoothing"]),
        ),
    ) if (passing or available) else None
    if passing:
        selection_reason = "best_passing_candidate"
        has_passing_candidate = True
    elif available:
        selection_reason = "best_available_but_gate_failed"
        has_passing_candidate = False
    else:
        selection_reason = "no_available_candidate"
        has_passing_candidate = False

    result = {
        "available": bool(available),
        "baseline_rmse": float(baseline_rmse) if np.isfinite(float(baseline_rmse)) else None,
        "baseline_p95": float(baseline_p95) if np.isfinite(float(baseline_p95)) else None,
        "candidate_rmse": best.get("candidate_rmse") if best else None,
        "candidate_p95": best.get("candidate_p95") if best else None,
        "smoothing": best.get("smoothing") if best else None,
        "selected_smoothing": best.get("smoothing") if best else None,
        "has_passing_candidate": has_passing_candidate,
        "selection_reason": selection_reason,
        "candidate_results": normalized,
    }
    for key in ("n_folds", "n_folds_attempted", "n_validation_controls", "validation_coverage"):
        result[key] = best.get(key) if best else 0 if key != "validation_coverage" else 0.0
    result["failed_group_ids"] = best.get("failed_group_ids", []) if best else []
    result["failure_reason"] = (
        best.get("failure_reason") if best and not best.get("available")
        else None
    )
    return result


def _accept_local_rbf_candidate(controls, params, cv_result=None, rematch_failures=None):
    """Apply the local-control and held-out-CV acceptance gates."""
    params = params or {}
    points = np.asarray(controls.get("points_xy", []), dtype=float)
    n_controls = int(controls.get("n_valid", len(points)))
    n_groups = int(len(np.unique(_local_spatial_group_labels(points)))) if len(points) else 0
    result = {
        "accepted": False,
        "reason": None,
        "n_controls": n_controls,
        "n_spatial_groups": n_groups,
        "cv_result": cv_result or {},
    }

    if rematch_failures:
        result["reason"] = "global-only fallback: required post-global rematch failed"
        result["rematch_failures"] = list(rematch_failures)
        return result
    if not params.get("enable_local_refinement", True):
        result["reason"] = "local refinement disabled"
        return result
    if n_controls < int(params.get("local_min_controls", 12)):
        result["reason"] = "too few local controls"
        return result
    if n_groups < int(params.get("local_min_spatial_groups", 3)):
        result["reason"] = "insufficient local spatial groups"
        return result
    if not cv_result:
        result["reason"] = "held-out CV unavailable"
        return result
    if cv_result.get("available", True) is False:
        result["reason"] = (
            "held-out CV unavailable: "
            + str(cv_result.get("failure_reason", "insufficient validation coverage"))
        )
        return result
    if "has_passing_candidate" in cv_result and not cv_result.get("has_passing_candidate"):
        result["reason"] = "no smoothing candidate passed held-out RMSE and P95 gates"
        result["selected_smoothing"] = cv_result.get("selected_smoothing")
        return result

    try:
        baseline_rmse = float(cv_result["baseline_rmse"])
        candidate_rmse = float(cv_result["candidate_rmse"])
        baseline_p95 = float(cv_result["baseline_p95"])
        candidate_p95 = float(cv_result["candidate_p95"])
    except (KeyError, TypeError, ValueError):
        result["reason"] = "held-out CV metrics unavailable"
        return result

    rmse_improvement = baseline_rmse - candidate_rmse
    p95_improvement = baseline_p95 - candidate_p95
    result["rmse_improvement"] = rmse_improvement
    result["p95_improvement"] = p95_improvement
    if not (np.isfinite(rmse_improvement) and np.isfinite(p95_improvement)):
        result["reason"] = "held-out CV metrics are not finite"
        return result
    if rmse_improvement < float(params.get("local_cv_min_rmse_improvement", 0.10)):
        result["reason"] = "held-out RMSE improvement below threshold"
        return result
    if p95_improvement < float(params.get("local_cv_min_p95_improvement", 0.15)):
        result["reason"] = "held-out P95 improvement below threshold"
        return result

    result["accepted"] = True
    result["reason"] = "held-out RMSE and P95 improvements passed"
    result["selected_smoothing"] = cv_result.get("selected_smoothing", cv_result.get("smoothing"))
    if "has_passing_candidate" in cv_result and result["selected_smoothing"] is None:
        result["accepted"] = False
        result["reason"] = "selected smoothing unavailable"
    return result


def _local_holdout_cv(controls, params):
    """Compare zero-residual translation with RBF on spatially held-out controls."""
    points = np.asarray(controls.get("points_xy", []), dtype=float)
    residual_dx = np.asarray(controls.get("residual_dx", []), dtype=float)
    residual_dy = np.asarray(controls.get("residual_dy", []), dtype=float)
    params = params or {}
    plan = _build_local_cv_fold_plan(points, params)
    validation = plan["validation_indices"]
    if len(validation) and len(residual_dx) == len(points) and len(residual_dy) == len(points):
        baseline_errors = np.hypot(residual_dx[validation], residual_dy[validation])
        baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
        baseline_p95 = float(np.percentile(baseline_errors, 95))
    else:
        baseline_rmse = float("nan")
        baseline_p95 = float("nan")

    smoothing_values = params.get("local_smoothing_candidates", [0.1])
    candidates = []
    seen = set()
    for value in smoothing_values or []:
        try:
            smoothing = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(smoothing) and smoothing not in seen:
            seen.add(smoothing)
            candidates.append(smoothing)
    candidate_results = [
        _evaluate_local_rbf_smoothing_candidate(controls, plan, smoothing, params)
        for smoothing in candidates
    ]
    result = _select_local_rbf_cv_candidate(
        candidate_results, baseline_rmse, baseline_p95, params
    )
    result.update({
        "n_folds": int(plan["n_folds"]),
        "n_folds_attempted": int(plan["n_folds_attempted"]),
        "n_validation_controls": int(plan["n_validation_controls"]),
        "validation_coverage": float(plan["validation_coverage"]),
    })
    if not plan["available"]:
        result["available"] = False
        result["failure_reason"] = plan.get("failure_reason")
    return result


def _collect_post_global_residual_pairs(
    global_only_arrays, registration_band_idx, transforms, nodata_values,
    matching_edges, params, rematch_fn, holdout_contexts=None,
):
    """Rematch global-only arrays while converting failures to diagnostics."""
    params = params or {}
    pairs = []
    failures = []
    block_size = int(params.get("local_block_size", 256))
    confidence = float(params.get("local_confidence_threshold", 0.60))
    max_shift = float(params.get("local_max_residual_shift", 3.0))
    search_max_shift = float(params.get("local_search_max_shift", 12.0))
    for i, j in matching_edges:
        try:
            context = (holdout_contexts or {}).get((i, j))
            rematch = rematch_fn(
                global_only_arrays[i][registration_band_idx],
                global_only_arrays[j][registration_band_idx],
                transforms[i], transforms[j], nodata_values[i], nodata_values[j],
                max_residual_shift=max_shift,
                local_search_max_shift=search_max_shift,
                block_size=block_size,
                confidence_threshold=confidence,
                holdout_exclusion_mask=(context or {}).get("holdout_exclusion_mask")
                if context and context.get("available") else None,
            )
        except Exception as exc:
            failures.append({
                "idx_i": i, "idx_j": j,
                "reason": f"post-global rematch failed: {exc}",
            })
            continue
        if not rematch or not rematch.get("available", True):
            failures.append({
                "idx_i": i, "idx_j": j,
                "reason": "post-global rematch unavailable",
            })
            continue
        pairs.append({
            "idx_i": i, "idx_j": j,
            "shift_dx": float(rematch.get("shift_dx", 0.0)),
            "shift_dy": float(rematch.get("shift_dy", 0.0)),
            "confidence": float(rematch.get("confidence", 0.0)),
            "n_blocks": int(rematch.get("n_blocks", 0)),
            "rmse": float(rematch.get("rmse", 0.0)),
            "p95": float(rematch.get("p95", rematch.get("rmse", 0.0))),
            "matches": rematch.get("matches", []),
            "available": True,
            "is_post_global_residual": True,
        })
    return {"pairs": pairs, "failures": failures}


def _training_points_for_edge(pair_measurements, idx_i, idx_j):
    """Return training block centers in the reference coordinates of an edge."""
    points = []
    for pair in pair_measurements:
        if {pair.get("idx_i"), pair.get("idx_j")} != {idx_i, idx_j}:
            continue
        matches = pair.get("matches") or []
        if pair.get("idx_i") == idx_i and pair.get("idx_j") == idx_j:
            pair_points = [(m.get("ref_x"), m.get("ref_y")) for m in matches]
        else:
            pair_points = [(m.get("tgt_x"), m.get("tgt_y")) for m in matches]
        points.extend(point for point in pair_points if None not in point)
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.unique(np.asarray(points, dtype=float).reshape((-1, 2)), axis=0)


def _build_pair_holdout_context(
    arr_ref, tr_ref, arr_tgt, tr_tgt, nodata_ref, nodata_tgt, params,
):
    """Build common-valid TRAIN/HOLDOUT masks in reference-patch pixels."""
    from src.coregistration import (
        build_pair_overlap_context,
        derive_training_window_requirements,
        reserve_validation_windows,
    )

    pair_grid = build_pair_overlap_context(
        arr_ref, tr_ref, arr_tgt, tr_tgt, nodata_ref, nodata_tgt,
    )
    if not pair_grid.get("available"):
        return pair_grid

    common_valid = pair_grid["common_valid_mask"]

    block_sizes = params.get(
        "validation_block_size_candidates",
        [params.get("validation_block_size", 384)],
    )
    block_sizes = [int(size) for size in block_sizes]
    reservation_step = params.get("validation_reservation_step")
    if reservation_step is None:
        reservation_step = max(
            1,
            min(int(params.get("validation_step", 256)), min(block_sizes) // 2),
        )
    training_requirements = derive_training_window_requirements(params)
    reservation_kwargs = {
        "training_block_sizes": list(training_requirements),
        "min_training_windows": training_requirements,
    }
    split = reserve_validation_windows(
        common_valid,
        block_size_candidates=block_sizes,
        final_min_blocks=int(params.get("final_min_blocks", 5)),
        step=int(reservation_step),
        offset_row=int(params.get("validation_reservation_offset_row", 0)),
        offset_col=int(params.get("validation_reservation_offset_col", 0)),
        min_common_valid_ratio=float(
            params.get("validation_min_common_valid_ratio", 0.30)
        ),
        seed=int(params.get("holdout_seed", 42)),
        reservation_margin=int(params.get("validation_reservation_margin", 2)),
        buffer_pixels=int(params.get("holdout_buffer_pixels", 0)),
        **reservation_kwargs,
    )
    split.update({
        "patch_window_ref": pair_grid["ref_window"],
        "patch_window_tgt": pair_grid["tgt_window"],
        "overlap_transform": pair_grid["overlap_transform"],
        "shape": pair_grid["shape"],
        "grid_compatible": pair_grid["grid_compatible"],
        "resampled_target": pair_grid["resampled_target"],
        "common_valid_pixels": int(common_valid.sum()),
        "common_valid_ratio": float(common_valid.mean()),
        "common_valid_mask": common_valid,
        "validation_reservation": dict(split),
    })
    return split


def _public_holdout_summary(context):
    """Return JSON-safe holdout metadata without serializing large masks."""
    if not context:
        return None
    reservation = context.get("validation_reservation")
    reservation_summary = None
    if reservation is not None:
        reservation_summary = {
            "selected_block_size": reservation.get("selected_block_size"),
            "candidates_before_spatial_filter": int(
                reservation.get("candidates_before_spatial_filter", 0)
            ),
            "reserved_windows": list(reservation.get("reserved_windows", [])),
            "reserved_count": int(reservation.get("reserved_count", 0)),
            "reserved_centers": list(reservation.get("reserved_centers", [])),
            "train_usable_cells": int(
                len(reservation.get("train_usable_cells", []))
            ),
            "unused_cells": list(reservation.get("unused_cells", [])),
            "unused_count": int(len(reservation.get("unused_cells", []))),
            "candidate_counts": dict(reservation.get("candidate_counts", {})),
            "training_feasibility": dict(
                reservation.get("training_feasibility", {})
            ),
            "buffer_pixels": int(reservation.get("buffer_pixels", 0)),
            "failure_reason": reservation.get("failure_reason"),
        }
    return {
        "available": bool(context.get("available", False)),
        "candidate_cells": len(context.get("candidate_cells", [])),
        "train_cells": len(context.get("train_cells", [])),
        "holdout_cells": len(context.get("holdout_cells", [])),
        "buffer_pixels": int(context.get("buffer_pixels", 0)),
        "common_valid_pixels": int(context.get("common_valid_pixels", 0)),
        "common_valid_ratio": float(context.get("common_valid_ratio", 0.0)),
        "failure_reason": context.get("failure_reason"),
        "validation_reservation": reservation_summary,
    }


def _percentiles(values):
    """Return compact percentiles for a local-control diagnostic array."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p100": None}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p100": float(np.max(values)),
    }


def _translate_training_measurements_to_crop(
    pair_measurements, original_transforms, cropped_transforms,
):
    """Translate pair-match coordinates from a full frame into a crop frame.

    Rasterio transforms encode the pixel origin.  A crop therefore changes
    the pixel coordinates even when the underlying geographic locations are
    unchanged.  Only the copied validation measurements are translated; the
    registration result keeps its original full-frame coordinates.
    """
    offsets = []
    for original, cropped in zip(original_transforms, cropped_transforms):
        try:
            x_resolution = float(original.a)
            y_resolution = float(original.e)
            if abs(x_resolution) < 1e-12 or abs(y_resolution) < 1e-12:
                raise ValueError("zero pixel resolution")
            col_offset = int(round((float(cropped.c) - float(original.c)) / x_resolution))
            row_offset = int(round((float(original.f) - float(cropped.f)) / abs(y_resolution)))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            col_offset, row_offset = 0, 0
        offsets.append((col_offset, row_offset))

    translated = []
    for pair in pair_measurements or []:
        translated_pair = dict(pair)
        idx_i = pair.get("idx_i")
        idx_j = pair.get("idx_j")
        matches = []
        for match in pair.get("matches") or []:
            if not isinstance(match, dict):
                matches.append(match)
                continue
            translated_match = dict(match)
            for x_key, y_key, scene_idx in (
                ("ref_x", "ref_y", idx_i),
                ("tgt_x", "tgt_y", idx_j),
            ):
                if x_key not in translated_match or y_key not in translated_match:
                    continue
                try:
                    col_offset, row_offset = offsets[int(scene_idx)]
                    translated_match[x_key] = float(translated_match[x_key]) - col_offset
                    translated_match[y_key] = float(translated_match[y_key]) - row_offset
                except (IndexError, TypeError, ValueError):
                    continue
            matches.append(translated_match)
        translated_pair["matches"] = matches
        translated.append(translated_pair)
    return translated, offsets


def _validate_final_registration_arrays(
    registered_arrays,
    registration_band_idx,
    transforms,
    nodata_values,
    validation_edges,
    training_measurements,
    params,
    holdout_contexts=None,
):
    """Validate the exact arrays that will be used by the quality gate."""
    from src.coregistration import (
        aggregate_final_validation_quality,
        validate_registration_independent_grid,
    )

    params = params or {}
    validation_results = []
    validation_block_size = int(params.get("validation_block_size", 384))
    validation_step = int(params.get("validation_step", 256))
    validation_offset_row = int(params.get("validation_offset_row", 128))
    validation_offset_col = int(params.get("validation_offset_col", 128))
    validation_min_distance = float(
        params.get("validation_min_distance_from_training", 256)
    )
    validation_confidence = float(
        params.get("validation_confidence_threshold", 0.45)
    )
    validation_max_shift = float(
        params.get("validation_max_residual_shift", 3.0)
    )
    validation_min_blocks = int(params.get("final_min_blocks", 5))
    validation_candidates = params.get(
        "validation_block_size_candidates", [validation_block_size]
    )
    required_candidate_count = int(params.get(
        "validation_required_candidate_count", max(1, 2 * validation_min_blocks)
    ))

    for idx_i, idx_j in validation_edges:
        training_points = _training_points_for_edge(
            training_measurements, idx_i, idx_j,
        )
        context = (holdout_contexts or {}).get((idx_i, idx_j))
        reserved_mask = None
        reserved_windows = None
        if bool(params.get("enable_spatial_holdout", False)):
            if context and context.get("available"):
                reserved_mask = context.get("holdout_region_full_mask")
                reservation = context.get("validation_reservation", {})
                reserved_windows = reservation.get("reserved_windows")
            if reserved_mask is None:
                reserved_mask = np.zeros_like(
                    registered_arrays[idx_i][registration_band_idx], dtype=bool
                )
        try:
            validation = validate_registration_independent_grid(
                registered_arrays[idx_i][registration_band_idx], transforms[idx_i],
                registered_arrays[idx_j][registration_band_idx], transforms[idx_j],
                nodata_values[idx_i], nodata_values[idx_j], training_points,
                block_size=validation_block_size,
                step=validation_step,
                offset_row=validation_offset_row,
                offset_col=validation_offset_col,
                min_distance_from_training=validation_min_distance,
                confidence_threshold=validation_confidence,
                max_residual_shift=validation_max_shift,
                min_accepted=validation_min_blocks,
                reserved_holdout_mask=reserved_mask,
                validation_block_size_candidates=validation_candidates
                if reserved_mask is not None else None,
                required_candidate_count=required_candidate_count
                if reserved_mask is not None else None,
                reserved_validation_windows=reserved_windows,
            )
        except Exception as exc:
            logger.warning(
                "最终独立验证边 (%d, %d) 异常: %s", idx_i, idx_j, exc,
            )
            validation = {
                "blocks": [], "stats": None, "coverage": None,
                "failure_reason": f"validation failed: {exc}",
            }
        validation["idx_i"] = idx_i
        validation["idx_j"] = idx_j
        if context:
            validation["holdout"] = _public_holdout_summary(context)
        validation_results.append(validation)

    aggregate_quality = aggregate_final_validation_quality(
        validation_results, params, required_edges=validation_edges,
    )
    final_quality = {
        "quality": aggregate_quality.get("quality", "fail"),
        "rmse": float(aggregate_quality.get("rmse", float("inf"))),
        "p95": float(aggregate_quality.get("p95", float("inf"))),
        "median": float(aggregate_quality.get("median", float("inf"))),
        "confidence": float(aggregate_quality.get(
            "mean_confidence", aggregate_quality.get("confidence", 0.0)
        )),
        "n_blocks": int(aggregate_quality.get(
            "n_blocks", aggregate_quality.get("n_accepted", 0)
        )),
    }
    if aggregate_quality.get("unavailable_edges"):
        final_quality["quality"] = "fail"
    return final_quality, {
        "edges": validation_results,
        "overall": final_quality,
    }


def _graph_components(n_images, edges):
    """Return geometric graph components in stable scene-index order."""
    adjacency = defaultdict(set)
    for idx_i, idx_j in edges:
        adjacency[idx_i].add(idx_j)
        adjacency[idx_j].add(idx_i)
    visited = set()
    components = []
    for node in range(n_images):
        if node in visited:
            continue
        component = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            queue.extend(nb for nb in adjacency[current] if nb not in component)
        components.append(sorted(component))
    return components


def _registration_failure_result(
    arrays,
    local_refinement_enabled,
    pair_matches,
    spanning_tree,
    geometric_edges,
    matching_edges,
    rejected_edges,
    connected_components,
    unreachable_scenes,
    reason,
):
    """Build the stable registration schema for a blocked registration."""
    n_images = len(arrays)
    quality = {
        "quality": "fail",
        "rmse": float("inf"),
        "p95": float("inf"),
        "median": float("inf"),
        "confidence": 0.0,
        "n_blocks": 0,
    }
    final_validation = {"edges": [], "overall": quality}
    local_refinement = {
        "enabled": bool(local_refinement_enabled),
        "used_for_scenes": [],
        "fallback_scenes": [],
        "cv_results": {},
    }
    return {
        "registered_arrays": [np.asarray(array).copy() for array in arrays],
        "global_shifts": np.zeros((n_images, 2), dtype=float),
        "local_dx_fields": [
            np.zeros(array.shape[1:], dtype=np.float64) for array in arrays
        ],
        "local_dy_fields": [
            np.zeros(array.shape[1:], dtype=np.float64) for array in arrays
        ],
        "local_refinement": local_refinement,
        "pair_matches": list(pair_matches),
        "connected": False,
        "spanning_tree": list(spanning_tree),
        "geometric_edges": list(geometric_edges),
        "matching_edges": list(matching_edges),
        "rejected_edges": list(rejected_edges),
        "connected_components": list(connected_components),
        "unreachable_scenes": list(unreachable_scenes),
        "status": "fail",
        "failure": {
            "code": "registration_blocked",
            "reason": reason,
        },
        "quality": quality,
        "final_validation": final_validation,
        "diagnostics": {
            "registration_blocked": True,
            "reason": reason,
            "local_refinement": local_refinement,
            "final_validation": final_validation,
            "quality": quality,
        },
    }


def _local_field_stats(field):
    """Return compact finite-field statistics for registration diagnostics."""
    values = np.asarray(field, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(finite.min()), "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std()),
    }


def _fit_local_rbf_field(controls, shape, params, *, smoothing=None):
    """Fit, fade, and component-clip one target-scene local field."""
    from src.coregistration import compute_hull_fade_mask, fit_local_rbf

    points = np.asarray(controls["points_xy"], dtype=float)
    smoothing_values = params.get("local_smoothing_candidates", [0.1])
    smoothing_values = [float(value) for value in smoothing_values]
    if smoothing is None:
        if len(smoothing_values) > 1:
            raise ValueError(
                "selected smoothing required when multiple candidates are configured"
            )
        smoothing = smoothing_values[0] if smoothing_values else 0.1
    smoothing = float(smoothing)
    rbf_dx, rbf_dy, coord_min, coord_max = fit_local_rbf(
        points,
        np.asarray(controls["residual_dx"], dtype=float),
        np.asarray(controls["residual_dy"], dtype=float),
        smoothing=smoothing,
        neighbors=min(20, len(points)),
    )

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = (xx.ravel() - coord_min[0]) / max(coord_max[0] - coord_min[0], 1e-10)
    ty = (yy.ravel() - coord_min[1]) / max(coord_max[1] - coord_min[1], 1e-10)
    normalized = np.column_stack([tx, ty])
    raw_dx = np.asarray(rbf_dx(normalized), dtype=float).reshape(h, w)
    raw_dy = np.asarray(rbf_dy(normalized), dtype=float).reshape(h, w)
    fade = compute_hull_fade_mask(
        points, h, w, buffer=int(params.get("local_hull_buffer", 128))
    )
    faded_dx = raw_dx * fade
    faded_dy = raw_dy * fade
    max_component = float(params.get(
        "local_hard_max_component",
        params.get("local_max_component", 2.5),
    ))
    clipped_dx = int(np.count_nonzero(np.abs(faded_dx) > max_component))
    clipped_dy = int(np.count_nonzero(np.abs(faded_dy) > max_component))
    field_dx = np.clip(faded_dx, -max_component, max_component)
    field_dy = np.clip(faded_dy, -max_component, max_component)
    magnitude = np.hypot(field_dx, field_dy)
    grad_dx = np.gradient(field_dx)
    grad_dy = np.gradient(field_dy)
    spatial_gradient = np.maximum(
        np.hypot(grad_dx[0], grad_dx[1]),
        np.hypot(grad_dy[0], grad_dy[1]),
    )
    return field_dx, field_dy, {
        "smoothing": smoothing,
        "clipping": {"dx": clipped_dx, "dy": clipped_dy, "total": clipped_dx + clipped_dy},
        "fade": {"min": float(fade.min()), "max": float(fade.max())},
        "local_dx": _local_field_stats(field_dx),
        "local_dy": _local_field_stats(field_dy),
        "max_dx": float(np.max(np.abs(field_dx))),
        "max_dy": float(np.max(np.abs(field_dy))),
        "p95_magnitude": float(np.percentile(magnitude, 95)),
        "max_spatial_gradient": float(np.max(spatial_gradient)),
    }


def _empty_local_controls():
    return {
        "points_xy": np.empty((0, 2), dtype=float),
        "residual_dx": np.array([], dtype=float),
        "residual_dy": np.array([], dtype=float),
        "confidence": np.array([], dtype=float),
        "n_valid": 0,
    }

def validate_band_consistency(scenes_config: List[Dict[str, Any]], required_bands=None) -> List[str]:
    """
    校验所有场景的波段一致性。

    检查：
      1. 每个场景都拥有所需的波段名集合。
      2. 分辨率是否一致（严格模式下必须完全一致）。

    参数
    ----------
    scenes_config : list of dict
        实验配置中的 scenes 列表。
    required_bands : set or None
        必须存在的波段集合（来自 selected_bands）。None 则使用交集。

    返回
    -------
    list of str
        错误列表，空列表表示全部通过。
    """
    errors: List[str] = []
    if not scenes_config:
        errors.append("scenes 列表为空")
        return errors

    # Use required_bands if provided (from selected_bands), else use intersection
    if required_bands:
        required = set(required_bands)
    else:
        # Fall back to intersection
        band_sets = []
        for scene in scenes_config:
            bands = scene.get("bands", {})
            band_sets.append(set(bands.keys()))
        if not band_sets:
            return errors
        required = band_sets[0]
        for s in band_sets[1:]:
            required = required & s
        if not required:
            errors.append("所有场景的公共波段集合为空（交集为空）")
            return errors

    for idx, scene in enumerate(scenes_config):
        if not isinstance(scene, dict):
            continue
        sid = scene.get("id", f"scene_{idx}")
        scene_bands = set(scene.get("bands", {}).keys())
        missing = required - scene_bands
        if missing:
            errors.append(
                f"scenes[{idx}] (id={sid}) 缺少波段: {sorted(missing)}"
            )
    return errors


def build_spanning_tree(
    overlaps: List[dict],
    n_images: int,
    reference_idx: int = 0,
) -> List[Tuple[int, int]]:
    """
    从重叠关系构建生成树（BFS）。

    以 reference_idx 为根，使用 BFS 找到一棵覆盖所有可达节点的生成树。
    未连接的节点被忽略。

    参数
    ----------
    overlaps : list of dict
        重叠信息列表，每项包含 idx_i, idx_j。
    n_images : int
        影像总数。
    reference_idx : int
        参考影像索引（生成树的根）。

    返回
    -------
    list of (int, int)
        生成树的边列表，(parent, child)。
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].append(j)
        adj[j].append(i)

    visited = set([reference_idx])
    queue = deque([reference_idx])
    tree_edges: List[Tuple[int, int]] = []

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                tree_edges.append((node, nb))
                queue.append(nb)

    return tree_edges


def propagate_control_scene(
    overlaps: List[dict],
    n_images: int,
    reference_idx: int = 0,
) -> Dict[int, Optional[int]]:
    """
    确定每个场景的父场景（基于 BFS 连通性）。

    返回
    -------
    dict
        scene_idx -> parent_idx，参考场景的 parent 为 None。
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].append(j)
        adj[j].append(i)

    parent_map: Dict[int, Optional[int]] = {reference_idx: None}
    queue = deque([reference_idx])

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in parent_map:
                parent_map[nb] = node
                queue.append(nb)

    # 未连接的节点也给一个默认值
    for idx in range(n_images):
        if idx not in parent_map:
            parent_map[idx] = None

    return parent_map


# ===========================================================================
# 主管线类
# ===========================================================================

class MultibandPipeline:
    """
    通用多景多波段辐射归一化管线。

    用法
    ----
    >>> from src.experiment_config import load_config
    >>> from src.multiband_pipeline import MultibandPipeline
    >>> cfg = load_config("config.yaml")
    >>> pipe = MultibandPipeline(cfg)
    >>> results = pipe.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        初始化管线。

        参数
        ----------
        config : ExperimentConfig
            实验配置对象。
        """
        self.config = config
        self.output_root = config.output_root
        self.smoke = config.smoke
        self.dry_run = config.dry_run
        self.smoke_crop_size = config.smoke_crop_size

        # 解析公共波段列表
        self.common_bands = get_common_bands(config)
        self.n_bands = len(self.common_bands)

        # 确定配准波段索引
        if config.registration_band in self.common_bands:
            self.registration_band_idx = self.common_bands.index(config.registration_band)
        else:
            self.registration_band_idx = self.n_bands - 1  # 回退到最后一波段

        # 控制场景索引
        self.control_idx = 0
        if config.control_scene and config.control_scene != "auto":
            for idx, scene in enumerate(config.scenes):
                if scene.get("id") == config.control_scene:
                    self.control_idx = idx
                    break

        # 缓存路径
        self._cache_dir = os.path.join(self.output_root, ".cache")

        # 冒烟模式：预收集各景元数据以确定裁剪中心
        self._smoke_meta: Optional[List[Dict[str, Any]]] = None
        if self.smoke:
            self._smoke_meta = self._collect_smoke_metadata()

    def _requested_normalization_methods(self) -> List[str]:
        """
        Get the list of requested normalization methods in order.
        
        Returns base methods (original, bagrn, bagrn_volrn) plus any
        ablation methods, with duplicates removed while preserving order.
        """
        base_methods = ["original", "bagrn", "bagrn_volrn"]
        all_methods = base_methods + list(self.config.ablation_methods)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for method in all_methods:
            if method not in seen:
                seen.add(method)
                result.append(method)
        
        return result

    # -----------------------------------------------------------------------
    # 冒烟模式辅助
    # -----------------------------------------------------------------------

    def _collect_smoke_metadata(self) -> List[Dict[str, Any]]:
        """冒烟模式：预读各景元数据，确定裁剪中心。

        策略：不做全局交集中心，而是记录各景 bounds，
        后续在 detect_overlaps 之后为每条生成树边选择真实重叠中心裁剪。
        """
        import rasterio
        meta_list = []
        for scene_cfg in self.config.scenes:
            bands_dict = scene_cfg.get("bands", {})
            band_path = bands_dict.get(self.common_bands[0])
            if band_path is None:
                meta_list.append({})
                continue
            with rasterio.open(band_path) as src:
                tr = src.transform
                h, w = src.height, src.width
                left = tr.c
                top = tr.f
                right = left + tr.a * w
                bottom = top + tr.e * h
            meta_list.append({
                "bounds": (left, bottom, right, top),
                "height": h, "width": w, "transform": tr,
            })

        if meta_list:
            # 限制场景数：保留距所有景几何中心最近的 smoke_scene_count 景
            all_lefts = [m["bounds"][0] for m in meta_list if m]
            all_rights = [m["bounds"][2] for m in meta_list if m]
            all_tops = [m["bounds"][3] for m in meta_list if m]
            all_bottoms = [m["bounds"][1] for m in meta_list if m]
            center_x = (min(all_lefts) + max(all_rights)) / 2
            center_y = (min(all_bottoms) + max(all_tops)) / 2
            self._smoke_geo_center = (center_x, center_y)

            if len(meta_list) > self.config.smoke_scene_count:
                import math
                def _dist_to_center(m):
                    if not m or "bounds" not in m:
                        return float("inf")
                    lb, bot, rb, tp = m["bounds"]
                    mx = (lb + rb) / 2
                    my = (bot + tp) / 2
                    return math.hypot(mx - center_x, my - center_y)
                indexed = list(enumerate(meta_list))
                indexed.sort(key=lambda t: _dist_to_center(t[1]))
                keep = {i for i, _ in indexed[: self.config.smoke_scene_count]}
                self.config.scenes = [
                    s for i, s in enumerate(self.config.scenes) if i in keep
                ]
                meta_list = [m for i, m in enumerate(meta_list) if i in keep]
                logger.info(
                    "冒烟模式: 保留最近 %d 景（共 %d 景）",
                    len(meta_list), len(self.config.scenes),
                )
        else:
            self._smoke_geo_center = None

        return meta_list

    def _compute_overlap_center(self, bounds_i, bounds_j):
        """计算两景重叠区的地理中心坐标。"""
        left = max(bounds_i[0], bounds_j[0])
        right = min(bounds_i[2], bounds_j[2])
        bottom = max(bounds_i[1], bounds_j[1])
        top = min(bounds_i[3], bounds_j[3])
        if left >= right or bottom >= top:
            return None
        return ((left + right) / 2, (bottom + top) / 2)

    def _smoke_crop_around_center(self, arr: np.ndarray, tr, center_xy=None):
        """冒烟模式：围绕指定地理中心裁剪 smoke_crop_size x smoke_crop_size 区域。"""
        from rasterio.transform import Affine
        crop_size = self.smoke_crop_size
        if arr.ndim == 2:
            rows, cols = arr.shape
        else:
            _, rows, cols = arr.shape

        if center_xy is not None:
            cx, cy = center_xy
            inv_tr = ~tr
            col_center, row_center = inv_tr * (cx, cy)
            col_center, row_center = int(round(col_center)), int(round(row_center))
        elif self._smoke_geo_center is not None:
            cx, cy = self._smoke_geo_center
            inv_tr = ~tr
            col_center, row_center = inv_tr * (cx, cy)
            col_center, row_center = int(round(col_center)), int(round(row_center))
        else:
            row_center = rows // 2
            col_center = cols // 2

        r0 = max(0, row_center - crop_size // 2)
        c0 = max(0, col_center - crop_size // 2)
        r1 = min(rows, r0 + crop_size)
        c1 = min(cols, c0 + crop_size)
        if r1 - r0 < crop_size and rows >= crop_size:
            r0 = max(0, rows - crop_size)
            r1 = rows
        if c1 - c0 < crop_size and cols >= crop_size:
            c0 = max(0, cols - crop_size)
            c1 = cols

        if arr.ndim == 2:
            cropped = arr[r0:r1, c0:c1]
        else:
            cropped = arr[:, r0:r1, c0:c1]
        new_tr = tr * Affine.translation(c0, r0)
        return cropped, new_tr

    def _smoke_recrop_by_spanning_tree(
        self, scene_data: Dict[str, Any], tree_edges: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """冒烟模式：按生成树边的真实重叠中心重新裁剪各景。

        对每条树边 (parent, child)，取两景重叠区中心作为裁剪中心。
        每景的裁剪中心 = 其所有树边重叠中心的均值（保证该景与所有邻居重叠）。
        """
        import rasterio
        from rasterio.transform import Affine

        if not self.smoke or not tree_edges or not self._smoke_meta:
            return scene_data

        n_scenes = len(scene_data["arrays"])
        crop_size = self.smoke_crop_size

        # 为每条树边计算重叠中心
        edge_centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for parent, child in tree_edges:
            bounds_p = self._smoke_meta[parent].get("bounds")
            bounds_c = self._smoke_meta[child].get("bounds")
            if bounds_p and bounds_c:
                center = self._compute_overlap_center(bounds_p, bounds_c)
                if center:
                    edge_centers[(parent, child)] = center

        if not edge_centers:
            logger.warning("冒烟模式: 无法计算任何树边重叠中心，保持原始裁剪")
            return scene_data

        # 为每景计算裁剪中心 = 其所有关联树边重叠中心的均值
        scene_centers: Dict[int, Tuple[float, float]] = {}
        for idx in range(n_scenes):
            related_centers = []
            for (p, c), center in edge_centers.items():
                if p == idx or c == idx:
                    related_centers.append(center)
            if related_centers:
                avg_x = np.mean([c[0] for c in related_centers])
                avg_y = np.mean([c[1] for c in related_centers])
                scene_centers[idx] = (avg_x, avg_y)

        # 重新读取并裁剪各景
        new_arrays = []
        new_transforms = []
        new_bounds = []
        scenes = self.config.scenes

        for idx in range(n_scenes):
            center = scene_centers.get(idx)
            if center is None:
                new_arrays.append(scene_data["arrays"][idx])
                new_transforms.append(scene_data["transforms"][idx])
                new_bounds.append(scene_data["bounds"][idx])
                continue

            scene_cfg = scenes[idx]
            bands_dict = scene_cfg.get("bands", {})
            band_arrays = []
            tr = None
            for band_name in self.common_bands:
                band_path = bands_dict.get(band_name)
                if band_path is None:
                    continue
                arr, tr, _, _ = read_geotiff(band_path)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                arr, tr = self._smoke_crop_around_center(arr, tr, center_xy=center)
                band_arrays.append(arr)

            if band_arrays:
                multiband = np.stack(band_arrays, axis=0)
                new_arrays.append(multiband)
                new_transforms.append(tr)
                h, w = multiband.shape[1], multiband.shape[2]
                left = tr.c
                top = tr.f
                right = left + tr.a * w
                bottom = top + tr.e * h
                new_bounds.append((left, bottom, right, top))
            else:
                new_arrays.append(scene_data["arrays"][idx])
                new_transforms.append(scene_data["transforms"][idx])
                new_bounds.append(scene_data["bounds"][idx])

        scene_data["arrays"] = new_arrays
        scene_data["transforms"] = new_transforms
        scene_data["bounds"] = new_bounds

        logger.info(
            "冒烟模式: 按 %d 条生成树边重叠中心重新裁剪 %d 景",
            len(edge_centers), n_scenes,
        )
        return scene_data

    # -----------------------------------------------------------------------
    # a. 加载场景
    # -----------------------------------------------------------------------

    def load_scenes(self) -> Dict[str, Any]:
        """
        加载所有场景，验证波段可用性，处理多分辨率。

        返回
        -------
        dict
            {
                'arrays': list of np.ndarray,         # (bands, rows, cols)
                'transforms': list of Affine,
                'crs': str,
                'nodata_values': list of float|None,
                'bounds': list of (left, bottom, right, top),
                'scene_ids': list of str,
                'band_names': list of str,
                'resolution': float,                   # 公共分辨率
            }
        """
        logger.info("加载场景数据...")
        t0 = time.time()

        scenes = self.config.scenes
        n_scenes = len(scenes)

        arrays: List[np.ndarray] = []
        transforms = []
        crs_list: List[str] = []
        nodata_values: List[Optional[float]] = []
        bounds_list: List[Tuple[float, float, float, float]] = []
        scene_ids: List[str] = []

        for idx, scene_cfg in enumerate(scenes):
            sid = scene_cfg.get("id", f"scene_{idx}")
            scene_ids.append(sid)
            bands_dict = scene_cfg.get("bands", {})

            # 加载公共波段
            band_arrays = []
            for band_name in self.common_bands:
                band_path = bands_dict.get(band_name)
                if band_path is None:
                    raise ValueError(
                        f"场景 {sid} 缺少波段 {band_name}，"
                        f"可用波段: {list(bands_dict.keys())}"
                    )
                if self.smoke:
                    # 冒烟模式：先读取完整影像（用于准确重叠检测）
                    arr, tr, crs, nd = read_geotiff(band_path)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]  # 单波段文件 → (rows, cols)
                    elif arr.ndim == 3 and arr.shape[0] > 1:
                        raise ValueError(
                            f"场景 {sid} 波段 {band_name} 文件包含 {arr.shape[0]} 个波段，"
                            f"期望单波段文件"
                        )
                    band_arrays.append(arr)
                    if not band_arrays or len(band_arrays) == 1:
                        transforms.append(tr)
                        crs_list.append(crs)
                        nodata_values.append(nd)
                else:
                    arr, tr, crs, nd = read_geotiff(band_path)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]  # 单波段文件 → (rows, cols)
                    elif arr.ndim == 3 and arr.shape[0] > 1:
                        raise ValueError(
                            f"场景 {sid} 波段 {band_name} 文件包含 {arr.shape[0]} 个波段，"
                            f"期望单波段文件"
                        )
                    band_arrays.append(arr)
                    if not band_arrays or len(band_arrays) == 1:
                        transforms.append(tr)
                        crs_list.append(crs)
                        nodata_values.append(nd)

            # 组装多波段数组
            multiband = np.stack(band_arrays, axis=0)

            # After loading all bands for a scene, check intra-scene consistency
            if len(band_arrays) > 1:
                ref_shape = band_arrays[0].shape
                for bi, ba in enumerate(band_arrays[1:], 1):
                    if ba.shape != ref_shape:
                        raise ValueError(
                            f"场景 {sid} 波段形状不一致: 波段0={ref_shape}, 波段{bi}={ba.shape}"
                        )

            arrays.append(multiband)

            # 计算地理范围
            if self.smoke:
                h, w = multiband.shape[1], multiband.shape[2]
            else:
                with open(bands_dict[self.common_bands[0]], "rb") as f:
                    import rasterio
                    with rasterio.open(bands_dict[self.common_bands[0]]) as src:
                        h, w = src.height, src.width
            tr = transforms[idx]
            left = tr.c
            top = tr.f
            right = left + tr.a * w
            bottom = top + tr.e * h
            bounds_list.append((left, bottom, right, top))

        # 冒烟模式：验证全幅影像至少存在一对重叠区域
        if self.smoke and len(bounds_list) >= 2:
            has_overlap_any = False
            for i in range(len(bounds_list)):
                for j in range(i + 1, len(bounds_list)):
                    if has_overlap(bounds_list[i], bounds_list[j]):
                        has_overlap_any = True
                        break
                if has_overlap_any:
                    break
            if not has_overlap_any:
                logger.warning(
                    "冒烟模式全幅影像无重叠区域！scene_count=%d",
                    len(bounds_list),
                )

        # 统一分辨率
        # Strict mode validation
        if self.config.common_bands_strategy == "strict":
            from src.scene_preflight import validate_strict_scene_grids
            validate_strict_scene_grids(scene_ids, transforms, crs_list)
            # Use first scene's resolution (all should be same in strict mode)
            resolution = abs(transforms[0].a)
        else:
            # Non-strict mode: use minimum resolution
            resolutions = [abs(tr.a) for tr in transforms]
            resolution = min(resolutions)
            
            # Still validate CRS consistency
            if len(set(crs_list)) > 1:
                raise ValueError(
                    f"CRS 不一致: {dict(zip(scene_ids, crs_list))}。"
                    f"所有场景必须使用相同 CRS"
                )
        
        crs = crs_list[0]

        elapsed = time.time() - t0
        logger.info(
            "加载完成: %d 景, %d 波段, 分辨率=%.4f, 耗时 %.1fs",
            n_scenes, self.n_bands, resolution, elapsed,
        )

        return {
            "arrays": arrays,
            "transforms": transforms,
            "crs": crs,
            "nodata_values": nodata_values,
            "bounds": bounds_list,
            "scene_ids": scene_ids,
            "band_names": self.common_bands,
            "resolution": resolution,
        }

    # -----------------------------------------------------------------------
    # b. 检测重叠
    # -----------------------------------------------------------------------

    def detect_overlaps(self, scene_data: Dict[str, Any]) -> List[dict]:
        """
        检测所有场景对之间的重叠区域。

        参数
        ----------
        scene_data : dict
            由 load_scenes() 返回的数据字典。

        返回
        -------
        list of dict
            每个元素包含:
                idx_i, idx_j, window_i, window_j, pixel_count,
                per_band_mean, per_band_std
        """
        logger.info("检测重叠区域...")
        t0 = time.time()

        transforms = scene_data["transforms"]
        bounds_list = scene_data["bounds"]
        arrays = scene_data["arrays"]
        nodata_values = scene_data["nodata_values"]
        n_images = len(arrays)
        min_pixels = 1000 if not self.smoke else 100

        overlaps: List[dict] = []
        from src.coregistration import build_common_valid_mask

        for i in range(n_images):
            for j in range(i + 1, n_images):
                win = get_overlap_window(
                    bounds_list[i], transforms[i],
                    bounds_list[j], transforms[j],
                )
                if win is None:
                    continue

                (ri_s, ri_e, ci_s, ci_e), (rj_s, rj_e, cj_s, cj_e) = win
                bbox_pixels = (ri_e - ri_s) * (ci_e - ci_s)

                if bbox_pixels < min_pixels:
                    continue

                ref_patch = arrays[i][self.registration_band_idx, ri_s:ri_e, ci_s:ci_e]
                tgt_patch = arrays[j][self.registration_band_idx, rj_s:rj_e, cj_s:cj_e]
                common_h = min(ref_patch.shape[0], tgt_patch.shape[0])
                common_w = min(ref_patch.shape[1], tgt_patch.shape[1])
                common_valid = build_common_valid_mask(
                    ref_patch[:common_h, :common_w],
                    tgt_patch[:common_h, :common_w],
                    nodata_values[i], nodata_values[j],
                )
                common_valid_pixels = int(common_valid.sum())
                common_area = max(common_valid.size, 1)
                rows_with_common = np.flatnonzero(common_valid.any(axis=1))
                cols_with_common = np.flatnonzero(common_valid.any(axis=0))

                # 计算重叠区逐波段统计
                per_band_stats = {}
                for b_idx in range(self.n_bands):
                    pi = arrays[i][b_idx, ri_s:ri_e, ci_s:ci_e]
                    pj = arrays[j][b_idx, rj_s:rj_e, cj_s:cj_e]

                    nd_i = nodata_values[i]
                    nd_j = nodata_values[j]

                    mi = np.isfinite(pi)
                    if nd_i is not None:
                        mi &= (pi != nd_i)
                    mj = np.isfinite(pj)
                    if nd_j is not None:
                        mj &= (pj != nd_j)
                    
                    # Use independent valid masks to support different-sized windows
                    valid_i = pi[mi]
                    valid_j = pj[mj]

                    if valid_i.size > 0 and valid_j.size > 0:
                        per_band_stats[b_idx] = {
                            "mean_i": float(valid_i.mean()),
                            "mean_j": float(valid_j.mean()),
                            "std_i": float(valid_i.std()),
                            "std_j": float(valid_j.std()),
                        }

                overlaps.append({
                    "idx_i": i,
                    "idx_j": j,
                    "window_i": (ri_s, ri_e, ci_s, ci_e),
                    "window_j": (rj_s, rj_e, cj_s, cj_e),
                    "pixel_count": bbox_pixels,
                    "bbox_overlap_pixels": bbox_pixels,
                    "common_valid_pixels": common_valid_pixels,
                    "common_valid_ratio": float(common_valid_pixels / common_area),
                    "common_valid_shape": (common_h, common_w),
                    "common_valid_row_span": (
                        int(rows_with_common[-1] - rows_with_common[0] + 1)
                        if len(rows_with_common) else 0
                    ),
                    "common_valid_col_span": (
                        int(cols_with_common[-1] - cols_with_common[0] + 1)
                        if len(cols_with_common) else 0
                    ),
                    "per_band_stats": per_band_stats,
                })

        elapsed = time.time() - t0
        logger.info(
            "检测到 %d 对重叠区域, 耗时 %.1fs", len(overlaps), elapsed,
        )

        return overlaps

    # -----------------------------------------------------------------------
    # c. 配准
    # -----------------------------------------------------------------------

    def register_scenes(
        self,
        scene_data: Dict[str, Any],
        overlaps: List[dict],
        registration_band_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对所有场景执行多波段配准。

        流程：
          1. 用配准波段执行逐对稳健匹配并检查图连通性
          2. 网络平差并从原始影像执行全局残差精化
          3. 对通过门控的后全局残差执行可选 RBF 局部精化
          4. 从原始影像执行一次最终全局+局部位移 warp
          5. 对最终数组执行独立验证并分类质量

        参数
        ----------
        scene_data : dict
            由 load_scenes() 返回的数据字典。
        overlaps : list of dict
            由 detect_overlaps() 返回的重叠列表。
        registration_band_idx : int or None
            配准使用的波段索引。None 则使用 self.registration_band_idx。

        返回
        -------
        dict
            {
                'registered_arrays': list of np.ndarray,
                'global_shifts': np.ndarray (n_images, 2),
                'pair_matches': list of dict,
                'connected': bool,
                'status': 'pass' or 'fail',
                'failure': dict,
                'quality': dict,
                'final_validation': {'edges': list, 'overall': dict},
                'local_refinement': dict,
                'diagnostics': dict,
            }
        """
        from src.coregistration import (
            collect_block_matches,
            build_robust_pair_measurement,
            multi_image_network_adjustment,
            warp_multiband_with_displacement_field,
            rematch_pair_on_registered,
            refine_global_residual_shifts_from_original,
        )

        if registration_band_idx is None:
            registration_band_idx = self.registration_band_idx

        logger.info(
            "开始配准 (波段索引=%d, 波段名=%s)...",
            registration_band_idx,
            self.common_bands[registration_band_idx] if registration_band_idx < len(self.common_bands) else "?",
        )
        t0 = time.time()

        arrays = scene_data["arrays"]
        transforms = scene_data["transforms"]
        nodata_values = scene_data["nodata_values"]
        n_images = len(arrays)
        config = getattr(self, "config", None)
        reg_params = getattr(config, "registration_params", {}) if config is not None else {}

        if n_images <= 1:
            local_refinement = {
                "enabled": bool(reg_params.get("enable_local_refinement", True)),
                "used_for_scenes": [],
                "fallback_scenes": [],
                "cv_results": {},
                "rematch_failures": [],
            }
            quality = {
                "quality": "pass",
                "rmse": 0.0,
                "p95": 0.0,
                "median": 0.0,
                "confidence": 1.0,
                "n_blocks": 0,
            }
            final_validation = {"edges": [], "overall": quality}
            status, failure = _registration_status_and_failure(
                True, quality, reg_params.get("required_quality", "pass")
            )
            return {
                "registered_arrays": [a.copy() for a in arrays],
                "global_shifts": np.zeros((n_images, 2)),
                "local_dx_fields": [np.zeros(a.shape[1:], dtype=np.float64) for a in arrays],
                "local_dy_fields": [np.zeros(a.shape[1:], dtype=np.float64) for a in arrays],
                "local_refinement": local_refinement,
                "pair_matches": [],
                "connected": True,
                "spanning_tree": [],
                "geometric_edges": [],
                "matching_edges": [],
                "rejected_edges": [],
                "connected_components": [list(range(n_images))] if n_images else [],
                "unreachable_scenes": [],
                "status": status,
                "failure": failure,
                "quality": quality,
                "final_validation": final_validation,
                "diagnostics": {
                    "skipped": True,
                    "reason": "单景无需配准",
                    "local_refinement": local_refinement,
                    "final_validation": final_validation,
                },
            }

        # ---- Step 1: 逐对匹配（用配准波段） ----
        from src.coregistration import phase_correlation_from_overlap

        pair_measurements: List[dict] = []
        raw_block_matches: List[dict] = []
        rejected_edges: List[dict] = []
        holdout_contexts: Dict[Tuple[int, int], dict] = {}
        global_block_size = int(reg_params.get("global_block_size", 512))
        max_global_shift = float(reg_params.get("max_global_shift", 40.0))
        global_confidence_threshold = float(
            reg_params.get("global_confidence_threshold", 0.5)
        )
        holdout_enabled = bool(reg_params.get("enable_spatial_holdout", False))

        # Reserve exact validation windows before any matching or refinement.
        # This makes insufficient independent geometry an immediate failure,
        # rather than a late failure after a full registration run.
        if holdout_enabled:
            geometric_edges = [(ov["idx_i"], ov["idx_j"]) for ov in overlaps]
            for ov in overlaps:
                i, j = ov["idx_i"], ov["idx_j"]
                holdout_context = _build_pair_holdout_context(
                    arrays[i][registration_band_idx], transforms[i],
                    arrays[j][registration_band_idx], transforms[j],
                    nodata_values[i], nodata_values[j], reg_params,
                )
                if holdout_context.get("available"):
                    ri_s, ri_e, ci_s, ci_e = holdout_context["patch_window_ref"]
                    full_holdout = np.zeros(
                        arrays[i][registration_band_idx].shape, dtype=bool,
                    )
                    full_holdout[ri_s:ri_e, ci_s:ci_e] = holdout_context[
                        "holdout_region_mask"
                    ]
                    holdout_context["holdout_region_full_mask"] = full_holdout
                holdout_contexts[(i, j)] = holdout_context
                required = int(reg_params.get("final_min_blocks", 5))
                if (not holdout_context.get("available")
                        or int(holdout_context.get("reserved_count", 0)) < required):
                    reason = holdout_context.get(
                        "failure_reason",
                        "insufficient independent validation geometry",
                    )
                    failure_result = _registration_failure_result(
                        arrays,
                        reg_params.get("enable_local_refinement", True),
                        [], [], geometric_edges, [], [],
                        _graph_components(n_images, geometric_edges),
                        [
                            scene_data.get("scene_ids", [str(k) for k in range(n_images)])[k]
                            for k in range(n_images) if k != self.control_idx
                        ],
                        reason,
                    )
                    failure_result["diagnostics"]["holdout"] = {
                        f"{i}-{j}": _public_holdout_summary(holdout_context)
                    }
                    failure_result["diagnostics"]["validation_reservation"] = {
                        f"{i}-{j}": _public_holdout_summary(holdout_context).get(
                            "validation_reservation"
                        )
                    }
                    return failure_result

        for ov in overlaps:
            i, j = ov["idx_i"], ov["idx_j"]
            arr_i_reg = arrays[i][registration_band_idx]
            arr_j_reg = arrays[j][registration_band_idx]
            nd_i = nodata_values[i]  # Keep None as None, don't convert to 0
            nd_j = nodata_values[j]  # Keep None as None, don't convert to 0

            sampling_mask = None
            if holdout_enabled:
                holdout_context = holdout_contexts[(i, j)]
                if holdout_context.get("available"):
                    sampling_mask = holdout_context["holdout_exclusion_mask"]

            # 尝试1: 块匹配
            matches, screening = collect_block_matches(
                arr_i_reg, transforms[i],
                arr_j_reg, transforms[j],
                nd_i, nd_j,
                block_size=global_block_size,
                max_global_shift=max_global_shift,
                confidence_threshold=global_confidence_threshold,
                **({"holdout_exclusion_mask": sampling_mask}
                   if sampling_mask is not None else {}),
            )
            raw_matches = [dict(match) for match in matches]
            raw_block_matches.append({
                "idx_i": i, "idx_j": j,
                "matches": raw_matches,
                "screening": dict(screening),
            })

            if matches:
                robust_pair = build_robust_pair_measurement(
                    matches, {**reg_params, "screening": screening})
                if robust_pair["status"] == "pass":
                    pair_measurements.append({
                        "idx_i": i, "idx_j": j,
                        "shift_dx": robust_pair["shift_dx"],
                        "shift_dy": robust_pair["shift_dy"],
                        "confidence": robust_pair["confidence"],
                        "n_blocks": robust_pair["n_blocks_inlier"],
                        "n_blocks_total": robust_pair["n_blocks_total"],
                        "rmse": robust_pair["rmse"], "p95": robust_pair["p95"],
                        "matches": robust_pair["matches"],
                        "raw_matches": raw_matches,
                        "screening": robust_pair["screening"],
                        "method": "block_match",
                        "holdout": _public_holdout_summary(
                            holdout_contexts.get((i, j))
                        ),
                    })
                    logger.info(
                        "  [%d]-[%d] block_match: dx=%.4f, dy=%.4f, conf=%.3f, "
                        "blocks=%d/%d, rmse=%.3f, p95=%.3f",
                        i, j, robust_pair["shift_dx"], robust_pair["shift_dy"],
                        robust_pair["confidence"], robust_pair["n_blocks_inlier"],
                        robust_pair["n_blocks_total"], robust_pair["rmse"], robust_pair["p95"],
                    )
                    continue
                logger.info("  [%d]-[%d] 稳健块匹配失败: %s", i, j, robust_pair.get("reason"))

            # 尝试2: overlap phase correlation fallback（地理重叠区）
            logger.info("  [%d]-[%d] 块匹配失败，尝试 phase correlation fallback...", i, j)
            try:
                sy, sx, conf_pc = phase_correlation_from_overlap(
                    arr_i_reg, transforms[i],
                    arr_j_reg, transforms[j],
                    nodata_ref=nd_i, nodata_tgt=nd_j,
                    holdout_exclusion_mask=(holdout_contexts.get((i, j)) or {}).get(
                        "holdout_exclusion_mask"
                    ) if holdout_enabled else None,
                )
            except Exception as exc:
                logger.warning("  [%d]-[%d] phase correlation 异常: %s", i, j, exc)
                sy, sx, conf_pc = 0.0, 0.0, 0.0

            if (conf_pc >= global_confidence_threshold
                    and abs(sy) < max_global_shift
                    and abs(sx) < max_global_shift):
                pair_measurements.append({
                    "idx_i": i, "idx_j": j,
                    "shift_dx": float(sx), "shift_dy": float(sy),
                    "confidence": float(conf_pc),
                    "n_blocks": 1,
                    "rmse": 0.0, "p95": 0.0,
                    "matches": [], "raw_matches": raw_matches,
                    "screening": screening,
                    "method": "overlap_phase_correlation",
                    "holdout": _public_holdout_summary(
                        holdout_contexts.get((i, j))
                    ),
                })
                logger.info(
                    "  [%d]-[%d] overlap_phase_correlation: dx=%.4f, dy=%.4f, conf=%.3f",
                    i, j, float(sx), float(sy), float(conf_pc),
                )
                continue

            # 尝试3: 拒绝该边
            reason = "稳健块匹配失败且phase_correlation失败"
            if matches:
                reason = f"稳健块匹配失败 ({robust_pair.get('reason', 'unknown')}) 且phase_correlation失败"
            if screening.get("low_valid", 0) > 0:
                reason += f" (low_valid={screening['low_valid']})"
            if screening.get("low_texture", 0) > 0:
                reason += f" (low_texture={screening['low_texture']})"
            rejected_edges.append({
                "idx_i": i, "idx_j": j,
                "reason": reason, "screening": screening,
                "raw_matches": raw_matches,
            })
            logger.warning("  [%d]-[%d] 拒绝: %s", i, j, reason)

        if not pair_measurements:
            geometric_edges = [(ov["idx_i"], ov["idx_j"]) for ov in overlaps]
            rejected_list = [
                (item["idx_i"], item["idx_j"], item["reason"])
                for item in rejected_edges
            ]
            scene_ids = scene_data.get(
                "scene_ids", [str(i) for i in range(n_images)]
            )
            unreachable_ids = [
                scene_ids[idx] for idx in range(n_images) if idx != self.control_idx
            ]
            failure_result = _registration_failure_result(
                arrays,
                reg_params.get("enable_local_refinement", True),
                [], [], geometric_edges, [], rejected_list,
                _graph_components(n_images, geometric_edges),
                unreachable_ids,
                "没有可用的匹配对，无法进行配准",
            )
            failure_result["diagnostics"]["raw_block_matches"] = raw_block_matches
            return failure_result

        # ---- Step 1.5: 连通性分析 ----
        # 几何重叠边（所有 detect_overlaps 发现的对）
        geometric_edges = [(ov["idx_i"], ov["idx_j"]) for ov in overlaps]
        # 有效匹配边
        matching_edges = [(p["idx_i"], p["idx_j"]) for p in pair_measurements]
        # 拒绝边
        rejected_list = [(r["idx_i"], r["idx_j"], r["reason"]) for r in rejected_edges]

        logger.info("几何重叠边: %d 条 %s", len(geometric_edges), geometric_edges)
        logger.info("有效匹配边: %d 条 %s", len(matching_edges), matching_edges)
        if rejected_list:
            logger.info("拒绝边: %d 条", len(rejected_list))
            for ri, rj, reason in rejected_list:
                logger.info("  [%d]-[%d] 原因: %s", ri, rj, reason)

        # 计算连通分量（BFS）
        components = [set(component) for component in _graph_components(
            n_images, geometric_edges,
        )]
        logger.info("连通分量: %d 个 %s", len(components),
                     [sorted(c) for c in components])

        # 从有效匹配边构建生成树（BFS from control_idx）
        adj_match = defaultdict(set)
        for p in pair_measurements:
            adj_match[p["idx_i"]].add(p["idx_j"])
            adj_match[p["idx_j"]].add(p["idx_i"])

        visited = set([self.control_idx])
        queue = deque([self.control_idx])
        spanning_tree_edges: List[Tuple[int, int]] = []
        while queue:
            node = queue.popleft()
            for nb in adj_match[node]:
                if nb not in visited:
                    visited.add(nb)
                    spanning_tree_edges.append((node, nb))
                    queue.append(nb)

        unreachable = set(range(n_images)) - visited
        scene_ids = scene_data.get("scene_ids", [str(i) for i in range(n_images)])
        unreachable_ids = [scene_ids[i] for i in sorted(unreachable)]

        logger.info("生成树边: %d 条 %s", len(spanning_tree_edges), spanning_tree_edges)
        logger.info("可达场景: %d/%d %s", len(visited), n_images, sorted(visited))
        if unreachable:
            logger.warning("不可达场景: %s", unreachable_ids)

        # 不可达时返回稳定失败结果，让 run() 使用统一质量门控。
        if unreachable:
            reason = f"网络不连通: 场景 {unreachable_ids} 无法从控制景到达"
            if self.smoke:
                reason = (
                    f"smoke 五景验收失败: 场景 {unreachable_ids} 不可达，"
                    f"要求所有 {n_images} 景连通"
                )
            failure_result = _registration_failure_result(
                arrays,
                reg_params.get("enable_local_refinement", True),
                pair_measurements,
                spanning_tree_edges,
                geometric_edges,
                matching_edges,
                rejected_list,
                [sorted(component) for component in components],
                unreachable_ids,
                reason,
            )
            failure_result["diagnostics"]["raw_block_matches"] = raw_block_matches
            return failure_result

        # ---- Step 2: 网络平差 ----
        # 用所有实测边做网络平差
        adj_result = multi_image_network_adjustment(
            pair_measurements, n_images, self.control_idx,
        )
        global_shifts = adj_result["global_shifts"]
        logger.info("网络平差完成, 闭环误差数: %d", len(adj_result.get("loop_errors", [])))

        # ---- Step 2.5: 用全局配准后的残余重新精化全局位移 ----
        refine_params = {**reg_params, "reference_idx": self.control_idx}
        refine_result = refine_global_residual_shifts_from_original(
            [a[registration_band_idx] for a in arrays],
            global_shifts,
            transforms,
            nodata_values,
            matching_edges,
            refine_params,
            sampling_masks={
                edge: context.get("holdout_exclusion_mask")
                for edge, context in holdout_contexts.items()
                if context.get("available")
            },
        )
        global_shifts = refine_result["global_shifts"]
        logger.info(
            "全局残余精化完成, iterations=%d, warnings=%d",
            len(refine_result["history"]), len(refine_result["warnings"]),
        )

        # ---- Step 3: 后全局残余控制与门控局部RBF ----
        from src.coregistration import (
            balance_edge_controls,
            build_local_residual_controls,
            build_parent_based_local_controls,
            rematch_pair_on_registered,
            warp_multiband_with_displacement_field,
        )

        local_enabled = bool(reg_params.get("enable_local_refinement", True))
        local_dx_fields = [
            np.zeros(arrays[idx].shape[1:], dtype=np.float64) for idx in range(n_images)
        ]
        local_dy_fields = [
            np.zeros(arrays[idx].shape[1:], dtype=np.float64) for idx in range(n_images)
        ]
        local_refinement = {
            "enabled": local_enabled,
            "used_for_scenes": [],
            "fallback_scenes": [],
            "cv_results": {},
            "scenes": {},
            "rematch_failures": [],
        }

        # Rematch only the global-only arrays generated from ORIGINAL inputs.
        global_only_arrays = []
        post_global_pairs = []
        post_global_failures = []
        if local_enabled:
            for idx in range(n_images):
                gdx, gdy = global_shifts[idx]
                h, w = arrays[idx].shape[1:]
                global_only_arrays.append(warp_multiband_with_displacement_field(
                    arrays[idx], gdx, gdy,
                    np.zeros((h, w), dtype=np.float64),
                    np.zeros((h, w), dtype=np.float64),
                    nodata_values[idx],
                ))
            local_confidence = float(reg_params.get("local_confidence_threshold", 0.60))
            post_global_result = _collect_post_global_residual_pairs(
                global_only_arrays, registration_band_idx, transforms, nodata_values,
                matching_edges, reg_params, rematch_pair_on_registered,
                holdout_contexts=holdout_contexts,
            )
            post_global_pairs = post_global_result["pairs"]
            post_global_failures = post_global_result["failures"]
            local_refinement["rematch_failures"] = post_global_failures

        parent_map = {child: parent for parent, child in spanning_tree_edges}
        for idx in range(n_images):
            if idx == self.control_idx:
                continue
            scene_result = {"accepted": False, "n_controls": 0, "n_spatial_groups": 0}
            if not local_enabled:
                scene_result["reason"] = "local refinement disabled"
                local_refinement["fallback_scenes"].append(idx)
                local_refinement["scenes"][str(idx)] = scene_result
                continue

            edge_controls = []
            parent_idx = parent_map.get(idx)
            parent_added = False
            if parent_idx is not None:
                parent_control = build_parent_based_local_controls(
                    idx, parent_idx, post_global_pairs, global_shifts,
                    confidence_threshold=local_confidence, min_points=1,
                    mad_scale=float(reg_params.get("local_outlier_mad_scale", 3.0)),
                    hard_max_shift=float(reg_params.get("local_hard_max_component", 8.0)),
                    neighbor_radius=int(reg_params.get("local_neighbor_k", 8)),
                )
                if parent_control.get("n_valid", 0) > 0:
                    edge_controls.append(parent_control)
                    parent_added = True

            # Add other post-global edges in the current target scene's pixel system.
            for pair in post_global_pairs:
                if parent_added and {pair["idx_i"], pair["idx_j"]} == {idx, parent_idx}:
                    continue
                if idx == pair["idx_j"]:
                    oriented_matches = pair["matches"]
                elif idx == pair["idx_i"]:
                    oriented_matches = [
                        {
                            **match,
                            "tgt_x": match["ref_x"], "tgt_y": match["ref_y"],
                            "shift_dx": -match["shift_dx"],
                            "shift_dy": -match["shift_dy"],
                        }
                        for match in pair["matches"]
                        if "ref_x" in match and "ref_y" in match
                    ]
                else:
                    continue
                controls = build_local_residual_controls(
                    oriented_matches, 0.0, 0.0,
                    confidence_threshold=local_confidence, min_points=1,
                    mad_scale=float(reg_params.get("local_outlier_mad_scale", 3.0)),
                    hard_max_shift=float(reg_params.get("local_hard_max_component", 8.0)),
                    neighbor_radius=int(reg_params.get("local_neighbor_k", 8)),
                )
                if controls.get("n_valid", 0) > 0:
                    edge_controls.append(controls)

            balanced = balance_edge_controls(
                edge_controls,
                max_total=int(reg_params.get("local_max_controls", 60)),
                grid_size=int(reg_params.get("local_block_size", 256)),
                min_per_edge=1,
                dedup_distance=float(reg_params.get("local_dedup_distance", 50.0)),
            )
            usable_balanced = [c for c in balanced if c.get("n_valid", 0) > 0]
            if usable_balanced:
                points = np.vstack([c["points_xy"] for c in usable_balanced])
                residual_dx = np.concatenate([
                    c["residual_dx"] for c in usable_balanced
                ])
                residual_dy = np.concatenate([
                    c["residual_dy"] for c in usable_balanced
                ])
                confidence = np.concatenate([
                    c["confidence"] for c in usable_balanced
                ])
                controls = {
                    "points_xy": points, "residual_dx": residual_dx,
                    "residual_dy": residual_dy, "confidence": confidence,
                    "n_valid": len(points),
                }
            else:
                controls = _empty_local_controls()

            scene_failures = [
                failure for failure in post_global_failures
                if idx in (failure["idx_i"], failure["idx_j"])
            ]
            cv_result = _local_holdout_cv(controls, reg_params)
            gate = _accept_local_rbf_candidate(
                controls, reg_params, cv_result, rematch_failures=scene_failures
            )
            scene_result.update(gate)
            scene_result["local_controls"] = {
                "n_raw": int(controls.get("n_raw", controls.get("n_valid", 0))),
                "n_confidence": int(controls.get("n_confidence", controls.get("n_valid", 0))),
                "n_spatial_consistent": int(controls.get("n_spatial_consistent", controls.get("n_valid", 0))),
                "n_rejected_gross": int(controls.get("n_rejected_gross", 0)),
                "residual_dx_percentiles": _percentiles(controls.get("residual_dx", [])),
                "residual_dy_percentiles": _percentiles(controls.get("residual_dy", [])),
            }
            if scene_failures:
                scene_result["rematch_failures"] = scene_failures
            local_refinement["cv_results"][str(idx)] = cv_result or {}
            candidate_results = (cv_result or {}).get("candidate_results", [])
            logger.info(
                "  [%d] local RBF CV: candidates=%d, selected=%s, passing=%s, reason=%s",
                idx,
                len(candidate_results),
                (cv_result or {}).get("selected_smoothing"),
                (cv_result or {}).get("has_passing_candidate"),
                (cv_result or {}).get("selection_reason"),
            )
            if gate["accepted"]:
                try:
                    selected_smoothing = gate.get("selected_smoothing")
                    local_dx_fields[idx], local_dy_fields[idx], field_stats = _fit_local_rbf_field(
                        controls, arrays[idx].shape[1:], reg_params,
                        smoothing=selected_smoothing,
                    )
                    scene_result["field_stats"] = field_stats
                    local_refinement["used_for_scenes"].append(idx)
                except Exception as exc:
                    scene_result["accepted"] = False
                    scene_result["reason"] = f"local RBF fitting failed: {exc}"
                    local_refinement["fallback_scenes"].append(idx)
            else:
                local_refinement["fallback_scenes"].append(idx)
            local_refinement["scenes"][str(idx)] = scene_result

        # ---- Step 4: 对所有波段从 ORIGINAL 影像施加一次最终位移 ----
        registered_arrays: List[np.ndarray] = []
        for idx in range(n_images):
            gdx = global_shifts[idx, 0]
            gdy = global_shifts[idx, 1]

            if (abs(gdx) < 1e-6 and abs(gdy) < 1e-6
                    and not np.any(local_dx_fields[idx])
                    and not np.any(local_dy_fields[idx])):
                registered_arrays.append(arrays[idx].astype(np.float64))
            else:
                h, w = arrays[idx].shape[1], arrays[idx].shape[2]
                nd_val = nodata_values[idx]  # Keep None as None, don't convert to 0.0

                warped = warp_multiband_with_displacement_field(
                    arrays[idx], gdx, gdy,
                    local_dx_fields[idx], local_dy_fields[idx], nd_val,
                )
                registered_arrays.append(warped)

            logger.info("  [%d] 全局位移: dx=%.4f, dy=%.4f", idx, gdx, gdy)

        # ---- Step 5: 在最终注册数组上执行独立验证 ----
        final_quality, final_validation = _validate_final_registration_arrays(
            registered_arrays,
            registration_band_idx,
            transforms,
            nodata_values,
            spanning_tree_edges,
            [*pair_measurements, *post_global_pairs],
            reg_params,
            holdout_contexts=holdout_contexts,
        )
        connected = not bool(unreachable)
        status, failure = _registration_status_and_failure(
            connected,
            final_quality,
            reg_params.get("required_quality", "pass"),
        )

        elapsed = time.time() - t0
        logger.info("配准完成, 耗时 %.1fs", elapsed)

        holdout_diagnostics = {
            f"{i}-{j}": _public_holdout_summary(context)
            for (i, j), context in holdout_contexts.items()
        }
        local_control_diagnostics = {
            str(idx): scene.get("local_controls", {})
            for idx, scene in local_refinement.get("scenes", {}).items()
            if scene.get("local_controls")
        }
        local_field_diagnostics = {
            str(idx): scene.get("field_stats", {})
            for idx, scene in local_refinement.get("scenes", {}).items()
            if scene.get("field_stats")
        }
        diagnostic_masks = {
            f"{i}-{j}": {
                "common_valid_mask": context.get("common_valid_mask", np.zeros((0, 0), dtype=bool)),
                "holdout_region_mask": context.get("holdout_region_mask", np.zeros((0, 0), dtype=bool)),
                "train_sampling_mask": context.get("train_sampling_mask", np.zeros((0, 0), dtype=bool)),
                "holdout_exclusion_mask": context.get("holdout_exclusion_mask", np.zeros((0, 0), dtype=bool)),
            }
            for (i, j), context in holdout_contexts.items()
            if context.get("available")
        }

        return {
            "registered_arrays": registered_arrays,
            "global_shifts": global_shifts,
            "local_dx_fields": local_dx_fields,
            "local_dy_fields": local_dy_fields,
            "local_refinement": local_refinement,
            "pair_matches": pair_measurements,
            "connected": connected,
            "status": status,
            "failure": failure,
            "spanning_tree": spanning_tree_edges,
            "geometric_edges": geometric_edges,
            "matching_edges": matching_edges,
            "rejected_edges": rejected_list,
            "quality": final_quality,
            "final_validation": final_validation,
            "diagnostic_masks": diagnostic_masks,
            "connected_components": [sorted(c) for c in components],
            "unreachable_scenes": unreachable_ids,
            "diagnostics": {
                "n_pairs_matched": len(pair_measurements),
                "n_rejected": len(rejected_edges),
                "n_geometric": len(geometric_edges),
                "n_components": len(components),
                "raw_block_matches": raw_block_matches,
                "global_shifts": global_shifts.tolist(),
                "loop_errors": adj_result.get("loop_errors", []),
                "global_refinement_history": refine_result["history"],
                "global_refinement_warnings": refine_result["warnings"],
                "local_refinement": local_refinement,
                "overlap": {
                    f"{ov['idx_i']}-{ov['idx_j']}": {
                        "bbox_pixels": int(ov.get("bbox_overlap_pixels", ov.get("pixel_count", 0))),
                        "common_valid_pixels": int(ov.get("common_valid_pixels", 0)),
                        "common_valid_ratio": float(ov.get("common_valid_ratio", 0.0)),
                        "common_valid_row_span": int(ov.get("common_valid_row_span", 0)),
                        "common_valid_col_span": int(ov.get("common_valid_col_span", 0)),
                    }
                    for ov in overlaps
                },
                "holdout": holdout_diagnostics,
                "local_controls": local_control_diagnostics,
                "local_field": local_field_diagnostics,
                "final_validation": final_validation,
                "quality": final_quality,
                "elapsed_sec": elapsed,
            },
        }

    # -----------------------------------------------------------------------
    # d. 辐射归一化
    # -----------------------------------------------------------------------

    def apply_radiometric_normalization(
        self,
        registered_data: Dict[str, Any],
        overlaps: List[dict],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """
        对配准后的数据应用多种辐射归一化方法。

        支持的方法：
          - 'original': 不做归一化（原始数据）
          - 'bagrn': 全局归一化（BAGRN）
          - 'volrn_only': 仅局部归一化（无 BAGRN 前处理）
          - 'bagrn_volrn': BAGRN + VOLRN（论文主方法）
          - 'histogram_matching': 直方图匹配（基线）
          - 'moment_matching': 矩匹配（基线）
          - 'wallis': Wallis 滤波（基线）

        参数
        ----------
        registered_data : dict
            由 register_scenes() 返回的数据字典。
        overlaps : list of dict
            重叠区域信息。
        methods : list of str or None
            要执行的方法列表。None 则使用 config 中的 ablation_methods。

        返回
        -------
        dict
            method_name -> list of np.ndarray (归一化后的数组)。
        """
        if methods is None:
            methods = ["original", "bagrn", "bagrn_volrn"] + list(self.config.ablation_methods)

        # 去重保持顺序
        seen = set()
        unique_methods = []
        for m in methods:
            if m not in seen:
                seen.add(m)
                unique_methods.append(m)
        methods = unique_methods

        logger.info("执行辐射归一化, 方法: %s", methods)
        t0 = time.time()

        arrays = registered_data["registered_arrays"]
        transforms = registered_data.get("transforms", [])
        bounds_list = registered_data.get("bounds", [])
        nodata_values_list = registered_data.get("nodata_values", [None] * len(arrays))

        control_idx = self.control_idx
        volrn_params = self.config.volrn_params

        # 构建生成树用于传播型方法
        tree_edges = build_spanning_tree(overlaps, len(arrays), control_idx)
        spanning_tree = tree_edges if tree_edges else None

        results: Dict[str, List[np.ndarray]] = {}
        bagrn_cache = None

        for method in methods:
            logger.info("  方法: %s", method)
            t_method = time.time()

            try:
                if method == "original":
                    results[method] = [a.copy() for a in arrays]

                elif method == "bagrn":
                    normalized, _, _ = bagrn_normalize(
                        arrays, nodata_values_list, overlaps, control_idx,
                    )
                    bagrn_cache = normalized
                    results[method] = normalized

                elif method == "volrn_only":
                    # 无 BAGRN 前处理的 VOLRN
                    normalized, _ = volrn_normalize(
                        arrays, transforms, bounds_list, nodata_values_list,
                        block_size_pixels=volrn_params.get("block_size", 400),
                        lambda_param=volrn_params.get("lambda", 0.5),
                        rho=volrn_params.get("rho", 1.0),
                        max_iter=volrn_params.get("max_iter", 200),
                        tol=volrn_params.get("tol", 1e-4),
                        verbose=False,
                    )
                    results[method] = normalized

                elif method == "bagrn_volrn":
                    if bagrn_cache is not None:
                        bagrn_normalized = bagrn_cache
                    else:
                        bagrn_normalized, _, _ = bagrn_normalize(
                            arrays, nodata_values_list, overlaps, control_idx,
                        )
                    volrn_normalized, volrn_block_coeffs, volrn_diagnostics = volrn_normalize(
                        bagrn_normalized, transforms, bounds_list, nodata_values_list,
                        block_size_pixels=volrn_params.get("block_size", 400),
                        lambda_param=volrn_params.get("lambda", 0.5),
                        rho=volrn_params.get("rho", 1.0),
                        max_iter=volrn_params.get("max_iter", 200),
                        tol=volrn_params.get("tol", 1e-4),
                        verbose=False,
                        return_diagnostics=True,
                    )
                    results[method] = volrn_normalized
                    results[f"{method}_block_coefficients"] = volrn_block_coeffs

                elif method in ("histogram_matching", "moment_matching", "wallis"):
                    normalized = run_comparison(
                        method, arrays, nodata_values_list, overlaps,
                        control_idx=control_idx,
                        spanning_tree=spanning_tree,
                    )
                    results[method] = normalized

                else:
                    logger.warning("未知方法 '%s'，跳过", method)
                    continue

            except Exception as exc:
                logger.error("  方法 '%s' 失败: %s", method, exc)
                continue

            elapsed_m = time.time() - t_method
            logger.info("    %s 完成, 耗时 %.1fs", method, elapsed_m)

        elapsed = time.time() - t0
        logger.info("所有归一化方法完成, 总耗时 %.1fs", elapsed)

        return results

    # -----------------------------------------------------------------------
    # e. 生成镶嵌
    # -----------------------------------------------------------------------

    def compute_mosaics(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        transforms: list,
        crs: str,
        nodata_values: list,
        output_dir: str,
    ) -> Dict[str, List[str]]:
        """
        对每种归一化方法生成镶嵌图。

        参数
        ----------
        normalized_dict : dict
            method_name -> list of np.ndarray。
        transforms : list
            仿射变换列表。
        crs : str
            坐标参考系。
        nodata_values : list
            各景 NoData 值。
        output_dir : str
            输出目录。

        返回
        -------
        dict
            method_name -> list of mosaic file paths。
        """
        logger.info("生成镶嵌图...")
        t0 = time.time()

        os.makedirs(output_dir, exist_ok=True)
        mosaic_results: Dict[str, List[str]] = {}

        for method, arrays in normalized_dict.items():
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            paths: List[str] = []

            for mode in self.config.mosaic_modes:
                mode = mode.strip()  # 清理可能的前导空格
                if mode == "narrow_feather":
                    for fw in self.config.feather_widths:
                        out_name = f"mosaic_{method}_{mode}_fw{fw}.tif"
                        out_path = os.path.join(method_dir, out_name)

                        if self.dry_run:
                            logger.info("  [DRY-RUN] 将生成: %s", out_path)
                            paths.append(out_path)
                            continue

                        try:
                            create_mosaic(
                                arrays, transforms, crs, nodata_values,
                                out_path,
                                mode=mode,
                                feather_width=fw,
                            )
                            paths.append(out_path)
                            logger.info("  已生成: %s", out_path)
                        except Exception as exc:
                            logger.error("  镶嵌失败 [%s, %s, fw=%d]: %s", method, mode, fw, exc)
                else:
                    out_name = f"mosaic_{method}_{mode}.tif"
                    out_path = os.path.join(method_dir, out_name)

                    if self.dry_run:
                        logger.info("  [DRY-RUN] 将生成: %s", out_path)
                        paths.append(out_path)
                        continue

                    try:
                        create_mosaic(
                            arrays, transforms, crs, nodata_values,
                            out_path,
                            mode=mode,
                        )
                        paths.append(out_path)
                        logger.info("  已生成: %s", out_path)
                    except Exception as exc:
                        logger.error("  镶嵌失败 [%s, %s]: %s", method, mode, exc)

            mosaic_results[method] = paths

        elapsed = time.time() - t0
        logger.info("镶嵌完成, 耗时 %.1fs", elapsed)

        return mosaic_results

    # -----------------------------------------------------------------------
    # f. 评价指标
    # -----------------------------------------------------------------------

    def evaluate_metrics(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        overlaps: List[dict],
        nodata_values: list,
        bands: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        对每种归一化方法计算评价指标。

        指标包括 ADM, ADSD, CD, GL, RDOA, Ave（论文 Section 3.1）。

        参数
        ----------
        normalized_dict : dict
            method_name -> list of np.ndarray。
        overlaps : list of dict
            重叠区域信息。
        nodata_values : list
            各景 NoData 值。
        bands : list of int or None
            波段索引列表。None 则处理所有波段。
        output_dir : str or None
            CSV 输出目录。None 则不保存。

        返回
        -------
        dict
            method_name -> {adm, adsd, cd, gl, rdoa, ave, per_pair, per_band}。
        """
        logger.info("计算评价指标...")
        t0 = time.time()

        if bands is None:
            bands = list(range(self.n_bands))

        # 获取原始数组用于 GL 计算
        original_arrays = normalized_dict.get("original", None)

        metrics_results: Dict[str, dict] = {}

        for method, arrays in normalized_dict.items():
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue
            t_method = time.time()
            logger.info("  评估方法: %s", method)

            # 全局指标
            metrics = compute_all(
                original_arrays if original_arrays is not None else arrays,
                arrays, nodata_values, overlaps, bands,
            )

            # 逐对指标
            per_pair = compute_per_pair(arrays, nodata_values, overlaps, bands)

            # 逐波段指标
            per_band = compute_per_band(arrays, nodata_values, overlaps, bands)

            metrics_results[method] = {
                **metrics,
                "per_pair": per_pair,
                "per_band": per_band,
            }

            # 保存 CSV
            if output_dir:
                method_dir = os.path.join(output_dir, method)
                os.makedirs(method_dir, exist_ok=True)
                save_metrics_csv(
                    metrics, method_dir,
                    bands=bands, overlaps=overlaps,
                    arrays_before=original_arrays if original_arrays is not None else arrays,
                    arrays_after=arrays,
                    nodata_values=nodata_values,
                )

            elapsed_m = time.time() - t_method
            logger.info("    %s 评估完成, 耗时 %.1fs", method, elapsed_m)

        elapsed = time.time() - t0
        logger.info("指标计算完成, 耗时 %.1fs", elapsed)

        return metrics_results

    # -----------------------------------------------------------------------
    # f2. 光谱保持指标
    # -----------------------------------------------------------------------

    def evaluate_spectral(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        original_arrays: List[np.ndarray],
        nodata_values: list,
        band_names: List[str],
        overlaps: List[dict],
        output_dir: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        计算光谱保持指标：SAM, spectral RMSE, correlation。

        对每种归一化方法（除 original 外），逐景比较归一化前后光谱。
        """
        from src.spectral_metrics import compute_all_spectral

        logger.info("计算光谱保持指标...")
        t0 = time.time()

        spectral_results: Dict[str, dict] = {}

        for method, arrays in normalized_dict.items():
            if method == "original":
                continue
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue

            logger.info("  光谱指标: %s", method)
            try:
                method_dir = os.path.join(output_dir, method) if output_dir else None
                spectral = compute_all_spectral(
                    original_arrays, arrays, nodata_values, overlaps,
                    band_names=band_names, output_dir=method_dir,
                )
                spectral_results[method] = spectral
            except Exception as exc:
                logger.error("  光谱指标计算失败 [%s]: %s", method, exc)
                spectral_results[method] = {"error": str(exc)}

        elapsed = time.time() - t0
        logger.info("光谱指标完成, 耗时 %.1fs", elapsed)
        return spectral_results

    # -----------------------------------------------------------------------
    # f3. 数据质量检查
    # -----------------------------------------------------------------------

    def check_data_quality(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        scene_data: Dict[str, Any],
    ) -> Dict[str, dict]:
        """
        检查每种方法输出的数据质量：NaN/Inf/valid pixels/shape/transform/CRS。

        如果有效数据区出现 NaN 或 Inf，标记该方法为失败。
        使用 normalized_dict["original"] 作为 reference valid mask。
        """
        logger.info("执行数据质量检查...")
        quality_results: Dict[str, dict] = {}

        nodata_values = scene_data["nodata_values"]
        transforms = scene_data["transforms"]
        crs = scene_data["crs"]
        resolution = scene_data["resolution"]
        band_names = scene_data["band_names"]
        
        # 获取 reference (original) 作为 valid mask 的基准
        reference_arrays = normalized_dict.get("original", None)

        for method, arrays in normalized_dict.items():
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue

            method_quality = {"status": "pass", "issues": []}
            n_scenes = len(arrays)

            for i, arr in enumerate(arrays):
                nd = nodata_values[i] if i < len(nodata_values) else None
                bands_shape = arr.shape
                n_bands_out = bands_shape[0]

                # 使用 reference 构建 valid mask
                if reference_arrays and i < len(reference_arrays):
                    ref = reference_arrays[i]
                    ref_valid = np.isfinite(ref).all(axis=0)
                    if nd is not None:
                        ref_valid &= ~np.any(ref == nd, axis=0)
                else:
                    # 无 reference 时使用 arr 自身
                    if nd is not None:
                        ref_valid = np.isfinite(arr).all(axis=0) & ~np.any(arr == nd, axis=0)
                    else:
                        ref_valid = np.isfinite(arr).all(axis=0)

                # 只统计 reference valid area 中的 NaN/Inf
                nan_in_valid = int(np.sum(np.isnan(arr) & ref_valid[np.newaxis, :, :]))
                inf_in_valid = int(np.sum(np.isinf(arr) & ref_valid[np.newaxis, :, :]))
                total_nan = int(np.sum(np.isnan(arr)))
                total_inf = int(np.sum(np.isinf(arr)))
                valid_pixels = int(ref_valid.sum())

                # 只有 reference valid area 中出现 NaN/Inf 才判 fail
                if nan_in_valid > 0 or inf_in_valid > 0:
                    method_quality["status"] = "fail"
                    method_quality["issues"].append(
                        f"scene_{i}: NaN={nan_in_valid}, Inf={inf_in_valid} in valid area"
                    )

                method_quality[f"scene_{i}"] = {
                    "shape": list(bands_shape),
                    "n_bands": n_bands_out,
                    "nan_in_valid": nan_in_valid,
                    "inf_in_valid": inf_in_valid,
                    "total_nan": total_nan,
                    "total_inf": total_inf,
                    "valid_pixels": valid_pixels,
                }

            # Check transform and CRS consistency (use first scene)
            if transforms:
                method_quality["transform"] = str(transforms[0])
            method_quality["crs"] = crs
            method_quality["resolution"] = resolution
            method_quality["band_names"] = band_names

            quality_results[method] = method_quality
            if method_quality["status"] == "fail":
                logger.warning("  %s: 数据质量检查失败: %s", method, method_quality["issues"])
            else:
                logger.info("  %s: 数据质量检查通过", method)

        return quality_results

    # -----------------------------------------------------------------------
    # g. 完整管线
    # -----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        执行完整的多景多波段处理管线。

        返回
        -------
        dict
            {
                'config': ExperimentConfig,
                'scene_data': dict,
                'overlaps': list,
                'registration': dict,
                'normalized': dict,
                'mosaics': dict,
                'metrics': dict,
                'elapsed_total': float,
                'pipeline_status': str,
                'registration_connected': bool,
                'n_scenes_requested': int,
                'n_scenes_processed': int,
                'unreachable_scenes': list,
                'failed_methods': list,
                'skipped_outputs': list,
            }
        """
        total_t0 = time.time()
        n_scenes_requested = len(self.config.scenes)
        logger.info("=" * 60)
        logger.info("管线启动: %s (请求 %d 景)", self.config.experiment_name, n_scenes_requested)
        logger.info("=" * 60)

        failed_methods: List[str] = []
        skipped_outputs: List[str] = []

        # 0. dry-run 提前返回（跳过文件校验）
        if self.dry_run:
            logger.info("[DRY-RUN] 以下为将要执行的步骤:")
            logger.info("  1. 加载 %d 景影像", n_scenes_requested)
            logger.info("  2. 检测重叠区域")
            logger.info("  3. 配准 (波段: %s)", self.config.registration_band)
            logger.info("  4. 辐射归一化: %s", self.config.ablation_methods)
            logger.info("  5. 生成镶嵌图")
            logger.info("  6. 计算评价指标")
            return {"dry_run": True, "config": self.config}

        # 0. 配置校验
        errors = validate_config(self.config)
        if errors:
            for e in errors:
                logger.error("配置校验失败: %s", e)
            raise ValueError(f"配置校验失败: {errors}")

        band_errors = validate_band_consistency(
            self.config.scenes,
            required_bands=set(self.config.selected_bands)
        )
        if band_errors:
            for e in band_errors:
                logger.error("波段一致性校验失败: %s", e)
            raise ValueError(f"波段一致性校验失败: {band_errors}")

        # 1. 加载场景（smoke 模式下先用全局中心裁剪）
        scene_data = self.load_scenes()

        # 2. 检测重叠
        overlaps = self.detect_overlaps(scene_data)
        if not overlaps:
            logger.warning("未检测到重叠区域，管线提前结束")
            return {
                "scene_data": scene_data,
                "overlaps": [],
                "registration": None,
                "normalized": None,
                "mosaics": None,
                "metrics": None,
                "elapsed_total": time.time() - total_t0,
                "pipeline_status": "failed",
                "registration_connected": False,
                "n_scenes_requested": n_scenes_requested,
                "n_scenes_processed": 0,
                "unreachable_scenes": [],
                "failed_methods": [],
                "skipped_outputs": ["no_overlaps"],
            }

        # 2.5 smoke 模式：从全幅重叠检测构建初始生成树，然后按树边重叠中心裁剪
        # 策略：先在全幅影像上完成配准（确保连通性），再裁剪到目标尺寸并复用全局位移
        if self.smoke:
            from src.multiband_pipeline import build_spanning_tree
            from src.coregistration import warp_multiband_with_displacement_field

            initial_tree = build_spanning_tree(overlaps, len(scene_data["arrays"]), self.control_idx)
            if initial_tree:
                # 先在全幅上配准，验证连通性并获取全局位移
                full_registration = self.register_scenes(scene_data, overlaps)
                full_global_shifts = full_registration["global_shifts"]
                full_local_dx_fields = full_registration.get("local_dx_fields", [])
                full_local_dy_fields = full_registration.get("local_dy_fields", [])
                full_transforms = list(scene_data["transforms"])

                # 裁剪到目标尺寸
                scene_data = self._smoke_recrop_by_spanning_tree(scene_data, initial_tree)
                overlaps = self.detect_overlaps(scene_data)
                logger.info("冒烟模式: 裁剪到 %dpx 后检测到 %d 对重叠",
                            self.smoke_crop_size, len(overlaps))

                # 对裁剪后的影像直接应用全幅配准的全局位移（不再重新配准）
                registered_arrays = []
                cropped_local_dx_fields = []
                cropped_local_dy_fields = []
                for idx in range(len(scene_data["arrays"])):
                    gdx = full_global_shifts[idx, 0]
                    gdy = full_global_shifts[idx, 1]
                    h, w = scene_data["arrays"][idx].shape[1:]
                    local_dx = np.zeros((h, w), dtype=np.float64)
                    local_dy = np.zeros((h, w), dtype=np.float64)
                    if idx < len(full_local_dx_fields) and idx < len(full_local_dy_fields):
                        old_tr = full_transforms[idx]
                        new_tr = scene_data["transforms"][idx]
                        col_offset = int(round((new_tr.c - old_tr.c) / old_tr.a))
                        row_offset = int(round((old_tr.f - new_tr.f) / abs(old_tr.e)))
                        source_dx = full_local_dx_fields[idx]
                        source_dy = full_local_dy_fields[idx]
                        if (row_offset >= 0 and col_offset >= 0
                                and row_offset + h <= source_dx.shape[0]
                                and col_offset + w <= source_dx.shape[1]):
                            local_dx = source_dx[row_offset:row_offset + h,
                                                 col_offset:col_offset + w]
                            local_dy = source_dy[row_offset:row_offset + h,
                                                 col_offset:col_offset + w]
                    cropped_local_dx_fields.append(local_dx)
                    cropped_local_dy_fields.append(local_dy)
                    if (abs(gdx) < 1e-6 and abs(gdy) < 1e-6
                            and not np.any(local_dx) and not np.any(local_dy)):
                        registered_arrays.append(scene_data["arrays"][idx].astype(np.float64))
                    else:
                        nd_val = scene_data["nodata_values"][idx]  # Keep None as None
                        warped = warp_multiband_with_displacement_field(
                            scene_data["arrays"][idx], gdx, gdy,
                            local_dx, local_dy, nd_val,
                        )
                        registered_arrays.append(warped)
                    logger.info("  [%d] 全局位移: dx=%.4f, dy=%.4f (复用全幅配准)", idx, gdx, gdy)

                validation_measurements, crop_offsets = (
                    _translate_training_measurements_to_crop(
                        full_registration["pair_matches"],
                        full_transforms,
                        scene_data["transforms"],
                    )
                )
                cropped_quality, cropped_final_validation = _validate_final_registration_arrays(
                    registered_arrays,
                    self.registration_band_idx,
                    scene_data["transforms"],
                    scene_data["nodata_values"],
                    full_registration["spanning_tree"],
                    validation_measurements,
                    self.config.registration_params,
                )
                smoke_diagnostics = dict(full_registration.get("diagnostics", {}))
                smoke_diagnostics["quality"] = cropped_quality
                smoke_diagnostics["final_validation"] = cropped_final_validation
                smoke_diagnostics["smoke_recropped_validation"] = True
                smoke_diagnostics["smoke_training_coordinate_offsets"] = [
                    {"col": col, "row": row} for col, row in crop_offsets
                ]
                status, failure = _registration_status_and_failure(
                    bool(full_registration["connected"]),
                    cropped_quality,
                    self.config.registration_params.get("required_quality", "pass"),
                )
                if full_registration.get("status") == "fail":
                    status = "fail"
                    failure = full_registration.get("failure", failure)

                # 构建 registration dict，复用全幅配准的连通性信息
                registration = {
                    "registered_arrays": registered_arrays,
                    "global_shifts": full_global_shifts,
                    "local_dx_fields": cropped_local_dx_fields,
                    "local_dy_fields": cropped_local_dy_fields,
                    "local_refinement": full_registration.get("local_refinement", {}),
                    "pair_matches": full_registration["pair_matches"],
                    "connected": full_registration["connected"],
                    "status": status,
                    "failure": failure,
                    "spanning_tree": full_registration["spanning_tree"],
                    "geometric_edges": full_registration["geometric_edges"],
                    "matching_edges": full_registration["matching_edges"],
                    "rejected_edges": full_registration["rejected_edges"],
                    "connected_components": full_registration["connected_components"],
                    "unreachable_scenes": full_registration["unreachable_scenes"],
                    "quality": cropped_quality,
                    "final_validation": cropped_final_validation,
                    "diagnostics": smoke_diagnostics,
                }
            else:
                registration = self.register_scenes(scene_data, overlaps)
        else:
            # 3. 配准（含连通性检查、匹配回退、详细日志）
            registration = self.register_scenes(scene_data, overlaps)

        # 将 transforms, bounds, nodata 传递到后续步骤
        registration["transforms"] = scene_data["transforms"]
        registration["bounds"] = scene_data["bounds"]
        registration["nodata_values"] = scene_data["nodata_values"]
        registration["crs"] = scene_data["crs"]

        n_scenes_processed = len(scene_data["arrays"])
        unreachable_scenes = registration.get("unreachable_scenes", [])
        registration_connected = bool(
            registration.get("connected", len(unreachable_scenes) == 0)
        )

        required_quality = getattr(
            self.config, "registration_params", {}
        ).get("required_quality", "pass")
        registration_status = str(registration.get("status", "fail")).lower()
        final_quality = registration.get("quality", {}).get("quality", "fail")
        registration_quality_ok = (
            registration_status == "pass"
            and registration_connected
            and registration_quality_meets_requirement(final_quality, required_quality)
        )
        if not registration_quality_ok:
            skipped_outputs.append("registration_quality_gate")
            logger.error(
                "注册质量门控失败: status=%s, connected=%s, final_quality=%s, required_quality=%s",
                registration_status, registration_connected, final_quality, required_quality,
            )
            return {
                "config": self.config,
                "scene_data": scene_data,
                "overlaps": overlaps,
                "registration": registration,
                "normalized": None,
                "mosaics": None,
                "metrics": None,
                "spectral": {},
                "quality": {},
                "elapsed_total": time.time() - total_t0,
                "pipeline_status": "failed",
                "registration_connected": registration_connected,
                "n_scenes_requested": n_scenes_requested,
                "n_scenes_processed": n_scenes_processed,
                "unreachable_scenes": unreachable_scenes,
                "failed_methods": [],
                "skipped_outputs": skipped_outputs,
            }

        # 4. 辐射归一化
        normalized = self.apply_radiometric_normalization(
            registration, overlaps,
        )

        # 检查哪些方法失败了
        method_arrays = {k: v for k, v in normalized.items()
                         if isinstance(v, list) and v and isinstance(v[0], np.ndarray)}
        for method_name in self.config.ablation_methods:
            if method_name not in method_arrays:
                failed_methods.append(method_name)

        output_dir = os.path.join(self.output_root, self.config.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save per-scene normalized GeoTIFFs
        for method, arrays_list in normalized.items():
            if not isinstance(arrays_list, list):
                continue
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            for i, arr in enumerate(arrays_list):
                if not isinstance(arr, np.ndarray):
                    continue
                if i >= len(scene_data["scene_ids"]):
                    skipped_outputs.append(f"{method}_scene_{i}_out_of_range")
                    continue
                scene_id = scene_data["scene_ids"][i]
                out_path = os.path.join(method_dir, f"{scene_id}_{method}.tif")
                if not self.dry_run:
                    write_geotiff(
                        out_path, arr,
                        scene_data["transforms"][i],
                        scene_data["crs"],
                        nodata=scene_data["nodata_values"][i],
                    )

        # 5. 生成镶嵌
        mosaics = self.compute_mosaics(
            method_arrays,
            scene_data["transforms"],
            scene_data["crs"],
            scene_data["nodata_values"],
            output_dir,
        )

        # 6. 计算指标
        metrics_dir = os.path.join(output_dir, "metrics")
        metrics = self.evaluate_metrics(
            normalized,
            overlaps,
            scene_data["nodata_values"],
            output_dir=metrics_dir,
        )

        # 7. 计算光谱保持指标（SAM, spectral RMSE, correlation）
        spectral = {}
        if self.config.enable_spectral_metrics:
            spectral_dir = os.path.join(output_dir, "spectral")
            original_arrays = normalized.get("original", None)
            if original_arrays is not None:
                spectral = self.evaluate_spectral(
                    normalized,
                    original_arrays,
                    scene_data["nodata_values"],
                    scene_data["band_names"],
                    overlaps,
                    output_dir=spectral_dir,
                )
        else:
            logger.info("光谱指标已禁用 (enable_spectral_metrics=False)")

        # 8. 数据质量检查
        quality = self.check_data_quality(normalized, scene_data)

        total_elapsed = time.time() - total_t0

        # 确定 pipeline_status - 使用请求的方法列表
        requested_methods = self._requested_normalization_methods()
        failed_methods = [
            m for m in requested_methods
            if m not in method_arrays
        ]
        quality_failed_methods = [
            m for m in requested_methods
            if quality.get(m, {}).get("status") == "fail"
        ]
        
        if registration_connected and not failed_methods and not quality_failed_methods:
            pipeline_status = "success"
        else:
            pipeline_status = "failed"
            if failed_methods:
                logger.warning("失败的方法: %s", failed_methods)
            if quality_failed_methods:
                logger.warning("质量检查失败的方法: %s", quality_failed_methods)

        logger.info("=" * 60)
        logger.info("管线完成: 总耗时 %.1fs, 状态=%s", total_elapsed, pipeline_status)
        logger.info("=" * 60)

        return {
            "config": self.config,
            "scene_data": scene_data,
            "overlaps": overlaps,
            "registration": registration,
            "normalized": normalized,
            "mosaics": mosaics,
            "metrics": metrics,
            "spectral": spectral,
            "quality": quality,
            "elapsed_total": total_elapsed,
            "pipeline_status": pipeline_status,
            "registration_connected": registration_connected,
            "n_scenes_requested": n_scenes_requested,
            "n_scenes_processed": n_scenes_processed,
            "unreachable_scenes": unreachable_scenes,
            "failed_methods": failed_methods,
            "skipped_outputs": skipped_outputs,
        }


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    """命令行入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="BAGRN-VOLRN 多景多波段辐射归一化管线"
    )
    parser.add_argument(
        "config", type=str,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印执行计划，不实际处理",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="冒烟测试模式（使用小区域裁剪）",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="输出根目录（覆盖配置文件中的值）",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="指定辐射归一化方法列表",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载配置
    from src.experiment_config import load_config
    config = load_config(args.config)

    # 命令行覆盖
    if args.dry_run:
        config.dry_run = True
    if args.smoke:
        config.smoke = True
    if args.output_root:
        config.output_root = args.output_root

    # 执行管线
    pipe = MultibandPipeline(config)
    results = pipe.run()

    # 输出摘要
    if not args.dry_run and results.get("metrics"):
        print("\n" + "=" * 60)
        print("指标摘要:")
        print("=" * 60)
        for method, m in results["metrics"].items():
            print(f"  {method}:")
            print(f"    ADM={m.get('adm', 0):.6f}  ADSD={m.get('adsd', 0):.6f}")
            print(f"    CD={m.get('cd', 0):.6f}  GL={m.get('gl', 0):.6f}")
            print(f"    RDOA={m.get('rdoa', 0):.6f}  Ave={m.get('ave', 0):.6f}")


if __name__ == "__main__":
    main()
