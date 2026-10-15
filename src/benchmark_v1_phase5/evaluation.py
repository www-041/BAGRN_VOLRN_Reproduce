"""Independent DEVELOPMENT scoring; no fitting and no TEST truth access."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from benchmark_phase2_core import pair_metrics

from .coordinates import georef_only_translation
from .inputs import PairInput


METHODS = ("SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR",
           "EfficientLoFTR", "LightGlue+DISK", "GEOREF_ONLY")
EXECUTION_STATUSES = {
    "SUCCESS", "ENV_UNAVAILABLE", "WEIGHTS_UNAVAILABLE", "NO_VALID_OVERLAP",
    "UNSUPPORTED_GRID", "PREPROCESSING_FAIL", "ALGORITHM_FAIL", "NOT_ATTEMPTED",
}
TRIAL_PAIRS = ("00_01", "02_03", "05_07", "08_10")


def load_approved_development_truth(snapshot) -> dict[str, list[dict]]:
    """Adapt Phase4's validating loader while preserving its approval contract.

    Phase4 checks the frozen source candidate has APPROVED first review before
    returning a point, but intentionally strips that source field. This narrow
    adapter restores the explicit status only after that validation succeeds.
    """
    from benchmark_phase4_development import load_development_truth

    verified = load_development_truth(snapshot)
    return {pair: [{**point, "human_review_status": "APPROVED"} for point in rows]
            for pair, rows in verified.items()}


def _validate_truth(pair_id: str, dev_points: Iterable[Mapping]) -> list[Mapping]:
    points = list(dev_points)
    seen: set[str] = set()
    for point in points:
        if point.get("split") != "DEVELOPMENT":
            raise ValueError("Only DEVELOPMENT points may enter the DEVELOPMENT scorer")
        if point.get("human_review_status") != "APPROVED":
            raise ValueError("DEVELOPMENT truth requires human_review_status=APPROVED")
        if point.get("pair_id") != pair_id:
            raise ValueError("DEVELOPMENT truth pair does not match result pair")
        point_id = str(point.get("point_id", ""))
        if not point_id or point_id in seen:
            raise ValueError("DEVELOPMENT points require unique nonempty point_id values")
        seen.add(point_id)
        for name in ("row_i", "col_i", "row_j", "col_j"):
            try:
                value = float(point[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid DEVELOPMENT coordinate {name}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite DEVELOPMENT coordinate {name}")
    return points


def score_development(result: Mapping, dev_points: Iterable[Mapping]) -> dict:
    """Score one native i->j translation against APPROVED DEVELOPMENT truth.

    Inputs contain only the already-persisted method result and the caller's
    DEVELOPMENT-only truth view. No transform fitting occurs here.
    """
    method, pair_id = result.get("method"), result.get("pair_id")
    status = result.get("status")
    if method not in METHODS or pair_id not in TRIAL_PAIRS:
        raise ValueError("Unexpected Phase5 method or trial pair")
    if status not in EXECUTION_STATUSES:
        raise ValueError("Unknown algorithm execution status")
    points = _validate_truth(pair_id, dev_points)
    base = {"method": method, "pair_id": pair_id, "execution_status": status,
            "evaluation_status": "NOT_SCORED", "metrics": None,
            "independent_point_count": 0, "diagnostic_only": method == "GEOREF_ONLY"}
    if status != "SUCCESS":
        if result.get("delta_row_col") is not None:
            raise ValueError("Non-success execution must not carry a translation")
        return base

    if result.get("direction") != "scene_i_to_scene_j":
        raise ValueError("Scoring requires scene_i_to_scene_j direction")
    if result.get("frame") != "native":
        raise ValueError("Scoring requires native pixel coordinates")
    if result.get("translation_semantics") != "full_native_i_to_j":
        raise ValueError("Scoring requires a full native i-to-j translation")
    delta = result.get("delta_row_col")
    if not isinstance(delta, (tuple, list, np.ndarray)) or len(delta) != 2:
        raise ValueError("SUCCESS requires a two-component native translation")
    try:
        dr, dc = map(float, delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("Translation must contain finite values") from exc
    if not np.isfinite([dr, dc]).all():
        raise ValueError("Translation must contain finite values")
    if not points:
        return {**base, "evaluation_status": "NO_INDEPENDENT_TRUTH"}

    errors = [math.hypot(float(point["row_i"]) + dr - float(point["row_j"]),
                         float(point["col_i"]) + dc - float(point["col_j"]))
              for point in points]
    return {**base, "evaluation_status": "SCORED",
            "metrics": pair_metrics(errors), "independent_point_count": len(points)}


def score_georef_only(pair: PairInput, dev_points: Iterable[Mapping]) -> dict:
    """Diagnostic no-additional-registration mapping scored by the same routine."""
    if pair.pair_id not in TRIAL_PAIRS:
        raise ValueError("GEOREF_ONLY pair is outside the frozen trial")
    if pair.status != "READY":
        result = {"method": "GEOREF_ONLY", "pair_id": pair.pair_id,
                  "status": pair.status, "delta_row_col": None}
    else:
        try:
            delta = georef_only_translation(pair.native_affine_i, pair.native_affine_j)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            result = {"method": "GEOREF_ONLY", "pair_id": pair.pair_id,
                      "status": "UNSUPPORTED_GRID", "delta_row_col": None}
        else:
            result = {"method": "GEOREF_ONLY", "pair_id": pair.pair_id,
                      "status": "SUCCESS", "delta_row_col": delta,
                      "direction": "scene_i_to_scene_j", "frame": "native",
                      "translation_semantics": "full_native_i_to_j"}
    return score_development(result, dev_points)


def summarize_development(scored_rows: Iterable[Mapping],
                          expected_pairs: Iterable[str] = TRIAL_PAIRS) -> dict:
    """Build per-method status denominators and full pair-level RMSE distribution."""
    pairs = tuple(expected_pairs)
    if not pairs or len(set(pairs)) != len(pairs) or any(p not in TRIAL_PAIRS for p in pairs):
        raise ValueError("Expected pairs must be a nonempty unique subset of frozen trial pairs")
    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    for row in scored_rows:
        method, pair_id = row.get("method"), row.get("pair_id")
        if method not in METHODS or pair_id not in pairs:
            raise ValueError("Unexpected scored result row")
        if pair_id in grouped[method]:
            raise ValueError("Duplicate method/pair result")
        if row.get("execution_status") not in EXECUTION_STATUSES:
            raise ValueError("Unknown execution status in summary")
        grouped[method][pair_id] = row

    methods = {}
    diagnostics = {}
    for method in METHODS:
        rows = grouped.get(method, {})
        complete = [rows.get(pair) for pair in pairs]
        for row in complete:
            if row is None:
                continue
            evaluation_status = row.get("evaluation_status")
            if evaluation_status not in {"SCORED", "NO_INDEPENDENT_TRUTH", "NOT_SCORED"}:
                raise ValueError("Unknown evaluation status in summary")
            if row["execution_status"] != "SUCCESS" and (
                    evaluation_status != "NOT_SCORED" or row.get("metrics") is not None):
                raise ValueError("Failed execution cannot carry evaluation metrics")
            if evaluation_status == "SCORED" and (
                    row.get("metrics") is None or row["execution_status"] != "SUCCESS"):
                raise ValueError("SCORED requires a successful execution and metrics")
            if evaluation_status != "SCORED" and row.get("metrics") is not None:
                raise ValueError("Unscored row cannot carry metrics")
        statuses = Counter((row["execution_status"] if row else "NOT_ATTEMPTED")
                           for row in complete)
        scored = [row for row in complete if row and row["evaluation_status"] == "SCORED"]
        values = [float(row["metrics"]["rmse_px"]) for row in scored]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Summary received invalid pair RMSE")
        summary = {
            "eligible_pairs": len(pairs),
            "attempted_pairs": sum(count for key, count in statuses.items()
                                    if key != "NOT_ATTEMPTED"),
            "execution_status_counts": dict(sorted(statuses.items())),
            "evaluation_status_counts": dict(sorted(Counter(
                row["evaluation_status"] if row else "NOT_SCORED"
                for row in complete).items())),
            "scored_pair_count": len(scored),
            "diagnostic_only": method == "GEOREF_ONLY",
            "pair_rmse_distribution": values,
        }
        if method == "GEOREF_ONLY":
            diagnostics[method] = summary
        else:
            summary["macro_rmse_success_only"] = float(np.mean(values)) if values else None
            methods[method] = summary
    return {"expected_pairs": list(pairs), "methods": methods,
            "diagnostic_baselines": diagnostics,
            "scope": "DEVELOPMENT_ONLY_NOT_SPATIALLY_HELD_OUT_GENERALIZATION"}
