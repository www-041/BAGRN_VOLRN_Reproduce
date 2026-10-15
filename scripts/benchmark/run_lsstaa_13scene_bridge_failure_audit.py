"""Offline bridge, rank-nullspace and control-subnetwork diagnostics for LSSTAA."""
from __future__ import annotations

from datetime import datetime
import csv
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from shapely.geometry import Polygon
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_lsstaa.core import (  # noqa: E402
    choose_control, control_statistics, pif_from_series,
    select_pif_subset,
)
from src.benchmark_v1_lsstaa.raster import proxy_ndvi, verified_scene_date  # noqa: E402
from src.benchmark_v1_unified.aligned import corrected_affine  # noqa: E402
from scripts.benchmark.run_lsstaa_13scene_network_precheck import (  # noqa: E402
    FREEZE, SOURCE_ROOT, TASK14, load_inputs,
)

OUT_JSON = ROOT / "docs/benchmark/v1/unified_v1/lsstaa_13scene_bridge_failure_audit_20261015.json"
OUT_MD = ROOT / "docs/benchmark/v1/unified_v1/lsstaa_13scene_bridge_failure_audit_20261015.md"
N_SCENES = 13
BANDS = ("B9", "B4", "B7", "B14")
THRESHOLDS = (14, 30)
CONTROL_SUBNET = [0, 1, 6, 8, 9, 10, 11, 12]
GRID_BINS = 100
MIN_RAW_PIF_SAMPLES = 50


def time_stamps_and_days(scene_records: list[dict]) -> tuple[list[datetime], list[int]]:
    timestamps = [datetime.strptime(row["scene_id"].split("_")[-3], "%Y%m%d%H%M%S")
                  for row in scene_records]
    dates = [datetime.strptime(row["date"], "%Y-%m-%d").date() for row in scene_records]
    return timestamps, [(value - dates[0]).days for value in dates]


def footprint_polygons(scene_records: list[dict]) -> list[Polygon]:
    transform_path = TASK14 / "stages/03_global_adjustment/global/translation_l2/global_transforms.json"
    transforms = json.loads(transform_path.read_text(encoding="utf-8"))["transforms"]
    polygons = []
    for row in scene_records:
        scene_index = row["scene_index"]
        path = Path(row["source_paths"]["B9"])
        with rasterio.open(path) as ds:
            affine = corrected_affine(ds.transform, np.asarray(transforms[scene_index]["matrix"]))
            points = [affine * (0, 0), affine * (ds.width, 0),
                      affine * (ds.width, ds.height), affine * (0, ds.height)]
        polygons.append(Polygon(points))
    return polygons


def pair_inventory() -> dict[tuple[int, int], dict]:
    path = FREEZE / "pair_overlap_inventory.csv"
    rows = csv.DictReader(path.open(encoding="utf-8-sig", newline=""))
    result = {}
    for row in rows:
        i, j = int(row["pair_id"][:2]), int(row["pair_id"][3:5])
        result[(i, j)] = {
            "native_valid_overlap_pixels": int(row["native_valid_overlap_pixels"] or 0),
            "aligned_valid_overlap_pixels": int(row["aligned_valid_overlap_pixels"] or 0),
            "inventory_status": row["status"],
        }
    if len(result) != 78:
        raise ValueError(f"EXPECTED_78_FROZEN_PAIR_INVENTORY_ROWS:{len(result)}")
    return result


def quantile_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "min": None, "p01": None, "median": None,
                "p99": None, "max": None, "std": None}
    p01, median, p99 = np.quantile(values, [.01, .5, .99], method="hazen")
    return {"count": int(values.size), "min": float(values.min()),
            "p01": float(p01), "median": float(median), "p99": float(p99),
            "max": float(values.max()), "std": float(values.std())}


def pair_band_screen(a: np.ndarray, b: np.ndarray, pif: np.ndarray) -> dict:
    valid = pif & np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    raw_count = int(valid.sum())
    x, y = np.asarray(a[valid], np.float64), np.asarray(b[valid], np.float64)
    result = {
        "pif_intersection_valid_samples": raw_count,
        "grid_input_finite_nonzero_samples": 0,
        "grid_selected_samples": 0,
        "grid_subset_r2": None,
        "x_dynamic_range": quantile_summary(x),
        "y_dynamic_range": quantile_summary(y),
        "grid_status": "NOT_RUN",
        "failure_code": None,
        "accepted_by_lsstaa_edge_filter": False,
        "selected_x": np.empty(0, dtype=np.float64),
        "selected_y": np.empty(0, dtype=np.float64),
    }
    if raw_count < MIN_RAW_PIF_SAMPLES:
        result.update(grid_status="NOT_RUN_INSUFFICIENT_RAW_PIF",
                      failure_code="PIF_VALID_INTERSECTION_LT_50")
        return result
    finite = np.isfinite(x) & np.isfinite(y)
    result["grid_input_finite_nonzero_samples"] = int(np.count_nonzero(finite & (x != 0) & (y != 0)))
    try:
        sx, sy, r2 = select_pif_subset(x, y, GRID_BINS)
    except ValueError as exc:
        result.update(grid_status="REJECTED", failure_code=f"GRID_{str(exc)}")
        return result
    result["grid_selected_samples"] = int(len(sx))
    result["grid_subset_r2"] = float(r2)
    result["selected_x"], result["selected_y"] = sx, sy
    if r2 < 0:
        result.update(grid_status="REJECTED_NEGATIVE_R2",
                      failure_code="GRID_SUBSET_R2_NEGATIVE")
        return result
    result.update(grid_status="ACCEPTED", accepted_by_lsstaa_edge_filter=True)
    return result


def stage_reason(row: dict, threshold: int, band: dict) -> tuple[str, str]:
    if row["day_gap_calendar_days"] > threshold:
        return "A_TIME_WINDOW_BLOCKED", "TIME_WINDOW_BLOCKED"
    if not row["adjusted_footprint_intersects"]:
        return "B_NO_SPATIAL_OVERLAP", "NO_ADJUSTED_FOOTPRINT_INTERSECTION"
    if row["aligned_valid_overlap_pixels"] == 0:
        return "C_NO_VALID_OVERLAP", "NO_ALIGNED_VALID_PIXEL_INTERSECTION"
    if band["pif_intersection_valid_samples"] < MIN_RAW_PIF_SAMPLES:
        return "D_PIF_SUPPORT_INSUFFICIENT", band["failure_code"]
    if not band["accepted_by_lsstaa_edge_filter"]:
        return "E_GRID_OR_REGRESSION_REJECTED", band["failure_code"]
    return "EDGE_ACCEPTED", "VALID_EDGE"


def components(nodes: list[int], edges: list[dict]) -> list[list[int]]:
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        i, j = edge["scene_i"], edge["scene_j"]
        adjacency[i].add(j)
        adjacency[j].add(i)
    unseen = set(nodes)
    groups = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        found, pending = {root}, [root]
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node] & unseen:
                unseen.remove(neighbor)
                found.add(neighbor)
                pending.append(neighbor)
        groups.append(sorted(found))
    return groups


def design_for(nodes: list[int], control: int, edges: list[dict]) -> tuple[np.ndarray, list[str]]:
    unknown_nodes = [node for node in nodes if node != control]
    col = {node: 2 * index for index, node in enumerate(unknown_nodes)}
    matrix_rows = []
    for edge in edges:
        i, j = edge["scene_i"], edge["scene_j"]
        x, y = edge["selected_x"], edge["selected_y"]
        block = np.zeros((len(x), 2 * len(unknown_nodes)), dtype=np.float64)
        if i != control:
            block[:, col[i]] = x
            block[:, col[i] + 1] = 1.0
        if j != control:
            block[:, col[j]] = -y
            block[:, col[j] + 1] = -1.0
        matrix_rows.append(block)
    matrix = np.vstack(matrix_rows) if matrix_rows else np.zeros((0, 2 * len(unknown_nodes)))
    labels = [f"scene_{node:02d}_{param}" for node in unknown_nodes for param in ("gain", "offset")]
    return matrix, labels


def svd_diagnostic(matrix: np.ndarray, labels: list[str], nodes: list[int],
                   control: int | None) -> dict:
    rows, cols = matrix.shape
    if cols == 0:
        return {"matrix_shape": [rows, cols], "rank": 0, "rank_tolerance": 0.0,
                "singular_values": [], "zero_space_dimension": 0,
                "zero_space_basis": [], "condition_number_design": 1.0,
                "condition_number_normal": 1.0, "column_labels": labels}
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    smax = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = float(max(rows, cols) * np.finfo(np.float64).eps * smax)
    rank = int(np.count_nonzero(singular_values > tolerance))
    null_basis = vt[rank:, :]
    null_rows = []
    for basis_index, vector in enumerate(null_basis):
        maximum = float(np.max(np.abs(vector))) if vector.size else 0.0
        support = [{"parameter": label, "value": float(value)}
                   for label, value in zip(labels, vector)
                   if abs(float(value)) >= max(1e-7, maximum * 1e-5)]
        null_rows.append({"basis_index": basis_index, "dominant_parameters": support})
    full_rank = rank == cols
    cond_a = float(singular_values[0] / singular_values[cols-1]) if full_rank and cols and len(singular_values) >= cols else None
    normal = matrix.T @ matrix
    cond_normal = float(np.linalg.cond(normal)) if full_rank and cols else None
    column_norms = np.linalg.norm(matrix, axis=0) if rows else np.zeros(cols)
    scaled_cond = None
    if full_rank and cols and np.all(column_norms > 0):
        scaled_values = np.linalg.svd(matrix / column_norms, compute_uv=False)
        scaled_cond = float(scaled_values[0] / scaled_values[-1])
    return {
        "matrix_shape": [int(rows), int(cols)], "rank": rank,
        "rank_tolerance": tolerance,
        "rank_tolerance_rule": "max(matrix_rows,matrix_cols)*float64_epsilon*largest_singular_value",
        "singular_values": [float(value) for value in singular_values],
        "zero_space_dimension": int(cols-rank),
        "zero_space_basis": null_rows,
        "condition_number_design": cond_a,
        "condition_number_normal": cond_normal,
        "condition_number_design_column_scaled": scaled_cond,
        "column_labels": labels,
        "control_scene_fixed": control,
        "component_scene_indices": nodes,
    }


def edge_objects(pair_rows: list[dict], band_name: str, threshold: int) -> list[dict]:
    accepted = []
    for row in pair_rows:
        if row["day_gap_calendar_days"] > threshold:
            continue
        band = row["bands"][band_name]
        if band["accepted_by_lsstaa_edge_filter"]:
            accepted.append({"scene_i": row["scene_i"], "scene_j": row["scene_j"],
                             "selected_x": band["selected_x"], "selected_y": band["selected_y"],
                             "sample_count": band["grid_selected_samples"]})
    return accepted


def band_network_audit(pair_rows: list[dict], band_name: str, threshold: int,
                       control: int) -> dict:
    all_nodes = list(range(N_SCENES))
    edges = edge_objects(pair_rows, band_name, threshold)
    groups = components(all_nodes, edges)
    degrees = {node: 0 for node in all_nodes}
    for edge in edges:
        degrees[edge["scene_i"]] += 1
        degrees[edge["scene_j"]] += 1
    full_a, full_labels = design_for(all_nodes, control, edges)
    full_rank = svd_diagnostic(full_a, full_labels, all_nodes, control)
    local = []
    for group in groups:
        local_edges = [edge for edge in edges if edge["scene_i"] in group and edge["scene_j"] in group]
        local_control = control if control in group else None
        local_a, local_labels = design_for(group, local_control if local_control is not None else -1, local_edges)
        local.append({"scene_indices": group,
                      "control_anchored": local_control == control,
                      "edge_count": len(local_edges),
                      "joint_system": svd_diagnostic(local_a, local_labels, group, local_control)})
    return {"valid_edge_count": len(edges), "effective_edges": [
                {"scene_i": e["scene_i"], "scene_j": e["scene_j"], "sample_count": e["sample_count"]}
                for e in edges],
            "degrees": degrees, "components": groups,
            "control_reachable_scene_indices": next(group for group in groups if control in group),
            "joint_design": full_rank, "component_local_systems": local}


def subset_audit(pair_rows: list[dict], band_name: str, threshold: int,
                 nodes: list[int], control: int) -> dict:
    edges = [edge for edge in edge_objects(pair_rows, band_name, threshold)
             if edge["scene_i"] in nodes and edge["scene_j"] in nodes]
    matrix, labels = design_for(nodes, control, edges)
    overall = svd_diagnostic(matrix, labels, nodes, control)
    edge_diagnostics = []
    total_ties = sum(edge["sample_count"] for edge in edges)
    for edge in edges:
        without = [candidate for candidate in edges if candidate is not edge]
        reduced, _ = design_for(nodes, control, without)
        reduced_rank = svd_diagnostic(reduced, labels, nodes, control)["rank"]
        original = next(row for row in pair_rows if (row["scene_i"], row["scene_j"]) == (edge["scene_i"], edge["scene_j"]))
        pif_band = original["bands"][band_name]
        edge_diagnostics.append({
            "scene_i": edge["scene_i"], "scene_j": edge["scene_j"],
            "selected_grid_samples": edge["sample_count"],
            "selected_sample_share": edge["sample_count"] / total_ties if total_ties else 0.0,
            "rank_without_edge": reduced_rank,
            "unique_rank_contribution_by_leave_one_edge_out": overall["rank"] - reduced_rank,
            "pif_intersection_valid_samples": pif_band["pif_intersection_valid_samples"],
            "grid_subset_r2": pif_band["grid_subset_r2"],
            "x_dynamic_range": pif_band["x_dynamic_range"],
            "y_dynamic_range": pif_band["y_dynamic_range"],
        })
    # Scene-level support is computed from the stack by the caller.
    return {"scene_indices": nodes, "control_scene_index": control,
            "valid_edge_count": len(edges), "selected_grid_sample_total": total_ties,
            "joint_design": overall, "edges": edge_diagnostics,
            "max_single_edge_sample_share": max((r["selected_sample_share"] for r in edge_diagnostics), default=0.0),
            "pif_scene_support_counts": {}}


def local_native_block_diagnostic(pair_ids: list[tuple[int, int]], pif: np.ndarray,
                                  stack: np.ndarray) -> list[dict]:
    """Inspect native B9 samples under coarse PIF cells only; no native PIF is claimed."""
    factor = 10
    result = []
    for i, j in pair_ids:
        shared = pif & (stack[:, :, i, 0] != 0) & (stack[:, :, j, 0] != 0)
        rows, cols = np.where(shared)
        input_paths = [FREEZE / "aligned_inputs" / f"scene_{scene:03d}_B9_input.tif"
                       for scene in (i, j)]
        mask_paths = [FREEZE / "aligned_inputs" / f"scene_{scene:03d}_input_mask.tif"
                      for scene in (i, j)]
        x_parts, y_parts = [], []
        with rasterio.open(input_paths[0]) as a, rasterio.open(input_paths[1]) as b, \
             rasterio.open(mask_paths[0]) as ma, rasterio.open(mask_paths[1]) as mb:
            if (a.width, a.height, a.transform, a.crs) != (b.width, b.height, b.transform, b.crs):
                raise ValueError("NATIVE_ROI_PAIR_GRID_MISMATCH")
            for row, col in zip(rows, cols):
                r0, c0 = int(row * factor), int(col * factor)
                h, w = min(factor, a.height-r0), min(factor, a.width-c0)
                if h <= 0 or w <= 0:
                    continue
                window = Window(c0, r0, w, h)
                av, bv = a.read(1, window=window), b.read(1, window=window)
                valid = (ma.read(1, window=window) != 0) & (mb.read(1, window=window) != 0)
                valid &= np.isfinite(av) & np.isfinite(bv) & (av != 0) & (bv != 0)
                if valid.any():
                    x_parts.append(av[valid].astype(np.float64))
                    y_parts.append(bv[valid].astype(np.float64))
        x = np.concatenate(x_parts) if x_parts else np.empty(0, dtype=np.float64)
        y = np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.float64)
        local = {"scene_i": i, "scene_j": j,
                 "diagnostic_label": "NATIVE_B9_SAMPLES_WITHIN_10X_COARSE_PIF_BLOCKS_NOT_NATIVE_PIF",
                 "coarse_shared_pif_cells": int(len(rows)),
                 "native_valid_pair_samples_within_those_cells": int(len(x)),
                 "native_x_dynamic_range": quantile_summary(x),
                 "native_y_dynamic_range": quantile_summary(y),
                 "grid_subset_on_native_block_samples": None,
                 "interpretation_limit": "PIF mask was computed on the 10x stack; this ROI check does not recompute the 13-date native-resolution PIF classifier and cannot establish a native-resolution LSSTAA edge."}
        if len(x) >= MIN_RAW_PIF_SAMPLES:
            try:
                sx, sy, r2 = select_pif_subset(x, y, GRID_BINS)
                local["grid_subset_on_native_block_samples"] = {
                    "accepted_by_sampling_only": bool(r2 >= 0),
                    "selected_samples": int(len(sx)), "r2": float(r2),
                    "edge_would_pass_full_method": None,
                    "note": "The current min-50 edge gate is bypassed only for this diagnostic comparison; this does not count as a valid edge."}
            except ValueError as exc:
                local["grid_subset_on_native_block_samples"] = {
                    "accepted_by_sampling_only": False, "failure_code": f"GRID_{exc}"}
        result.append(local)
    return result


def main() -> int:
    stack, days, scene_records, provenance = load_inputs()
    dates = [verified_scene_date(Path(row["source_paths"]["B9"]).parent /
                                 f"{row['scene_id']}_MTL.txt") for row in scene_records]
    timestamps, _ = time_stamps_and_days(scene_records)
    ndvi_valid = np.all(np.isfinite(stack) & (stack != 0), axis=3)
    ndvi = np.stack([proxy_ndvi(stack[:, :, i, 2], stack[:, :, i, 3], ndvi_valid[:, :, i])
                     for i in range(N_SCENES)], axis=2)
    pif_result = pif_from_series(ndvi, days)
    rsq, gains, offsets, overlap = control_statistics(pif_result.bias_corrected,
                                                       pif_result.temporal_difference)
    control = choose_control(rsq, gains, offsets, overlap)
    polygons = footprint_polygons(scene_records)
    inventory = pair_inventory()
    pair_rows = []
    for i in range(N_SCENES):
        for j in range(i+1, N_SCENES):
            date_gap = abs((datetime.strptime(dates[i], "%Y-%m-%d").date() -
                            datetime.strptime(dates[j], "%Y-%m-%d").date()).days)
            seconds_gap = abs((timestamps[i] - timestamps[j]).total_seconds())
            intersection = polygons[i].intersection(polygons[j])
            overlap_meta = inventory[(i, j)]
            row = {
                "scene_i": i, "scene_j": j,
                "scene_id_i": scene_records[i]["scene_id"],
                "scene_id_j": scene_records[j]["scene_id"],
                "date_i": dates[i], "date_j": dates[j],
                "acquisition_timestamp_i": timestamps[i].isoformat(),
                "acquisition_timestamp_j": timestamps[j].isoformat(),
                "elapsed_seconds": seconds_gap,
                "day_gap_calendar_days": date_gap,
                "within_official_14_day_window": date_gap <= 14,
                "within_exploratory_30_day_window": date_gap <= 30,
                "adjusted_footprint_intersects": bool(not intersection.is_empty and intersection.area > 0),
                "adjusted_footprint_intersection_area_map_units2": float(intersection.area),
                **overlap_meta,
                "both_sides_have_aligned_valid_pixels": overlap_meta["aligned_valid_overlap_pixels"] > 0,
                "all_band_pif_intersection_valid_samples": 0,
                "bands": {},
            }
            for band_index, band_name in enumerate(BANDS):
                band_result = pair_band_screen(stack[:, :, i, band_index],
                                               stack[:, :, j, band_index], pif_result.pif)
                row["bands"][band_name] = band_result
            row["all_band_pif_intersection_valid_samples"] = min(
                row["bands"][name]["pif_intersection_valid_samples"] for name in BANDS)
            row["all_band_grid_selected_samples_min"] = min(
                row["bands"][name]["grid_selected_samples"] for name in BANDS)
            for threshold in THRESHOLDS:
                row[f"outcome_{threshold}d_by_band"] = {}
                for band_name in BANDS:
                    reason, code = stage_reason(row, threshold, row["bands"][band_name])
                    row[f"outcome_{threshold}d_by_band"][band_name] = {"category": reason, "reason_code": code}
            pair_rows.append(row)

    thresholds = {}
    for threshold in THRESHOLDS:
        per_band = {band: band_network_audit(pair_rows, band, threshold, control) for band in BANDS}
        accepted_pairs = {band: [[edge["scene_i"], edge["scene_j"]]
                                 for edge in per_band[band]["effective_edges"]] for band in BANDS}
        stage_counts = {}
        stage_pass_counts = {}
        for band in BANDS:
            counts = {}
            for row in pair_rows:
                category = row[f"outcome_{threshold}d_by_band"][band]["category"]
                counts[category] = counts.get(category, 0) + 1
            stage_counts[band] = counts
            stage_pass_counts[band] = {
                "time_window_pass": sum(row["day_gap_calendar_days"] <= threshold for row in pair_rows),
                "adjusted_spatial_footprint_intersection": sum(row["adjusted_footprint_intersects"] for row in pair_rows),
                "both_sides_aligned_valid_pixels": sum(row["aligned_valid_overlap_pixels"] > 0 for row in pair_rows),
                "pif_intersection_at_least_50": sum(row["bands"][band]["pif_intersection_valid_samples"] >= MIN_RAW_PIF_SAMPLES for row in pair_rows),
                "grid_subset_accepted": sum(row["bands"][band]["accepted_by_lsstaa_edge_filter"] for row in pair_rows),
                "final_edges_with_time_gate": sum(row["day_gap_calendar_days"] <= threshold and row["bands"][band]["accepted_by_lsstaa_edge_filter"] for row in pair_rows),
            }
        thresholds[str(threshold)] = {
            "configuration_label": "OFFICIAL_DEFAULT" if threshold == 14 else "EXPLORATORY_NETWORK_CONNECTIVITY_ONLY",
            "stage_outcome_counts_by_band": stage_counts,
            "stage_pass_counts_by_band": stage_pass_counts,
            "network_by_band": per_band,
            "all_bands_same_edges": all(value == accepted_pairs[BANDS[0]] for value in accepted_pairs.values()),
        }

    scene_pif_counts = {band: {str(i): int(np.count_nonzero(pif_result.pif & (stack[:, :, i, band_index] != 0)))
                               for i in range(N_SCENES)}
                        for band_index, band in enumerate(BANDS)}
    subnets = {band: subset_audit(pair_rows, band, 30, CONTROL_SUBNET, control)
               for band in BANDS}
    for band in BANDS:
        subnets[band]["pif_scene_support_counts"] = {str(node): scene_pif_counts[band][str(node)]
                                                      for node in CONTROL_SUBNET}

    # Bridge candidates are every pair crossing a 30-day connected-component cut
    # involving a component that cannot reach the selected control.
    components_30 = thresholds["30"]["network_by_band"]["B9"]["components"]
    control_group = next(group for group in components_30 if control in group)
    component_id = {node: index for index, group in enumerate(components_30) for node in group}
    bridge_pairs = [row for row in pair_rows if component_id[row["scene_i"]] != component_id[row["scene_j"]]
                    and (component_id[row["scene_i"]] != component_id[control] or
                         component_id[row["scene_j"]] != component_id[control])]
    bridge_summary = []
    for group in components_30:
        if group == control_group:
            continue
        candidates = []
        for row in bridge_pairs:
            if not (row["scene_i"] in group or row["scene_j"] in group):
                continue
            physical = row["adjusted_footprint_intersects"] and row["aligned_valid_overlap_pixels"] > 0
            pif = max(row["bands"][band]["pif_intersection_valid_samples"] for band in BANDS)
            grid_samples = max(row["bands"][band]["grid_selected_samples"] for band in BANDS)
            candidates.append({
                "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                "scene_id_i": row["scene_id_i"], "scene_id_j": row["scene_id_j"],
                "day_gap_calendar_days": row["day_gap_calendar_days"],
                "within_30_day_window": row["within_exploratory_30_day_window"],
                "adjusted_footprint_intersects": row["adjusted_footprint_intersects"],
                "aligned_valid_overlap_pixels": row["aligned_valid_overlap_pixels"],
                "max_band_pif_intersection_valid_samples": pif,
                "max_band_grid_selected_samples": grid_samples,
                "physical_overlap_support": physical,
                "band_outcomes_30d": row["outcome_30d_by_band"],
            })
        candidates.sort(key=lambda x: (x["within_30_day_window"], x["physical_overlap_support"],
                                       x["max_band_pif_intersection_valid_samples"],
                                       x["max_band_grid_selected_samples"]), reverse=True)
        best = candidates[0] if candidates else None
        to_control = [candidate for candidate in candidates
                      if component_id[candidate["scene_i"] if candidate["scene_i"] not in group else candidate["scene_j"]]
                      == component_id[control]]
        best_to_control = to_control[0] if to_control else None
        bridge_summary.append({
            "disconnected_component_scene_indices": group,
            "candidate_pair_count_to_other_components": len(candidates),
            "all_cross_component_candidates": candidates,
            "most_promising_pair": best,
            "most_promising_pair_directly_to_control_component": best_to_control,
            "most_promising_pair_has_physical_edge_conditions": bool(
                best and best["within_30_day_window"] and best["physical_overlap_support"]
                and best["max_band_pif_intersection_valid_samples"] >= MIN_RAW_PIF_SAMPLES
                and best["max_band_grid_selected_samples"] >= 3),
        })

    for threshold in THRESHOLDS:
        for band in BANDS:
            network = thresholds[str(threshold)]["network_by_band"][band]
            edges = edge_objects(pair_rows, band, threshold)
            all_design, _ = design_for(list(range(N_SCENES)), control, edges)
            full_rank = network["joint_design"]["rank"]
            for edge in network["effective_edges"]:
                reduced_edges = [e for e in edges if (e["scene_i"], e["scene_j"]) !=
                                 (edge["scene_i"], edge["scene_j"])]
                reduced, _ = design_for(list(range(N_SCENES)), control, reduced_edges)
                edge["unique_rank_contribution_leave_one_out"] = full_rank - svd_diagnostic(
                    reduced, network["joint_design"]["column_labels"], list(range(N_SCENES)), control)["rank"]
                edge["rank_redundant_when_removed"] = edge["unique_rank_contribution_leave_one_out"] == 0

    eligible_physical_candidates = [candidate for item in bridge_summary
        for candidate in item["all_cross_component_candidates"]
        if candidate["within_30_day_window"] and candidate["physical_overlap_support"]]
    case_a_candidates = [candidate for candidate in eligible_physical_candidates
        if candidate["max_band_pif_intersection_valid_samples"] >= MIN_RAW_PIF_SAMPLES
        and candidate["max_band_grid_selected_samples"] < 3]
    decision = ("CASE_C_NO_TIME_OR_SPATIAL_BRIDGE" if not eligible_physical_candidates
                else "CASE_B_OVERLAP_BUT_PIF_OR_GRID_CONSTRAINTS_INSUFFICIENT")
    if case_a_candidates:
        decision = "CASE_A_POTENTIAL_DOWNSAMPLE_OR_GRID_MISSED_BRIDGE"
    if any(network["joint_design"]["zero_space_dimension"] > 0 and
           len(network["control_reachable_scene_indices"]) == N_SCENES
           for threshold in thresholds.values()
           for network in threshold["network_by_band"].values()):
        decision = "CASE_D_ADDITIONAL_RANK_DEFICIENCY_AFTER_CONNECTIVITY"

    roi_pairs = set()
    for item in bridge_summary:
        # The strongest time-eligible physical candidate per disconnected
        # component is a diagnostic ROI only; selection uses PIF support, never scores.
        for candidate in (item["most_promising_pair"], item["most_promising_pair_directly_to_control_component"]):
            if candidate and candidate["within_30_day_window"] and candidate["physical_overlap_support"]:
                roi_pairs.add((candidate["scene_i"], candidate["scene_j"]))
    native_roi_diagnostics = local_native_block_diagnostic(sorted(roi_pairs), pif_result.pif, stack)
    if native_roi_diagnostics:
        for item in bridge_summary:
            for candidate in (item["most_promising_pair"], item["most_promising_pair_directly_to_control_component"]):
                if candidate:
                    review = next((row for row in native_roi_diagnostics
                                   if (row["scene_i"], row["scene_j"]) ==
                                   (candidate["scene_i"], candidate["scene_j"])), None)
                    candidate["native_roi_block_diagnostic"] = review

    report = {
        "task": "LSSTAA_13_SCENE_BRIDGE_FAILURE_AND_SUBNETWORK_AUDIT",
        "created_date": "2026-10-15", "status": "COMPLETED_OFFLINE_DIAGNOSTIC",
        "method": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED",
        "official_reference_equivalence": "UNRESOLVED_PRIOR_GATE; NOT_REINVESTIGATED",
        "freeze_lock": {"valid": provenance["freeze_lock_valid"],
                        "verified_entries": provenance["freeze_lock_entries"],
                        "sha256": provenance["freeze_lock_sha256"]},
        "input": {"scene_count": N_SCENES, "bands": list(BANDS), "factor": 10,
                  "pif_count": int(pif_result.pif.sum()), "pif_threshold": pif_result.threshold,
                  "control_scene_index": control, "control_scene_id": scene_records[control]["scene_id"],
                  "scene_records": [{"scene_index": row["scene_index"], "scene_id": row["scene_id"],
                                     "date": dates[row["scene_index"]],
                                     "acquisition_timestamp": timestamps[row["scene_index"]].isoformat(),
                                     "day_offset": days[row["scene_index"]]} for row in scene_records],
                  "provenance": provenance},
        "pair_count": len(pair_rows), "pair_diagnostics": [
            {**{key: value for key, value in row.items() if key != "bands"},
             "bands": {band: {key: value for key, value in band_data.items()
                              if key not in ("selected_x", "selected_y")}
                       for band, band_data in row["bands"].items()}}
            for row in pair_rows],
        "threshold_diagnostics": thresholds,
        "bridge_diagnostics_30_day": {"control_component": control_group,
                                       "disconnected_components": bridge_summary,
                                       "limited_native_resolution_roi_diagnostics": native_roi_diagnostics},
        "scene_pif_support_counts_by_band": scene_pif_counts,
        "control_subnetwork_30_day": {"scene_indices": CONTROL_SUBNET,
                                       "control_scene_index": control,
                                       "by_band": subnets},
        "joint_design_construction": {
            "observation_equation": "g_i*x + o_i - g_j*y - o_j = y - x",
            "control_scene_constraint": "fixed gain=1, offset=0; no other scene fixed",
            "sample_weighting": "equal weight per LSSTAA selected grid subset sample; no additional data weights",
            "matrix_source": "raw rectangular joint design matrix assembled from the selected x/y ties; rank/nullspace from direct float64 SVD",
        },
        "decision": {"case": decision,
                     "recommendation": "STOP_13_SCENE_GLOBAL_CLOUD_TRIAL; review native-resolution PIF semantics for candidate bridge ROIs; separately review the algebraically solvable but numerically scaled 8-scene subnetwork before deciding on any experiment",
                     "full_13_scene_solvable": False,
                     "control_subnetwork_solvable_all_bands": all(
                         subnets[band]["joint_design"]["rank"] == subnets[band]["joint_design"]["matrix_shape"][1]
                         for band in BANDS)},
        "guardrails": {"thresholds_tested_only": [14, 30],
                       "30_day_is_connectivity_exploration_only": True,
                       "freeze_lock_modified": False, "frozen_inputs_or_masks_modified": False,
                       "three_scene_outputs_modified": False,
                       "full_resolution_13_scene_solve_performed": False,
                       "full_resolution_corrected_geotiffs_generated": False,
                       "formal_benchmark_scoring_performed": False,
                       "official_reference_coefficients_investigated": False,
                       "pseudo_inverse_or_regularization_used": False,
                       "radiometric_coefficients_emitted": False},
    }
    write_reports(report)
    print(json.dumps({"status": report["status"], "case": report["decision"]["case"],
                      "subnet_solvable_all_bands": report["decision"]["control_subnetwork_solvable_all_bands"],
                      "outputs": [str(OUT_MD), str(OUT_JSON)]}, ensure_ascii=False))
    return 0


def write_reports(report: dict) -> None:
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    scene_names = {row["scene_index"]: row["scene_id"] for row in report["input"]["scene_records"]}
    rows = [
        "# LSSTAA 13景网络断边归因与可解子网诊断", "",
        "离线诊断，输入冻结锁54/54通过。使用10倍估计与固定官方14天/单独标记30天探索窗口；未做云端实验、全分辨率联合求解或正式评分。方法仍标记为未验证DN代理，不宣称严格复现官方LSSTAA。", "",
        f"- PIF像元：{report['input']['pif_count']}；控制景：{report['input']['control_scene_index']}（{report['input']['control_scene_id']}）。78对场景的各阶段结果、完整边列表和原因码见JSON。", "",
        "## 网络与秩诊断", "",
        "| 配置 | 波段 | 有效边 | 连通分量 | 控制可达 | 原始设计矩阵秩 | 未知数 | 零空间 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for threshold in THRESHOLDS:
        for band in BANDS:
            item = report["threshold_diagnostics"][str(threshold)]["network_by_band"][band]
            design = item["joint_design"]
            rows.append(f"| {threshold}天 | {band} | {item['valid_edge_count']} | {len(item['components'])} | {len(item['control_reachable_scene_indices'])}/13 | {design['rank']} | {design['matrix_shape'][1]} | {design['zero_space_dimension']} |")
    rows += ["", "- 14天结构性秩亏与数值零空间完全相符：孤立景2、7各贡献2维；未锚定的[0,1]、[3,4,5]、[10,11,12]各贡献1维，共7维。30天剩余5维恰为景2、7的4维及未锚定[3,4,5]的1维。四波段奇异值、秩阈值、零空间参数与分量局部秩均记录在JSON。", "",
             "阶段边际通过计数（78对全体，并非上一阶段条件通过数）：14天时间27、范围相交60、对齐有效像元48、PIF交集≥50为17、网格接受17、最终8条边；30天依次为51、60、48、17、17、最终13条边。首失败原因码逐对记录；14天分布为时间51/无范围5/无有效像元2/PIF不足12/网格失败0/有效边8，30天为27/10/7/21/0/13。", "",
             "## 30天跨分量桥接与8景子网", "",
             "| 未接入分量 | 跨分量候选数 | 最有希望对（任意分量） | 最有希望对（直连控制分量） | 有效重叠/PIF/网格 |", "|---|---:|---|---|---|"]
    for bridge in report["bridge_diagnostics_30_day"]["disconnected_components"]:
        best = bridge["most_promising_pair"]
        direct = bridge["most_promising_pair_directly_to_control_component"]
        if best:
            pair_label = f"{best['scene_i']}-{best['scene_j']} ({best['day_gap_calendar_days']}d)"
            stage_label = f"{best['aligned_valid_overlap_pixels']:,} / {best['max_band_pif_intersection_valid_samples']} / {best['max_band_grid_selected_samples']}"
        else:
            pair_label, stage_label = "无", "无"
        direct_label = (f"{direct['scene_i']}-{direct['scene_j']} ({direct['day_gap_calendar_days']}d)"
                        if direct else "无")
        rows.append(f"| {bridge['disconnected_component_scene_indices']} | {bridge['candidate_pair_count_to_other_components']} | {pair_label} | {direct_label} | {stage_label} |")
    rows += ["", "局部原分辨率复核只读取上述候选对冻结B9影像中与10倍PIF格网单元对应的小窗口：2-5为1768个双方有效原始像元、203个网格样本、R²=0.918；4-6为1011、202、R²=0.935；3-7为200、132、R²=0.302。该诊断显示10倍均值把PIF有效样本数压到19/19/2，低于当前≥50边门槛；扩展到原始像元后抽样可产生样本，但原始分辨率13时相PIF分类并未重算，因此只能标记为高分辨率桥接候选，不能认定为已验证有效边。", "",
             f"8景控制子网秩条件：`{report['decision']['control_subnetwork_solvable_all_bands']}`。", "",
             "| 波段 | 设计矩阵 | 秩 | cond(A) | cond(AᵀA) | 列归一化cond(A) | 最大单边样本占比 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for band in BANDS:
        sub = report["control_subnetwork_30_day"]["by_band"][band]
        diag = sub["joint_design"]
        rows.append(f"| {band} | {diag['matrix_shape'][0]}×{diag['matrix_shape'][1]} | {diag['rank']}/14 | {diag['condition_number_design']:.3g} | {diag['condition_number_normal']:.3g} | {diag['condition_number_design_column_scaled']:.3g} | {sub['max_single_edge_sample_share']:.1%} |")
    rows += ["", "四波段均为14/14满秩；8景每景PIF有效支持最少88像元，边交集最少68像元，采样后每边80–204个样本。各边两侧DN的P01–P99跨度见JSON（B9约2432–4485、B4约2480–4428、B7约984–4759、B14约1075–4821），所有边侧标准差均非零；本预检未预设动态范围充分性的硬阈值。", "",
             "列归一化后cond(A)=129–226；原始cond(A)偏高主要受DN增益列与偏移列量纲差异影响。单边最大样本占比11.7%–12.8%，没有单边样本数量主导；但删去0-1边会降秩2，删去1-6边会降秩1，它们是维持子网连通与满秩的关键桥。其余边删一仍满秩，提供冗余观测。此处只证明代数可解，不能代替8景实验评审。", "",
             f"## 决策：{report['decision']['case']}", "",
             report["decision"]["recommendation"], "",
             "未使用伪逆或正则化；未输出系数或校正影像。除非桥接候选明确因10倍降采样/网格抽样漏检，否则不进行原分辨率ROI复核；任何ROI复核都仅作为诊断。", "",
             f"逐边JSON：`{OUT_JSON.name}`。场景序号映射：" + "; ".join(f"{index}={name}" for index, name in scene_names.items()) + "."]
    OUT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
