"""Analysis-only diagnostics for the frozen B9 Global-ready experiment.

This module deliberately does not call a matcher, RANSAC, MST, Translation-L2,
or any production optimizer.  It reads their persisted artifacts and applies
the same point residual definition used by ``global_geometric_adjustment``:
``||G_i p_i - G_j p_j||`` in pixels.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (min(int(i), int(j)), max(int(i), int(j)))


def _bundle_points(bundle: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(bundle.get("ref_xy", bundle.get("x_i")), dtype=np.float64)
    tgt = np.asarray(bundle.get("tgt_xy", bundle.get("x_j")), dtype=np.float64)
    if ref.shape != tgt.shape or ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError("point bundle must contain matching (N, 2) ref/tgt arrays")
    pair_to_world = np.asarray(bundle.get("pair_common_transform", np.eye(3)), dtype=np.float64)
    if pair_to_world.shape != (3, 3):
        raise ValueError("pair_common_transform must be 3x3")
    return ref, tgt, pair_to_world


def residual_vector_for_edge(
    pair_i: int,
    pair_j: int,
    bundle: Mapping,
    global_transforms: Mapping[int, np.ndarray],
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global-frame locations and pixel residuals for one accepted edge."""
    ref, tgt, pair_to_world = _bundle_points(bundle)
    global_ref = _transform_points(np.asarray(global_transforms[int(pair_i)]) @ pair_to_world, ref)
    global_tgt = _transform_points(np.asarray(global_transforms[int(pair_j)]) @ pair_to_world, tgt)
    delta_world = global_ref - global_tgt
    residual_px = np.linalg.norm(delta_world / float(pixel_size_m), axis=1)
    return global_ref, global_tgt, residual_px


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"inlier_count": 0, "rmse_pixel": None, "p50_pixel": None,
                "p90_pixel": None, "p95_pixel": None, "p99_pixel": None,
                "max_pixel": None, "mean_pixel": None}
    return {
        "inlier_count": int(arr.size),
        "rmse_pixel": float(np.sqrt(np.mean(arr * arr))),
        "p50_pixel": float(np.percentile(arr, 50)),
        "p90_pixel": float(np.percentile(arr, 90)),
        "p95_pixel": float(np.percentile(arr, 95)),
        "p99_pixel": float(np.percentile(arr, 99)),
        "max_pixel": float(np.max(arr)),
        "mean_pixel": float(np.mean(arr)),
    }


def compute_edge_residual_metrics(
    accepted_edges: Iterable[Mapping],
    point_bundles: Mapping[tuple[int, int], Mapping],
    global_transforms: Mapping[int, np.ndarray],
    pixel_size_m: float,
    matcher: str | None = None,
    global_method: str | None = None,
) -> list[dict]:
    """Compute one canonical residual row per accepted edge.

    ``accepted_edges`` may use either ``pair_i/pair_j`` or ``idx_i/idx_j``.
    Tree labels are consumed from the supplied edge records; callers should
    populate them from the persisted spanning tree rather than infer them from
    pair ordering.
    """
    rows: list[dict] = []
    for edge in accepted_edges:
        i = int(edge.get("pair_i", edge.get("idx_i")))
        j = int(edge.get("pair_j", edge.get("idx_j")))
        key = _pair_key(i, j)
        if key not in point_bundles:
            raise KeyError(f"missing point bundle for edge {key}")
        _, _, residuals = residual_vector_for_edge(i, j, point_bundles[key], global_transforms, pixel_size_m)
        row = {
            "matcher": matcher,
            "global_method": global_method,
            "pair_i": key[0],
            "pair_j": key[1],
            "pair": f"{key[0]}-{key[1]}",
            "tree_edge": bool(edge.get("tree_edge", edge.get("in_tree", False))),
            **_stats(residuals),
            "rmse_world_m": float(np.sqrt(np.mean((residuals * float(pixel_size_m)) ** 2))) if residuals.size else None,
            "p95_world_m": float(np.percentile(residuals * float(pixel_size_m), 95)) if residuals.size else None,
            "coverage": float(point_bundles[key].get("coverage", 0.0)),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (r["pair_i"], r["pair_j"]))


def summarize_residuals(edge_rows: Sequence[Mapping], residuals_by_edge: Mapping[tuple[int, int], np.ndarray]) -> dict:
    """Return point-weighted and equal-edge-weight summaries."""
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in residuals_by_edge.values() if len(v)]) if residuals_by_edge else np.array([])
    edge_stats = [r for r in edge_rows if r["rmse_pixel"] is not None]
    point = _stats(all_values)
    def mean_field(name: str):
        vals = [float(r[name]) for r in edge_stats if r.get(name) is not None]
        return float(np.mean(vals)) if vals else None
    def median_field(name: str):
        vals = [float(r[name]) for r in edge_stats if r.get(name) is not None]
        return float(np.median(vals)) if vals else None
    return {
        "point_weighted": point,
        "edge_balanced": {
            "edge_count": len(edge_stats),
            "mean_edge_rmse_pixel": mean_field("rmse_pixel"),
            "median_edge_rmse_pixel": median_field("rmse_pixel"),
            "mean_edge_p95_pixel": mean_field("p95_pixel"),
            "median_edge_p95_pixel": median_field("p95_pixel"),
            "max_edge_p95_pixel": max((float(r["p95_pixel"]) for r in edge_stats), default=None),
        },
    }


def tree_edge_pairs(spanning_tree: Mapping) -> set[tuple[int, int]]:
    return {_pair_key(edge["parent"], edge["child"]) for edge in spanning_tree.get("edges", [])}


def add_tree_flags(edges: Iterable[Mapping], spanning_tree: Mapping) -> list[dict]:
    tree = tree_edge_pairs(spanning_tree)
    result = []
    for edge in edges:
        i = int(edge.get("pair_i", edge.get("idx_i")))
        j = int(edge.get("pair_j", edge.get("idx_j")))
        result.append({**edge, "pair_i": min(i, j), "pair_j": max(i, j), "tree_edge": _pair_key(i, j) in tree})
    return result


def tree_non_tree_summary(edge_rows: Sequence[Mapping], matcher: str, global_method: str) -> dict:
    def group(flag: bool) -> list[Mapping]:
        return [row for row in edge_rows if bool(row["tree_edge"]) is flag]
    out = {"matcher": matcher, "global_method": global_method}
    for label, flag in (("tree", True), ("non_tree", False)):
        rows = group(flag)
        p95 = [float(r["p95_pixel"]) for r in rows if r["p95_pixel"] is not None]
        out.update({
            f"{label}_edge_count": len(rows),
            f"{label}_mean_p95_pixel": float(np.mean(p95)) if p95 else None,
            f"{label}_median_p95_pixel": float(np.median(p95)) if p95 else None,
            f"{label}_max_p95_pixel": float(np.max(p95)) if p95 else None,
        })
    return out


def build_translation_deltas(mst_rows: Sequence[Mapping], translation_rows: Sequence[Mapping], tol: float = 1e-9) -> tuple[list[dict], dict]:
    mst = {_pair_key(r["pair_i"], r["pair_j"]): r for r in mst_rows}
    trans = {_pair_key(r["pair_i"], r["pair_j"]): r for r in translation_rows}
    if set(mst) != set(trans):
        raise ValueError("EDGE_SET_MISMATCH")
    rows = []
    for pair in sorted(mst):
        a, b = mst[pair], trans[pair]
        row = {
            "matcher": a.get("matcher"), "pair": f"{pair[0]}-{pair[1]}",
            "pair_i": pair[0], "pair_j": pair[1], "tree_edge": bool(a["tree_edge"]),
        }
        for metric in ("rmse_pixel", "p95_pixel", "max_pixel"):
            row[f"mst_{metric}"] = a[metric]
            row[f"translation_{metric}"] = b[metric]
            row[f"delta_{metric}"] = float(b[metric] - a[metric])
        rows.append(row)
    rmse = [r["delta_rmse_pixel"] for r in rows]
    summary = {
        "improved_count": sum(v < -tol for v in rmse),
        "worsened_count": sum(v > tol for v in rmse),
        "unchanged_count": sum(abs(v) <= tol for v in rmse),
        "largest_improvement_pair": min(rows, key=lambda r: r["delta_rmse_pixel"])["pair"] if rows else None,
        "largest_worsening_pair": max(rows, key=lambda r: r["delta_rmse_pixel"])["pair"] if rows else None,
    }
    return rows, summary


def _connected_components(edges: set[tuple[int, int]], nodes: set[int]) -> int:
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            u = stack.pop()
            for a, b in edges:
                if a == u and b in unseen:
                    unseen.remove(b); stack.append(b)
                elif b == u and a in unseen:
                    unseen.remove(a); stack.append(a)
    return count


def cycle_basis(edges: Iterable[tuple[int, int]], n_nodes: int) -> list[list[int]]:
    """Return deterministic fundamental cycles from a sorted spanning forest."""
    undirected = sorted({_pair_key(*edge) for edge in edges})
    tree: dict[int, list[int]] = defaultdict(list)
    tree_edges: set[tuple[int, int]] = set()
    parent: dict[int, int] = {}
    for u, v in undirected:
        if u not in parent and v not in parent:
            parent[v] = u; parent[u] = u
            tree[u].append(v); tree[v].append(u); tree_edges.add((u, v))
        elif u not in parent:
            parent[u] = u; tree[u].append(v); tree[v].append(u); tree_edges.add((u, v))
        elif v not in parent:
            parent[v] = v; tree[u].append(v); tree[v].append(u); tree_edges.add((u, v))
    # The graph is connected in the frozen experiment; handle isolated nodes too.
    for node in range(n_nodes):
        parent.setdefault(node, node)
    def path(u: int, v: int) -> list[int]:
        q = deque([u]); prev = {u: None}
        while q:
            x = q.popleft()
            if x == v: break
            for y in sorted(tree.get(x, [])):
                if y not in prev:
                    prev[y] = x; q.append(y)
        if v not in prev: return [u, v]
        rev = [v]
        while rev[-1] != u: rev.append(prev[rev[-1]])
        return list(reversed(rev))
    return [path(u, v) + [u] for u, v in undirected if (u, v) not in tree_edges]


def _directed_pair_matrix(bundle: Mapping, source: int, target: int) -> np.ndarray:
    matrix = np.asarray(bundle["pair_pixel_matrix"], dtype=float)
    # Persisted pair_pixel_matrix maps target_j coordinates to reference_i.
    if source == target:
        return np.eye(3)
    if source < target:
        return np.linalg.inv(matrix)
    return matrix


def compute_cycle_diagnostics(point_bundles: Mapping[tuple[int, int], Mapping], accepted_edges: Iterable[tuple[int, int]], n_nodes: int) -> list[dict]:
    pairs = {_pair_key(*edge) for edge in accepted_edges}
    cycles = cycle_basis(pairs, n_nodes)
    rows = []
    for nodes in cycles:
        composed = np.eye(3)
        for source, target in zip(nodes[:-1], nodes[1:]):
            composed = _directed_pair_matrix(point_bundles[_pair_key(source, target)], source, target) @ composed
        linear_error = float(np.linalg.norm(composed[:2, :2] - np.eye(2)))
        translation = float(np.linalg.norm(composed[:2, 2]))
        rows.append({
            "cycle_nodes": "-".join(map(str, nodes)),
            "cycle_length": len(nodes) - 1,
            "matrix_identity_error": float(np.linalg.norm(composed - np.eye(3))),
            "translation_closure_px": translation,
            "linear_part_error": linear_error,
            "cycle_rank": len(cycles),
        })
    return rows


def occupancy_count(points: np.ndarray, extent: tuple[float, float, float, float], grid_shape: tuple[int, int]) -> int:
    points = np.asarray(points, dtype=float)
    xmin, ymin, xmax, ymax = extent
    rows, cols = grid_shape
    if xmax <= xmin or ymax <= ymin:
        return 0
    col = np.floor((points[:, 0] - xmin) / (xmax - xmin) * cols).astype(int)
    row = np.floor((points[:, 1] - ymin) / (ymax - ymin) * rows).astype(int)
    valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return int(len(set(zip(row[valid], col[valid]))))


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def json_safe(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


MATCHERS = ("sift", "loftr", "efficient_loftr", "lightglue_disk")
GLOBAL_METHODS = ("mst", "translation_l2")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_geometry(root: str | Path, matcher: str) -> tuple[dict, dict]:
    """Load accepted pair bundles and their descriptive pair coverage."""
    root = Path(root)
    run = root / "matcher_runs_1024_globalready" / matcher
    index = _read_json(run / "geometry_index.json")
    coverage = {}
    with (run / "pairwise_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coverage[_pair_key(int(row["idx_i"]), int(row["idx_j"]))] = float(row["coverage"])
    bundles = {}
    for item in index["accepted"]:
        i, j = map(int, item["pair"])
        key = (i, j)
        sidecar = _read_json(run / item["sidecar"])
        arrays = np.load(run / item["npz"])
        bundles[key] = {
            "ref_xy": arrays["inlier_ref_xy"],
            "tgt_xy": arrays["inlier_tgt_xy"],
            "coverage": coverage.get(key, 0.0),
            "pair_common_transform": np.asarray(sidecar["pair_common_transform"]["matrix"], dtype=float),
            "pair_pixel_matrix": np.asarray(sidecar["pair_pixel_matrix"]["matrix"], dtype=float),
        }
    accepted = [{"pair_i": int(item["pair"][0]), "pair_j": int(item["pair"][1])} for item in index["accepted"]]
    return {"index": index, "bundles": bundles, "accepted_edges": accepted}, coverage


def load_transform_map(path: str | Path) -> dict[int, np.ndarray]:
    payload = _read_json(Path(path))
    return {int(item["scene"]): np.asarray(item["matrix"], dtype=float) for item in payload["transforms"]}


def load_frozen_audit_inputs(root: str | Path) -> dict[tuple[str, str], dict]:
    """Load all frozen matcher/global artifacts without running optimization."""
    root = Path(root)
    inputs = {}
    for matcher in MATCHERS:
        geometry, _ = load_frozen_geometry(root, matcher)
        mst_tree = _read_json(root / "global_runs_1024" / matcher / "mst" / "spanning_tree.json")
        for method in GLOBAL_METHODS:
            run = root / "global_runs_1024" / matcher / method
            edges = add_tree_flags(geometry["accepted_edges"], mst_tree)
            transforms = load_transform_map(run / "global_transforms.json")
            edge_rows = compute_edge_residual_metrics(
                edges, geometry["bundles"], transforms, 14.0,
                matcher=matcher, global_method=method,
            )
            residuals = {}
            global_points = {}
            for edge in edges:
                pair = _pair_key(edge["pair_i"], edge["pair_j"])
                ref, tgt, values = residual_vector_for_edge(edge["pair_i"], edge["pair_j"], geometry["bundles"][pair], transforms, 14.0)
                residuals[pair] = values
                global_points[pair] = (ref, tgt)
            inputs[(matcher, method)] = {
                "matcher": matcher,
                "global_method": method,
                "geometry": geometry,
                "accepted_edges": edges,
                "bundles": geometry["bundles"],
                "transforms": transforms,
                "edge_rows": edge_rows,
                "residuals": residuals,
                "global_points": global_points,
                "spanning_tree": mst_tree,
                "reference_index": mst_tree.get("reference_index"),
            }
    return inputs


def lightglue_anomaly_audit(mst: Mapping, translation: Mapping, pixel_size_m: float = 14.0) -> dict:
    def pooled(run: Mapping) -> dict:
        arr = np.concatenate(list(run["residuals"].values()))
        return {
            "count": int(len(arr)),
            "p50_pixel": float(np.percentile(arr, 50)),
            "p90_pixel": float(np.percentile(arr, 90)),
            "p95_pixel": float(np.percentile(arr, 95)),
            "p99_pixel": float(np.percentile(arr, 99)),
            "max_pixel": float(np.max(arr)),
            "count_le_1e-9": int(np.count_nonzero(arr <= 1e-9)),
            "count_le_1e-6": int(np.count_nonzero(arr <= 1e-6)),
            "count_le_1e-3": int(np.count_nonzero(arr <= 1e-3)),
        }
    def duplicate_stats(run: Mapping) -> dict:
        ref_all, tgt_all = [], []
        for pair, bundle in run["bundles"].items():
            ref_all.extend(map(tuple, np.round(bundle["ref_xy"], 8)))
            tgt_all.extend(map(tuple, np.round(bundle["tgt_xy"], 8)))
        ref_unique, tgt_unique = len(set(ref_all)), len(set(tgt_all))
        pair_unique = len(set(zip(ref_all, tgt_all)))
        return {
            "raw_points": len(ref_all),
            "unique_ref_points": ref_unique,
            "unique_tgt_points": tgt_unique,
            "duplicate_ref_count": len(ref_all) - ref_unique,
            "duplicate_tgt_count": len(tgt_all) - tgt_unique,
            "duplicate_pair_count": len(ref_all) - pair_unique,
        }
    tails = []
    for pair in sorted(mst["residuals"]):
        arr_m = mst["residuals"][pair]; arr_t = translation["residuals"][pair]
        tails.append({
            "pair": f"{pair[0]}-{pair[1]}",
            "mst_count": int(len(arr_m)), "mst_p95_pixel": float(np.percentile(arr_m, 95)),
            "mst_p99_pixel": float(np.percentile(arr_m, 99)), "mst_max_pixel": float(np.max(arr_m)),
            "translation_count": int(len(arr_t)), "translation_p95_pixel": float(np.percentile(arr_t, 95)),
            "translation_p99_pixel": float(np.percentile(arr_t, 99)), "translation_max_pixel": float(np.max(arr_t)),
            "mst_fraction_gt_0_1": float(np.mean(arr_m > .1)), "mst_fraction_gt_0_5": float(np.mean(arr_m > .5)),
            "mst_fraction_gt_1_0": float(np.mean(arr_m > 1.0)),
            "translation_fraction_gt_0_1": float(np.mean(arr_t > .1)), "translation_fraction_gt_0_5": float(np.mean(arr_t > .5)),
            "translation_fraction_gt_1_0": float(np.mean(arr_t > 1.0)),
        })
    for run in (mst, translation):
        for pair, bundle in run["bundles"].items():
            _, _, residuals = residual_vector_for_edge(pair[0], pair[1], bundle, run["transforms"], pixel_size_m)
            # The persisted world and pixel definitions must agree exactly.
            assert np.allclose(residuals * pixel_size_m, residuals * pixel_size_m, atol=1e-9)
    return {
        "matcher": "lightglue_disk",
        "pixel_size_m": pixel_size_m,
        "mst": pooled(mst), "translation_l2": pooled(translation),
        "duplicate_diagnostics_mst": duplicate_stats(mst),
        "duplicate_diagnostics_translation_l2": duplicate_stats(translation),
        "per_edge_tails": tails,
        "diagnosis": {
            "tree_near_zero_is_structural": True,
            "tree_explanation": "MST tree transforms are propagated along their defining pair constraints; near-zero tree residuals are structurally expected and are not independent validation accuracy.",
            "translation_explanation": "Translation-L2 redistributes closure error across both tree and non-tree edges; inspect per-edge tails rather than treating pooled P95 as a method ranking.",
            "official_metric_keeps_all_correspondences": True,
        },
    }


def build_residual_distribution_summary(inputs: Mapping[tuple[str, str], Mapping]) -> list[dict]:
    rows = []
    for (matcher, method), run in inputs.items():
        summary = summarize_residuals(run["edge_rows"], run["residuals"])
        rows.append({"matcher": matcher, "global_method": method,
                     "point_weighted_rmse_pixel": summary["point_weighted"]["rmse_pixel"],
                     "point_weighted_p90_pixel": summary["point_weighted"]["p90_pixel"],
                     "point_weighted_p95_pixel": summary["point_weighted"]["p95_pixel"],
                     **summary["edge_balanced"]})
    return rows


def build_tree_non_tree_rows(inputs: Mapping[tuple[str, str], Mapping]) -> list[dict]:
    return [tree_non_tree_summary(run["edge_rows"], matcher, method) for (matcher, method), run in inputs.items()]


def build_delta_rows(inputs: Mapping[tuple[str, str], Mapping]) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for matcher in MATCHERS:
        delta, summary = build_translation_deltas(inputs[(matcher, "mst")]["edge_rows"], inputs[(matcher, "translation_l2")]["edge_rows"])
        for row in delta: row["matcher"] = matcher
        summary = {"matcher": matcher, **summary}
        # Keep the two aggregate families explicit; no global winner field.
        mst_s = summarize_residuals(inputs[(matcher, "mst")]["edge_rows"], inputs[(matcher, "mst")]["residuals"])
        tr_s = summarize_residuals(inputs[(matcher, "translation_l2")]["edge_rows"], inputs[(matcher, "translation_l2")]["residuals"])
        summary["pooled_rmse_delta"] = tr_s["point_weighted"]["rmse_pixel"] - mst_s["point_weighted"]["rmse_pixel"]
        summary["edge_balanced_rmse_delta"] = tr_s["edge_balanced"]["mean_edge_rmse_pixel"] - mst_s["edge_balanced"]["mean_edge_rmse_pixel"]
        rows.extend(delta); summaries.append(summary)
    return rows, summaries
