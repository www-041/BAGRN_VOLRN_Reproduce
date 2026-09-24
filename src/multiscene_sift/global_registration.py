"""Multi-scene global registration: accepted matcher graph, spanning tree, transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.multiscene_sift.models import PairwiseRegistration, Scene
from src.multiscene_sift.band_geometry import pixel_affine_to_world

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 5 — Accepted matcher graph
# ---------------------------------------------------------------------------


def build_accepted_graph(
    pairwise_results: list[PairwiseRegistration],
    matcher_name: str = "matcher",
) -> tuple[dict[int, set[int]], list[PairwiseRegistration]]:
    """Build adjacency dict of pairs with ``status == "OK"``.

    Returns:
        ``(adj, accepted)`` where *adj* is ``{scene_idx: {neighbour_idx}}``
        and *accepted* is the sub-list of passed registrations.

    Raises:
        RuntimeError: If the accepted graph is disconnected.
    """
    accepted = [r for r in pairwise_results if r.status == "OK"]
    n = max(
        max(r.idx_i, r.idx_j) for r in pairwise_results
    ) + 1 if pairwise_results else 0

    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for r in accepted:
        adj[r.idx_i].add(r.idx_j)
        adj[r.idx_j].add(r.idx_i)

    # Check connectivity
    if n > 0 and not _is_connected(n, adj):
        components = _connected_components(n, adj)
        raise RuntimeError(
            f"Accepted {matcher_name} graph is DISCONNECTED. Components: {components}"
        )

    return adj, accepted


def build_accepted_sift_graph(
    pairwise_results: list[PairwiseRegistration],
) -> tuple[dict[int, set[int]], list[PairwiseRegistration]]:
    """Backward-compatible alias for legacy SIFT callers."""
    return build_accepted_graph(pairwise_results, matcher_name="SIFT")


def _is_connected(n: int, adj: dict[int, set[int]]) -> bool:
    """Check graph connectivity via DFS."""
    if n <= 1:
        return True
    visited = set()
    stack = [0]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        stack.extend(adj[v] - visited)
    return len(visited) == n


def _connected_components(n: int, adj: dict[int, set[int]]) -> list[list[int]]:
    """Return connected components of the graph."""
    visited: set[int] = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        comp = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(adj[v] - visited)
        if comp:
            components.append(sorted(comp))
    return components


# ---------------------------------------------------------------------------
# Task 6 — Reference scene selection
# ---------------------------------------------------------------------------


def select_reference_scene(
    adj: dict[int, set[int]],
    accepted: list[PairwiseRegistration],
) -> dict:
    """Select the reference scene using degree, sum(Q), and index tie-break.

    Edge quality: Q = n_inlier × inlier_ratio × coverage

    Returns:
        Dict with ``reference_index``, ``reference_name``, ``reason``, etc.
    """
    n = len(adj)

    # Compute Q for each accepted edge
    edge_q: dict[tuple[int, int], float] = {}
    for r in accepted:
        q = r.inliers * r.inlier_ratio * r.coverage
        edge_q[(r.idx_i, r.idx_j)] = q
        edge_q[(r.idx_j, r.idx_i)] = q

    # Degree and sum Q per scene
    degree = {i: len(adj[i]) for i in range(n)}
    sum_q = {}
    for i in range(n):
        total = 0.0
        for j in adj[i]:
            total += edge_q.get((i, j), 0.0)
        sum_q[i] = total

    # Sort: highest degree, then highest sum Q, then lowest index
    candidates = sorted(
        range(n),
        key=lambda i: (-degree[i], -sum_q[i], i),
    )
    ref_idx = candidates[0]

    return {
        "reference_index": ref_idx,
        "degree": degree[ref_idx],
        "sum_q": round(sum_q[ref_idx], 4),
        "all_degrees": {i: degree[i] for i in range(n)},
        "all_sum_q": {i: round(sum_q[i], 4) for i in range(n)},
        "selection_reason": (
            f"highest degree ({degree[ref_idx]}), "
            f"highest sum(Q) ({sum_q[ref_idx]:.2f}), "
            f"lowest index ({ref_idx})"
        ),
    }


# ---------------------------------------------------------------------------
# Task 7 — Maximum-quality spanning tree
# ---------------------------------------------------------------------------


def build_spanning_tree(
    adj: dict[int, set[int]],
    accepted: list[PairwiseRegistration],
    ref_idx: int,
) -> list[dict]:
    """Build a maximum spanning tree rooted at *ref_idx*.

    Edge weight: Q = n_inlier × inlier_ratio × coverage

    Returns:
        List of tree edge dicts with ``parent``, ``child``, ``depth``,
        ``edge`` (PairwiseRegistration), and ``weight``.
    """
    n = len(adj)

    # Build weighted edge list (direction-independent)
    edge_map: dict[tuple[int, int], float] = {}
    for r in accepted:
        q = r.inliers * r.inlier_ratio * r.coverage
        key = tuple(sorted((r.idx_i, r.idx_j)))
        edge_map[key] = q

    # Prim's algorithm for max spanning tree
    # Start from ref_idx
    in_tree: set[int] = {ref_idx}
    tree_edges: list[dict] = []
    depth: dict[int, int] = {ref_idx: 0}
    parent: dict[int, int | None] = {ref_idx: None}

    while len(in_tree) < n:
        best_weight = -1.0
        best_edge = None
        best_node = None
        best_parent = None

        for i in in_tree:
            for j in adj[i]:
                if j in in_tree:
                    continue
                key = tuple(sorted((i, j)))
                w = edge_map.get(key, 0.0)
                if w > best_weight:
                    best_weight = w
                    best_edge = (i, j)
                    best_node = j
                    best_parent = i

        if best_edge is None:
            # Graph may be disconnected (should have been caught earlier)
            break

        in_tree.add(best_node)
        depth[best_node] = depth[best_parent] + 1
        parent[best_node] = best_parent
        tree_edges.append({
            "parent": best_parent,
            "child": best_node,
            "depth": depth[best_node],
            "weight": round(best_weight, 4),
        })

    return tree_edges


# ---------------------------------------------------------------------------
# Task 8 — Compose global world transforms
# ---------------------------------------------------------------------------


def compose_global_transforms(
    scenes: list[Scene],
    accepted: list[PairwiseRegistration],
    tree_edges: list[dict],
    ref_idx: int,
) -> list[np.ndarray]:
    """Compose world-correction matrices for every scene via the spanning tree.

    Reference: G_ref = I (identity).

    For each tree edge (parent → child):
        G_child = G_parent @ A_{parent←child}

    where A_{parent←child} is the pair's world-coordinate affine.

    If the stored pair is in reversed orientation (child→parent), the
    matrix is inverted before composition.

    Returns:
        List of 3×3 numpy arrays, ``G[i]`` for each scene.
    """
    n = len(scenes)
    G: list[np.ndarray | None] = [None] * n
    G[ref_idx] = np.eye(3, dtype=np.float64)

    # Build quick lookup: (i, j) → PairwiseRegistration
    pair_lookup: dict[tuple[int, int], PairwiseRegistration] = {}
    for r in accepted:
        pair_lookup[(r.idx_i, r.idx_j)] = r

    # Process in BFS order (tree edges are already in insertion order from Prim)
    # But we need to process by depth to ensure parent transforms exist
    # Sort by depth
    sorted_edges = sorted(tree_edges, key=lambda e: e["depth"])

    for edge in sorted_edges:
        p = edge["parent"]
        c = edge["child"]

        # Find the pair registration for (p, c) or (c, p)
        if (p, c) in pair_lookup:
            pair_reg = pair_lookup[(p, c)]
            A = pixel_affine_to_world(
                np.array(pair_reg.pair_pixel_matrix),
                pair_reg.pair_common_transform,
            )
        elif (c, p) in pair_lookup:
            pair_reg = pair_lookup[(c, p)]
            A_inv = pixel_affine_to_world(
                np.array(pair_reg.pair_pixel_matrix),
                pair_reg.pair_common_transform,
            )
            A = np.linalg.inv(A_inv)
        else:
            logger.warning("No pair registration found for edge %d→%d", p, c)
            A = np.eye(3)

        G[c] = G[p] @ A

    # Ensure all scenes have transforms
    for i in range(n):
        if G[i] is None:
            logger.warning("Scene %d has no global transform, using identity", i)
            G[i] = np.eye(3)

    return G


# ---------------------------------------------------------------------------
# Task 9 — Global consistency diagnostics
# ---------------------------------------------------------------------------


def global_consistency_diagnostics(
    accepted: list[PairwiseRegistration],
    G: list[np.ndarray],
    tree_edges: list[dict],
    pixel_size_m: float | None = None,
    *,
    pixel_size: float | None = None,
    pixel_size_x: float | None = None,
    pixel_size_y: float | None = None,
) -> list[dict]:
    """Evaluate global consistency using ALL accepted edges' inlier points.

    For each accepted pair (i, j) and every RANSAC inlier point k:
        P_i_world = pair_common_transform * ref_xy[k]
        P_j_world = pair_common_transform * tgt_xy[k]
        e_k = || G_i @ P_i_world - G_j @ P_j_world ||
        e_world_k = sqrt(dx**2 + dy**2), in CRS metres
        e_px_k = sqrt((dx/res_x)^2 + (dy/res_y)^2)

    Reports point-level median, RMSE, P90, P95, and max per edge in both
    explicit world-metre and pixel units. Pixel resolution must be supplied
    explicitly; the deprecated ``pixel_size`` keyword is accepted only as a
    compatibility alias and has no default.

    Returns:
        List of per-edge diagnostics dicts.
    """
    if pixel_size_m is not None and pixel_size is not None:
        raise ValueError("pass only one of pixel_size_m or deprecated pixel_size")
    if pixel_size_m is None:
        pixel_size_m = pixel_size
    if pixel_size_x is None and pixel_size_y is None:
        if pixel_size_m is None:
            raise ValueError("pixel_size_m is required for pixel metrics")
        pixel_size_x = pixel_size_m
        pixel_size_y = pixel_size_m
    elif pixel_size_x is None or pixel_size_y is None:
        raise ValueError("pixel_size_x and pixel_size_y must be provided together")

    pixel_size_x = float(pixel_size_x)
    pixel_size_y = float(pixel_size_y)
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("pixel sizes must be positive")
    isotropic_pixel_size_m = (
        pixel_size_x if np.isclose(pixel_size_x, pixel_size_y) else None
    )

    tree_pairs: set[tuple[int, int]] = set()
    for e in tree_edges:
        tree_pairs.add((e["parent"], e["child"]))
        tree_pairs.add((e["child"], e["parent"]))

    results = []
    for r in accepted:
        # Skip pairs with no inlier point data
        if r.inlier_ref_xy is None or r.inlier_tgt_xy is None:
            continue
        ref_pts = np.asarray(r.inlier_ref_xy)
        tgt_pts = np.asarray(r.inlier_tgt_xy)
        if len(ref_pts) == 0:
            continue

        # Convert pixel coords to world coords using pair's common-grid transform
        ref_world = np.array([
            r.pair_common_transform * (px, py)
            for px, py in ref_pts
        ])
        tgt_world = np.array([
            r.pair_common_transform * (px, py)
            for px, py in tgt_pts
        ])

        # Apply global transforms (G_i, G_j are 3×3 in world coords)
        Gi, Gj = G[r.idx_i], G[r.idx_j]
        # Homogeneous coords
        ref_h = np.column_stack([ref_world, np.ones(len(ref_world))])
        tgt_h = np.column_stack([tgt_world, np.ones(len(tgt_world))])

        ref_transformed = (Gi @ ref_h.T).T[:, :2]  # (N, 2)
        tgt_transformed = (Gj @ tgt_h.T).T[:, :2]  # (N, 2)

        # Per-point residuals
        dx_world = ref_transformed[:, 0] - tgt_transformed[:, 0]
        dy_world = ref_transformed[:, 1] - tgt_transformed[:, 1]

        dx_px = dx_world / pixel_size_x
        dy_px = dy_world / pixel_size_y
        errors_world_m = np.sqrt(dx_world**2 + dy_world**2)
        errors_px = np.sqrt(dx_px**2 + dy_px**2)

        world_median = float(np.median(errors_world_m))
        world_rmse = float(np.sqrt(np.mean(errors_world_m**2)))
        world_p90 = float(np.percentile(errors_world_m, 90))
        world_p95 = float(np.percentile(errors_world_m, 95))
        world_max = float(np.max(errors_world_m))
        pixel_median = float(np.median(errors_px))
        pixel_rmse = float(np.sqrt(np.mean(errors_px**2)))
        pixel_p90 = float(np.percentile(errors_px, 90))
        pixel_p95 = float(np.percentile(errors_px, 95))
        pixel_max = float(np.max(errors_px))

        is_tree = (r.idx_i, r.idx_j) in tree_pairs

        results.append({
            "idx_i": r.idx_i,
            "idx_j": r.idx_j,
            "in_tree": is_tree,
            "n_points": len(errors_px),
            "global_median_world_m": round(world_median, 4),
            "global_rmse_world_m": round(world_rmse, 4),
            "global_p90_world_m": round(world_p90, 4),
            "global_p95_world_m": round(world_p95, 4),
            "global_max_world_m": round(world_max, 4),
            "global_median_pixel": round(pixel_median, 4),
            "global_rmse_pixel": round(pixel_rmse, 4),
            "global_p90_pixel": round(pixel_p90, 4),
            "global_p95_pixel": round(pixel_p95, 4),
            "global_max_pixel": round(pixel_max, 4),
            "pixel_size_m": isotropic_pixel_size_m,
            # Deprecated aliases retained for legacy diagnostic consumers.
            "global_median_px": round(pixel_median, 4),
            "global_rmse_px": round(pixel_rmse, 4),
            "global_p90_px": round(pixel_p90, 4),
            "global_p95_px": round(pixel_p95, 4),
            "global_max_px": round(pixel_max, 4),
        })

    return results


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------


def save_global_registration_info(
    ref_info: dict,
    tree_edges: list[dict],
    G: list[np.ndarray],
    out_dir: Path,
) -> None:
    """Save global_registration.json and spanning_tree.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "global_registration.json", "w") as f:
        json.dump(ref_info, f, indent=2)

    tree_data = {
        "reference_index": ref_info["reference_index"],
        "edges": tree_edges,
        "n_edges": len(tree_edges),
    }
    with open(out_dir / "spanning_tree.json", "w") as f:
        json.dump(tree_data, f, indent=2)

    transforms_data = {
        "reference_index": ref_info["reference_index"],
        "transforms": [
            {"scene": i, "matrix": g.tolist()} for i, g in enumerate(G)
        ],
    }
    with open(out_dir / "global_transforms.json", "w") as f:
        json.dump(transforms_data, f, indent=2)

    logger.info("Global registration info saved")


def save_consistency_diagnostics(
    results: list[dict],
    out_dir: Path,
) -> None:
    """Save global_edge_consistency.csv and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV. Keep the deprecated *_px aliases so existing loop-diagnostic
    # readers remain compatible while the new explicit-unit fields are used by
    # the modern benchmark.
    csv_path = out_dir / "global_edge_consistency.csv"
    with open(csv_path, "w") as f:
        f.write(
            "idx_i,idx_j,in_tree,n_points,pixel_size_m,"
            "global_median_world_m,global_rmse_world_m,global_p90_world_m,"
            "global_p95_world_m,global_max_world_m,global_median_pixel,"
            "global_rmse_pixel,global_p90_pixel,global_p95_pixel,"
            "global_max_pixel,global_median_px,global_rmse_px,"
            "global_p90_px,global_p95_px,global_max_px\n"
        )
        for r in results:
            pixel_size_m = r.get("pixel_size_m", "")
            world_values = [
                r.get("global_median_world_m", ""),
                r.get("global_rmse_world_m", ""),
                r.get("global_p90_world_m", ""),
                r.get("global_p95_world_m", ""),
                r.get("global_max_world_m", ""),
            ]
            pixel_values = [
                r.get("global_median_pixel", r.get("global_median_px", "")),
                r.get("global_rmse_pixel", r.get("global_rmse_px", "")),
                r.get("global_p90_pixel", r.get("global_p90_px", "")),
                r.get("global_p95_pixel", r.get("global_p95_px", "")),
                r.get("global_max_pixel", r.get("global_max_px", "")),
            ]
            f.write(
                f"{r['idx_i']},{r['idx_j']},{r['in_tree']},"
                f"{r['n_points']},{pixel_size_m},"
                f"{','.join(map(str, world_values))},{','.join(map(str, pixel_values))},"
                f"{r['global_median_px']},{r['global_rmse_px']},"
                f"{r['global_p90_px']},{r['global_p95_px']},"
                f"{r['global_max_px']}\n"
            )

    # JSON
    with open(out_dir / "global_edge_consistency.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

    logger.info("Global consistency diagnostics saved: %d edges", len(results))
