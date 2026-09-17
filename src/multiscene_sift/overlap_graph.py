"""Geographic overlap graph from scene footprints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.multiscene_sift.models import OverlapEdge, Scene

logger = logging.getLogger(__name__)


def build_geographic_overlap_graph(
    scenes: list[Scene],
    registration_band: str = "B14",
) -> list[OverlapEdge]:
    """Build an overlap graph from geographic footprints (B14 band).

    An edge exists between scene *i* and scene *j* iff their bounding
    rectangles intersect with non-zero area.

    Returns:
        List of :class:`OverlapEdge` for all overlapping pairs (i < j).

    Raises:
        ValueError: If the resulting graph is disconnected.
    """
    n = len(scenes)
    # Extract footprint bounds for each scene
    footprints: list[tuple[float, float, float, float]] = []
    areas: list[float] = []
    for s in scenes:
        b = s.bounds[registration_band]
        # (left, bottom, right, top)
        w = b.right - b.left
        h = b.top - b.bottom
        footprints.append((b.left, b.bottom, b.right, b.top))
        areas.append(w * h)

    edges: list[OverlapEdge] = []

    for i in range(n):
        for j in range(i + 1, n):
            intersection = _intersection_area(footprints[i], footprints[j])
            if intersection <= 0:
                continue
            ratio_i = intersection / areas[i] if areas[i] > 0 else 0.0
            ratio_j = intersection / areas[j] if areas[j] > 0 else 0.0
            edges.append(OverlapEdge(
                idx_i=i, idx_j=j,
                intersection_area=intersection,
                overlap_ratio_i=ratio_i,
                overlap_ratio_j=ratio_j,
            ))

    # Check connectivity
    if not _is_connected(n, edges):
        components = _connected_components(n, edges)
        raise ValueError(
            f"Geographic overlap graph is DISCONNECTED. "
            f"Components: {components}"
        )

    return edges


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute intersection area of two axis-aligned rectangles."""
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if left >= right or bottom >= top:
        return 0.0
    return (right - left) * (top - bottom)


def _is_connected(n: int, edges: list[OverlapEdge]) -> bool:
    """Check if the overlap graph is connected."""
    if n <= 1:
        return True
    adj = {i: set() for i in range(n)}
    for e in edges:
        adj[e.idx_i].add(e.idx_j)
        adj[e.idx_j].add(e.idx_i)
    visited = set()
    stack = [0]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        stack.extend(adj[v] - visited)
    return len(visited) == n


def _connected_components(
    n: int, edges: list[OverlapEdge],
) -> list[list[int]]:
    """Return list of connected components."""
    adj = {i: set() for i in range(n)}
    for e in edges:
        adj[e.idx_i].add(e.idx_j)
        adj[e.idx_j].add(e.idx_i)
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


def save_overlap_graph(
    edges: list[OverlapEdge],
    out_dir: str | Path,
) -> None:
    """Save overlap graph as CSV and JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "overlap_graph.csv"
    with open(csv_path, "w") as f:
        f.write("idx_i,idx_j,intersection_area,overlap_ratio_i,overlap_ratio_j\n")
        for e in edges:
            f.write(f"{e.idx_i},{e.idx_j},{e.intersection_area:.1f},"
                    f"{e.overlap_ratio_i:.4f},{e.overlap_ratio_j:.4f}\n")

    # JSON
    json_path = out / "overlap_graph.json"
    data = {
        "edges": [
            {
                "idx_i": e.idx_i,
                "idx_j": e.idx_j,
                "intersection_area": e.intersection_area,
                "overlap_ratio_i": e.overlap_ratio_i,
                "overlap_ratio_j": e.overlap_ratio_j,
            }
            for e in edges
        ],
        "n_edges": len(edges),
        "connected": True,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Overlap graph saved: %d edges", len(edges))