"""Metadata-only B9 footprint overlap graph and audit artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.multiscene_sift.overlap_graph import _intersection_area


PAIR_FIELDS = [
    "idx_i",
    "idx_j",
    "scene_i",
    "scene_j",
    "intersection_area",
    "overlap_area_i_ratio",
    "overlap_area_j_ratio",
    "symmetric_overlap_ratio",
    "has_overlap",
]


def build_b9_overlap_graph(records: list[dict]) -> list[dict]:
    """Return one explicit overlap row for every scene pair.

    ``symmetric_overlap_ratio`` is defined as
    ``intersection_area / min(area_i, area_j)``. The rows retain non-overlap
    pairs so the audit can account for the complete 13-scene matrix.
    """
    pairs = []
    for idx_i, record_i in enumerate(records):
        bounds_i = _record_bounds(record_i)
        area_i = _area(bounds_i)
        for idx_j in range(idx_i + 1, len(records)):
            record_j = records[idx_j]
            bounds_j = _record_bounds(record_j)
            area_j = _area(bounds_j)
            intersection = _intersection_area(bounds_i, bounds_j)
            has_overlap = intersection > 0.0
            ratio_i = intersection / area_i if area_i > 0 else 0.0
            ratio_j = intersection / area_j if area_j > 0 else 0.0
            smaller_area = min(area_i, area_j)
            pairs.append({
                "idx_i": idx_i,
                "idx_j": idx_j,
                "scene_i": record_i["scene_id"],
                "scene_j": record_j["scene_id"],
                "intersection_area": intersection,
                "overlap_area_i_ratio": ratio_i,
                "overlap_area_j_ratio": ratio_j,
                "symmetric_overlap_ratio": (
                    intersection / smaller_area if smaller_area > 0 else 0.0
                ),
                "has_overlap": has_overlap,
            })
    return pairs


def connected_components(n_scenes: int, pairs: list[dict]) -> list[list[int]]:
    """Return connected components using only pairs with geographic overlap."""
    adjacency = {index: set() for index in range(n_scenes)}
    for pair in pairs:
        if pair["has_overlap"]:
            adjacency[pair["idx_i"]].add(pair["idx_j"])
            adjacency[pair["idx_j"]].add(pair["idx_i"])
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_scenes):
        if start in visited:
            continue
        stack = [start]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        components.append(sorted(component))
    return components


def write_overlap_outputs(
    records: list[dict],
    pairs: list[dict],
    output_dir: str | Path,
) -> dict:
    """Write the four Task 4 overlap artifacts and return graph metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = connected_components(len(records), pairs)
    edges = [pair for pair in pairs if pair["has_overlap"]]
    payload = {
        "n_scenes": len(records),
        "n_pairs": len(pairs),
        "n_edges": len(edges),
        "connected": len(components) <= 1,
        "connected_components": components,
        "symmetric_overlap_definition": "intersection_area / min(area_i, area_j)",
        "scene_ids": [record["scene_id"] for record in records],
        "pairs": pairs,
        "edges": edges,
    }

    with open(output_dir / "02_overlap_edges.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)
    with open(output_dir / "02_overlap_graph.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    scene_ids = [record["scene_id"] for record in records]
    matrix = [[0.0 for _ in scene_ids] for _ in scene_ids]
    for pair in edges:
        i, j = pair["idx_i"], pair["idx_j"]
        matrix[i][j] = matrix[j][i] = pair["symmetric_overlap_ratio"]
    with open(output_dir / "02_overlap_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", *scene_ids])
        for scene_id, row in zip(scene_ids, matrix):
            writer.writerow([scene_id, *row])

    _draw_overlap_graph(records, edges, output_dir / "02_overlap_graph.png")
    return payload


def _record_bounds(record: dict) -> tuple[float, float, float, float]:
    return (
        float(record["left"]),
        float(record["bottom"]),
        float(record["right"]),
        float(record["top"]),
    )


def _area(bounds: tuple[float, float, float, float]) -> float:
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _draw_overlap_graph(records: list[dict], edges: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(records)))
    for edge in edges:
        graph.add_edge(
            edge["idx_i"],
            edge["idx_j"],
            label=f"{edge['symmetric_overlap_ratio']:.3f}",
        )
    figure, axis = plt.subplots(figsize=(10, 8))
    positions = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        labels={i: str(i) for i in range(len(records))},
        node_color="lightblue",
        node_size=800,
        font_size=9,
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels={(u, v): data["label"] for u, v, data in graph.edges(data=True)},
        ax=axis,
        font_size=7,
    )
    axis.set_title("B9 geographic overlap graph (symmetric ratio)")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
