"""Footprint-only recommendation of a five-scene B9 validation subgraph."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path


SMALL_OVERLAP_THRESHOLD = 0.10
SCORING_FORMULA = (
    "score = 3 * cycle_rank + 2 * min_degree + mean_overlap "
    "- small_overlap_penalty; "
    "small_overlap_penalty = sum(max(0, 0.10 - edge_ratio))"
)


def rank_five_scene_candidates(
    records: list[dict],
    pairs: list[dict],
    *,
    top_n: int = 5,
) -> list[dict]:
    """Return the highest-scoring hard-valid five-scene combinations."""
    pair_map = {
        (pair["idx_i"], pair["idx_j"]): pair
        for pair in pairs
        if pair["has_overlap"]
    }
    candidates = []
    for combo in itertools.combinations(range(len(records)), 5):
        combo_set = set(combo)
        edges = [
            pair for (idx_i, idx_j), pair in pair_map.items()
            if idx_i in combo_set and idx_j in combo_set
        ]
        degrees = {index: 0 for index in combo}
        adjacency = {index: set() for index in combo}
        for edge in edges:
            i, j = edge["idx_i"], edge["idx_j"]
            degrees[i] += 1
            degrees[j] += 1
            adjacency[i].add(j)
            adjacency[j].add(i)
        if not _is_connected(combo[0], adjacency):
            continue
        cycle_rank = len(edges) - len(combo) + 1
        if len(edges) < 6 or cycle_rank < 1 or min(degrees.values()) < 1:
            continue
        ratios = [float(edge["symmetric_overlap_ratio"]) for edge in edges]
        mean_overlap = sum(ratios) / len(ratios)
        small_overlap_penalty = sum(
            max(0.0, SMALL_OVERLAP_THRESHOLD - ratio) for ratio in ratios
        )
        candidate = {
            "scene_indices": list(combo),
            "scene_ids": [records[index]["scene_id"] for index in combo],
            "edge_count": len(edges),
            "cycle_rank": cycle_rank,
            "min_degree": min(degrees.values()),
            "degree_sequence": [degrees[index] for index in combo],
            "mean_overlap": mean_overlap,
            "small_overlap_penalty": small_overlap_penalty,
            "score": 3 * cycle_rank + 2 * min(degrees.values()) + mean_overlap - small_overlap_penalty,
            "edge_pairs": [[edge["scene_i"], edge["scene_j"]] for edge in edges],
        }
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: (-candidate["score"], candidate["scene_indices"]))
    return candidates[:top_n]


def write_candidate_outputs(
    records: list[dict],
    candidates: list[dict],
    output_dir: str | Path,
) -> dict:
    """Write Top-5 CSV/JSON/PNG recommendation artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recommended = candidates[0] if candidates else None
    payload = {
        "status": "COMPLETE" if recommended else "NO_VALID_FIVE_SCENE",
        "n_scenes": len(records),
        "scoring_formula": SCORING_FORMULA,
        "hard_constraints": {
            "scene_count": 5,
            "induced_graph_connected": True,
            "minimum_edge_count": 6,
            "minimum_cycle_rank": 1,
            "minimum_degree": 1,
        },
        "recommended": recommended,
        "top_candidates": candidates,
    }
    fields = [
        "rank", "scene_indices", "scene_ids", "edge_count", "cycle_rank",
        "min_degree", "degree_sequence", "mean_overlap",
        "small_overlap_penalty", "score", "edge_pairs",
    ]
    with open(output_dir / "03_five_scene_candidates.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            row = {"rank": rank, **candidate}
            for field in ("scene_indices", "scene_ids", "degree_sequence", "edge_pairs"):
                row[field] = ";".join(
                    ",".join(value) if isinstance(value, list) and value and isinstance(value[0], str) else str(value)
                    for value in candidate[field]
                ) if field == "edge_pairs" else ";".join(map(str, candidate[field]))
            writer.writerow(row)
    with open(output_dir / "03_recommended_five_scene.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _draw_recommended_subgraph(records, recommended, output_dir / "03_recommended_subgraph.png")
    return payload


def _is_connected(start: int, adjacency: dict[int, set[int]]) -> bool:
    visited = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    return len(visited) == len(adjacency)


def _draw_recommended_subgraph(records: list[dict], candidate: dict | None, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.Graph()
    if candidate:
        graph.add_nodes_from(candidate["scene_indices"])
        for scene_i, scene_j in candidate["edge_pairs"]:
            i = next(index for index in candidate["scene_indices"] if records[index]["scene_id"] == scene_i)
            j = next(index for index in candidate["scene_indices"] if records[index]["scene_id"] == scene_j)
            graph.add_edge(i, j)
    figure, axis = plt.subplots(figsize=(9, 7))
    positions = nx.spring_layout(graph, seed=42) if graph.nodes else {}
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        labels={index: str(index) for index in graph.nodes},
        node_color="lightgreen",
        node_size=900,
        font_size=10,
    )
    axis.set_title("Recommended B9 five-scene induced subgraph")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
