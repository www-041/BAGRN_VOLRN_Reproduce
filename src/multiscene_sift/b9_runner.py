"""Lightweight B9 matcher runner built on the existing pairwise API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.multiscene_sift.global_registration import build_accepted_graph
from src.multiscene_sift.models import OverlapEdge, PairwiseRegistration, Scene
from src.multiscene_sift.pairwise import (
    SUPPORTED_MATCHERS,
    run_all_pairs,
    save_pairwise_summary,
)
from src.registration_benchmark.matchers.efficient_loftr import (
    EFFICIENT_LOFTR_UNAVAILABLE,
    is_efficient_loftr_available,
)


def run_b9_registration(
    frozen_config: dict,
    output_dir: str | Path,
    *,
    matcher: str,
    device: str = "auto",
    pair: tuple[int, int] | None = None,
    protocol_config_path: str | Path | None = None,
    pair_runner: Callable = run_all_pairs,
) -> dict:
    """Run one requested matcher over the frozen B9 geographic edges.

    This function deliberately delegates matching and geometry to the shared
    ``run_all_pairs``/``register_pair`` path. It stops at pairwise geometry;
    no global adjustment, radiometric normalization, or mosaicking is run.
    """
    matcher = str(matcher).strip().lower()
    if matcher not in SUPPORTED_MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; expected {SUPPORTED_MATCHERS}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenes = scenes_from_frozen_config(frozen_config)
    edges = edges_from_frozen_config(frozen_config)
    if pair is not None:
        requested_pair = tuple(sorted((int(pair[0]), int(pair[1]))))
        candidate_edges = {
            tuple(sorted((edge.idx_i, edge.idx_j))) for edge in edges
        }
        if requested_pair not in candidate_edges:
            raise ValueError(
                f"pair {requested_pair} is not a frozen candidate edge; "
                f"available={sorted(candidate_edges)}"
            )
        edges = [
            edge
            for edge in edges
            if tuple(sorted((edge.idx_i, edge.idx_j))) == requested_pair
        ]
    t0 = time.perf_counter()

    if matcher == "efficient_loftr" and not is_efficient_loftr_available():
        results = [
            _unavailable_result(edge, matcher, EFFICIENT_LOFTR_UNAVAILABLE)
            for edge in edges
        ]
        status = EFFICIENT_LOFTR_UNAVAILABLE
    else:
        results = pair_runner(
            scenes,
            edges,
            output,
            band=frozen_config["band"],
            match_max_side=int(frozen_config["registration"]["match_max_side"]),
            ransac_threshold=float(
                frozen_config["ransac"]["residual_threshold_px"]
            ),
            random_seed=int(frozen_config["registration"]["random_seed"]),
            matcher=matcher,
            device=device,
        )
        status = None

    save_pairwise_summary(results, output)
    graph = _write_accepted_graph(
        results,
        n_scenes=len(scenes),
        matcher=matcher,
        output=output,
        forced_status=status,
    )
    runtime = _write_runtime(results, matcher, device, t0, output, graph["status"])
    run_config = {
        "band": frozen_config["band"],
        "matcher": matcher,
        "device": device,
        "scene_manifest_indices": frozen_config["selection"]["manifest_indices"],
        "match_max_side": frozen_config["registration"]["match_max_side"],
        "protocol_config_path": (
            str(protocol_config_path) if protocol_config_path is not None else None
        ),
        "ransac": frozen_config["ransac"],
        "global_adjustment": False,
        "radiometric_normalization": False,
        "mosaic": False,
    }
    if pair is not None:
        run_config["selected_pair"] = list(requested_pair)
    _write_json(run_config, output / "run_config.json")
    return {"status": graph["status"], "accepted_graph": graph, "runtime": runtime}


def scenes_from_frozen_config(config: dict) -> list[Scene]:
    """Convert frozen scene metadata to the existing ``Scene`` model."""
    scenes = []
    for index, record in enumerate(config["scenes"]):
        pixel_x = float(record.get("pixel_size_x_m", record.get("resolution_x_m", 14.0)))
        pixel_y = float(record.get("pixel_size_y_m", record.get("resolution_y_m", 14.0)))
        transform = Affine(pixel_x, 0.0, float(record["left"]), 0.0, -pixel_y, float(record["top"]))
        crs = CRS.from_string(record["crs"]) if record.get("crs") else None
        scenes.append(
            Scene(
                index=index,
                name=record["scene_id"],
                directory=record.get("scene_dir", ""),
                band_paths={config["band"]: record["b9_path"]},
                crs=crs,
                transforms={config["band"]: transform},
                shapes={config["band"]: (int(record["height"]), int(record["width"]))},
                nodata={config["band"]: None},
                bounds={
                    config["band"]: (
                        float(record["left"]),
                        float(record["bottom"]),
                        float(record["right"]),
                        float(record["top"]),
                    )
                },
            )
        )
    return scenes


def edges_from_frozen_config(config: dict) -> list[OverlapEdge]:
    """Map manifest-index edges to the local 0..4 runner indices."""
    manifest_to_local = {
        manifest_index: local_index
        for local_index, manifest_index in enumerate(
            config["selection"]["manifest_indices"]
        )
    }
    edges = []
    for pair in config["graph_edges"]:
        i = manifest_to_local[pair["manifest_idx_i"]]
        j = manifest_to_local[pair["manifest_idx_j"]]
        edges.append(
            OverlapEdge(
                idx_i=i,
                idx_j=j,
                intersection_area=float(pair["intersection_area"]),
                overlap_ratio_i=float(pair["overlap_area_i_ratio"]),
                overlap_ratio_j=float(pair["overlap_area_j_ratio"]),
            )
        )
    return edges


def _write_accepted_graph(
    results: list[PairwiseRegistration],
    *,
    n_scenes: int,
    matcher: str,
    output: Path,
    forced_status: str | None,
) -> dict:
    accepted = [result for result in results if result.status == "OK"]
    adjacency = {index: set() for index in range(n_scenes)}
    for result in accepted:
        adjacency[result.idx_i].add(result.idx_j)
        adjacency[result.idx_j].add(result.idx_i)
    components = _connected_components(adjacency)

    if forced_status:
        status = forced_status
        error = forced_status
    else:
        try:
            build_accepted_graph(results, matcher_name=matcher)
            status = "CONNECTED" if len(components) <= 1 else "NETWORK_DISCONNECTED"
            error = None
        except RuntimeError as exc:
            status = "NETWORK_DISCONNECTED"
            error = str(exc)

    payload = {
        "status": status,
        "matcher": matcher,
        "n_scenes": n_scenes,
        "accepted_edges": [
            {"idx_i": r.idx_i, "idx_j": r.idx_j, "status": r.status}
            for r in accepted
        ],
        "accepted_edge_count": len(accepted),
        "connected_components": components,
        "error": error,
        "thresholds_unchanged": True,
    }
    _write_json(payload, output / "accepted_graph.json")
    return payload


def _write_runtime(
    results: list[PairwiseRegistration],
    matcher: str,
    device: str,
    started: float,
    output: Path,
    status: str,
) -> dict:
    payload = {
        "matcher": matcher,
        "device": device,
        "status": status,
        "matcher_runtime_sec": sum(float(r.matcher_runtime_sec) for r in results),
        "geometry_runtime_sec": sum(float(r.geometry_runtime_sec) for r in results),
        "total_runtime_sec": sum(float(r.runtime_sec) for r in results),
        "runner_wall_runtime_sec": time.perf_counter() - started,
        "peak_gpu_memory_mb": max(
            (float(r.peak_gpu_memory_mb) for r in results if r.peak_gpu_memory_mb is not None),
            default=None,
        ),
    }
    _write_json(payload, output / "runtime.json")
    return payload


def _unavailable_result(edge: OverlapEdge, matcher: str, status: str):
    return PairwiseRegistration(
        idx_i=edge.idx_i,
        idx_j=edge.idx_j,
        status=status,
        raw_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        coverage=0.0,
        residual_median=float("nan"),
        residual_rmse=float("nan"),
        residual_p95=float("nan"),
        pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pair_common_transform=None,
        runtime_sec=0.0,
        matcher=matcher,
    )


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    visited = set()
    components = []
    for start in adjacency:
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


def _write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
