"""Matcher-only five-scene benchmark for the frozen geometry pipeline.

This module evaluates SIFT, LoFTR, EfficientLoFTR, and LightGlue+DISK through
the existing pairwise shared-RANSAC path and the existing network diagnostics.
It deliberately stops before radiometric normalization and mosaicking.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from src.multiscene_sift.dataset import discover_five_scenes, save_manifest
from src.multiscene_sift.global_registration import (
    build_accepted_graph,
    build_spanning_tree,
    compose_global_transforms,
    global_consistency_diagnostics,
    save_consistency_diagnostics,
    save_global_registration_info,
    select_reference_scene,
)
from src.multiscene_sift.overlap_graph import (
    build_geographic_overlap_graph,
    save_overlap_graph,
)
from src.multiscene_sift.pairwise import SUPPORTED_MATCHERS, run_all_pairs
from src.multiscene_sift.summary import collect_registration_summary

logger = logging.getLogger(__name__)

MODERN_MATCHERS = ("sift", "loftr", "efficient_loftr", "lightglue_disk")


def run_modern_matcher_benchmark(
    input_root: str,
    output_dir: str,
    registration_band: str = "B14",
    matchers: tuple[str, ...] = MODERN_MATCHERS,
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    random_seed: int = 0,
    scene_names: list[str] | None = None,
    device: str = "auto",
) -> dict:
    """Run pairwise and network geometry evaluation for each matcher.

    No BAGRN, VOLRN, cloud-mask processing, radiometric normalization, or
    mosaic output is performed here.
    """
    requested = tuple(str(name).strip().lower() for name in matchers)
    unknown = set(requested) - set(SUPPORTED_MATCHERS)
    if unknown:
        raise ValueError(f"Unknown modern matcher(s): {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one matcher is required")

    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "input_root": input_root,
        "output_dir": str(out),
        "registration_band": registration_band,
        "matchers": list(requested),
        "match_max_side": match_max_side,
        "ransac_threshold": ransac_threshold,
        "random_seed": random_seed,
        "device": device,
        "radiometric_normalization": False,
        "mosaic": False,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    scenes, manifest = discover_five_scenes(
        input_root,
        scene_names,
        bands=(registration_band,),
    )
    save_manifest(manifest, out / "dataset_manifest.json")
    edges = build_geographic_overlap_graph(scenes, registration_band)
    save_overlap_graph(edges, out)

    method_summaries: dict[str, dict] = {}
    for matcher in requested:
        method_out = out / matcher
        method_out.mkdir(parents=True, exist_ok=True)
        logger.info("=== Modern matcher: %s ===", matcher)
        pairwise_results = run_all_pairs(
            scenes,
            edges,
            method_out,
            band=registration_band,
            match_max_side=match_max_side,
            ransac_threshold=ransac_threshold,
            random_seed=random_seed,
            matcher=matcher,
            device=device,
        )
        method_summaries[matcher] = _evaluate_network(
            scenes,
            pairwise_results,
            method_out,
            matcher,
            len(edges),
            random_seed,
        )

    comparison = {
        "status": "COMPLETE",
        "n_scenes": len(scenes),
        "geographic_edges": len(edges),
        "matchers": list(requested),
        "methods": method_summaries,
        "runtime_sec": time.perf_counter() - t0,
    }
    with open(out / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, allow_nan=False)
    _write_comparison_csv(out / "comparison.csv", method_summaries)
    _write_comparison_png(out / "comparison.png", method_summaries)
    return comparison


def _evaluate_network(
    scenes,
    pairwise_results,
    method_out: Path,
    matcher: str,
    geographic_edges: int,
    random_seed: int,
) -> dict:
    try:
        adj, accepted = build_accepted_graph(pairwise_results, matcher_name=matcher)
    except RuntimeError as exc:
        accepted_count = sum(1 for result in pairwise_results if result.status == "OK")
        summary = collect_registration_summary(
            matcher=matcher,
            random_seed=random_seed,
            geographic_edges=geographic_edges,
            pairwise_results=pairwise_results,
            consistency=[],
            bagrn_runtime_sec=0.0,
            volrn_runtime_sec=0.0,
            total_runtime_sec=sum(float(r.runtime_sec) for r in pairwise_results),
        )
        network = {
            **summary,
            "status": "DISCONNECTED",
            "graph_connected": False,
            "error": str(exc),
            "reference_index": None,
            "spanning_tree_edges": 0,
            "accepted_graph_edges": accepted_count,
        }
        _write_network_summary(method_out, network)
        return network

    ref_info = select_reference_scene(adj, accepted)
    reference_idx = ref_info["reference_index"]
    tree_edges = build_spanning_tree(adj, accepted, reference_idx)
    transforms = compose_global_transforms(scenes, accepted, tree_edges, reference_idx)
    save_global_registration_info(ref_info, tree_edges, transforms, method_out)
    consistency = global_consistency_diagnostics(
        accepted,
        transforms,
        tree_edges,
        pixel_size=1.0,
    )
    save_consistency_diagnostics(consistency, method_out)
    summary = collect_registration_summary(
        matcher=matcher,
        random_seed=random_seed,
        geographic_edges=geographic_edges,
        pairwise_results=pairwise_results,
        consistency=consistency,
        bagrn_runtime_sec=0.0,
        volrn_runtime_sec=0.0,
        total_runtime_sec=sum(float(r.runtime_sec) for r in pairwise_results),
    )
    network = {
        **summary,
        "status": "CONNECTED",
        "graph_connected": True,
        "error": None,
        "reference_index": reference_idx,
        "spanning_tree_edges": len(tree_edges),
        "accepted_graph_edges": len(accepted),
    }
    _write_network_summary(method_out, network)
    return network


def _write_network_summary(out: Path, summary: dict) -> None:
    with open(out / "network_summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)


def _write_comparison_csv(path: Path, summaries: dict[str, dict]) -> None:
    if not summaries:
        return
    fields = sorted({key for summary in summaries.values() for key in summary})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["matcher", *fields], extrasaction="ignore")
        writer.writeheader()
        for matcher, summary in summaries.items():
            writer.writerow({"matcher": matcher, **summary})


def _write_comparison_png(path: Path, summaries: dict[str, dict]) -> None:
    """Write a compact geometry/runtime comparison figure.

    The figure is descriptive only: it does not select a matcher or alter any
    registration result. ``None`` values are rendered as zero-height bars with
    a ``not available`` annotation so disconnected methods remain visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    metrics = (
        ("mean_pairwise_rmse_px", "Mean pairwise RMSE (px)"),
        ("mean_pairwise_p95_px", "Mean pairwise P95 (px)"),
        ("matching_runtime_sec", "Matcher runtime (s)"),
        ("peak_gpu_memory_mb", "Peak GPU memory (MB)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        values = []
        missing = []
        for index, name in enumerate(names):
            value = summaries[name].get(key)
            try:
                finite = value is not None and float(value) == float(value)
            except (TypeError, ValueError):
                finite = False
            values.append(float(value) if finite else 0.0)
            if not finite:
                missing.append(index)
        axis.bar(names, values, color="#4472C4")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        for index in missing:
            axis.text(index, 0.0, "N/A", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Modern matcher geometry benchmark (descriptive)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
