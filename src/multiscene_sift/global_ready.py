"""Strict loader adapting B9 Global-ready bundles to existing global code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.multiscene_sift.b9_runner import scenes_from_frozen_config
from src.multiscene_sift.geometry_artifacts import load_pair_geometry_bundle
from src.multiscene_sift.models import PairwiseRegistration


class GlobalInputIncompleteError(RuntimeError):
    """The run cannot supply the point-level Global input contract."""


class ProvenanceMismatchError(RuntimeError):
    """The run mixes protocol/configuration provenance."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_pairs(config: dict[str, Any]) -> set[tuple[int, int]]:
    mapping = {
        int(manifest): local
        for local, manifest in enumerate(config["selection"]["manifest_indices"])
    }
    return {
        tuple(sorted((mapping[int(edge["manifest_idx_i"])], mapping[int(edge["manifest_idx_j"])])))
        for edge in config["graph_edges"]
    }


def _float_or_nan(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _incomplete(message: str) -> GlobalInputIncompleteError:
    return GlobalInputIncompleteError(f"GLOBAL_INPUT_INCOMPLETE: {message}")


def _provenance(message: str) -> ProvenanceMismatchError:
    return ProvenanceMismatchError(f"PROVENANCE_MISMATCH: {message}")


def load_global_ready_run(
    run_dir: str | Path,
    frozen_config: dict[str, Any],
    protocol_config_path: str | Path,
) -> dict[str, Any]:
    """Load one rerun without reconstructing points from scalar summaries."""
    run = Path(run_dir)
    index_path = run / "geometry_index.json"
    if not index_path.is_file():
        raise _incomplete("POINT_DATA_UNRECOVERABLE_FROM_SUMMARY")
    required = [
        run / "run_config.json",
        run / "pairwise_summary.json",
        run / "accepted_graph.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise _incomplete("missing files: " + ", ".join(missing))

    expected_path = Path(protocol_config_path)
    if not expected_path.is_file():
        raise _provenance(f"protocol config not found: {expected_path}")
    expected_hash = _sha256(expected_path)
    expected_scale = frozen_config.get("registration", {}).get("match_max_side")
    if expected_scale != 1024:
        raise _provenance(f"frozen config match_max_side={expected_scale}, expected 1024")

    index = _read(index_path)
    run_config = _read(run / "run_config.json")
    if index.get("config_sha256") != expected_hash:
        raise _provenance("geometry_index config hash differs from frozen config")
    if index.get("match_max_side") != 1024 or run_config.get("match_max_side") != 1024:
        raise _provenance("run is not match_max_side=1024")
    recorded_path = run_config.get("protocol_config_path")
    if not recorded_path or Path(str(recorded_path)).name != expected_path.name:
        raise _provenance("run_config protocol_config_path differs from frozen config")
    if index.get("scene_manifest_indices") != frozen_config["selection"]["manifest_indices"]:
        raise _provenance("scene manifest indices differ")
    if index.get("config_sha256") != run_config.get("protocol_config_sha256", index.get("config_sha256")):
        # Older global-ready runs do not need a duplicate hash field, but if it
        # is present it must agree rather than silently override the index.
        if "protocol_config_sha256" in run_config:
            raise _provenance("run_config protocol hash differs from geometry index")

    summary = _read(run / "pairwise_summary.json")
    rows = summary.get("results")
    if not isinstance(rows, list):
        raise _incomplete("pairwise_summary.results is not a list")
    expected = _expected_pairs(frozen_config)
    observed = {
        tuple(sorted((int(row["idx_i"]), int(row["idx_j"]))))
        for row in rows
        if "idx_i" in row and "idx_j" in row
    }
    if observed != expected:
        missing_pairs = sorted(expected - observed)
        raise _incomplete(f"pairwise candidate set mismatch; missing={missing_pairs}")

    bundle_records = {
        tuple(sorted(map(int, record["pair"] ))): record
        for record in index.get("bundles", [])
    }
    failed_pairs = {
        tuple(sorted(map(int, record["pair"])))
        for record in index.get("failed", [])
    }
    scenes = scenes_from_frozen_config(frozen_config)
    results: list[PairwiseRegistration] = []
    for row in rows:
        pair = tuple(sorted((int(row["idx_i"]), int(row["idx_j"]))))
        status = str(row["status"])
        record = bundle_records.get(pair)
        bundle = None
        if record is not None:
            sidecar = run / record["sidecar"]
            try:
                bundle = load_pair_geometry_bundle(sidecar)
            except (OSError, ValueError, KeyError) as exc:
                raise _incomplete(f"invalid geometry bundle for pair {pair}: {exc}") from exc
            metadata = bundle["metadata"]
            if (
                metadata.get("config_sha256") != expected_hash
                or metadata.get("match_max_side") != 1024
                or metadata.get("matcher") != row.get("matcher")
                or metadata.get("coordinate_frame") != "pair_common_grid"
                or metadata.get("transform_direction") != "target_to_reference"
                or metadata.get("status") != status
                or metadata.get("accepted") != (status == "OK")
            ):
                raise _provenance(f"bundle metadata mismatch for pair {pair}")
            expected_scene_names = [scenes[pair[0]].name, scenes[pair[1]].name]
            actual_scene_names = [metadata["pair"]["scene_i"], metadata["pair"]["scene_j"]]
            if actual_scene_names != expected_scene_names:
                raise _provenance(f"bundle scene identity mismatch for pair {pair}")
        elif status == "OK":
            raise _incomplete(f"accepted pair {pair} has no geometry bundle")
        elif pair not in failed_pairs:
            raise _incomplete(f"non-OK pair {pair} is absent from geometry index failure list")

        results.append(PairwiseRegistration(
            idx_i=int(row["idx_i"]),
            idx_j=int(row["idx_j"]),
            status=status,
            raw_matches=int(row.get("raw_matches", 0)),
            inliers=int(row.get("inliers", 0)),
            inlier_ratio=float(row.get("inlier_ratio", 0.0)),
            coverage=float(row.get("coverage", 0.0)),
            residual_median=_float_or_nan(row.get("residual_median")),
            residual_rmse=_float_or_nan(row.get("residual_rmse")),
            residual_p95=_float_or_nan(row.get("residual_p95")),
            pair_pixel_matrix=(
                bundle["pair_pixel_matrix"].tolist() if bundle is not None else np.eye(3).tolist()
            ),
            pair_common_transform=(
                bundle["pair_common_transform"] if bundle is not None else None
            ),
            runtime_sec=float(row.get("runtime_sec", 0.0)),
            matcher=str(row.get("matcher", index.get("matcher", "unknown"))),
            matcher_runtime_sec=float(row.get("matcher_runtime_sec", 0.0)),
            geometry_runtime_sec=float(row.get("geometry_runtime_sec", 0.0)),
            peak_gpu_memory_mb=row.get("peak_gpu_memory_mb"),
            inlier_ref_xy=(bundle["inlier_ref_xy"] if bundle is not None else None),
            inlier_tgt_xy=(bundle["inlier_tgt_xy"] if bundle is not None else None),
        ))

    graph = _read(run / "accepted_graph.json")
    graph_edges = {
        tuple(sorted((int(item["idx_i"]), int(item["idx_j"]))))
        for item in graph.get("accepted_edges", [])
    }
    accepted_edges = {
        tuple(sorted((result.idx_i, result.idx_j)))
        for result in results
        if result.status == "OK"
    }
    if graph_edges != accepted_edges:
        raise _incomplete("accepted_graph does not match pairwise accepted edges")
    return {
        "run_dir": run,
        "scenes": scenes,
        "results": results,
        "accepted_results": [result for result in results if result.status == "OK"],
        "accepted_graph": graph,
        "geometry_index": index,
        "config_sha256": expected_hash,
        "protocol_config_path": expected_path,
    }
