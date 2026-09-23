"""Offline global geometric-adjustment diagnostics for an existing five-scene run.

This module deliberately consumes saved artifacts only.  It never invokes a
matcher or reconstructs pairwise correspondences from imagery.
"""

from __future__ import annotations

import csv
import json
import csv
from pathlib import Path
from typing import Any

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    pd = None


class _SimpleLoc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, key):
        row, column = key
        return self._frame._rows[row][column]


class _SimpleDataFrame:
    """Small pandas-compatible subset used when pandas is unavailable."""

    def __init__(self, rows: list[dict], columns: list[str]):
        self._rows = rows
        self.columns = columns
        self.loc = _SimpleLoc(self)

    def __getitem__(self, column):
        return [row[column] for row in self._rows]

    def __len__(self):
        return len(self._rows)

    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError("fallback dataframe only supports orient='records'")
        return [dict(row) for row in self._rows]

    def to_csv(self, path, index=False):
        if index:
            raise ValueError("fallback dataframe does not support index output")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self._rows)

def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _edge_key(i: int, j: int) -> tuple[int, int]:
    return min(i, j), max(i, j)


def _point_artifact_edges(run_dir: Path) -> set[tuple[int, int]]:
    """Find edges explicitly represented by stored point-level CSVs.

    A CSV is accepted only when it contains ``edge_i``/``edge_j`` columns.
    A generic residual CSV without edge identity is intentionally not guessed
    to belong to an accepted edge.
    """
    found: set[tuple[int, int]] = set()
    for path in run_dir.rglob("*.csv"):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                point_fields = {"point_id", "ref_x", "ref_y", "tgt_x", "tgt_y"}
                if not {"edge_i", "edge_j"}.issubset(fields) or not point_fields.issubset(fields):
                    continue
                for row in reader:
                    if row.get("edge_i") in (None, "") or row.get("edge_j") in (None, ""):
                        continue
                    found.add(_edge_key(int(row["edge_i"]), int(row["edge_j"])))
        except (OSError, ValueError, UnicodeError):
            continue
    return found


def discover_five_scene_adjustment_inputs(five_scene_run_dir: str | Path) -> dict:
    """Audit saved five-scene inputs before any global adjustment.

    The returned manifest is ``READY`` only when every accepted direct edge
    has explicitly recoverable point-level observations.  Missing point data
    is a hard stop; this function has no rematching fallback by design.
    """
    run_dir = Path(five_scene_run_dir)
    dataset = _read_json(run_dir / "dataset_manifest.json", {})
    registration = _read_json(run_dir / "global_registration.json", {})
    spanning = _read_json(run_dir / "spanning_tree.json", {})
    pairwise = _read_json(run_dir / "pairwise_summary.json", {})
    results = pairwise.get("results", []) if isinstance(pairwise, dict) else []
    accepted = [row for row in results if row.get("status") == "OK"]
    point_edges = _point_artifact_edges(run_dir)

    accepted_edges = []
    for row in accepted:
        i, j = int(row["idx_i"]), int(row["idx_j"])
        key = _edge_key(i, j)
        accepted_edges.append(
            {
                "edge": [i, j],
                "status": row.get("status"),
                "inliers": int(row.get("inliers", 0)),
                "point_artifact_status": "READY" if key in point_edges else "MISSING",
            }
        )

    missing = [row["edge"] for row in accepted_edges if row["point_artifact_status"] == "MISSING"]
    return {
        "status": "READY" if not missing else "INSUFFICIENT_EXISTING_ARTIFACTS",
        "rematch_allowed": False,
        "run_dir": str(run_dir),
        "scene_indices": [int(scene["index"]) for scene in dataset.get("scenes", [])],
        "reference_idx": registration.get("reference_index", spanning.get("reference_index")),
        "spanning_tree_edges": spanning.get("edges", []),
        "accepted_edges": accepted_edges,
        "missing_point_edges": missing,
    }


def write_adjustment_input_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Write the Task 0 audit without changing the source run artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "00_adjustment_input_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_historical_pair_baselines(five_scene_run_dir: str | Path) -> dict:
    """Freeze the historical pair summary without re-running registration."""
    run_dir = Path(five_scene_run_dir)
    summary_path = run_dir / "pairwise_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    payload = _read_json(summary_path, {})
    rows = payload.get("results", []) if isinstance(payload, dict) else []

    def convert(row: dict) -> dict:
        i, j = int(row["idx_i"]), int(row["idx_j"])
        return {
            "edge": [i, j],
            "status": row.get("status"),
            "raw_matches": int(row.get("raw_matches", 0)),
            "inliers": int(row.get("inliers", 0)),
            "inlier_ratio": float(row.get("inlier_ratio", 0.0)),
            "coverage": float(row.get("coverage", 0.0)),
            "rmse_px": row.get("residual_rmse"),
            "p95_px": row.get("residual_p95"),
            "affine_matrix": row.get("pixel_matrix"),
            "artifact_sources": [str(summary_path)],
        }

    converted = [convert(row) for row in rows]
    return {
        "run_dir": str(run_dir),
        "source": str(summary_path),
        "accepted_edges": [row for row in converted if row["status"] == "OK"],
        "rejected_edges": [row for row in converted if row["status"] != "OK"],
    }


def write_historical_pair_baselines(baselines: dict, output_dir: str | Path) -> Path:
    """Persist the frozen historical pair baseline artifact."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "00_historical_pair_baselines.json"
    path.write_text(json.dumps(baselines, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_frozen_registration_config(five_scene_run_dir: str | Path) -> dict:
    """Recover the exact historical registration configuration.

    Values absent from the run manifest are read only from the production
    call-chain constants and implementations, never from this function's
    defaults.
    """
    run_dir = Path(five_scene_run_dir)
    config_path = run_dir / "run_config.json"
    dataset_path = run_dir / "dataset_manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not dataset_path.is_file():
        raise ValueError("FROZEN_CONFIG_INCOMPLETE: dataset_manifest.json")
    run_config = _read_json(config_path, {})
    required_run_keys = {
        "matcher", "registration_band", "match_max_side", "random_seed",
        "ransac_threshold",
    }
    missing = sorted(key for key in required_run_keys if key not in run_config)
    if missing:
        raise ValueError(
            "FROZEN_CONFIG_INCOMPLETE: missing historical keys " + ", ".join(missing)
        )

    from src.multiscene_sift.pairwise import (
        LOWE_RATIO,
        MIN_INLIER_RATIO,
        MIN_INLIERS,
        RANSAC_MAX_TRIALS,
        SIFT_NFEATURES,
    )

    dataset = _read_json(dataset_path, {})
    nodata_by_band: dict[str, list[float | None]] = {}
    for scene in dataset.get("scenes", []):
        for band, metadata in scene.get("bands", {}).items():
            if "nodata" not in metadata:
                raise ValueError(f"FROZEN_CONFIG_INCOMPLETE: nodata for {band}")
            value = metadata["nodata"]
            if value not in nodata_by_band.setdefault(band, []):
                nodata_by_band[band].append(value)

    return {
        "matcher": str(run_config["matcher"]).upper(),
        "registration_band": run_config["registration_band"],
        "match_max_side": int(run_config["match_max_side"]),
        "sift_nfeatures": int(SIFT_NFEATURES),
        "lowe_ratio": float(LOWE_RATIO),
        "mutual_check": True,
        "ransac_model": "AffineTransform",
        "ransac_residual_threshold": float(run_config["ransac_threshold"]),
        "ransac_max_trials": int(RANSAC_MAX_TRIALS),
        "minimum_inliers": int(MIN_INLIERS),
        "minimum_inlier_ratio": float(MIN_INLIER_RATIO),
        "seed": int(run_config["random_seed"]),
        "nodata_policy": nodata_by_band,
        "valid_mask_policy": "nodata exclusion plus finite-value mask",
        "pair_common_grid_policy": {
            "resolution": "finer input resolution",
            "extent": "union extent",
            "data_resampling": "bilinear",
            "mask_resampling": "nearest",
            "match_view": "overlap crop, percentile stretch 2-98, shared max-side resize",
        },
        "source_artifacts": [str(config_path), str(dataset_path)],
        "source_code": [
            "src/multiscene_sift/pairwise.py",
            "src/registration_benchmark/matchers/sift.py",
            "src/registration_benchmark/geometry.py",
            "src/registration_benchmark/common_grid.py",
        ],
    }


def write_frozen_registration_config(config: dict, output_dir: str | Path) -> Path:
    """Persist the frozen registration configuration artifact."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "01_frozen_registration_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([arr, np.ones(len(arr))])
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def evaluate_edge_point_residuals(
    global_transforms: dict[int, np.ndarray],
    edge_observations: dict,
) -> pd.DataFrame:
    """Evaluate every saved inlier in the supplied global frame."""
    rows: list[dict] = []
    for edge, observation in edge_observations.items():
        i, j = (int(edge[0]), int(edge[1]))
        points_i = np.asarray(observation["x_i"], dtype=np.float64)
        points_j = np.asarray(observation["x_j"], dtype=np.float64)
        if points_i.shape != points_j.shape or points_i.ndim != 2 or points_i.shape[1] != 2:
            raise ValueError(f"invalid point arrays for edge {i}-{j}")
        pair_to_world = np.asarray(
            observation.get("pair_common_transform", np.eye(3)), dtype=np.float64
        )
        global_i = _transform_points(global_transforms[i] @ pair_to_world, points_i)
        global_j = _transform_points(global_transforms[j] @ pair_to_world, points_j)
        delta_world = global_i - global_j
        pixel_size = float(observation.get("pixel_size", 1.0))
        delta = delta_world / pixel_size
        errors = np.linalg.norm(delta, axis=1)
        for point_id, (pi, pj, gi, gj, d, error) in enumerate(
            zip(points_i, points_j, global_i, global_j, delta, errors)
        ):
            rows.append({
                "edge_i": i, "edge_j": j, "point_id": point_id,
                "x_i": float(pi[0]), "y_i": float(pi[1]),
                "x_j": float(pj[0]), "y_j": float(pj[1]),
                "global_x_i": float(gi[0]), "global_y_i": float(gi[1]),
                "global_x_j": float(gj[0]), "global_y_j": float(gj[1]),
                "dx": float(d[0]), "dy": float(d[1]),
                "residual_px": float(error),
                "is_tree_edge": bool(observation.get("is_tree_edge", False)),
            })
    columns = [
        "edge_i", "edge_j", "point_id", "x_i", "y_i", "x_j", "y_j",
        "global_x_i", "global_y_i", "global_x_j", "global_y_j", "dx", "dy",
        "residual_px", "is_tree_edge",
    ]
    if pd is not None:
        return pd.DataFrame(rows, columns=columns)
    return _SimpleDataFrame(rows, columns)


def _residual_stats(rows: list[dict]) -> dict:
    errors = np.asarray([row["residual_px"] for row in rows], dtype=np.float64)
    if errors.size == 0:
        return {
            "n_points": 0, "median_px": None, "rmse_px": None, "p95_px": None,
            "max_px": None, "dx_mean": None, "dy_mean": None,
        }
    return {
        "n_points": int(errors.size),
        "median_px": float(np.median(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
        "dx_mean": float(np.mean([row["dx"] for row in rows])),
        "dy_mean": float(np.mean([row["dy"] for row in rows])),
    }


def summarize_network_residuals(point_residuals) -> dict:
    """Return point-weighted, edge-balanced, and explicitly split aggregates."""
    rows = point_residuals.to_dict(orient="records")
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["edge_i"]), int(row["edge_j"])), []).append(row)
    edge_summary = []
    for (i, j), edge_rows in sorted(grouped.items()):
        stats = _residual_stats(edge_rows)
        edge_summary.append({"edge_i": i, "edge_j": j,
                             "is_tree_edge": bool(edge_rows[0]["is_tree_edge"]), **stats})

    def aggregate(selected):
        selected_rows = [row for edge_rows in selected for row in edge_rows]
        stats = _residual_stats(selected_rows)
        selected_stats = [_residual_stats(edge_rows) for edge_rows in selected]
        rmses = [item["rmse_px"] for item in selected_stats if item["rmse_px"] is not None]
        p95s = [item["p95_px"] for item in selected_stats if item["p95_px"] is not None]
        return {
            **stats,
            "edge_count": len(selected),
            "mean_edge_rmse_px": float(np.mean(rmses)) if rmses else None,
            "mean_edge_p95_px": float(np.mean(p95s)) if p95s else None,
            "max_edge_rmse_px": float(np.max(rmses)) if rmses else None,
            "max_edge_p95_px": float(np.max(p95s)) if p95s else None,
        }

    tree = [edge_rows for edge_rows in grouped.values()
            if edge_rows[0]["is_tree_edge"]]
    non_tree = [edge_rows for edge_rows in grouped.values()
                if not edge_rows[0]["is_tree_edge"]]
    zero_one = grouped.get((0, 1), [])
    return {
        "metric_frame": "mst_global_frame",
        "point_weighted": _residual_stats(rows),
        "edge_balanced": aggregate(list(grouped.values())),
        "zero_one": _residual_stats(zero_one),
        "tree_edges": aggregate(tree),
        "non_tree_edges": aggregate(non_tree),
        "per_edge": edge_summary,
    }


def write_mst_residual_artifacts(point_residuals, summary: dict, output_dir: str | Path) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_path = out_dir / "01_mst_point_residuals.csv"
    point_residuals.to_csv(point_path, index=False)
    edge_path = out_dir / "01_mst_edge_summary.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        rows = summary["per_edge"]
        fields = ["edge_i", "edge_j", "is_tree_edge", "n_points", "median_px",
                  "rmse_px", "p95_px", "max_px", "dx_mean", "dy_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    network_path = out_dir / "01_mst_network_summary.json"
    network_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"point_residuals": point_path, "edge_summary": edge_path, "network_summary": network_path}


def load_mst_global_transforms(path: str | Path) -> dict[int, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(item["scene"]): np.asarray(item["matrix"], dtype=np.float64)
            for item in payload["transforms"]}


def load_frozen_edge_observations(
    inliers_csv: str | Path,
    spanning_tree_json: str | Path,
    dataset_manifest: str | Path | None = None,
) -> dict:
    tree_payload = json.loads(Path(spanning_tree_json).read_text(encoding="utf-8"))
    tree_edges = {tuple(sorted((int(item["parent"]), int(item["child"]))))
                  for item in tree_payload["edges"]}
    pair_transforms = {}
    if dataset_manifest is not None:
        manifest = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        bounds = {int(scene["index"]): scene["bands"]["B14"]["bounds"]
                  for scene in manifest["scenes"]}
        resolution = float(manifest["scenes"][0]["bands"]["B14"]["resolution"][0])
        for i, j in {tuple(sorted((int(row["edge_i"]), int(row["edge_j"]))))
                     for row in csv.DictReader(Path(inliers_csv).open(encoding="utf-8"))}:
            left = min(bounds[i][0], bounds[j][0])
            top = max(bounds[i][3], bounds[j][3])
            pair_transforms[(i, j)] = np.array([
                [resolution, 0.0, left], [0.0, -resolution, top], [0.0, 0.0, 1.0]
            ], dtype=np.float64)
    grouped: dict[tuple[int, int], dict[str, list]] = {}
    with Path(inliers_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            edge = (int(row["edge_i"]), int(row["edge_j"]))
            observation = grouped.setdefault(edge, {"x_i": [], "x_j": []})
            observation["x_i"].append([float(row["x_i"]), float(row["y_i"])])
            observation["x_j"].append([float(row["x_j"]), float(row["y_j"])])
    for edge, observation in grouped.items():
        observation["x_i"] = np.asarray(observation["x_i"], dtype=np.float64)
        observation["x_j"] = np.asarray(observation["x_j"], dtype=np.float64)
        observation["is_tree_edge"] = tuple(sorted(edge)) in tree_edges
        if pair_transforms:
            observation["pair_common_transform"] = pair_transforms[tuple(sorted(edge))]
            observation["pixel_size"] = resolution
    return grouped


def build_translation_adjustment_system(
    mst_global_transforms: dict[int, np.ndarray],
    edge_observations: dict,
    reference_idx: int,
    weight_mode: str = "equal_edge",
) -> dict:
    """Build the unweighted linear system for global-frame node translations."""
    if reference_idx not in mst_global_transforms:
        raise ValueError(f"reference scene {reference_idx} is missing from transforms")
    if weight_mode != "equal_edge":
        raise ValueError(f"unsupported translation weight mode: {weight_mode}")

    scenes = sorted(int(scene) for scene in mst_global_transforms)
    unknown_scene_order = [scene for scene in scenes if scene != reference_idx]
    scene_set = set(scenes)
    adjacency = {scene: set() for scene in scenes}
    rows = []
    rhs = []
    point_weights = []
    edge_weight_definition = "each point in edge e has weight 1 / N_e"

    def add_coefficients(row, scene_i, scene_j, sign):
        if scene_i != reference_idx:
            row[2 * unknown_scene_order.index(scene_i) + sign] += 1.0
        if scene_j != reference_idx:
            row[2 * unknown_scene_order.index(scene_j) + sign] -= 1.0

    for edge, observation in sorted(edge_observations.items()):
        i, j = int(edge[0]), int(edge[1])
        if i not in scene_set or j not in scene_set or i == j:
            raise ValueError(f"invalid edge {i}-{j} for transform scene set")
        points_i = np.asarray(observation["x_i"], dtype=np.float64)
        points_j = np.asarray(observation["x_j"], dtype=np.float64)
        if points_i.shape != points_j.shape or points_i.ndim != 2 or points_i.shape[1] != 2:
            raise ValueError(f"invalid point arrays for edge {i}-{j}")
        adjacency[i].add(j)
        adjacency[j].add(i)
        pair_to_world = np.asarray(
            observation.get("pair_common_transform", np.eye(3)), dtype=np.float64
        )
        qi = _transform_points(mst_global_transforms[i] @ pair_to_world, points_i)
        qj = _transform_points(mst_global_transforms[j] @ pair_to_world, points_j)
        weight = 1.0 / len(points_i) if len(points_i) else 0.0
        for point_i, point_j in zip(qi, qj):
            for axis in (0, 1):
                row = np.zeros(2 * len(unknown_scene_order), dtype=np.float64)
                add_coefficients(row, i, j, axis)
                rows.append(row)
                rhs.append(float(-(point_i[axis] - point_j[axis])))
                point_weights.append(weight)

    reachable = {reference_idx}
    frontier = [reference_idx]
    while frontier:
        scene = frontier.pop()
        for neighbour in adjacency[scene]:
            if neighbour not in reachable:
                reachable.add(neighbour)
                frontier.append(neighbour)
    if reachable != scene_set:
        raise ValueError(f"translation graph is disconnected from reference {reference_idx}")

    matrix = np.asarray(rows, dtype=np.float64).reshape((-1, 2 * len(unknown_scene_order)))
    vector = np.asarray(rhs, dtype=np.float64)
    return {
        "A": matrix,
        "b": vector,
        "point_weights": np.asarray(point_weights, dtype=np.float64)[::2],
        "unknown_scene_order": unknown_scene_order,
        "reference_idx": int(reference_idx),
        "edge_weight_definition": edge_weight_definition,
        "rank_expectation": 2 * len(unknown_scene_order),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)) if matrix.size else None,
    }


def write_translation_system_summary(system: dict, output_path: str | Path) -> Path:
    """Write matrix metadata without embedding the potentially large matrix."""
    summary = {
        "unknown_scene_order": system["unknown_scene_order"],
        "reference_idx": system["reference_idx"],
        "n_rows": int(system["A"].shape[0]),
        "n_columns": int(system["A"].shape[1]),
        "rank": system["rank"],
        "rank_expectation": system["rank_expectation"],
        "condition_number": system["condition_number"],
        "n_points": int(len(system["point_weights"])),
        "point_weight_sum": float(np.sum(system["point_weights"])),
        "edge_weight_definition": system["edge_weight_definition"],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
