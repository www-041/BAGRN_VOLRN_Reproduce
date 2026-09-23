"""Diagnostic replay/export wrapper for frozen accepted SIFT pairs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.multiscene_sift.dataset import discover_five_scenes
from src.multiscene_sift.global_geometric_adjustment import (
    load_frozen_registration_config,
    load_historical_pair_baselines,
    write_frozen_registration_config,
    write_historical_pair_baselines,
)
from src.multiscene_sift.pairwise import register_pair


def _apply_affine(matrix: list[list[float]], points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    hom = np.column_stack([arr, np.ones(len(arr))])
    return (np.asarray(matrix, dtype=np.float64) @ hom.T).T[:, :2]


def replay_pair_and_capture_inliers(
    scene_i,
    scene_j,
    frozen_config: dict,
    output_dir: str | Path,
) -> dict:
    """Run the existing pair registration once and export its stored inliers."""
    capture_data: dict = {}

    def capture(payload: dict) -> None:
        capture_data.update(payload)

    result = register_pair(
        scene_i,
        scene_j,
        band=frozen_config["registration_band"],
        match_max_side=frozen_config["match_max_side"],
        ransac_threshold=frozen_config["ransac_residual_threshold"],
        random_seed=frozen_config["seed"],
        matcher=frozen_config["matcher"].lower(),
        diagnostic_capture=capture,
    )
    ref = np.asarray(result.inlier_ref_xy if result.inlier_ref_xy is not None else np.empty((0, 2)))
    tgt = np.asarray(result.inlier_tgt_xy if result.inlier_tgt_xy is not None else np.empty((0, 2)))
    if ref.shape != tgt.shape or ref.ndim != 2 or ref.shape[1:] != (2,):
        raise ValueError("registration result contains inconsistent inlier coordinates")

    matrix = result.pair_pixel_matrix
    predicted = _apply_affine(matrix, tgt) if len(tgt) else np.empty((0, 2))
    residual = np.linalg.norm(predicted - ref, axis=1) if len(ref) else np.empty(0)
    out_dir = Path(output_dir)
    pair_dir = out_dir / "02_replayed_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.idx_i}_{result.idx_j}"
    csv_path = pair_dir / f"{stem}_inliers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["point_id", "x_i", "y_i", "x_j", "y_j", "coordinate_frame", "residual_px"]
        )
        for point_id, (p_i, p_j, err) in enumerate(zip(ref, tgt, residual)):
            writer.writerow(
                [point_id, p_i[0], p_i[1], p_j[0], p_j[1], "pair_common_grid", err]
            )

    raw_ref = capture_data.get("raw_ref_xy")
    raw_tgt = capture_data.get("raw_tgt_xy")
    raw_mask = capture_data.get("inlier_mask")
    summary = {
        "edge": [result.idx_i, result.idx_j],
        "status": result.status,
        "raw_matches": int(result.raw_matches),
        "inliers": int(result.inliers),
        "inlier_ratio": float(result.inlier_ratio),
        "coverage": float(result.coverage),
        "rmse_px": float(result.residual_rmse),
        "p95_px": float(result.residual_p95),
        "affine_matrix": matrix,
        "coordinate_frame": "pair_common_grid",
        "transform_direction": "target_to_reference",
        "inlier_count_exported": int(len(ref)),
        "inliers_csv": str(csv_path),
        "raw_coordinates": {
            "ref_xy": np.asarray(raw_ref).tolist() if raw_ref is not None else None,
            "tgt_xy": np.asarray(raw_tgt).tolist() if raw_tgt is not None else None,
            "inlier_mask": np.asarray(raw_mask).tolist() if raw_mask is not None else None,
        },
        "raw_match_count": int(len(raw_ref)) if raw_ref is not None else 0,
        "raw_coordinates_available": raw_ref is not None and raw_tgt is not None and raw_mask is not None,
    }
    (pair_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def load_scene_names(five_scene_run_dir: str | Path) -> list[str]:
    path = Path(five_scene_run_dir) / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(scene["name"]) for scene in payload.get("scenes", [])]


def recover_accepted_pairs(
    five_scene_run_dir: str | Path,
    input_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Replay exactly the frozen historical accepted edge list."""
    baselines = load_historical_pair_baselines(five_scene_run_dir)
    config = load_frozen_registration_config(five_scene_run_dir)
    scenes, _ = discover_five_scenes(
        str(input_root), load_scene_names(five_scene_run_dir), bands=(config["registration_band"],)
    )
    scene_by_index = {scene.index: scene for scene in scenes}
    out_dir = Path(output_dir)
    write_historical_pair_baselines(baselines, out_dir)
    write_frozen_registration_config(config, out_dir)
    replayed = []
    accepted = baselines["accepted_edges"]
    for ordinal, edge_record in enumerate(accepted, start=1):
        i, j = edge_record["edge"]
        print(f"Replaying {ordinal}/{len(accepted)}: {i}-{j}")
        replayed.append(
            replay_pair_and_capture_inliers(
                scene_by_index[i], scene_by_index[j], config, out_dir
            )
        )
    return {
        "accepted_count": len(accepted),
        "replayed_edges": [item["edge"] for item in replayed],
    }
