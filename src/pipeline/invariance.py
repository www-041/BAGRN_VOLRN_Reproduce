"""Task15.1 fresh scene/correction-order invariance audits.

The audits deliberately start after the frozen Task15 Stage06/07 boundary.
They do not rerun matching, geometric adjustment, BAGRN, seam search, or
local-coefficient fitting.  Scene-order replay reconstructs only the frozen
pairwise preference fields and label aggregation.  Correction-order replay
uses the frozen local coefficients and evaluates the existing correction
operator with normal and reversed pair iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from src.seam_local.adapter import aggregate_labels_with_ties
from src.seam_local.multiscene_label import PairwisePreferenceField, build_pairwise_preference_field


FIXED_PERMUTATION = [3, 8, 1, 12, 0, 6, 10, 2, 11, 5, 9, 4, 7]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _scene_paths(root: Path) -> list[Path]:
    manifest = _read_json(root / "stages/06_bagrn/bagrn/normalized_scenes_manifest.json")
    count = len(manifest.get("scenes", []))
    if count < 1:
        raise RuntimeError("HARD_STOP_INVARIANCE_MISSING_SCENE_MANIFEST")
    return [root / "stages/06_bagrn/bagrn/normalized_scenes" / f"scene_{i:03d}.tif" for i in range(count)]


def _distance_paths(root: Path, count: int) -> list[Path]:
    base = root / "stages/05_valid_distance_cache"
    return [base / f"distance_scene_{i:03d}.tif" for i in range(count)]


def _pair_rows(root: Path) -> list[dict[str, Any]]:
    rows = json.loads((root / "stages/07_pairwise_seam_local/pairwise_results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("pairwise_results.json must contain a list")
    return [row for row in rows if row.get("status") == "PASS"]


def _grid(paths: list[Path]) -> tuple[Affine, int, int]:
    with rasterio.open(paths[0]) as src:
        return src.transform, src.height, src.width


def _load_p95(root: Path, paths: list[Path]) -> np.ndarray:
    result = np.ones(len(paths), dtype=np.float32)
    for index, path in enumerate(_distance_paths(root, len(paths))):
        with rasterio.open(path) as src:
            values = src.read(1)
        positive = values[values > 0]
        if positive.size:
            result[index] = float(np.percentile(positive, 95))
    return result


def _build_pair_info(root: Path, rows: Iterable[dict[str, Any]], transform: Affine, height: int, width: int) -> list[tuple[int, int, bool, str, np.ndarray]]:
    from scripts.run_task14a_resume_13 import _load_seam_centers

    result = []
    pair_root = root / "stages/07_pairwise_seam_local/pairs"
    for row in rows:
        pair_id = row["pair_id"]
        orientation, centers = _load_seam_centers(pair_root / pair_id / "seam_refined.geojson", transform, height, width)
        result.append((int(row["scene_i"]), int(row["scene_j"]), row.get("side_1_source", "A") == "A", orientation, centers))
    return result


def _labels_for_order(
    root: Path,
    paths: list[Path],
    distances: list[Path],
    pair_info: list[tuple[int, int, bool, str, np.ndarray]],
    p95: np.ndarray,
    order: list[int],
    reference_path: Path,
    transform: Affine,
    height: int,
    width: int,
) -> tuple[int, int]:
    from scripts.run_task14a_resume_13 import _local_tile_seam_path

    positions = {scene: index for index, scene in enumerate(order)}
    remapped_pairs = []
    for scene_i, scene_j, side_a, orientation, centers in pair_info:
        ai, bj = positions[scene_i], positions[scene_j]
        if ai < bj:
            remapped_pairs.append((ai, bj, side_a, orientation, centers))
        else:
            remapped_pairs.append((bj, ai, not side_a, orientation, centers))

    reference_support = 0
    difference = 0
    with rasterio.open(reference_path) as reference:
        sources = [rasterio.open(paths[index]) for index in order]
        distance_sources = [rasterio.open(distances[index]) for index in order]
        try:
            for r0 in range(0, height, 1024):
                hh = min(1024, height - r0)
                window = Window(0, r0, width, hh)
                masks = np.stack([src.read_masks(1, window=window) > 0 for src in sources])
                raw = np.stack([src.read(1, window=window).astype(np.float32) for src in distance_sources])
                fields: dict[tuple[int, int], PairwisePreferenceField] = {}
                for scene_a, scene_b, side_a, orientation, centers in remapped_pairs:
                    path = _local_tile_seam_path(orientation, centers, r0, hh, width)
                    c = centers[r0:r0 + hh] if orientation == "vertical" else centers
                    domain = np.isfinite(c)[:, None] if orientation == "vertical" else np.broadcast_to(np.isfinite(c)[None, :], (hh, width))
                    field = build_pairwise_preference_field(
                        masks[scene_a], masks[scene_b], path, "A" if side_a else "B",
                        orientation=orientation, scene_a=scene_a, scene_b=scene_b,
                    )
                    fields[(scene_a, scene_b)] = PairwisePreferenceField(
                        scene_a,
                        scene_b,
                        np.where(domain, field.vote, 0).astype(np.float32),
                        field.available & domain,
                        np.where(domain, field.confidence, 0).astype(np.float32),
                        orientation,
                    )
                labels, _, _ = aggregate_labels_with_ties(masks, fields, raw, p95_edt=p95[order])
                mapped = np.full(labels.shape, -1, dtype=np.int16)
                for permuted_index, scene_identity in enumerate(order):
                    mapped[labels == permuted_index] = int(scene_identity)
                reference_tile = reference.read(1, window=window)
                reference_support += int(np.count_nonzero(reference_tile >= 0))
                difference += int(np.count_nonzero(mapped != reference_tile))
        finally:
            for source in sources + distance_sources:
                source.close()
    return difference, reference_support


def run_scene_order_invariance(root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Replay frozen Stage08 labeling for original, reverse, and fixed orders."""
    root = Path(root)
    paths = _scene_paths(root)
    distances = _distance_paths(root, len(paths))
    transform, height, width = _grid(paths)
    rows = _pair_rows(root)
    pair_info = _build_pair_info(root, rows, transform, height, width)
    p95 = _load_p95(root, paths)
    reference_path = root / "stages/08_multiscene_labeling/source_label_map.tif"
    original = list(range(len(paths)))
    reverse = list(reversed(original))
    if len(paths) != len(FIXED_PERMUTATION):
        raise RuntimeError("HARD_STOP_INVARIANCE_SCENE_COUNT_NOT_13")
    fixed = list(FIXED_PERMUTATION)
    original_diff, support = _labels_for_order(root, paths, distances, pair_info, p95, original, reference_path, transform, height, width)
    reverse_diff, reverse_support = _labels_for_order(root, paths, distances, pair_info, p95, reverse, reference_path, transform, height, width)
    fixed_diff, fixed_support = _labels_for_order(root, paths, distances, pair_info, p95, fixed, reference_path, transform, height, width)
    result = {
        "status": "PASS" if original_diff == 0 and reverse_diff == 0 and fixed_diff == 0 else "HARD_STOP_13SCENE_SCENE_ORDER_DEPENDENCE",
        "evidence_type": "fresh_task15_frozen_stage08_minimal_replay",
        "original_order": original,
        "reverse_order": reverse,
        "fixed_permutation": fixed,
        "original_order_difference_pixels": original_diff,
        "reverse_order_difference_pixels": reverse_diff,
        "fixed_permutation_difference_pixels": fixed_diff,
        "reference_label_support": support,
        "reverse_label_support": reverse_support,
        "permutation_label_support": fixed_support,
        "pairwise_preference_edges": len(rows),
        "inputs": {
            "stage06": str(root / "stages/06_bagrn/bagrn/normalized_scenes"),
            "stage07": str(root / "stages/07_pairwise_seam_local"),
            "stage08_reference": str(reference_path),
            "stage05_distance_cache": str(root / "stages/05_valid_distance_cache"),
            "new_pairwise_edges": False,
            "refit_coefficients": False,
        },
    }
    _write_json(Path(output_path) if output_path else root / "11_metrics/invariance/scene_order_invariance.json", result)
    return result


def _raster_difference(first_path: Path, second_path: Path, height: int, width: int) -> dict[str, Any]:
    max_abs = 0.0
    sum_abs = 0.0
    finite_pixels = 0
    with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
        for r0 in range(0, height, 1024):
            hh = min(1024, height - r0)
            window = Window(0, r0, width, hh)
            first_tile = first.read(1, window=window).astype(np.float64)
            second_tile = second.read(1, window=window).astype(np.float64)
            valid = np.isfinite(first_tile) & np.isfinite(second_tile)
            if np.any(valid):
                delta = np.abs(first_tile[valid] - second_tile[valid])
                max_abs = max(max_abs, float(np.max(delta)))
                sum_abs += float(np.sum(delta))
                finite_pixels += int(delta.size)
    return {"max_abs_difference": max_abs, "mean_abs_difference": sum_abs / max(finite_pixels, 1), "finite_pixels": finite_pixels}


def _correction_outputs_complete(directory: Path, count: int) -> bool:
    return (directory / "_SUCCESS.json").is_file() and all((directory / f"corrected_scene_{i:03d}.tif").is_file() for i in range(count))


def run_correction_order_invariance(root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate frozen Stage09 correction with normal/reversed pair iteration."""
    root = Path(root)
    paths = _scene_paths(root)
    transform, height, width = _grid(paths)
    rows = json.loads((root / "stages/07_pairwise_seam_local/pairwise_results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("pairwise_results.json must contain a list")
    output_root = root / "11_metrics/invariance"
    normal_dir = output_root / "correction_order_normal"
    reverse_dir = output_root / "correction_order_reversed"
    import scripts.run_task14a_resume_13 as frozen

    old_stage07, old_stage09 = frozen.STAGE07, frozen.STAGE09
    try:
        frozen.STAGE07 = root / "stages/07_pairwise_seam_local"
        if not _correction_outputs_complete(normal_dir, len(paths)):
            if normal_dir.exists():
                raise RuntimeError("HARD_STOP_INVARIANCE_INCOMPLETE_NORMAL_CORRECTION_ARTIFACT")
            frozen.STAGE09 = normal_dir
            normal = frozen._stage09(rows, paths, transform, height, width)
        else:
            normal = [normal_dir / f"corrected_scene_{i:03d}.tif" for i in range(len(paths))]
        if not _correction_outputs_complete(reverse_dir, len(paths)):
            if reverse_dir.exists():
                raise RuntimeError("HARD_STOP_INVARIANCE_INCOMPLETE_REVERSED_CORRECTION_ARTIFACT")
            frozen.STAGE09 = reverse_dir
            reverse = frozen._stage09(list(reversed(rows)), paths, transform, height, width)
        else:
            reverse = [reverse_dir / f"corrected_scene_{i:03d}.tif" for i in range(len(paths))]
    finally:
        frozen.STAGE07, frozen.STAGE09 = old_stage07, old_stage09
    differences = [_raster_difference(a, b, height, width) for a, b in zip(normal, reverse)]
    max_abs = max((item["max_abs_difference"] for item in differences), default=0.0)
    mean_abs = max((item["mean_abs_difference"] for item in differences), default=0.0)
    scale = max(1.0, abs(mean_abs))
    result = {
        "status": "PASS" if max_abs <= 1e-3 + 1e-6 * scale else "HARD_STOP_13SCENE_CORRECTION_ORDER_DEPENDENCE",
        "evidence_type": "fresh_task15_frozen_stage09_replay_no_refit",
        "normal_pair_order": "pairwise_results.json order",
        "reverse_pair_order": "reverse(pairwise_results.json)",
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "rtol": 1e-6,
        "atol": 1e-3,
        "exact_equality": max_abs == 0.0 and mean_abs == 0.0,
        "scene_differences": differences,
        "inputs": {
            "stage07": str(root / "stages/07_pairwise_seam_local"),
            "stage09_formal": str(root / "stages/09_correction"),
            "refit_local_coefficients": False,
        },
    }
    _write_json(Path(output_path) if output_path else output_root / "correction_order_invariance.json", result)
    return result


def run_invariance_audit(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    scene = run_scene_order_invariance(root)
    correction = run_correction_order_invariance(root)
    return {"scene_order": scene, "correction_order": correction}


__all__ = [
    "FIXED_PERMUTATION",
    "run_scene_order_invariance",
    "run_correction_order_invariance",
    "run_invariance_audit",
]
