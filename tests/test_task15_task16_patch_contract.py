from __future__ import annotations

import numpy as np
import rasterio
import inspect
from rasterio.transform import from_origin

from src.multiscene_sift.structural_metrics import stream_structure_metrics, structure_metrics
from src.pipeline.final_metrics import classify_structural_gate
from src.seam_local.config import SeamLocalRuntimeConfig
from src.seam_local.seam import SeamResult
from src.task16_volrn_comparison import run_strict_local_ablation, run_volrn_end_to_end
from scripts.run_task14_13scene_scale import run_task14


def _grid(width: int, height: int) -> dict:
    return {
        "crs": "EPSG:32650", "width": width, "height": height,
        "transform": list(from_origin(100.0, 200.0, 1.0, 1.0)),
    }


def _write(path, data, grid):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs=grid["crs"],
                       transform=from_origin(100.0, 200.0, 1.0, 1.0), nodata=np.nan) as dst:
        dst.write(data.astype(np.float32), 1)


def test_structural_metrics_are_identity_and_nodata_safe():
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    valid = np.ones_like(image, dtype=bool)
    valid[:2, :] = False
    image[~valid] = np.nan
    metrics = structure_metrics(image, image.copy(), valid)
    assert metrics["gradient_magnitude_ncc"] == 1.0
    assert metrics["cgl_rad"] == 0.0
    assert metrics["support_pixels"] > 0


def test_streaming_structure_metrics_matches_identity_contract(tmp_path):
    grid = _grid(12, 12)
    image = np.arange(144, dtype=np.float32).reshape(12, 12)
    image[:1, :] = np.nan
    path = tmp_path / "scene.tif"
    _write(path, image, grid)
    mask = np.isfinite(image)
    metrics = stream_structure_metrics(path, path, mask, tile_size=4, halo=1)
    assert metrics["finite"] is True
    assert metrics["gradient_magnitude_ncc"] == 1.0


def test_ncc_quality_review_does_not_fail_finite_gate():
    gate = classify_structural_gate([{"finite": True, "gradient_magnitude_ncc": 0.8}])
    assert gate == {"finite": True, "quality_review": "REVIEW"}


def test_task14_stage02_registration_parameters_are_defined():
    signature = inspect.signature(run_task14)
    assert signature.parameters["band"].default == "B9"
    assert signature.parameters["match_max_side"].default == 1024
    assert signature.parameters["ransac_threshold"].default == 2.0
    assert signature.parameters["random_seed"].default == 0


def test_strict_ablation_uses_one_v1_weight_layout(tmp_path):
    grid = _grid(8, 8)
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    paths = []
    ours = []
    volrn = []
    weights = []
    for index in range(2):
        base = np.full((8, 8), index + 1, dtype=np.float32)
        p = tmp_path / f"base{index}.tif"; _write(p, base, grid); paths.append(p)
        p = tmp_path / f"ours{index}.tif"; _write(p, base + 2, grid); ours.append(p)
        p = tmp_path / f"volrn{index}.tif"; _write(p, base + 3, grid); volrn.append(p)
        p = tmp_path / f"weight{index}.tif"; _write(p, np.ones((8, 8), dtype=np.float32), grid); weights.append(p)
    result = run_strict_local_ablation(paths, ours, volrn, weights, masks, ["a", "b"], grid, tmp_path / "strict")
    assert result["weight_manifest"]["same_weight_hashes"] is True
    assert set(result["routes"]) == {"A0_BAGRN_V1", "A1_OURS_V1", "A2_VOLRN_V1"}


def test_volrn_end_to_end_calls_shared_refinement(tmp_path):
    grid = _grid(8, 8)
    paths = []
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    for index in range(2):
        path = tmp_path / f"scene{index}.tif"
        _write(path, np.arange(64, dtype=np.float32).reshape(8, 8) + index, grid)
        paths.append(path)
    seam = SeamResult("vertical", np.column_stack((np.arange(8), np.full(8, 4))), 0.0, 0.0, 0.0, "OK")
    calls = []

    def fake_refine(a, b, va, vb, initial, *, refine_half_width, cost_config):
        calls.append((a.shape, refine_half_width, cost_config))
        return initial

    result = run_volrn_end_to_end(
        paths, masks, [{"pair_id": "00_01", "scene_i": 0, "scene_j": 1, "initial_seam": seam}],
        grid, tmp_path / "e2e", refine_fn=fake_refine, runtime_config=SeamLocalRuntimeConfig(),
    )
    assert len(calls) == 1
    assert result["shared_refine_function"] == "fake_refine"
    assert (tmp_path / "e2e/mosaic.tif").is_file()
