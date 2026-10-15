"""Fixed three-scene LSSTAA model and native-grid application."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio

from .core import (pif_from_series, control_statistics, choose_control,
                   connection_graph, solve_joint, apply_coefficients)
from .raster import proxy_ndvi


def derive_model(stack: np.ndarray, days: list[int]) -> dict:
    if stack.ndim != 4 or stack.shape[2:] != (3, 4) or len(days) != 3:
        raise ValueError("EXPECTED_THREE_SCENES_FOUR_BANDS")
    ndvi = np.stack([
        proxy_ndvi(stack[:, :, i, 2], stack[:, :, i, 3],
                   np.all(np.isfinite(stack[:, :, i, :]) & (stack[:, :, i, :] != 0), axis=2))
        for i in range(3)
    ], axis=2)
    pif = pif_from_series(ndvi, days)
    rsq, gains, offsets, overlaps = control_statistics(pif.bias_corrected, pif.temporal_difference)
    control = choose_control(rsq, gains, offsets, overlaps)
    edges, graph = connection_graph(stack[:, :, :, 0], days, pif.pif)
    solution = solve_joint(3, control, edges)
    return {"method": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED", "control_scene_index": control,
            "pif_count": int(pif.pif.sum()), "pif_threshold": pif.threshold,
            "bias_medians": pif.bias_medians.tolist(),
            "control_r2_matrix": rsq.tolist(), "graph": graph.astype(int).tolist(),
            "edges": [{"scene_i": i, "scene_j": j, "subset_count": len(x)} for i, j, x, y in edges],
            "gains_b9": solution.gains.tolist(), "offsets_b9": solution.offsets.tolist(),
            "rank": solution.rank, "condition": solution.condition,
            "pre_adjustment_rmse": solution.residual_rmse, "tie_count": solution.tie_count}


def write_corrected_scene(input_data: Path, input_mask: Path, output_dir: Path,
                          scene_index: int, gain: float, offset: float) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"scene_{scene_index:03d}_B9_lsstaa_float32.tif"
    mask_path = output_dir / f"scene_{scene_index:03d}_valid_mask.tif"
    if data_path.exists() or mask_path.exists():
        raise FileExistsError(f"refusing overwrite: {data_path} or {mask_path}")
    with rasterio.open(input_data) as source, rasterio.open(input_mask) as mask:
        if (source.width, source.height, source.transform, source.crs) != (mask.width, mask.height, mask.transform, mask.crs):
            raise ValueError("FROZEN_INPUT_MASK_GRID_MISMATCH")
        profile = source.profile.copy()
        profile.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
        mask_profile = mask.profile.copy()
        mask_profile.update(dtype="uint8", nodata=0, compress="deflate", predictor=2)
        with rasterio.open(data_path, "w", **profile) as output, rasterio.open(mask_path, "w", **mask_profile) as mout:
            for _, window in source.block_windows(1):
                values = source.read(1, window=window)
                valid = mask.read(1, window=window) != 0
                output.write(apply_coefficients(values, valid, gain, offset), 1, window=window)
                mout.write(valid.astype(np.uint8), 1, window=window)
    return data_path, mask_path
