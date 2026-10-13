from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.benchmark_v1_spectralmatch_phase2 import METHOD_ID, UPSTREAM_COMMIT, UPSTREAM_VERSION

NODATA_SENTINEL = float(np.finfo(np.float32).min)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_masked_copy(source_path, frozen_mask_path, output_path, *, block_size=512, crop_window=None):
    """Create a separate Spectralmatch input with the frozen mask encoded as NaN nodata.

    The input raster and frozen mask are read-only. Valid source values are copied
    unchanged; invalid pixels use one shared finite sentinel because Spectralmatch
    checks nodata equality and NaN is not equal to itself.
    """
    source_path, frozen_mask_path, output_path = map(Path, (source_path, frozen_mask_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite working input: {output_path}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    with rasterio.open(source_path) as source, rasterio.open(frozen_mask_path) as mask_src:
        if source.count != 1:
            raise ValueError("DZ01V frozen B9 input must be single-band")
        if (source.width, source.height, source.transform, source.crs) != (
            mask_src.width,
            mask_src.height,
            mask_src.transform,
            mask_src.crs,
        ):
            raise ValueError("frozen source and mask grids differ")
        if source.crs is None or source.transform is None:
            raise ValueError("frozen source has no georeferencing")
        if np.dtype(source.dtypes[0]).kind != "f":
            raise ValueError("Spectralmatch working copy expects frozen float B9 values")

        profile = source.profile.copy()
        if crop_window is not None:
            crop_window = Window(*(int(v) for v in (crop_window.col_off, crop_window.row_off, crop_window.width, crop_window.height)))
            if crop_window.col_off < 0 or crop_window.row_off < 0 or crop_window.width <= 0 or crop_window.height <= 0:
                raise ValueError("crop window must be positive and inside the frozen grid")
            if crop_window.col_off + crop_window.width > source.width or crop_window.row_off + crop_window.height > source.height:
                raise ValueError("crop window extends outside the frozen grid")
            if crop_window.width != int(crop_window.width) or crop_window.height != int(crop_window.height):
                raise ValueError("crop window dimensions must be integral")
            profile.update(width=int(crop_window.width), height=int(crop_window.height),
                           transform=source.window_transform(crop_window))
        width, height = profile["width"], profile["height"]
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=NODATA_SENTINEL,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        valid_pixels = 0
        total_pixels = width * height
        with rasterio.open(output_path, "w", **profile) as output:
            for row_off in range(0, height, block_size):
                for col_off in range(0, width, block_size):
                    window = Window(col_off, row_off, min(block_size, width - col_off), min(block_size, height - row_off))
                    source_window = Window(
                        col_off + (crop_window.col_off if crop_window else 0),
                        row_off + (crop_window.row_off if crop_window else 0),
                        window.width,
                        window.height,
                    )
                    values = source.read(1, window=source_window).astype(np.float32, copy=False)
                    valid = mask_src.read(1, window=source_window) > 0
                    if np.any(valid & ~np.isfinite(values)):
                        raise ValueError("frozen mask declares non-finite source pixels valid")
                    if np.any(valid & (values == NODATA_SENTINEL)):
                        raise ValueError("finite nodata sentinel collides with a valid frozen source pixel")
                    output.write(np.where(valid, values, NODATA_SENTINEL).astype(np.float32, copy=False), 1, window=window)
                    output.write_mask(valid.astype(np.uint8) * 255, window=window)
                    valid_pixels += int(valid.sum())

    return {
        "path": str(output_path.resolve()),
        "source_path": str(source_path.resolve()),
        "frozen_mask_path": str(frozen_mask_path.resolve()),
        "source_sha256": _sha256(source_path),
        "mask_sha256": _sha256(frozen_mask_path),
        "output_sha256": _sha256(output_path),
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "support_preserved": True,
        "nodata_encoding": {"value": NODATA_SENTINEL, "reason": "shared finite sentinel; NaN equality rejected by upstream input validation"},
        "crop_window": None if crop_window is None else {"col_off": int(crop_window.col_off), "row_off": int(crop_window.row_off), "width": int(crop_window.width), "height": int(crop_window.height)},
    }


def execute_fixed_grid_stages(api, input_paths, output_dir, *, global_options=None, local_options=None):
    """Run only Spectralmatch radiometric/seam stages; never invoke geometry correction."""
    output_dir = Path(output_dir)
    if len(input_paths) != 2:
        raise ValueError("Phase 2 smoke supports exactly one frozen pair")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [Path(path).stem for path in input_paths]
    stage_wall_seconds = {}

    global_dir = output_dir / "global_regression"
    global_dir.mkdir(exist_ok=True)
    global_outputs = [str(global_dir / f"{name}_Global.tif") for name in names]
    shared_runtime = {
        "calculation_dtype": "float32",
        "output_dtype": "float32",
        "custom_nodata_value": NODATA_SENTINEL,
        "cache": 0.5,
        "image_threads": None,
        "io_threads": 1,
        "tile_threads": 1,
        "window_size": 512,
        "save_as_cog": True,
        "debug_logs": True,
    }
    global_kwargs = {
        **shared_runtime,
        "estimate_stats": False,
        "pif_method": "flood_from_match_points",
        # B9 is one band. Do not invent red/NIR inputs or an NDVI filter.
        "pif_red_band_index": None,
        "pif_nir_band_index": None,
        "save_adjustments": str(global_dir / "global_adjustments.json"),
        **(global_options or {}),
    }
    stage_start = time.monotonic()
    global_result = api.global_regression(input_images=input_paths, output_images=global_outputs, **global_kwargs)
    stage_wall_seconds["global_regression"] = time.monotonic() - stage_start
    global_outputs = list(global_result or global_outputs)

    local_dir = output_dir / "local_block_adjustment"
    local_dir.mkdir(exist_ok=True)
    local_outputs = [str(local_dir / f"{name}_Local.tif") for name in names]
    local_kwargs = {
        **shared_runtime,
        "correction_method": "offset",
        "alpha": 1.0,
        "number_of_blocks": 100,
        **(local_options or {}),
    }
    stage_start = time.monotonic()
    local_result = api.local_block_adjustment(
        input_images=global_outputs,
        output_images=local_outputs,
        **local_kwargs,
    )
    stage_wall_seconds["local_block_adjustment"] = time.monotonic() - stage_start
    local_outputs = list(local_result or local_outputs)

    footprints_path = output_dir / "footprints.gpkg"
    stage_start = time.monotonic()
    footprints_result = api.create_footprints(
        input_images=local_outputs,
        output_polygons=str(footprints_path),
        image_field_name="image",
        output_layer="footprints",
        band=1,
        image_threads=None,
        debug_logs=True,
    )
    stage_wall_seconds["create_footprints"] = time.monotonic() - stage_start
    footprints_path = Path(footprints_result or footprints_path)

    seam_path = output_dir / "seamlines.gpkg"
    stage_start = time.monotonic()
    seam_result = api.Seamline.voronoi(
        input_polygons=str(footprints_path),
        output_mask=str(seam_path),
        input_layer="footprints",
        output_layer="seamlines",
        image_field_name="image",
        debug_logs=True,
        debug_vectors_path=str(output_dir / "seamline_debug.gpkg"),
    )
    stage_wall_seconds["voronoi_seamline"] = time.monotonic() - stage_start
    seam_path = Path(seam_result or seam_path)

    clipped_dir = output_dir / "seam_clipped_scenes"
    clipped_dir.mkdir(exist_ok=True)
    clipped_outputs = [str(clipped_dir / f"{name}_Seam.tif") for name in names]
    stage_start = time.monotonic()
    clipped_result = api.mask_rasters(
        input_images=local_outputs,
        output_images=clipped_outputs,
        vector_mask=("include", str(seam_path), "image"),
        window_size=512,
        image_threads=None,
        io_threads=1,
        tile_threads=1,
        debug_logs=True,
        custom_nodata_value=NODATA_SENTINEL,
    )
    stage_wall_seconds["mask_rasters"] = time.monotonic() - stage_start
    clipped_outputs = list(clipped_result or clipped_outputs)

    mosaic_path = output_dir / "mosaic.tif"
    stage_start = time.monotonic()
    mosaic_result = api.merge_rasters(
        input_images=clipped_outputs,
        output_image_path=str(mosaic_path),
        output_tiles=False,
        output_dtype="float32",
        custom_nodata_value=NODATA_SENTINEL,
        resolution="highest",
        resampling_method="nearest",
        window_size=512,
        io_threads=1,
        tile_threads=1,
        debug_logs=True,
    )
    stage_wall_seconds["merge_rasters"] = time.monotonic() - stage_start
    mosaic_path = Path(mosaic_result or mosaic_path)
    return {
        "global_scene_outputs": global_outputs,
        "scene_outputs": local_outputs,
        "footprints": str(footprints_path),
        "seamlines": str(seam_path),
        "seam_clipped_scene_outputs": clipped_outputs,
        "mosaic": str(mosaic_path),
        "geometry_registration": "NOT_RUN_FROZEN_CANONICAL_GRID",
        "stages": [
            "global_regression",
            "local_block_adjustment",
            "create_footprints",
            "voronoi_seamline",
            "mask_rasters",
            "merge_rasters",
        ],
        "stage_wall_seconds": stage_wall_seconds,
    }


def build_pair_submission(
    *,
    pair_id,
    scene_i,
    scene_j,
    frozen,
    corrected_scene_paths,
    output_mosaic_path,
    output_mosaic_mask_path,
    run_id,
    status="COMPLETED_DIAGNOSTIC_ONLY",
    outputs=None,
):
    """Build a PAIR_ONLY record accepted by both frozen schema and checker fields."""
    return {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "method_kind": "PAIR_ONLY",
        "input_freeze_sha256": frozen["input_freeze_sha256"],
        "geometry_freeze_sha256": frozen["geometry_freeze_sha256"],
        "band": "B9",
        "pixel_value_unit": "UNKNOWN",
        "value_unit": "UNKNOWN",
        "status": status,
        "output_mosaic_path": str(output_mosaic_path),
        "output_mosaic_mask_path": str(output_mosaic_mask_path),
        "corrected_scene_paths": dict(corrected_scene_paths),
        "pair_specific_outputs": {
            pair_id: {
                "scene_i": scene_i,
                "scene_j": scene_j,
                "corrected_scene_paths": dict(corrected_scene_paths),
                "output_mosaic_path": str(output_mosaic_path),
                "output_mosaic_mask_path": str(output_mosaic_mask_path),
                **(outputs or {}),
            }
        },
        "seam_type": "HARD",
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "evaluation_scope": "TWO_PAIR_COMPATIBILITY_SMOKE_NOT_FORMAL_RANKING",
    }
