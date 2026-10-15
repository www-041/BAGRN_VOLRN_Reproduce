"""Deterministic, blockwise hard-source mosaic for benchmark baselines."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import rasterio
from rasterio.windows import Window


def _check_grid(dataset, shape, transform, crs, path):
    if dataset.count != 1:
        raise ValueError(f"INVALID_BAND_COUNT:{path}:{dataset.count}")
    if (dataset.height, dataset.width) != tuple(shape):
        raise ValueError(f"INVALID_SHAPE:{path}:{dataset.height}x{dataset.width}")
    if dataset.crs is None or dataset.crs.to_string() != crs:
        raise ValueError(f"INVALID_CRS:{path}:{dataset.crs}")
    if not np.allclose(tuple(dataset.transform)[:6], transform, rtol=0.0, atol=1e-9):
        raise ValueError(f"INVALID_TRANSFORM:{path}:{tuple(dataset.transform)[:6]}")
    dtype = np.dtype(dataset.dtypes[0])
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"INVALID_PIXEL_DTYPE:{path}:{dtype}")


def _profile(shape, transform, crs, dtype, *, nodata=None, tiled=False):
    height, width = shape
    profile = {"driver": "GTiff", "width": width, "height": height, "count": 1,
               "dtype": dtype, "crs": crs, "transform": transform,
               "compress": "deflate", "predictor": 2 if np.issubdtype(np.dtype(dtype), np.floating) else 1,
               "nodata": nodata, "BIGTIFF": "IF_SAFER"}
    if tiled and height >= 16 and width >= 16:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    return profile


def _build_fixed_source_mosaic_impl(scene_paths, frozen_mask_paths, mosaic_path, mosaic_mask_path,
                                    source_labels_path, *, scene_ids, expected_shape,
                                    expected_transform, expected_crs="EPSG:32650",
                                    priority=None, block_shape=(512, 512), check=None) -> dict:
    """Copy the first valid source in fixed priority into the canonical grid.

    The explicit frozen mask is authoritative. Input dataset masks and finite
    pixel checks may remove invalid source pixels, but no output pixel is
    created outside the union of the frozen masks.
    """
    scene_paths = [Path(p) for p in scene_paths]
    frozen_mask_paths = [Path(p) for p in frozen_mask_paths]
    scene_ids = [str(x) for x in scene_ids]
    n = len(scene_paths)
    if n == 0 or len(frozen_mask_paths) != n or len(scene_ids) != n or len(set(scene_ids)) != n:
        raise ValueError("SCENE_PATH_MASK_IDENTITY_MISMATCH")
    priority = list(range(n)) if priority is None else [int(x) for x in priority]
    if len(priority) != n or set(priority) != set(range(n)):
        raise ValueError("INVALID_FIXED_PRIORITY")
    if n > 254:
        raise ValueError("SOURCE_LABEL_UINT8_CAPACITY_EXCEEDED")
    shape = tuple(map(int, expected_shape))
    transform = tuple(float(x) for x in expected_transform[:6])
    block_h, block_w = map(int, block_shape)
    if block_h < 1 or block_w < 1:
        raise ValueError("INVALID_BLOCK_SHAPE")
    paths = [Path(mosaic_path), Path(mosaic_mask_path), Path(source_labels_path)]
    if len({p.resolve() for p in paths}) != 3:
        raise ValueError("OUTPUT_PATHS_MUST_BE_DISTINCT")
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_OUTPUT:{existing}")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    write_seconds = 0.0
    input_union_pixels = holes = extras = 0
    source_counts = np.zeros(n, dtype=np.int64)
    with ExitStack() as stack:
        inputs = [stack.enter_context(rasterio.open(p)) for p in scene_paths]
        masks = [stack.enter_context(rasterio.open(p)) for p in frozen_mask_paths]
        for ds, p in zip(inputs, scene_paths):
            _check_grid(ds, shape, transform, expected_crs, p)
        for ds, p in zip(masks, frozen_mask_paths):
            _check_grid(ds, shape, transform, expected_crs, p)
            if ds.dtypes[0] not in ("uint8", "int8", "uint16", "int16", "uint32", "int32"):
                raise ValueError(f"INVALID_FROZEN_MASK_DTYPE:{p}:{ds.dtypes[0]}")
        dtype = inputs[0].dtypes[0]
        if any(ds.dtypes[0] != dtype for ds in inputs):
            raise ValueError("INCONSISTENT_SOURCE_DTYPES")

        mosaic_profile = _profile(shape, transform, expected_crs, dtype, tiled=True)
        mask_profile = _profile(shape, transform, expected_crs, "uint8", tiled=True)
        label_profile = _profile(shape, transform, expected_crs, "uint8", nodata=255, tiled=True)
        with stack.enter_context(rasterio.open(paths[0], "w", **mosaic_profile)) as mosaic_ds, \
             stack.enter_context(rasterio.open(paths[1], "w", **mask_profile)) as mask_out_ds, \
             stack.enter_context(rasterio.open(paths[2], "w", **label_profile)) as labels_ds:
            height, width = shape
            for row in range(0, height, block_h):
                for col in range(0, width, block_w):
                    if check:
                        check()
                    window = Window(col, row, min(block_w, width-col), min(block_h, height-row))
                    h, w = int(window.height), int(window.width)
                    mosaic = np.zeros((h, w), dtype=np.dtype(dtype))
                    valid_out = np.zeros((h, w), dtype=bool)
                    labels = np.full((h, w), 255, dtype=np.uint8)
                    union = np.zeros((h, w), dtype=bool)
                    for ds, input_mask_ds, source_index in zip(inputs, masks, range(n)):
                        mask_values = input_mask_ds.read(1, window=window)
                        frozen_valid = (mask_values > 0) & (input_mask_ds.dataset_mask(window=window) > 0)
                        values = ds.read(1, window=window)
                        source_valid = ds.dataset_mask(window=window) > 0
                        if np.issubdtype(values.dtype, np.floating):
                            source_valid &= np.isfinite(values)
                        if np.any(frozen_valid & ~source_valid):
                            raise ValueError(f"FROZEN_MASK_OUTSIDE_SOURCE_VALIDITY:{scene_ids[source_index]}:{row}:{col}")
                        union |= frozen_valid

                    for source_index in priority:
                        ds, input_mask_ds = inputs[source_index], masks[source_index]
                        mask_values = input_mask_ds.read(1, window=window)
                        frozen_valid = (mask_values > 0) & (input_mask_ds.dataset_mask(window=window) > 0)
                        values = ds.read(1, window=window)
                        source_valid = ds.dataset_mask(window=window) > 0
                        if np.issubdtype(values.dtype, np.floating):
                            source_valid &= np.isfinite(values)
                        take = frozen_valid & source_valid & ~valid_out
                        mosaic[take] = values[take]
                        labels[take] = source_index
                        valid_out[take] = True
                        source_counts[source_index] += int(take.sum())
                    input_union_pixels += int(union.sum())
                    holes += int((union & ~valid_out).sum())
                    extras += int((valid_out & ~union).sum())
                    t0 = time.perf_counter()
                    mosaic_ds.write(mosaic, 1, window=window)
                    mosaic_ds.write_mask(valid_out.astype(np.uint8) * 255, window=window)
                    mask_out_ds.write(valid_out.astype(np.uint8), 1, window=window)
                    mask_out_ds.write_mask(np.full((h, w), 255, dtype=np.uint8), window=window)
                    labels_ds.write(labels, 1, window=window)
                    labels_ds.write_mask(valid_out.astype(np.uint8) * 255, window=window)
                    write_seconds += time.perf_counter() - t0
    elapsed = time.perf_counter() - started
    return {
        "status": "COMPLETED" if holes == 0 and extras == 0 else "INCOMPLETE_COVERAGE",
        "backend": "DETERMINISTIC_FIXED_PRIORITY_HARD_SOURCE_V1",
        "priority_scene_ids": [scene_ids[i] for i in priority],
        "scene_id_to_label": {str(i): scene_ids[i] for i in range(n)},
        "source_pixel_counts": {scene_ids[i]: int(source_counts[i]) for i in range(n)},
        "input_union_pixels": input_union_pixels,
        "output_valid_pixels": int(source_counts.sum()),
        "holes": holes,
        "extras": extras,
        "elapsed_seconds": elapsed,
        "output_write_seconds": write_seconds,
        "mosaic_path": str(paths[0].resolve()),
        "mosaic_mask_path": str(paths[1].resolve()),
        "source_labels_path": str(paths[2].resolve()),
    }


def build_fixed_source_mosaic(scene_paths, frozen_mask_paths, mosaic_path, mosaic_mask_path,
                              source_labels_path, *, scene_ids, expected_shape,
                              expected_transform, expected_crs="EPSG:32650",
                              priority=None, block_shape=(512, 512), check=None) -> dict:
    """Atomically publish a complete mosaic triplet; clean partials on failure."""
    final_paths = [Path(mosaic_path), Path(mosaic_mask_path), Path(source_labels_path)]
    if len({p.resolve() for p in final_paths}) != 3:
        raise ValueError("OUTPUT_PATHS_MUST_BE_DISTINCT")
    existing = [str(p) for p in final_paths if p.exists()]
    if existing:
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_OUTPUT:{existing}")
    token = uuid4().hex
    temp_paths = [p.with_name(f"{p.stem}.partial-{token}{p.suffix}") for p in final_paths]
    for path in final_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _build_fixed_source_mosaic_impl(
            scene_paths, frozen_mask_paths, *temp_paths, scene_ids=scene_ids,
            expected_shape=expected_shape, expected_transform=expected_transform,
            expected_crs=expected_crs, priority=priority, block_shape=block_shape, check=check,
        )
        for source, destination in zip(temp_paths, final_paths):
            os.replace(source, destination)
            sidecar = Path(str(source) + ".msk")
            if sidecar.exists():
                os.replace(sidecar, Path(str(destination) + ".msk"))
        result["mosaic_path"] = str(final_paths[0].resolve())
        result["mosaic_mask_path"] = str(final_paths[1].resolve())
        result["source_labels_path"] = str(final_paths[2].resolve())
        return result
    except BaseException:
        for source in temp_paths:
            source.unlink(missing_ok=True)
            Path(str(source) + ".msk").unlink(missing_ok=True)
        for destination in final_paths:
            destination.unlink(missing_ok=True)
            Path(str(destination) + ".msk").unlink(missing_ok=True)
        raise
