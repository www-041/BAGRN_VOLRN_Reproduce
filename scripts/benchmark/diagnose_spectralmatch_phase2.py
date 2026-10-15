"""Read-only pixel-level diagnostics for preserved Spectralmatch Phase 2 crop smokes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.radiometric import (  # noqa: E402
    compute_pair_direct_metrics,
    compute_pair_distribution_metrics,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _valid(path: Path):
    with rasterio.open(path) as ds:
        values = ds.read(1)
        valid = (ds.dataset_mask() > 0) & np.isfinite(values)
        if ds.nodata is not None:
            valid &= values != ds.nodata
        return values, valid, ds.transform, ds.crs


def _vector_masks(gpkg: Path, layer: str, id_field: str, ids: list[str], shape, transform):
    import fiona

    geometries = {value: [] for value in ids}
    with fiona.open(gpkg, layer=layer) as source:
        for feature in source:
            value = str(feature["properties"].get(id_field, ""))
            if value in geometries and feature["geometry"]:
                geometries[value].append(feature["geometry"])
    return {
        value: rasterize(((geom, 1) for geom in geometries[value]), out_shape=shape,
                         transform=transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
        for value in ids
    }, {value: len(items) for value, items in geometries.items()}


def _save_mask(path: Path, values: np.ndarray, transform, crs):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": values.shape[1], "height": values.shape[0],
               "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
               "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.uint8), 1)
        dst.write_mask(np.full(values.shape, 255, dtype=np.uint8))


def _mosaic_valid_on_reference_grid(mosaic_path: Path, mosaic_mask_path: Path, reference_transform,
                                    reference_shape, reference_crs):
    with rasterio.open(mosaic_path) as mosaic, rasterio.open(mosaic_mask_path) as mask:
        if mosaic.crs != reference_crs or mask.crs != reference_crs:
            raise ValueError("mosaic CRS differs from frozen crop CRS")
        if (mosaic.width, mosaic.height, mosaic.transform) != (mask.width, mask.height, mask.transform):
            raise ValueError("mosaic and mosaic validity-mask grids differ")
        if any(abs(x) > 1e-12 for x in (mosaic.transform.b, mosaic.transform.d,
                                        reference_transform.b, reference_transform.d)):
            raise ValueError("rotated grids are not supported by the exact-offset audit")
        if not np.isclose(mosaic.transform.a, reference_transform.a) or not np.isclose(mosaic.transform.e, reference_transform.e):
            raise ValueError("mosaic pixel size differs from frozen crop")
        dc = (mosaic.transform.c - reference_transform.c) / reference_transform.a
        dr = (mosaic.transform.f - reference_transform.f) / reference_transform.e
        dc_i, dr_i = round(dc), round(dr)
        if not np.isclose(dc, dc_i, atol=1e-7) or not np.isclose(dr, dr_i, atol=1e-7):
            raise ValueError("mosaic has a non-integer pixel-grid offset")
        mapped = np.zeros(reference_shape, dtype=bool)
        vals = mosaic.read(1)
        valid = (mask.read(1) > 0) & np.isfinite(vals)
        if mosaic.nodata is not None:
            valid &= vals != mosaic.nodata
        src_r0, src_c0 = max(0, dr_i), max(0, dc_i)
        src_r1 = min(reference_shape[0], dr_i + mosaic.height)
        src_c1 = min(reference_shape[1], dc_i + mosaic.width)
        if src_r1 > src_r0 and src_c1 > src_c0:
            dst_r0, dst_c0 = src_r0 - dr_i, src_c0 - dc_i
            mapped[src_r0:src_r1, src_c0:src_c1] = valid[dst_r0:dst_r0 + src_r1-src_r0,
                                                          dst_c0:dst_c0 + src_c1-src_c0]
        extra = int(valid.sum() - mapped.sum())
        return mapped, {"row_offset_on_reference": dr_i, "col_offset_on_reference": dc_i,
                        "valid_pixels_outside_reference_grid": extra}


def _valid_on_reference_grid(path: Path, reference_transform, reference_shape, reference_crs):
    with rasterio.open(path) as ds:
        values = ds.read(1)
        valid = (ds.dataset_mask() > 0) & np.isfinite(values)
        if ds.nodata is not None:
            valid &= values != ds.nodata
        if ds.crs != reference_crs or any(abs(x) > 1e-12 for x in
                                          (ds.transform.b, ds.transform.d, reference_transform.b, reference_transform.d)):
            raise ValueError(f"unsupported CRS/rotated grid for {path}")
        if not np.isclose(ds.transform.a, reference_transform.a) or not np.isclose(ds.transform.e, reference_transform.e):
            raise ValueError(f"pixel resolution mismatch for {path}")
        dc = (ds.transform.c - reference_transform.c) / reference_transform.a
        dr = (ds.transform.f - reference_transform.f) / reference_transform.e
        dc_i, dr_i = round(dc), round(dr)
        if not np.isclose(dc, dc_i, atol=1e-7) or not np.isclose(dr, dr_i, atol=1e-7):
            raise ValueError(f"non-integer pixel-grid shift for {path}")
        mapped = np.zeros(reference_shape, dtype=bool)
        src_r0, src_c0 = max(0, dr_i), max(0, dc_i)
        src_r1 = min(reference_shape[0], dr_i + ds.height)
        src_c1 = min(reference_shape[1], dc_i + ds.width)
        if src_r1 > src_r0 and src_c1 > src_c0:
            dst_r0, dst_c0 = src_r0 - dr_i, src_c0 - dc_i
            mapped[src_r0:src_r1, src_c0:src_c1] = valid[dst_r0:dst_r0 + src_r1-src_r0,
                                                          dst_c0:dst_c0 + src_c1-src_c0]
        extra = int(valid.sum() - mapped.sum())
        return mapped, {"row_offset_on_reference": dr_i, "col_offset_on_reference": dc_i,
                        "valid_pixels_outside_reference_grid": extra}


def _stage_record(name: str, valid: np.ndarray, union: np.ndarray):
    return {"stage": name, "valid_pixels": int(valid.sum()),
            "missing_from_input_union": int((union & ~valid).sum()),
            "extra_outside_input_union": int((valid & ~union).sum()),
            "input_union_pixels": int(union.sum())}


def _draw_missing(path: Path, scene_i_only, scene_j_only, both):
    height, width = scene_i_only.shape
    scale = min(1200 / width, 900 / height, 1.0)
    canvas = Image.new("RGB", (max(1, round(width*scale)), max(1, round(height*scale))), (35, 35, 35))
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[scene_i_only] = (255, 70, 70)
    rgb[scene_j_only] = (70, 150, 255)
    rgb[both] = (255, 220, 50)
    image = Image.fromarray(rgb).resize(canvas.size, Image.Resampling.NEAREST)
    canvas.paste(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 52), fill=(255, 255, 255))
    draw.text((8, 7), "Final missing pixels: red=i only, blue=j only, yellow=both", fill="black")
    draw.text((8, 28), f"red={int(scene_i_only.sum())} blue={int(scene_j_only.sum())} yellow={int(both.sum())}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def diagnose_coverage(run_root: Path, output_dir: Path):
    pair_dir = run_root / "00_10" / "worker_output" / "00_10" / "crop_smoke"
    smoke = _read_json(pair_dir / "smoke_result.json")
    artifacts = smoke["input_artifacts"]
    mask_paths = [pair_dir / "spectralmatch_inputs" / f"scene_{idx:03d}_input_mask_crop.tif"
                  for idx in (0, 10)]
    masks = []
    for path in mask_paths:
        with rasterio.open(path) as ds:
            masks.append(ds.read(1) > 0)
            shape, transform, crs = (ds.height, ds.width), ds.transform, ds.crs
    mi, mj = masks
    union = mi | mj
    scene_ids = [Path(path).stem for path in smoke["stage_outputs"]["scene_outputs"]]
    stage_dir = pair_dir / "stages"
    footprint_masks, footprint_feature_counts = _vector_masks(stage_dir / "footprints.gpkg", "footprints", "image",
                                                               scene_ids, shape, transform)
    seam_masks, seam_feature_counts = _vector_masks(stage_dir / "seamlines.gpkg", "seamlines", "image",
                                                     scene_ids, shape, transform)
    if any(count == 0 for count in footprint_feature_counts.values()) or any(count == 0 for count in seam_feature_counts.values()):
        raise ValueError("expected per-scene footprint and seamline features were not found by scene identity")
    footprints_union = footprint_masks[scene_ids[0]] | footprint_masks[scene_ids[1]]
    seam_union = seam_masks[scene_ids[0]] | seam_masks[scene_ids[1]]
    clipped, clipped_grid_audits = [], []
    for path in smoke["stage_outputs"]["seam_clipped_scene_outputs"]:
        valid, grid_audit = _valid_on_reference_grid(Path(path), transform, shape, crs)
        clipped.append(valid); clipped_grid_audits.append(grid_audit)
    clipped_union = clipped[0] | clipped[1]
    mosaic, mosaic_grid = _mosaic_valid_on_reference_grid(
        Path(smoke["stage_outputs"]["mosaic"]), pair_dir / "output_masks" / "mosaic_mask.tif",
        transform, shape, crs)
    final_missing = union & ~mosaic
    i_only = final_missing & mi & ~mj
    j_only = final_missing & mj & ~mi
    both = final_missing & mi & mj
    if int(i_only.sum() + j_only.sum() + both.sum()) != int(final_missing.sum()):
        raise AssertionError("final missing pixels are not fully classified by frozen input masks")

    stages = [
        ("frozen_input_union", union),
        ("footprints_union", footprints_union),
        ("voronoi_seamline_union", seam_union),
        ("seam_clipped_union", clipped_union),
        ("final_mosaic_on_input_grid", mosaic),
    ]
    stage_rows = [_stage_record(name, valid, union) for name, valid in stages]
    stage_rows[2]["per_scene_assigned_pixels"] = {sid: int(seam_masks[sid].sum()) for sid in scene_ids}
    stage_rows[2]["overlap_assigned_to_both"] = int((seam_masks[scene_ids[0]] & seam_masks[scene_ids[1]]).sum())
    stage_rows[2]["unassigned_inside_input_union"] = int((union & ~seam_union).sum())
    stage_rows[1]["per_scene_rasterized_pixels"] = {sid: int(footprint_masks[sid].sum()) for sid in scene_ids}
    stage_rows[1]["per_scene_feature_count"] = footprint_feature_counts
    stage_rows[2]["per_scene_feature_count"] = seam_feature_counts
    stage_rows[3]["per_scene_output_grid_mapping"] = clipped_grid_audits
    stage_rows[4]["output_grid_mapping"] = mosaic_grid
    classes = {"scene_i_only": int(i_only.sum()), "scene_j_only": int(j_only.sum()),
               "both_overlap": int(both.sum())}

    masks_to_write = {
        "frozen_input_union.tif": union,
        "footprints_union.tif": footprints_union,
        "footprints_missing_from_input_union.tif": union & ~footprints_union,
        "voronoi_seamline_union.tif": seam_union,
        "voronoi_unassigned_inside_input_union.tif": union & ~seam_union,
        "seam_clipped_union.tif": clipped_union,
        "seam_clipped_missing_from_input_union.tif": union & ~clipped_union,
        "final_mosaic_valid_on_input_grid.tif": mosaic,
        "final_mosaic_missing_from_input_union.tif": final_missing,
        "final_missing_scene_i_only.tif": i_only,
        "final_missing_scene_j_only.tif": j_only,
        "final_missing_both_overlap.tif": both,
        "seamline_assigned_to_both.tif": seam_masks[scene_ids[0]] & seam_masks[scene_ids[1]],
    }
    for name, arr in masks_to_write.items():
        _save_mask(output_dir / "00_10_masks" / name, arr, transform, crs)
    _draw_missing(output_dir / "00_10_masks" / "final_missing_classes.png", i_only, j_only, both)

    # Compare vector footprint/assignment areas. The difference isolates what
    # polygon segmentation itself removed before raster masking and GDAL merge.
    vector_areas = {}
    import fiona
    from shapely.geometry import shape as to_shape
    for label, gpkg, layer in (("footprints", stage_dir / "footprints.gpkg", "footprints"),
                               ("seamlines", stage_dir / "seamlines.gpkg", "seamlines")):
        areas = {sid: 0.0 for sid in scene_ids}
        with fiona.open(gpkg, layer=layer) as src:
            for feat in src:
                sid = str(feat["properties"].get("image", ""))
                if sid in areas and feat["geometry"]:
                    areas[sid] += to_shape(feat["geometry"]).area
        vector_areas[label] = areas
    report = {
        "pair_id": "00_10", "scope": "1024x1024 crop only; diagnostic, not formal score",
        "crop_window": smoke["crop_definition"]["window"], "scene_ids": scene_ids,
        "affine_mapping": mosaic_grid,
        "stage_counts": stage_rows,
        "final_missing_class_counts": classes,
        "final_missing_total": int(final_missing.sum()),
        "vector_areas_m2": vector_areas,
        "interpretation": "All masks are rasterized/mapped on the frozen crop grid; no resampling or mask regeneration.",
        "artifacts": [str(output_dir / "00_10_masks" / name) for name in masks_to_write] +
                     [str(output_dir / "00_10_masks" / "final_missing_classes.png")],
    }
    _write_json(output_dir / "00_10_coverage_pixel_audit.json", report)
    with (output_dir / "00_10_stage_pixel_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "valid_pixels", "missing_from_input_union",
                                                    "extra_outside_input_union", "input_union_pixels", "details_json"])
        writer.writeheader()
        for row in stage_rows:
            base = {key: row.get(key) for key in ("stage", "valid_pixels", "missing_from_input_union",
                                                  "extra_outside_input_union", "input_union_pixels")}
            base["details_json"] = json.dumps({k: v for k, v in row.items() if k not in base}, ensure_ascii=False)
            writer.writerow(base)
    return report


def diagnose_radiometry(run_root: Path, output_dir: Path):
    pair_dir = run_root / "00_01" / "worker_output" / "00_01" / "crop_smoke"
    smoke = _read_json(pair_dir / "smoke_result.json")
    mask_paths = [pair_dir / "spectralmatch_inputs" / f"scene_{idx:03d}_input_mask_crop.tif"
                  for idx in (0, 1)]
    with rasterio.open(mask_paths[0]) as a, rasterio.open(mask_paths[1]) as b:
        support = (a.read(1) > 0) & (b.read(1) > 0)
        transform, crs = a.transform, a.crs
        if (a.width, a.height, a.transform, a.crs) != (b.width, b.height, b.transform, b.crs):
            raise ValueError("00_01 frozen crop masks are not on one grid")
    stage_paths = {
        "Input": [Path(smoke["input_artifacts"][k]["path"]) for k in ("i", "j")],
        "Global Regression": [Path(p) for p in smoke["stage_outputs"]["global_scene_outputs"]],
        "Local Block Adjustment": [Path(p) for p in smoke["stage_outputs"]["scene_outputs"]],
    }
    rows, values_by_stage = [], {}
    for stage, paths in stage_paths.items():
        arrays, records = [], []
        for scene_index, path in enumerate(paths):
            values, valid, t, c = _valid(path)
            if values.shape != support.shape or t != transform or c != crs:
                raise ValueError(f"{stage} scene {scene_index} grid differs from frozen crop")
            valid_support = support & valid
            sample = values[valid_support].astype(np.float64, copy=False)
            if sample.size != int(support.sum()):
                raise ValueError(f"{stage} scene {scene_index} has incomplete frozen pair support")
            arrays.append(sample)
            rec = {"stage": stage, "scene_index": scene_index, "support_pixels": int(sample.size),
                   "mean": float(sample.mean()), "std_population": float(sample.std(ddof=0)),
                   "p05": float(np.quantile(sample, .05)), "p95": float(np.quantile(sample, .95))}
            records.append(rec); rows.append(rec)
        values_by_stage[stage] = arrays
    pair_metrics = {}
    for stage, (left, right) in values_by_stage.items():
        same = np.ones(left.size, dtype=bool)
        pair_metrics[stage] = {
            "direct": compute_pair_direct_metrics(left, right, same).__dict__,
            "distribution": compute_pair_distribution_metrics(left, right, same).__dict__,
        }
    params_path = pair_dir / "stages/global_regression/global_adjustments.json"
    parameters = _read_json(params_path)
    report = {
        "pair_id": "00_01", "scope": "same frozen 360,981-pixel intersection support for Input/Global/Local",
        "support_pixels": int(support.sum()), "per_scene_stats": rows,
        "pair_metrics": pair_metrics,
        "global_regression_parameters_actual_saved": parameters,
        "local_block_adjustment_parameters": {
            "status": "NOT_EXPORTED_AS_EXPLICIT_GAINS_OFFSETS",
            "evidence": "Preserved local_block_adjustment output contains corrected rasters only; no parameter/map manifest was written.",
            "no_backsolving": True,
        },
        "artifact_paths": {"global_parameter_json": str(params_path),
                           "global_outputs": [str(p) for p in stage_paths["Global Regression"]],
                           "local_outputs": [str(p) for p in stage_paths["Local Block Adjustment"]]},
        "unit": "UNKNOWN; values are stored pixel values, not physical reflectance",
    }
    _write_json(output_dir / "00_01_radiometric_stage_audit.json", report)
    with (output_dir / "00_01_stage_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "00_01_pair_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["stage", "support_pixels", "mae", "rmse", "bias", "p95ae", "mamd", "msdd", "rdd"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for stage, metric in pair_metrics.items():
            writer.writerow({"stage": stage, "support_pixels": metric["direct"]["n"],
                             **{k: metric["direct"][k] for k in ("mae", "rmse", "bias", "p95ae")},
                             **{k: metric["distribution"][k] for k in ("mamd", "msdd", "rdd")}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    expected_parent = (Path("data/output/benchmark_v1_unified")).resolve()
    if run_root.parent != expected_parent or not run_root.name.startswith("spectralmatch_phase2_"):
        raise ValueError("run root must be a task-scoped spectralmatch_phase2_* output directory")
    output = args.output_dir.resolve() if args.output_dir else run_root / "diagnostics"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty diagnostics directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    coverage = diagnose_coverage(run_root, output)
    radiometry = diagnose_radiometry(run_root, output)
    print(json.dumps({"diagnostics": str(output), "00_10_missing": coverage["final_missing_class_counts"],
                      "00_01_support": radiometry["support_pixels"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
