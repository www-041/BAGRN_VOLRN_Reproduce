"""Read-only V-sensor metadata audit for the frozen 13-scene benchmark input.

Only GeoTIFF headers and small, bounded samples are read. The script writes two
inventory CSVs; Markdown interpretation is kept in docs/benchmark/v1.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds


UNKNOWN = "UNKNOWN"
BAND_RE = re.compile(r"^(DZ01V_.*)_B(\d+)\.TIF$", re.IGNORECASE)
META_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*;?\s*$")
QUALITY_RE = re.compile(r"(?:cloud|quality|_qa|_qc|_mask)", re.IGNORECASE)


def parse_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = META_RE.match(line)
        if match:
            result[match.group(1).split(".")[-1].upper()] = match.group(2).strip().rstrip(";").strip().strip('"')
    return result


def value(meta: dict[str, str], key: str) -> str:
    return meta.get(key, UNKNOWN) or UNKNOWN


def field(meta: dict[str, str], stem: str, band: int) -> str:
    return value(meta, f"{stem}_BAND_{band}")


def explicit_unit(tags: dict[str, str], band_tags: dict[str, str], units: str | None, meta: dict[str, str], band: int) -> str:
    for candidate in (units, *(band_tags.get(k) for k in ("UNITTYPE", "PIXEL_UNIT", "DATA_UNIT", "RADIOMETRIC_UNIT")),
                      *(tags.get(k) for k in ("UNITTYPE", "PIXEL_UNIT", "DATA_UNIT", "RADIOMETRIC_UNIT")),
                      meta.get(f"DATA_UNIT_BAND_{band}"), meta.get("DATA_UNIT")):
        if candidate and candidate.strip():
            return candidate.strip()
    return UNKNOWN


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_scene(scene_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    files = list(scene_dir.rglob("*"))
    tif_files = [p for p in files if p.is_file() and BAND_RE.fullmatch(p.name)]
    mtls = [p for p in files if p.is_file() and p.name.upper().endswith("_MTL.TXT")]
    if len(mtls) != 1:
        raise RuntimeError(f"Expected one MTL for {scene_dir.name}; found {len(mtls)}")
    mtl = mtls[0]
    meta = parse_metadata(mtl)
    quality = [p for p in files if p.is_file() and QUALITY_RE.search(p.name) and p.suffix.lower() in {".tif", ".tiff", ".xml", ".json"}]
    rows: list[dict[str, object]] = []
    for tif in tif_files:
        match = BAND_RE.fullmatch(tif.name)
        assert match is not None
        product_id, band_text = match.groups()
        band = int(band_text)
        imd_path = tif.with_name(tif.name + ".IMD")
        imd = parse_metadata(imd_path if imd_path.exists() else None)
        combined = {**meta, **imd}
        start = value(combined, "START_IMAGING_TIME")
        end = value(combined, "END_IMAGING_TIME")
        wmin = field(combined, "MEASURED_MIN_WAVELENGTH", band)
        wcenter = field(combined, "MEASURED_CENTER_WAVELENGTH", band)
        wmax = field(combined, "MEASURED_MAX_WAVELENGTH", band)
        bandwidth: str | float = UNKNOWN
        if wmin != UNKNOWN and wmax != UNKNOWN:
            try:
                bandwidth = float(wmax) - float(wmin)
            except ValueError:
                pass
        with rasterio.open(tif) as ds:
            tags = ds.tags()
            band_tags = ds.tags(1)
            unit = explicit_unit(tags, band_tags, ds.units[0], combined, band)
            t = ds.transform
            crs = ds.crs.to_string() if ds.crs else UNKNOWN
            row = {
                "scene_id": scene_dir.name,
                "product_id": product_id,
                "file_path": str(tif),
                "mtl_path": str(mtl),
                "imd_path": str(imd_path) if imd_path.is_file() else UNKNOWN,
                "satellite": value(combined, "SPACECRAFT_ID"),
                "sensor": value(combined, "SENSOR_ID"),
                "product_level": value(combined, "DATA_TYPE"),
                "acquisition_start": start,
                "acquisition_end": end,
                "band": band,
                "wavelength_min_nm": wmin,
                "wavelength_center_nm": wcenter,
                "wavelength_max_nm": wmax,
                "bandwidth_nm": bandwidth,
                "data_unit": unit,
                "width_px": ds.width,
                "height_px": ds.height,
                "pixel_size_x": ds.res[0],
                "pixel_size_y": ds.res[1],
                "crs": crs,
                "affine_a": t.a, "affine_b": t.b, "affine_c": t.c,
                "affine_d": t.d, "affine_e": t.e, "affine_f": t.f,
                "bounds_left": ds.bounds.left, "bounds_bottom": ds.bounds.bottom,
                "bounds_right": ds.bounds.right, "bounds_top": ds.bounds.top,
                "raster_band_count": ds.count,
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata if ds.nodata is not None else UNKNOWN,
                "raster_scale": ds.scales[0], "raster_offset": ds.offsets[0],
                "radiance_mult_mtl": field(combined, "RADIANCE_MULT", band),
                "radiance_add_mtl": field(combined, "RADIANCE_ADD", band),
                "reflectance_mult_mtl": field(combined, "REFLECTANCE_MULT", band),
                "reflectance_add_mtl": field(combined, "REFLECTANCE_ADD", band),
                "mask_flags": "+".join(flag.name for flag in ds.mask_flag_enums[0]),
                "quality_status_tag": tags.get("QUALITY_STATUS", UNKNOWN),
                "jitter_compensation_tag": tags.get("JITTER_COMPENSATION", UNKNOWN),
                "quality_files": ";".join(str(p) for p in quality) if quality else UNKNOWN,
                "file_size_bytes": tif.stat().st_size,
                "mtl_expected_name_matches": value(meta, f"FILE_NAME_BAND_{band}") == tif.name,
                "naming_rule_matches": product_id == value(meta, "PRODUCT_ID"),
            }
            rows.append(row)
    rows.sort(key=lambda row: int(row["band"]))
    present = {int(row["band"]) for row in rows}
    missing = sorted(set(range(1, 17)) - present)
    grids = {(row["crs"], row["width_px"], row["height_px"], row["pixel_size_x"],
              row["pixel_size_y"], row["affine_a"], row["affine_b"], row["affine_c"],
              row["affine_d"], row["affine_e"], row["affine_f"]) for row in rows}
    b9 = next((row for row in rows if row["band"] == 9), None)
    b9_lonlat = (transform_bounds(
        str(b9["crs"]), "EPSG:4326", b9["bounds_left"], b9["bounds_bottom"],
        b9["bounds_right"], b9["bounds_top"], densify_pts=21
    ) if b9 and b9["crs"] != UNKNOWN else None)
    scene = {
        "scene_id": scene_dir.name,
        "product_id": value(meta, "PRODUCT_ID"),
        "satellite": value(meta, "SPACECRAFT_ID"),
        "sensor": value(meta, "SENSOR_ID"),
        "product_level": value(meta, "DATA_TYPE"),
        "acquisition_start": value(meta, "START_IMAGING_TIME"),
        "acquisition_end": value(meta, "END_IMAGING_TIME"),
        "scene_path": str(scene_dir), "mtl_path": str(mtl),
        "v_band_count": len(rows), "v_bands": ";".join(f"B{n}" for n in sorted(present)),
        "missing_bands_B1_B16": ";".join(f"B{n}" for n in missing) if missing else "NONE",
        "all_band_names_match_mtl": all(row["mtl_expected_name_matches"] and row["naming_rule_matches"] for row in rows),
        "same_grid_within_scene": len(grids) == 1,
        "crs_values": ";".join(sorted({str(row["crs"]) for row in rows})),
        "pixel_sizes": ";".join(sorted({f'{row["pixel_size_x"]}x{row["pixel_size_y"]}' for row in rows})),
        "nodata_values": ";".join(sorted({str(row["nodata"]) for row in rows})),
        "data_units": ";".join(sorted({str(row["data_unit"]) for row in rows})),
        "b9_width_px": b9["width_px"] if b9 else UNKNOWN,
        "b9_height_px": b9["height_px"] if b9 else UNKNOWN,
        "mtl_reflective_lines": value(meta, "REFLECTIVE_LINES"),
        "mtl_reflective_samples": value(meta, "REFLECTIVE_SAMPLES"),
        "mtl_dimensions_match_b9_tif": (
            str(b9["height_px"]) == value(meta, "REFLECTIVE_LINES")
            and str(b9["width_px"]) == value(meta, "REFLECTIVE_SAMPLES")
        ) if b9 else False,
        "b9_quality_status_tag": b9["quality_status_tag"] if b9 else UNKNOWN,
        "b9_bounds_left": b9["bounds_left"] if b9 else UNKNOWN,
        "b9_bounds_bottom": b9["bounds_bottom"] if b9 else UNKNOWN,
        "b9_bounds_right": b9["bounds_right"] if b9 else UNKNOWN,
        "b9_bounds_top": b9["bounds_top"] if b9 else UNKNOWN,
        "b9_lon_min": b9_lonlat[0] if b9_lonlat else UNKNOWN,
        "b9_lat_min": b9_lonlat[1] if b9_lonlat else UNKNOWN,
        "b9_lon_max": b9_lonlat[2] if b9_lonlat else UNKNOWN,
        "b9_lat_max": b9_lonlat[3] if b9_lonlat else UNKNOWN,
        "quality_files": ";".join(str(p) for p in quality) if quality else UNKNOWN,
        "total_v_tif_bytes": sum(int(row["file_size_bytes"]) for row in rows),
    }
    return scene, rows


def sample_stats(ds: rasterio.io.DatasetReader) -> dict[str, object]:
    # Bounded 256 x 256 nearest-neighbour sample; never a full-image statistic.
    arr = ds.read(1, out_shape=(256, 256), resampling=Resampling.nearest, masked=True)
    values = arr.compressed()
    return {
        "sample_shape": [256, 256], "valid_sample_count": int(values.size),
        "valid_fraction": float(values.size / arr.size),
        "min": float(np.min(values)) if values.size else None,
        "p05": float(np.percentile(values, 5)) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def compare_one_b9(processed_root: Path, raw_root: Path, scene_id: str) -> dict[str, object]:
    processed = list((processed_root / scene_id).rglob("DZ01V*_B9.TIF"))
    if len(processed) != 1:
        return {"scene_id": scene_id, "status": "NO_UNIQUE_PROCESSED_B9", "processed_candidates": [str(p) for p in processed]}
    expected = processed[0].name
    raw = list((raw_root / scene_id).rglob(expected))
    if len(raw) != 1:
        return {"scene_id": scene_id, "status": "NO_UNIQUE_RAW_B9", "expected_raw_path": str(raw_root / scene_id / processed[0].parent.name / expected)}
    product_id = expected.removesuffix("_B9.TIF")
    raw_mtl_path = raw[0].with_name(product_id + "_MTL.txt")
    processed_mtl_path = processed[0].with_name(product_id + "_MTL.txt")
    raw_mtl = parse_metadata(raw_mtl_path)
    processed_mtl = parse_metadata(processed_mtl_path)
    identity = (raw_mtl.get("PRODUCT_ID") == processed_mtl.get("PRODUCT_ID") == product_id
                and raw_mtl.get("SENSOR_ID") == processed_mtl.get("SENSOR_ID") == "VNIR"
                and raw_mtl.get("SPACECRAFT_ID") == processed_mtl.get("SPACECRAFT_ID") == "DZ01"
                and raw_mtl.get("FILE_NAME_BAND_9") == processed_mtl.get("FILE_NAME_BAND_9") == expected)
    if not identity:
        return {"scene_id": scene_id, "status": "PRODUCT_IDENTITY_NOT_CONFIRMED",
                "raw_path": str(raw[0]), "processed_path": str(processed[0]),
                "raw_mtl_path": str(raw_mtl_path), "processed_mtl_path": str(processed_mtl_path)}
    result: dict[str, object] = {"scene_id": scene_id, "status": "MATCHED",
                                 "product_id": product_id, "band": 9,
                                 "raw_path": str(raw[0]), "processed_path": str(processed[0]),
                                 "raw_mtl_path": str(raw_mtl_path), "processed_mtl_path": str(processed_mtl_path)}
    with rasterio.open(raw[0]) as src, rasterio.open(processed[0]) as dst:
        for label, ds, p in (("raw", src, raw[0]), ("processed", dst, processed[0])):
            result[label] = {
                "file_size_bytes": p.stat().st_size, "width": ds.width, "height": ds.height,
                "count": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata,
                "crs": ds.crs.to_string() if ds.crs else UNKNOWN,
                "resolution": list(ds.res), "transform": list(ds.transform)[:6],
                "bounds": list(ds.bounds), "mask_flags": [x.name for x in ds.mask_flag_enums[0]],
                "tags": ds.tags(), "sample": sample_stats(ds),
            }
        bounds = (max(src.bounds.left, dst.bounds.left), max(src.bounds.bottom, dst.bounds.bottom),
                  min(src.bounds.right, dst.bounds.right), min(src.bounds.top, dst.bounds.top))
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            result["window"] = {"status": "NO_COMMON_BOUNDS"}
            return result
        # Compare validity on one shared 256 x 256 geographic grid. This is
        # explicitly sampled coverage, not an exact full-resolution mask audit.
        common_transform = from_bounds(*bounds, 256, 256)
        coarse_src = np.zeros((256, 256), dtype=src.dtypes[0])
        coarse_dst = np.zeros((256, 256), dtype=dst.dtypes[0])
        for ds, buffer in ((src, coarse_src), (dst, coarse_dst)):
            reproject(source=rasterio.band(ds, 1), destination=buffer,
                      src_transform=ds.transform, src_crs=ds.crs, src_nodata=ds.nodata,
                      dst_transform=common_transform, dst_crs=dst.crs,
                      dst_nodata=ds.nodata, resampling=Resampling.nearest,
                      warp_mem_limit=64)
        valid_src = coarse_src != src.nodata
        valid_dst = coarse_dst != dst.nodata
        result["common_grid_mask_sample"] = {
            "shape": [256, 256], "resampling": "nearest",
            "common_bounds": list(bounds),
            "raw_valid": int(np.count_nonzero(valid_src)),
            "processed_valid": int(np.count_nonzero(valid_dst)),
            "both_valid": int(np.count_nonzero(valid_src & valid_dst)),
            "raw_only_valid": int(np.count_nonzero(valid_src & ~valid_dst)),
            "processed_only_valid": int(np.count_nonzero(valid_dst & ~valid_src)),
        }
        center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        row, col = dst.index(center_x, center_y)
        width = height = 512
        row0 = max(0, min(dst.height - height, row - height // 2))
        col0 = max(0, min(dst.width - width, col - width // 2))
        window = Window(col0, row0, width, height)
        dst_values = dst.read(1, window=window, masked=True)
        src_aligned = np.zeros((height, width), dtype=src.dtypes[0])
        reproject(source=rasterio.band(src, 1), destination=src_aligned,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=dst.window_transform(window), dst_crs=dst.crs,
                  dst_nodata=src.nodata, resampling=Resampling.nearest,
                  warp_mem_limit=64)
        paired = (~np.ma.getmaskarray(dst_values)) & (src_aligned != src.nodata)
        old = src_aligned[paired].astype(np.float64)
        new = dst_values.data[paired].astype(np.float64)
        delta = new - old
        result["window"] = {
            "status": "PAIRED" if len(delta) else "NO_COMMON_VALID_SAMPLES",
            "location": "common_bounds_center", "post_window_row": row0,
            "post_window_col": col0, "window_size_px": [width, height],
            "geographic_center_projected": [center_x, center_y],
            "resampling": "nearest", "valid_pair_count": int(len(delta)),
            "post_window_valid_count": int(np.count_nonzero(~np.ma.getmaskarray(dst_values))),
            "raw_reprojected_valid_count": int(np.count_nonzero(src_aligned != src.nodata)),
            "equal_pair_fraction": float(np.mean(delta == 0)) if len(delta) else None,
            "mae": float(np.mean(np.abs(delta))) if len(delta) else None,
            "delta_p05": float(np.percentile(delta, 5)) if len(delta) else None,
            "delta_median": float(np.median(delta)) if len(delta) else None,
            "delta_p95": float(np.percentile(delta, 95)) if len(delta) else None,
            "raw_median": float(np.median(old)) if len(old) else None,
            "processed_median": float(np.median(new)) if len(new) else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/b9_13scene.yaml")
    parser.add_argument("--raw-root", default=r"D:\科研\地质一号\文献\数据")
    parser.add_argument("--output-dir", default="docs/benchmark/v1")
    parser.add_argument("--compare-scene", default="20251214025253")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["dataset"]["input_root"])
    if not root.is_dir():
        raise FileNotFoundError(root)
    scenes = sorted(p for p in root.iterdir() if p.is_dir())
    if len(scenes) != 13:
        raise RuntimeError(f"Expected 13 scene directories, found {len(scenes)}")
    scene_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    for scene_dir in scenes:
        scene, bands = audit_scene(scene_dir)
        scene_rows.append(scene)
        band_rows.extend(bands)
    output = Path(args.output_dir)
    write_csv(output / "scene_inventory.csv", scene_rows, list(scene_rows[0]))
    write_csv(output / "v_band_inventory.csv", band_rows, list(band_rows[0]))
    prepost = compare_one_b9(root, Path(args.raw_root), args.compare_scene)
    summary = {
        "source_root": str(root), "scene_count": len(scene_rows),
        "band_file_count": len(band_rows), "bands_by_scene": {x["scene_id"]: x["v_band_count"] for x in scene_rows},
        "complete_bands": [n for n in range(1, 17) if all(any(int(x["band"]) == n and x["scene_id"] == scene["scene_id"] for x in band_rows) for scene in scene_rows)],
        "crs": sorted({str(x["crs"]) for x in band_rows}),
        "resolution": sorted({f'{x["pixel_size_x"]}x{x["pixel_size_y"]}' for x in band_rows}),
        "nodata": sorted({str(x["nodata"]) for x in band_rows}),
        "dtype": sorted({str(x["dtype"]) for x in band_rows}),
        "units": sorted({str(x["data_unit"]) for x in band_rows}),
        "same_grid_scene_count": sum(bool(x["same_grid_within_scene"]) for x in scene_rows),
        "naming_consistent_scene_count": sum(bool(x["all_band_names_match_mtl"]) for x in scene_rows),
        "quality_file_scene_count": sum(x["quality_files"] != UNKNOWN for x in scene_rows),
        "b9_centers_nm": sorted({str(x["wavelength_center_nm"]) for x in band_rows if x["band"] == 9}),
        "prepost": prepost,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
