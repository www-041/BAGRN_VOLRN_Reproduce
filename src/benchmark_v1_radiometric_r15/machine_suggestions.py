from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import mapping, shape

from .multiband_view import read_window_padded


SUGGESTION_FIELDS = (
    "suggested_landcover_type",
    "suggested_terrain_type",
    "suggested_temporal_status",
    "suggested_interference_type",
)
FIELD_LABELS = {
    "suggested_landcover_type": {"VEGETATION", "BARE_GROUND", "WATER", "BUILT", "MIXED", "OTHER", "UNKNOWN"},
    "suggested_terrain_type": {"RIDGE", "SLOPE", "VALLEY", "FLAT", "OTHER", "UNKNOWN"},
    "suggested_temporal_status": {"NO_CHANGE", "POSSIBLE_CHANGE", "CONFIRMED_CHANGE", "UNCERTAIN", "UNKNOWN"},
    "suggested_interference_type": {"NONE", "CLOUD", "CLOUD_SHADOW", "TERRAIN_SHADOW", "HAZE", "SATURATION", "OTHER", "UNKNOWN"},
}
_FORBIDDEN_AUTOMATIC_LABELS = {"CONFIRMED_CHANGE"}


def _valid(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(data))


def _measure(values: np.ndarray, valid: np.ndarray, roi_mask: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float32)
    use = _valid(data, valid) & np.asarray(roi_mask, dtype=bool)
    total = int(np.asarray(roi_mask, dtype=bool).sum())
    selected = data[use].astype(np.float64, copy=False)
    if not selected.size:
        return {"valid_count": 0, "valid_fraction": 0.0 if total else None,
                "median_raw_dn": None, "mad_raw_dn": None, "p10_raw_dn": None,
                "p90_raw_dn": None, "normalized_gradient_texture": None}
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected - median)))
    p10, p90 = (float(v) for v in np.percentile(selected, (10, 90)))
    if data.shape[0] >= 2 and data.shape[1] >= 2:
        safe = np.where(use, data, median)
        gy, gx = np.gradient(safe.astype(np.float64))
        gradient = np.hypot(gx, gy)[use]
        texture = float(np.median(gradient) / max(mad, 1e-12)) if gradient.size else None
    else:
        texture = None
    return {"valid_count": int(selected.size),
            "valid_fraction": float(selected.size / total) if total else None,
            "median_raw_dn": median, "mad_raw_dn": mad,
            "p10_raw_dn": p10, "p90_raw_dn": p90,
            "normalized_gradient_texture": texture}


def extract_pair_roi_evidence(
    geometry: Any,
    paths_i: Mapping[int, str | Path],
    paths_j: Mapping[int, str | Path],
    *,
    max_window_pixels: int = 2_000_000,
) -> dict[str, Any]:
    """Read bounded per-band windows, mapping scene-j to scene-i B9 for comparison.

    Values remain raw DN. No reflectance index or temporal semantic label is computed.
    The affine mapping is only for common-window evidence, not proof of registration.
    """
    geom = shape(geometry) if isinstance(geometry, dict) else geometry
    if geom.is_empty or not geom.is_valid:
        raise ValueError("candidate ROI geometry must be non-empty and valid")
    if 9 not in paths_i or 9 not in paths_j:
        raise ValueError("both scenes require B9 reference grids")
    shared_bands = sorted(set(paths_i) & set(paths_j))
    if not shared_bands:
        raise ValueError("no common bands available")
    evidence: dict[str, Any] = {"bands": {}, "grid_mapping": "scene_j warped to scene_i B9 affine",
                                "resampling": "bilinear values; source validity masks nearest-neighbor",
                                "data_unit": "UNKNOWN_RAW_DN"}
    with rasterio.open(paths_i[9]) as ref_i, rasterio.open(paths_j[9]) as ref_j:
        if ref_i.crs is None or ref_i.crs.to_string() != "EPSG:32650" or ref_j.crs is None:
            raise ValueError("B9 reference rasters must have projected CRS; scene-i must be EPSG:32650")
        window = geometry_window(ref_i, [mapping(geom)], boundless=True).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError("candidate ROI produces an empty raster window")
        if window.width * window.height > max_window_pixels:
            raise ValueError(f"candidate ROI window exceeds configured limit: {window.width}x{window.height}")
        height, width = int(window.height), int(window.width)
        transform = ref_i.window_transform(window)
        inside = geometry_mask([mapping(geom)], out_shape=(height, width), transform=transform,
                               invert=True, all_touched=False)
        for band in shared_bands:
            with rasterio.open(paths_i[band]) as ds_i:
                if (ds_i.crs != ref_i.crs or ds_i.transform != ref_i.transform or
                        ds_i.width != ref_i.width or ds_i.height != ref_i.height):
                    raise ValueError(f"scene-i B{band} does not match its B9 grid")
                data_i, valid_i = read_window_padded(ds_i, window)
            with rasterio.open(paths_j[band]) as ds_j:
                if ds_j.crs != ref_j.crs or ds_j.transform != ref_j.transform or ds_j.width != ref_j.width or ds_j.height != ref_j.height:
                    raise ValueError(f"scene-j B{band} does not match its B9 grid")
                with WarpedVRT(ds_j, crs=ref_i.crs, transform=ref_i.transform,
                               width=ref_i.width, height=ref_i.height,
                               resampling=Resampling.bilinear, nodata=ds_j.nodata) as vrt:
                    data_j, valid_j = read_window_padded(vrt, window)
            if data_i.shape != (height, width) or data_j.shape != (height, width):
                raise ValueError(f"B{band} window shape mismatch")
            stats_i = _measure(data_i, valid_i, inside)
            stats_j = _measure(data_j, valid_j, inside)
            common = inside & _valid(data_i, valid_i) & _valid(data_j, valid_j)
            common_fraction = float(common.sum() / inside.sum()) if inside.any() else 0.0
            stats_i_common = _measure(data_i, common, inside)
            stats_j_common = _measure(data_j, common, inside)
            delta = None
            if common.any():
                delta = float(np.median(data_i[common].astype(np.float64) - data_j[common].astype(np.float64)))
            evidence["bands"][str(band)] = {
                "scene_i": stats_i, "scene_j": stats_j,
                "common_valid_fraction": common_fraction,
                "common_scene_i": stats_i_common, "common_scene_j": stats_j_common,
                "median_scene_i_minus_scene_j_raw_dn": delta,
            }
    return evidence


def _evidence_lines(band_features: Mapping[int | str, Mapping[str, Any]]) -> list[str]:
    lines = []
    for band in sorted(band_features, key=lambda x: int(x)):
        feat = band_features[band]
        texture_i = feat.get("common_scene_i", {}).get("normalized_gradient_texture", feat.get("texture_i"))
        texture_j = feat.get("common_scene_j", {}).get("normalized_gradient_texture", feat.get("texture_j"))
        lines.append(
            f"B{int(band)} common_valid_fraction={feat.get('common_valid_fraction')}; "
            f"texture_i/j={texture_i}/{texture_j}; "
            f"raw_median_delta={feat.get('median_scene_i_minus_scene_j_raw_dn')}"
        )
    return lines


def advise_fields(*, band_features: Mapping[int | str, Mapping[str, Any]],
                  available_bands: Sequence[int], missing_auxiliary: Sequence[str],
                  data_unit: str, geometry_verified: bool) -> dict[str, dict[str, Any]]:
    """Return conservative fieldwise machine recommendations, never review decisions.

    The current dataset has UNKNOWN physical units, no reliable DEM/cloud mask,
    and no independent change reference. Consequently observed DN/texture evidence
    is reported for human inspection but cannot justify semantic classes here.
    """
    bands = sorted({int(v) for v in available_bands})
    missing = sorted(set(str(v) for v in missing_auxiliary))
    evidence = _evidence_lines(band_features)
    evidence = evidence or ["No valid per-band ROI measurements were available."]
    common = {
        "suggested_landcover_type": (
            "LOW", ["NO_VALIDATED_LANDCOVER_CLASSIFIER", "DN_UNITS_UNKNOWN_NO_REFLECTANCE_INDICES"]),
        "suggested_terrain_type": (
            "MEDIUM", ["DEM_UNAVAILABLE", "LOCAL_TEXTURE_CANNOT_VERIFY_RIDGE_SLOPE_OR_VALLEY"]),
        "suggested_temporal_status": (
            "MEDIUM", ["NO_INDEPENDENT_CHANGE_REFERENCE", "CROSS_SCENE_GEOMETRY_UNVERIFIED",
                       "RAW_DN_DIFFERENCE_NOT_CHANGE_EVIDENCE"]),
        "suggested_interference_type": (
            "MEDIUM", ["CLOUD_QUALITY_MASK_UNAVAILABLE", "SHADOW_CANNOT_BE_SEPARATED_FROM_WATER_OR_DARK_VEGETATION"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for field in SUGGESTION_FIELDS:
        confidence, reasons = common[field]
        field_evidence = evidence
        if field == "suggested_terrain_type":
            field_evidence = [line for line in evidence if line.startswith(("B9 ", "B14 "))] or evidence
        elif field == "suggested_temporal_status":
            field_evidence = [line for line in evidence if "raw_median_delta" in line] or evidence
        elif field == "suggested_interference_type":
            field_evidence = ["No independently verified cloud, shadow, or quality mask is present.", *evidence[:4]]
        result[field] = {"label": "UNKNOWN", "confidence": confidence,
                         "evidence": field_evidence,
                         "uncertainty_reasons": reasons,
                         "missing_auxiliary_data": missing,
                         "evidence_bands": bands,
                         "data_unit": data_unit,
                         "geometry_verified": bool(geometry_verified)}
    return result


def build_machine_suggestion(feature: Mapping[str, Any], *, run_id: str,
                             pair_inputs, missing_auxiliary: Sequence[str] | None = None) -> dict[str, Any]:
    props = dict(feature.get("properties", {}))
    evidence = extract_pair_roi_evidence(feature["geometry"],
        {int(k[1:]): v.path for k, v in pair_inputs.scenes[0].bands.items() if v.usable and v.path},
        {int(k[1:]): v.path for k, v in pair_inputs.scenes[1].bands.items() if v.usable and v.path})
    bands = sorted(int(k) for k in evidence["bands"])
    missing = list(missing_auxiliary if missing_auxiliary is not None else
                   ["DEM", "CLOUD_MASK_OR_QA", "INDEPENDENT_CHANGE_REFERENCE", "VALIDATED_LANDCOVER_REFERENCE"])
    fields = advise_fields(band_features=evidence["bands"], available_bands=bands,
                           missing_auxiliary=missing, data_unit="UNKNOWN",
                           geometry_verified=False)
    kept = {name: props[name] for name in ("role", "grid_cell_id", "valid_support_fraction") if name in props}
    output_props = {"pair_id": props["pair_id"], "candidate_id": props["candidate_id"],
                    "scene_i": props.get("scene_i"), "scene_j": props.get("scene_j"),
                    "role_at_generation": props.get("frozen_role", props.get("role", "UNASSIGNED")),
                    "candidate_role_metadata": kept,
                    "machine_suggestion_status": "SUGGESTED",
                    "machine_run_id": run_id,
                    "recommendation_method": "R1.5_CONSERVATIVE_MULTIBAND_TEXTURE_EVIDENCE_V1",
                    "band_evidence": evidence,
                    **fields}
    if any(output_props[field]["label"] == "CONFIRMED_CHANGE" for field in SUGGESTION_FIELDS):
        raise AssertionError("machine generator must never issue CONFIRMED_CHANGE")
    return {"type": "Feature", "geometry": feature["geometry"], "properties": output_props}


def _write_csv(stream, features: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["pair_id", "candidate_id", "role_at_generation", "machine_run_id",
                  "machine_suggestion_status"]
    for field in SUGGESTION_FIELDS:
        fieldnames.extend((field, f"{field}_confidence", f"{field}_evidence",
                           f"{field}_uncertainty_reasons", f"{field}_missing_auxiliary_data"))
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for feature in features:
        props = feature["properties"]
        row = {name: props.get(name, "") for name in fieldnames[:5]}
        for field in SUGGESTION_FIELDS:
            item = props[field]
            row[field] = item["label"]
            row[f"{field}_confidence"] = item["confidence"]
            row[f"{field}_evidence"] = json.dumps(item["evidence"], ensure_ascii=False)
            row[f"{field}_uncertainty_reasons"] = json.dumps(item["uncertainty_reasons"], ensure_ascii=False)
            row[f"{field}_missing_auxiliary_data"] = json.dumps(item["missing_auxiliary_data"], ensure_ascii=False)
        writer.writerow(row)


def write_machine_suggestion_artifacts(output_dir: str | Path, pair_id: str,
                                       features: Sequence[Mapping[str, Any]], *,
                                       source_candidate_sha256: str = "",
                                       run_id: str = "") -> tuple[Path, Path]:
    """Write separate, immutable per-pair GeoJSON and CSV; never overwrite."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{pair_id}.machine_suggestions.geojson"
    csv_path = folder / f"{pair_id}.machine_suggestions.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite machine suggestion artifact for {pair_id}")
    keys = [(str(f.get("properties", {}).get("pair_id", "")),
             str(f.get("properties", {}).get("candidate_id", ""))) for f in features]
    if any(pair != pair_id or not candidate for pair, candidate in keys) or len(set(keys)) != len(keys):
        raise ValueError("machine suggestions must have unique matching pair/candidate identities")
    for feature in features:
        props = feature["properties"]
        if props.get("machine_suggestion_status") != "SUGGESTED":
            raise ValueError("machine records must remain SUGGESTED, never human-approved")
        for field in SUGGESTION_FIELDS:
            item = props[field]
            if item.get("label") not in FIELD_LABELS[field] or item.get("label") in _FORBIDDEN_AUTOMATIC_LABELS:
                raise ValueError(f"invalid or forbidden machine label for {field}: {item.get('label')}")
            if item.get("confidence") not in {"LOW", "MEDIUM"}:
                raise ValueError(f"invalid confidence for {field}")
    collection = {"type": "FeatureCollection", "name": "R1.5_MACHINE_SUGGESTIONS_NOT_GROUND_TRUTH",
                  "schema_version": 1, "machine_run_id": run_id,
                  "source_candidate_sha256": source_candidate_sha256,
                  "features": list(features)}
    targets = ((json_path, lambda f: json.dump(collection, f, ensure_ascii=False, indent=2)),
               (csv_path, lambda f: _write_csv(f, features)))
    temps: list[tuple[Path, Path]] = []
    try:
        for target, write in targets:
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=folder)
            os.close(fd)
            temp = Path(name); temps.append((temp, target))
            with temp.open("w", encoding="utf-8", newline="") as stream:
                write(stream); stream.flush(); os.fsync(stream.fileno())
        if json_path.exists() or csv_path.exists():
            raise FileExistsError("machine output appeared during write; refusing replacement")
        for temp, target in temps:
            temp.replace(target)
        return json_path, csv_path
    except Exception:
        for temp, _ in temps:
            temp.unlink(missing_ok=True)
        raise


def load_machine_suggestion_map(path: str | Path, *, expected_pair_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        collection = json.load(stream)
    if collection.get("type") != "FeatureCollection" or collection.get("schema_version") != 1:
        raise ValueError("unsupported machine suggestion artifact/schema")
    result = {}
    source_hash = str(collection.get("source_candidate_sha256", ""))
    for feature in collection.get("features", []):
        props = feature.get("properties", {})
        pair = str(props.get("pair_id", "")); candidate = str(props.get("candidate_id", ""))
        if pair != expected_pair_id or not candidate:
            raise ValueError(f"machine suggestion identity mismatch: {(pair, candidate)}")
        key = (pair, candidate)
        if key in result:
            raise ValueError(f"duplicate machine suggestion identity: {key}")
        for field in SUGGESTION_FIELDS:
            item = props.get(field, {})
            if item.get("label") not in FIELD_LABELS[field] or item.get("confidence") not in {"LOW", "MEDIUM"}:
                raise ValueError(f"invalid machine suggestion field {field} for {key}")
        if props.get("machine_suggestion_status") != "SUGGESTED":
            raise ValueError(f"unexpected machine suggestion status for {key}")
        props = dict(props)
        props["_source_candidate_sha256"] = source_hash
        result[key] = props
    return result


def summarize_human_agreement(machine_features: Sequence[Mapping[str, Any]],
                              human_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare to explicit eligible human approvals; this is agreement, not accuracy."""
    machine = {(str(f.get("properties", {}).get("pair_id", "")),
                str(f.get("properties", {}).get("candidate_id", ""))): f.get("properties", {})
               for f in machine_features}
    eligible = {}
    excluded = 0
    for row in human_rows:
        key = (str(row.get("pair_id", "")), str(row.get("candidate_id", "")))
        if row.get("multiband_review_decision") != "APPROVED":
            excluded += 1; continue
        role = str(row.get("role", row.get("frozen_role", "DEVELOPMENT")))
        if role == "TEST" and (row.get("multiband_second_review_status") != "APPROVED" or
                                row.get("multiband_dispute_status") != "NONE"):
            excluded += 1; continue
        eligible[key] = row
    fields = {}
    for field in SUGGESTION_FIELDS:
        label_name = field.removeprefix("suggested_")
        compared = [(machine[key][field]["label"], row.get(label_name, "UNKNOWN"))
                    for key, row in eligible.items() if key in machine]
        matches = sum(a == b for a, b in compared)
        fields[field] = {"n": len(compared), "agreement_count": matches,
                         "agreement_fraction": matches / len(compared) if compared else None,
                         "label_pairs": [{"machine": a, "human": b, "count": compared.count((a, b))}
                                         for a, b in sorted(set(compared))]}
    count = sum(1 for key in eligible if key in machine)
    return {"status": "COMPARISON_ONLY_NOT_ACCURACY" if count else "NO_HUMAN_APPROVED_LABELS",
            "interpretation": "Descriptive agreement with eligible human review only; not classifier accuracy or truth validation.",
            "eligible_human_record_count": count,
            "excluded_unapproved_or_incomplete_test_reviews": excluded,
            "fields": fields}


def select_human_review_sample(features: Sequence[Mapping[str, Any]], *,
                               max_samples: int = 12) -> list[dict[str, Any]]:
    """Select deterministic pair/role coverage without using machine score or label."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    role_order = {"DEVELOPMENT": 0, "TEST": 1, "UNASSIGNED": 2}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for feature in features:
        props = feature.get("properties", {})
        pair = str(props.get("pair_id", "")); role = str(props.get("role_at_generation", "UNASSIGNED"))
        grouped.setdefault((pair, role), []).append(feature)
    groups = sorted(grouped, key=lambda key: (role_order.get(key[1], 3), key[0]))
    selected: list[Mapping[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for group in groups:
        ordered = sorted(grouped[group], key=lambda feature: (
            shape(feature["geometry"]).centroid.x,
            shape(feature["geometry"]).centroid.y,
            str(feature["properties"].get("candidate_id", ""))))
        if ordered:
            feature = ordered[0]
            identity = (group[0], str(feature["properties"].get("candidate_id", "")))
            if identity not in used and len(selected) < max_samples:
                selected.append(feature); used.add(identity)
    if len(selected) < max_samples:
        remainder = sorted(features, key=lambda feature: (
            str(feature.get("properties", {}).get("pair_id", "")),
            str(feature.get("properties", {}).get("candidate_id", ""))))
        for feature in remainder:
            props = feature.get("properties", {})
            identity = (str(props.get("pair_id", "")), str(props.get("candidate_id", "")))
            if identity not in used:
                selected.append(feature); used.add(identity)
                if len(selected) >= max_samples:
                    break
    rows = []
    for feature in selected:
        props = feature["properties"]
        center = shape(feature["geometry"]).centroid
        rows.append({"pair_id": props.get("pair_id"), "candidate_id": props.get("candidate_id"),
                     "role": props.get("role_at_generation", "UNASSIGNED"),
                     "center_x_epsg32650": float(center.x), "center_y_epsg32650": float(center.y),
                     "all_machine_labels_unknown": all(props[field]["label"] == "UNKNOWN" for field in SUGGESTION_FIELDS),
                     "selection_basis": "DETERMINISTIC_PAIR_AND_FROZEN_ROLE_COVERAGE_NOT_MACHINE_SCORE"})
    return rows


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
