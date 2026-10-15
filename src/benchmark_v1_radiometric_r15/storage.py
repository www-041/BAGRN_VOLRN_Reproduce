from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape


def _normalize(features: list[dict[str, Any]], *, force_unverified: bool) -> dict[str, Any]:
    seen = set()
    result = []
    for source in features:
        props = dict(source.get("properties", {}))
        key = (str(props.get("pair_id", "")), str(props.get("candidate_id", "")))
        if not all(key) or key in seen:
            raise ValueError(f"missing or duplicate candidate identity: {key}")
        seen.add(key)
        geometry = shape(source["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"invalid candidate geometry: {key}")
        props.setdefault("review_status", "UNVERIFIED")
        if force_unverified:
            props["review_status"] = "UNVERIFIED"
        props.setdefault("reviewer", "")
        props.setdefault("review_time_utc", "")
        props.setdefault("eligible_for_test", False)
        if force_unverified:
            props["eligible_for_test"] = False
        result.append({"type": "Feature", "geometry": mapping(geometry), "properties": props})
    return {"type": "FeatureCollection", "name": "R1.5_auto_roi_proposals_unverified",
            "crs": {"type": "name", "properties": {"name": "EPSG:32650"}}, "features": result}


def write_candidate_artifacts(geojson_path: str | Path, features: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Create immutable candidate GeoJSON and CSV atomically; never overwrite either."""
    target = Path(geojson_path)
    csv_path = target.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate artifact: {target}")
    collection = _normalize(features, force_unverified=True)
    fields = sorted({key for feature in collection["features"] for key in feature["properties"]})
    temps: list[tuple[Path, Path]] = []
    try:
        for output, writer in (
            (target, lambda stream: json.dump(collection, stream, ensure_ascii=False, indent=2)),
            (csv_path, lambda stream: _write_csv(stream, collection["features"], fields)),
        ):
            fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
            os.close(fd)
            temp = Path(temp_name)
            temps.append((temp, output))
            with temp.open("w", encoding="utf-8", newline="") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
        if target.exists() or csv_path.exists():
            raise FileExistsError("candidate output appeared during write; refusing replacement")
        for temp, output in temps:
            temp.replace(output)
        return target, csv_path
    except Exception:
        for temp, _ in temps:
            temp.unlink(missing_ok=True)
        raise


def _write_csv(stream, features, fields):
    writer = csv.DictWriter(stream, fieldnames=["geometry"] + fields)
    writer.writeheader()
    for feature in features:
        writer.writerow({"geometry": json.dumps(feature["geometry"], separators=(",", ":")),
                         **feature["properties"]})


def load_candidate_collection(path: str | Path, expected_pair_id: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        collection = json.load(stream)
    if collection.get("type") != "FeatureCollection":
        raise ValueError("candidate artifact must be a GeoJSON FeatureCollection")
    normalized = _normalize(collection.get("features", []), force_unverified=False)
    for feature in normalized["features"]:
        pair = str(feature["properties"].get("pair_id", ""))
        if pair != expected_pair_id:
            raise ValueError(f"candidate pair mismatch: expected {expected_pair_id}, found {pair}")
    return normalized
