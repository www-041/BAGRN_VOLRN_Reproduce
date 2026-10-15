from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .inputs import ALLOWED_PAIRS, PairInputs, load_pair_inputs
from .proposals import ProposalConfig, generate_pair_proposals
from .spatial import assign_frozen_role
from .storage import write_candidate_artifacts


OUTPUT_REL = Path("data/output/benchmark_v1_radiometric_r1_5/auto_roi_candidates")
PROTECTED_PARTS = ("docs/benchmark/v1/radiometric_r1", "data/output/benchmark_v1_radiometric_r1",
                   "data/output/final_pipeline", "docs/benchmark/v1/phase3", "docs/benchmark/v1/phase4",
                   "docs/benchmark/v1/phase5", "data/input")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _watch_files(root: Path) -> dict[str, str]:
    candidates = [
        root / "docs/benchmark/v1/scene_inventory.csv",
        root / "docs/benchmark/v1/v_band_inventory.csv",
        root / "docs/benchmark/v1/temporal_pair_inventory.csv",
        root / "docs/benchmark/v1/phase2/input_identity_manifest.csv",
        root / "docs/benchmark/v1/phase3/spatial_split_protocol.md",
        root / "docs/benchmark/v1/phase3/checkpoint_split_assignments.csv",
        root / "docs/benchmark/v1/radiometric_r1/radiometric_roi_candidates.geojson",
    ]
    review_dir = root / "data/output/benchmark_v1_radiometric_r1/roi_reviews"
    if review_dir.is_dir():
        candidates.extend(p for p in review_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".jsonl", ".json"})
    result = {}
    for path in candidates:
        if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
            result[str(path.relative_to(root))] = _sha256(path)
    return result


def _pair_header_manifest(pair: PairInputs) -> dict[str, Any]:
    result = {}
    for scene in pair.scenes:
        result[scene.scene_id] = {
            band: {"path": str(item.path), "usable": item.usable,
                   "reason": item.exclusion_reason, "header": item.header_fingerprint,
                   "recorded_sha256": item.manifest_sha256 or "NOT_FROZEN"}
            for band, item in scene.bands.items()
        }
    # Normalize tuples and scalar metadata to the exact JSON representation used on disk.
    return json.loads(json.dumps(result, sort_keys=True, default=str))


def _json_atomic(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
    temp.replace(path)


def _protected_output(path: Path, root: Path):
    resolved = path.resolve()
    expected = (root / OUTPUT_REL).resolve()
    if not resolved.is_relative_to(expected):
        raise ValueError(f"output must remain under {expected}")
    for protected in PROTECTED_PARTS:
        if resolved.is_relative_to((root / protected).resolve()):
            raise ValueError(f"refusing protected output path: {resolved}")


def execute_generation(repo_root: str | Path, pair_ids: list[str], run_id: str,
                       *, max_candidates: int = 15,
                       config: ProposalConfig | None = None) -> Path:
    root = Path(repo_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("invalid run-id; use only letters, digits, underscore or hyphen")
    if not pair_ids or any(pair not in ALLOWED_PAIRS for pair in pair_ids):
        raise ValueError("pair is not in the frozen R1.5 allowlist")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate pair requested")
    if not 0 <= max_candidates <= 30:
        raise ValueError("max_candidates must be between 0 and 30")
    cfg = config or ProposalConfig(max_candidates=max_candidates)
    if cfg.max_candidates != max_candidates:
        cfg = ProposalConfig(**{**cfg.__dict__, "max_candidates": max_candidates})

    # Validate all requested pairs and their headers before creating any output artifact.
    pairs = {pair_id: load_pair_inputs(pair_id, root) for pair_id in pair_ids}
    run_dir = root / OUTPUT_REL / run_id
    _protected_output(run_dir, root)
    watch = _watch_files(root)
    cfg_doc = {**cfg.__dict__, "auxiliary_bands": list(cfg.auxiliary_bands)}
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"run directory already exists without a valid manifest: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != cfg_doc or manifest.get("protected_file_sha256") != watch:
            raise ValueError("run config or protected input hashes changed; use a new run-id")
        for pair_id, pair in pairs.items():
            fingerprint = _pair_header_manifest(pair)
            prior = manifest.get("pair_input_headers", {}).get(pair_id)
            if prior is not None and prior != fingerprint:
                raise ValueError(f"frozen input identity changed for {pair_id}")
            if any((run_dir / f"{pair_id}{suffix}").exists()
                   for suffix in (".geojson", ".csv", ".quality.json")):
                raise FileExistsError(f"pair {pair_id} output already exists; candidates are immutable")
    else:
        manifest = {"schema_version": 1, "run_id": run_id, "config": cfg_doc,
                    "protected_file_sha256": watch, "pair_input_headers": {},
                    "generated_pairs": [], "status": "GENERATING"}
        run_dir.mkdir(parents=True, exist_ok=False)
        _json_atomic(manifest_path, manifest)

    for pair_id, pair in pairs.items():
        diag: dict[str, Any] = {}
        candidates = generate_pair_proposals(pair, cfg, diagnostics=diag)
        features = []
        for item in candidates:
            props = dict(item)
            geometry = props.pop("geometry")
            sampling_cell = props.get("grid_cell_id", "UNKNOWN")
            role, cell_id, spatial_status = assign_frozen_role(geometry)
            props.update({"role": role, "grid_cell_id": cell_id, "sampling_grid_cell_id": sampling_cell,
                          "spatial_status": spatial_status,
                          "pairwise_leakage_audit": "PENDING_ALL_PAIR_AUDIT",
                          "candidate_source": "AUTO_R1_5_BOUNDED_WINDOW",
                          "review_status": "UNVERIFIED", "reviewer": "", "review_time_utc": "",
                          "human_label": "UNKNOWN", "eligible_for_test": False,
                          "source_manifest_reference": "docs/benchmark/v1/phase2/input_identity_manifest.csv"})
            features.append({"type": "Feature", "geometry": geometry, "properties": props})
        write_candidate_artifacts(run_dir / f"{pair_id}.geojson", features)
        _json_atomic(run_dir / f"{pair_id}.quality.json", {
            "pair_id": pair_id, "candidate_count": len(features), "label_suggestions": _count(features, "suggested_label"),
            "confidence_suggestions": _count(features, "suggestion_confidence"),
            "role_counts": _count(features, "role"), "diagnostics": diag,
            "sampling_bias": {"fixed_window_px": cfg.window_px,
                "edge_fraction_max": cfg.strong_edge_fraction_max,
                "likely_underrepresented": ["narrow/boundary-rich targets", "small fragmented features",
                    "low-texture terrain", "shadowed/low-contrast areas"],
                "thresholds_relaxed": False}})
        manifest["pair_input_headers"][pair_id] = _pair_header_manifest(pair)
        manifest["generated_pairs"] = [p for p in ["00_01", "02_03", "05_07", "08_10"]
                                        if p in set(manifest.get("generated_pairs", [])) | {pair_id}]
        manifest["status"] = "PARTIAL" if len(manifest["generated_pairs"]) < 4 else "FOUR_PAIR_INPUTS_GENERATED"
        _json_atomic(manifest_path, manifest)
    return run_dir


def _count(features, key):
    counts = {}
    for feature in features:
        value = str(feature["properties"].get(key, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    return counts
