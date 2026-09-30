"""Task15 end-to-end orchestrator.

This module is intentionally an adapter: registration, canonical warping,
BAGRN, seam/local correction, labeling, and blending stay in their existing
scientific modules.  It owns only configuration, checkpoints, provenance, and
the generic 5/13-scene wiring.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .artifacts import artifact_identity, quarantine_incomplete, stage_is_reusable, write_stage_manifest
from .config import PipelineConfig
from .provenance import canonical_sha256
from .resources import discover_selected_records, resource_snapshot, validate_environment
from .stages import STAGE_NAMES, stage_directory, stage_name_to_index


PROTOCOL_VERSION = "task15-v1"
HARD_STOP_FRESH_OUTPUT_EXISTS = "HARD_STOP_FRESH_OUTPUT_EXISTS"
HARD_STOP_UPSTREAM_PROVENANCE = "HARD_STOP_UPSTREAM_PROVENANCE"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_root(config: PipelineConfig) -> Path:
    root = config.outputs.root
    return root if root.is_absolute() else _repo_root() / root


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _files_in(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file() and item.name not in {"stage_manifest.json"})


def _write_manifest_for_stage(root: Path, index: int, config_hash: str, upstream: dict[str, dict[str, Any]] | None = None) -> Path:
    directory = stage_directory(root, index)
    files = _files_in(directory) if directory.is_dir() else []
    outputs = {str(path.relative_to(root)): artifact_identity(path) for path in files}
    return write_stage_manifest(
        directory,
        stage_name=STAGE_NAMES[index],
        config_sha256=config_hash,
        upstream_artifacts=upstream or {},
        outputs=outputs,
        protocol_version=PROTOCOL_VERSION,
        critical_parameters={"stage_index": index},
    )


def _configure_legacy_adapters(root: Path, config: PipelineConfig) -> None:
    os.environ["EFFICIENT_LOFTR_REPO"] = str(config.registration.matcher_repo.resolve())
    os.environ["EFFICIENT_LOFTR_WEIGHTS"] = str(config.registration.checkpoint.resolve())


def _run_existing_prefix(root: Path, config: PipelineConfig) -> dict[str, Any]:
    """Run raw-input Stage00-06 through the established Task14 primitives."""
    from scripts import run_task14_13scene_scale as legacy

    _configure_legacy_adapters(root, config)
    prefix = legacy.run_task14(
        stage="all",
        resume=False,
        input_root=config.dataset.input_root,
        output_root=root,
        device=config.registration.device,
        expected_scene_count=config.dataset.expected_scene_count,
        scene_selection=config.dataset.scene_selection,
        include_streaming_equivalence=False,
    )
    prefix_stops = [str(item) for item in prefix.get("hard_stops", []) if str(item) != "HARD_STOP_TASK14_STAGE_NOT_IMPLEMENTED"]
    if prefix_stops:
        raise RuntimeError(";".join(prefix_stops))
    continuation = legacy.continue_task14_after_stage02(
        root,
        repo_dir=config.registration.matcher_repo,
        weights_path=config.registration.checkpoint,
        device=config.registration.device,
    )
    continuation_stops = [str(item) for item in continuation.get("hard_stops", []) if str(item) != "HARD_STOP_TASK14_MULTISCENE_ADAPTER_UNAVAILABLE"]
    if continuation_stops:
        raise RuntimeError(";".join(continuation_stops))
    return {"prefix": prefix, "continuation": continuation}


def _run_existing_downstream(root: Path, config: PipelineConfig) -> dict[str, Any]:
    """Run the proven Stage07-12 seam/label/correction adapter on this root."""
    from scripts import run_task14a_resume_13 as legacy

    legacy.OUT = root
    legacy.STAGE07 = root / "stages/07_pairwise_seam_local"
    legacy.STAGE08 = root / "stages/08_multiscene_labeling"
    legacy.STAGE09 = root / "stages/09_correction"
    legacy.STAGE10 = root / "stages/10_mosaics"
    legacy.STAGE11 = root / "stages/11_metrics"
    legacy.STAGE12 = root / "stages/12_report"
    legacy.TILE = config.streaming.tile_size
    legacy.HALO = config.streaming.halo
    from src.seam_local.config import SeamLocalRuntimeConfig
    legacy.RUNTIME_CONFIG = SeamLocalRuntimeConfig(
        intensity_weight=config.seam.intensity_weight,
        gradient_weight=config.seam.gradient_weight,
        normalization=config.seam.normalization,
        coarse_factor=config.seam.coarse_factor,
        refine_half_width=config.seam.refine_half_width,
        corridor_half_width=config.local_correction.corridor_half_width,
        segment_length=config.local_correction.segment_length,
        min_valid_pixels=config.local_correction.min_valid_pixels,
        percentile_low=config.local_correction.percentile_low,
        percentile_high=config.local_correction.percentile_high,
        stability_gain_min=config.local_correction.stability_gain_min,
        stability_gain_max=config.local_correction.stability_gain_max,
        preference_distance_scale=config.labeling.preference_distance_scale,
        tie_tolerance=config.labeling.tie_tolerance,
        blend_method=config.blend.method,
        blend_half_width=config.blend.half_width,
    )
    legacy.run()
    stage10 = legacy.STAGE10
    renames = {
        "v0_bagrn_mosaic.tif": "v0_bagrn_weighted.tif",
        "v1_mosaic.tif": "v1_multiscene_label_blend.tif",
        "v2_mosaic.tif": "v2_local_corrected_multiscene.tif",
    }
    for old_name, new_name in renames.items():
        source = stage10 / old_name
        if not source.is_file() and (stage10 / new_name).is_file():
            continue
        if not source.is_file():
            raise RuntimeError(f"HARD_STOP_PIPELINE_ARTIFACT_MISSING: {source}")
        shutil.copyfile(source, stage10 / new_name)
    mosaic_summary = stage10 / "mosaic_summary.json"
    payload = json.loads(mosaic_summary.read_text(encoding="utf-8"))
    payload["v0"] = "BAGRN + weighted feather semantics; Stage06 distance-weighted stream"
    mosaic_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "stage07": legacy.STAGE07 / "pairwise_results.json",
        "stage08": legacy.STAGE08 / "labeling_summary.json",
        "stage10": stage10 / "mosaic_summary.json",
        "stage11": legacy.STAGE11 / "metrics_summary.json",
        "stage12": legacy.STAGE12 / "scale_summary.json",
    }


def _write_pipeline_status(root: Path, *, status: str, scientific_decision: str, error: str | None = None, **extra: Any) -> Path:
    payload = {"pipeline_status": status, "scientific_decision": scientific_decision, "protocol_version": PROTOCOL_VERSION, "peak_ram_measured": False, "peak_vram_measured": False, "resource_snapshot": resource_snapshot(), **extra}
    if error:
        payload["error"] = error
    path = root / "pipeline_status.json"
    _write_json(path, payload)
    return path


def _publish_required_artifacts(root: Path) -> None:
    """Publish stable root-level paths while retaining stage-owned artifacts."""
    source_dir = root / "stages/10_mosaics"
    target_dir = root / "10_mosaics"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("v0_bagrn_weighted.tif", "v1_multiscene_label_blend.tif", "v2_local_corrected_multiscene.tif"):
        source = source_dir / name
        if source.is_file():
            shutil.copyfile(source, target_dir / name)
    source_previews = source_dir / "previews"
    target_previews = target_dir / "previews"
    target_previews.mkdir(parents=True, exist_ok=True)
    for name in ("v0_mosaic.png", "v1_mosaic.png", "v2_mosaic.png", "v0_v1_v2_comparison.png"):
        source = source_previews / name
        if source.is_file():
            shutil.copyfile(source, target_previews / name)


def run_pipeline(config: PipelineConfig, mode: str = "fresh", from_stage: int | None = None) -> dict[str, Any]:
    if mode not in {"fresh", "resume"}:
        raise ValueError("mode must be fresh or resume")
    root = _resolve_output_root(config)
    config_hash = canonical_sha256(config.as_dict())
    if mode == "fresh":
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"{HARD_STOP_FRESH_OUTPUT_EXISTS}: {root}")
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise RuntimeError(f"{HARD_STOP_UPSTREAM_PROVENANCE}: resume root does not exist: {root}")

    if mode == "resume" and from_stage is None:
        prefix_ready = all((root / relative).is_file() for relative in (
            "stages/06_bagrn/bagrn/normalized_scenes_manifest.json",
            "stages/06_bagrn/bagrn/mosaic.tif",
            "stages/06_bagrn/bagrn/radiometric_summary.json",
        ))
        final_manifest = stage_directory(root, 12)
        final_output = final_manifest / "stage_manifest.json"
        if stage_is_reusable(final_manifest, expected_config_sha256=config_hash, expected_upstream_artifacts={}, required_outputs=(), protocol_version=PROTOCOL_VERSION):
            return {"status": "RESUMED", "scientific_decision": "RECOVERED_FROM_CHECKPOINT", "output_root": str(root)}
        if not prefix_ready:
            for index in range(len(STAGE_NAMES)):
                directory = stage_directory(root, index)
                if directory.exists() and not stage_is_reusable(directory, expected_config_sha256=config_hash, expected_upstream_artifacts={}, required_outputs=(), protocol_version=PROTOCOL_VERSION):
                    quarantine_incomplete(directory)

    if from_stage is not None:
        if from_stage < 0 or from_stage >= len(STAGE_NAMES):
            raise ValueError(f"invalid from_stage: {from_stage}")
        for index in range(from_stage):
            if not (stage_directory(root, index) / "stage_manifest.json").is_file():
                raise RuntimeError(f"{HARD_STOP_UPSTREAM_PROVENANCE}: missing manifest for stage {index}")
        if mode == "resume" and from_stage < 12:
            for index in range(from_stage, len(STAGE_NAMES)):
                directory = stage_directory(root, index)
                if directory.exists():
                    quarantine_incomplete(directory)

    try:
        validate_environment(config)
        records = discover_selected_records(config)
        _write_json(root / "00_preflight" / "input_inventory.json", {"scene_count": len(records), "scene_ids": [r["scene_id"] for r in records], "records": records})
        _write_manifest_for_stage(root, 0, config_hash)
        if not (mode == "resume" and (root / "stages/06_bagrn/bagrn/normalized_scenes_manifest.json").is_file()):
            _run_existing_prefix(root, config)
        for index in range(1, 7):
            _write_manifest_for_stage(root, index, config_hash)
        if from_stage is not None and from_stage >= 12:
            from .final_report import write_final_report

            _publish_required_artifacts(root)
            report = write_final_report(root)
            _publish_required_artifacts(root)
            scientific_decision = report.get("scientific_decision", "NOT_MEASURED")
            five_gate = report.get("metrics", {}).get("quality", {}).get("five_scene_regression")
            if config.dataset.expected_scene_count == 5 and five_gate and five_gate.get("status") != "PASS":
                status_path = _write_pipeline_status(root, status="HARD_STOP", scientific_decision=five_gate["status"], error=five_gate["status"], five_scene_regression=five_gate, output_root=str(root), config_sha256=config_hash)
                raise RuntimeError(f"{five_gate['status']}; status={status_path}")
            status_path = _write_pipeline_status(root, status="SUCCESS", scientific_decision=scientific_decision, output_root=str(root), config_sha256=config_hash)
            _write_json(root / "pipeline_manifest.json", {"protocol_version": PROTOCOL_VERSION, "config_sha256": config_hash, "stage_names": list(STAGE_NAMES), "scientific_decision": scientific_decision, "task14a2_reference_is_input": False})
            (root / "pipeline.log").write_text("Task15 completed through the generic fresh-input orchestrator; see stage manifests and final report.\n", encoding="utf-8")
            return {"status": "SUCCESS", "scientific_decision": scientific_decision, "output_root": str(root), "status_path": str(status_path)}
        _run_existing_downstream(root, config)
        from .final_report import write_final_report

        _publish_required_artifacts(root)
        report = write_final_report(root)
        _publish_required_artifacts(root)
        for index in range(7, 13):
            _write_manifest_for_stage(root, index, config_hash)
        scientific_decision = report.get("scientific_decision", "NOT_MEASURED")
        five_gate = report.get("metrics", {}).get("quality", {}).get("five_scene_regression")
        if config.dataset.expected_scene_count == 5 and five_gate and five_gate.get("status") != "PASS":
            status_path = _write_pipeline_status(root, status="HARD_STOP", scientific_decision=five_gate["status"], error=five_gate["status"], five_scene_regression=five_gate, output_root=str(root), config_sha256=config_hash)
            raise RuntimeError(f"{five_gate['status']}; status={status_path}")
        status_path = _write_pipeline_status(root, status="SUCCESS", scientific_decision=scientific_decision, output_root=str(root), config_sha256=config_hash)
        _write_json(root / "pipeline_manifest.json", {"protocol_version": PROTOCOL_VERSION, "config_sha256": config_hash, "stage_names": list(STAGE_NAMES), "scientific_decision": scientific_decision, "task14a2_reference_is_input": False})
        (root / "pipeline.log").write_text("Task15 completed through the generic fresh-input orchestrator; see stage manifests and final report.\n", encoding="utf-8")
        return {"status": "SUCCESS", "scientific_decision": scientific_decision, "output_root": str(root), "status_path": str(status_path)}
    except Exception as exc:
        status_path = _write_pipeline_status(root, status="HARD_STOP", scientific_decision="NOT_READY", error=str(exc), output_root=str(root), config_sha256=config_hash)
        raise RuntimeError(f"Task15 stopped: {exc}; status={status_path}") from exc
