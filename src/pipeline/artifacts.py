"""Artifact identity, stage manifests, and conservative resume checks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import file_sha256


MANIFEST_NAME = "stage_manifest.json"
SUCCESS_STATUS = "PASS"


def artifact_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "sha256": file_sha256(path)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_stage_manifest(
    stage_dir: str | Path,
    *,
    stage_name: str,
    config_sha256: str,
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    protocol_version: str,
    status: str = SUCCESS_STATUS,
    critical_parameters: Mapping[str, Any] | None = None,
) -> Path:
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage_name": stage_name,
        "status": status,
        "protocol_version": protocol_version,
        "created_utc": _utc_now(),
        "config_sha256": config_sha256,
        "upstream_artifacts": dict(upstream_artifacts),
        "outputs": dict(outputs),
        "critical_parameters": dict(critical_parameters or {}),
    }
    target = stage_dir / MANIFEST_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def read_stage_manifest(stage_dir: str | Path) -> dict[str, Any] | None:
    path = Path(stage_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def stage_is_reusable(
    stage_dir: str | Path,
    *,
    expected_config_sha256: str,
    expected_upstream_artifacts: Mapping[str, Mapping[str, Any]],
    required_outputs: Sequence[str | Path],
    protocol_version: str,
) -> bool:
    manifest = read_stage_manifest(stage_dir)
    if not manifest or manifest.get("status") != SUCCESS_STATUS:
        return False
    if manifest.get("protocol_version") != protocol_version:
        return False
    if manifest.get("config_sha256") != expected_config_sha256:
        return False
    if manifest.get("upstream_artifacts") != dict(expected_upstream_artifacts):
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return False
    expected_outputs = list(outputs.values())
    paths_to_check = list(required_outputs)
    for expected in expected_outputs:
        if isinstance(expected, Mapping) and expected.get("path") not in {None, ""}:
            paths_to_check.append(expected["path"])
    for required in paths_to_check:
        required = Path(required)
        if not required.is_file():
            return False
        expected = next((item for item in expected_outputs if isinstance(item, Mapping) and item.get("path") == str(required)), None)
        if expected is None and required in [Path(item) for item in required_outputs]:
            return False
        if expected is None:
            continue
        try:
            current = artifact_identity(required)
        except OSError:
            return False
        if current.get("sha256") != expected.get("sha256") or current.get("size") != expected.get("size"):
            return False
    return True


def quarantine_incomplete(stage_dir: str | Path) -> Path | None:
    """Move a partial stage aside so a rerun cannot mistake it for a result."""
    stage_dir = Path(stage_dir)
    if not stage_dir.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = stage_dir.with_name(f"{stage_dir.name}_incomplete_{stamp}")
    os.replace(stage_dir, target)
    return target
