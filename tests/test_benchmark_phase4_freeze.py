"""Phase4 must trust only byte-identical Phase3 and first-review inputs."""

import csv
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_phase4_freeze import FrozenInputError, load_frozen_trial, quality_summary


def test_real_frozen_snapshot_has_exact_trial_roles_and_quality_labels():
    frozen = load_frozen_trial(ROOT)
    assert len(frozen.assignments) == len(frozen.candidates) == 160
    assert {role: sum(row["split"] == role for row in frozen.assignments.values())
            for role in ("DEVELOPMENT", "TEST", "UNASSIGNED")} == {
                "DEVELOPMENT": 43, "TEST": 84, "UNASSIGNED": 33}
    assert {pair: sum(key[0] == pair for key in frozen.candidates)
            for pair in ("00_01", "02_03", "05_07", "08_10")} == {
                "00_01": 40, "02_03": 40, "05_07": 40, "08_10": 40}
    quality = quality_summary(frozen)
    assert quality["high_confidence_labels"] == 160
    assert quality["human_adjusted_count"] == 0
    assert quality["half_pixel_like_notes"] == 156
    assert quality["one_pixel_like_notes"] == 4


def test_changed_candidate_bytes_hard_stop(tmp_path):
    for relative in [*(f"docs/benchmark/v1/phase2/auto_candidates/{pair}.csv"
                       for pair in ("00_01", "02_03", "05_07", "08_10")),
                     *(f"docs/benchmark/v1/phase3/{name}" for name in (
                         "checkpoint_split_assignments.csv", "split_leakage_audit.json",
                         "spatial_split_regions.geojson", "spatial_split_protocol.md",
                         "spatial_split_map.png"))]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    path = tmp_path / "docs/benchmark/v1/phase2/auto_candidates/00_01.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FrozenInputError, match="SHA256"):
        load_frozen_trial(tmp_path)


def test_changed_audit_cannot_redefine_its_own_hashes(tmp_path):
    source_root = ROOT / "docs/benchmark/v1"
    for relative in [*(f"phase2/auto_candidates/{pair}.csv"
                       for pair in ("00_01", "02_03", "05_07", "08_10")),
                     *(f"phase3/{name}" for name in (
                         "checkpoint_split_assignments.csv", "split_leakage_audit.json",
                         "spatial_split_regions.geojson", "spatial_split_protocol.md",
                         "spatial_split_map.png"))]:
        target = tmp_path / "docs/benchmark/v1" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    audit_path = tmp_path / "docs/benchmark/v1/phase3/split_leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["git_head"] = "altered"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(FrozenInputError, match="audit SHA256"):
        load_frozen_trial(tmp_path)
