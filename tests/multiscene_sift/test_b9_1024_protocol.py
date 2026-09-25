"""Contract tests for the isolated B9 1024 matcher protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD_CONFIG = ROOT / "data/output/b9_five_scene_validation/04_frozen_five_scene_config.json"
NEW_CONFIG = ROOT / "data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_protocol_scale(config: dict) -> dict:
    normalized = json.loads(json.dumps(config))
    normalized["registration"].pop("match_max_side")
    return normalized


def test_b9_1024_config_isolated_from_frozen_1600_protocol():
    old = _load(OLD_CONFIG)
    new = _load(NEW_CONFIG)

    assert old["registration"]["match_max_side"] == 1600
    assert new["registration"]["match_max_side"] == 1024
    assert _without_protocol_scale(old) == _without_protocol_scale(new)
