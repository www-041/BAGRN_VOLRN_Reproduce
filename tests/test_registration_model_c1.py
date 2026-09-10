"""Frozen protocol tests for the diagnostic-only MODEL-C1 experiment."""

import hashlib
import json
from pathlib import Path

import yaml

from src.experiment_config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dz01_model_c1_affine_b12.yaml"
MANIFEST_PATH = ROOT / "configs" / "dz01_model_c1_holdout_manifest.json"
BAND_CONFIG_PATH = ROOT / "configs" / "dz01_registration_band_ab_b12.yaml"

AFFINE_KEYS = {
    "affine_min_controls",
    "affine_min_spatial_groups",
    "affine_min_inliers",
    "affine_min_inlier_ratio",
    "affine_ransac_residual_threshold",
    "affine_ransac_max_trials",
    "affine_ransac_seed",
    "affine_max_scale_delta",
    "affine_max_rotation_deg",
    "affine_max_shear_deg",
    "affine_max_component",
    "affine_cv_min_rmse_improvement",
    "affine_cv_min_p95_improvement",
}

EXPECTED_WINDOWS = {
    (2944, 256, 384, 384),
    (256, 640, 384, 384),
    (1536, 64, 384, 384),
    (2176, 704, 384, 384),
    (1088, 768, 384, 384),
    (640, 192, 384, 384),
    (2560, 512, 384, 384),
}


def _load(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _base_fingerprint(params):
    base = {key: value for key, value in params.items() if key not in AFFINE_KEYS}
    payload = json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_model_c1_manifest_freezes_exact_b14_holdout_windows():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pair = manifest["pairs"]["0-1"]
    actual = {
        (item["row"], item["col"], item["height"], item["width"])
        for item in pair["reserved_windows"]
    }
    assert manifest["source_registration_band"] == "B14"
    assert manifest["validation_band"] == "B14"
    assert pair["selected_block_size"] == 384
    assert actual == EXPECTED_WINDOWS
    assert len(actual) == 7


def test_model_c1_config_is_b12_with_same_base_protocol_and_explicit_affine_params():
    config = _load(CONFIG_PATH)
    band_config = _load(BAND_CONFIG_PATH)
    assert config["registration_band"] == "B12"
    assert config["selected_bands"] == ["B12", "B14"]
    assert [scene["id"] for scene in config["scenes"]] == [
        scene["id"] for scene in band_config["scenes"]
    ]
    assert config["scenes"] == band_config["scenes"]
    assert _base_fingerprint(load_config(str(CONFIG_PATH)).registration_params) == (
        "08864b73e8a53990aa4d74038cdb269d10b0cddd7ebc2df26d5d6654713ccb98"
    )
    assert set(AFFINE_KEYS).issubset(config["registration_params"])
    assert config["registration_params"]["affine_ransac_seed"] == 42


def test_model_c1_manifest_records_expected_base_fingerprint():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["base_registration_params_sha256"] == (
        "08864b73e8a53990aa4d74038cdb269d10b0cddd7ebc2df26d5d6654713ccb98"
    )
