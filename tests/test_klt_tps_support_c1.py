import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "dz01_model_c1_holdout_manifest.json"


def _config():
    return {
        "registration_band": "B12",
        "selected_bands": ["B12", "B14"],
        "scenes": [
            {"id": "scene_20251114"},
            {"id": "scene_20251120"},
        ],
        "registration_params": {"registration_backend": "klt_tps"},
    }


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_tps_support_c1_protocol_accepts_frozen_b14_manifest():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    result = validate_tps_support_c1_protocol(_config(), _manifest())

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["registration_band"] == "B12"
    assert result["validation_band"] == "B14"
    assert result["reserved_count"] == 7
    assert result["selected_block_size"] == 384
    assert result["taper_pixels"] == 64


def test_tps_support_c1_protocol_requires_klt_tps_backend():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["registration_params"]["registration_backend"] = "legacy"

    result = validate_tps_support_c1_protocol(config, _manifest())

    assert result["valid"] is False
    assert any("klt_tps" in error for error in result["errors"])


def test_tps_support_c1_protocol_requires_b12_registration_and_b14_validation():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["registration_band"] = "B14"
    config["selected_bands"] = ["B12", "B14"]

    result = validate_tps_support_c1_protocol(
        config, _manifest(), validation_band="B12"
    )

    assert result["valid"] is False
    assert any("registration band" in error.lower() for error in result["errors"])
    assert any("validation band" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_rejects_empty_pair_windows():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    manifest = _manifest()
    manifest["pairs"]["0-1"]["reserved_windows"] = []

    result = validate_tps_support_c1_protocol(_config(), manifest)

    assert result["valid"] is False
    assert any("window" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_requires_exactly_seven_384_windows():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    manifest = _manifest()
    manifest["pairs"]["0-1"]["reserved_windows"].append(
        {"row": 10, "col": 10, "height": 384, "width": 384}
    )

    result = validate_tps_support_c1_protocol(_config(), manifest)

    assert result["valid"] is False
    assert any("seven" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_rejects_scene_id_mismatch():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["scenes"][1]["id"] = "other_scene"

    result = validate_tps_support_c1_protocol(config, _manifest())

    assert result["valid"] is False
    assert any("scene" in error.lower() for error in result["errors"])


def test_tps_support_c1_array_sha256_is_shape_dtype_and_content_sensitive():
    from src.klt_tps_support_c1 import array_sha256

    base = np.array([[1, 2], [3, 4]], dtype=np.int16)

    assert array_sha256(base) != array_sha256(base.astype(np.float32))
    assert array_sha256(base) != array_sha256(base.reshape(1, 4))
    changed = base.copy()
    changed[0, 0] = 9
    assert array_sha256(base) != array_sha256(changed)
    assert array_sha256(base) == array_sha256(np.ascontiguousarray(base))


def test_validation_block_keys_are_sorted_and_ignore_acceptance_status():
    from src.klt_tps_support_c1 import validation_block_keys

    validation = {
        "edges": [
            {
                "idx_i": 0,
                "idx_j": 1,
                "validation_block_size_selected": 384,
                "blocks": [
                    {
                        "validation_row": 512,
                        "validation_col": 64,
                        "block_size": 384,
                        "accepted": False,
                    },
                    {
                        "validation_row": 256,
                        "validation_col": 640,
                        "block_size": 384,
                        "accepted": True,
                    },
                ],
            }
        ]
    }

    assert validation_block_keys(validation) == [
        (0, 1, 256, 640, 384),
        (0, 1, 512, 64, 384),
    ]
