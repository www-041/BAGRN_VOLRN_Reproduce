"""RED tests for the frozen Task10D metric protocol."""

from copy import deepcopy
from pathlib import Path

import pytest

from src.multiscene_sift.task10d_metric_protocol import (
    load_task10d_metric_protocol,
    validate_task10d_metric_protocol,
)


PROTOCOL_PATH = Path("data/output/b9_five_scene_validation/task10d_metric_protocol.json")


def test_task10d_protocol_freezes_exact_metric_definitions():
    protocol = load_task10d_metric_protocol(PROTOCOL_PATH)

    assert protocol["local_grid"]["tile_size_pixels"] == 256
    assert protocol["local_grid"]["min_shared_valid_pixels"] == 4096
    assert protocol["rdd"]["implementation"] == "scipy.stats.wasserstein_distance"
    assert protocol["cgl"]["gradient_operator"] == "scipy.ndimage.sobel"
    assert protocol["cgl"]["angle_difference"] == "abs(atan2(sin(delta), cos(delta)))"
    assert protocol["cgl"]["gradient_threshold"] == 1e-12
    assert protocol["seam_zone"]["min_normalized_weight"] == 0.25
    assert protocol["paper_metrics"]["cd_status"] == "UNVERIFIED"
    assert protocol["paper_metrics"]["gl_status"] == "UNVERIFIED"
    assert validate_task10d_metric_protocol(protocol) is None


def test_task10d_protocol_rejects_runtime_definition_override():
    protocol = load_task10d_metric_protocol(PROTOCOL_PATH)
    overridden = deepcopy(protocol)
    overridden["local_grid"]["tile_size_pixels"] = 400

    with pytest.raises(ValueError, match="tile_size_pixels"):
        validate_task10d_metric_protocol(overridden)
