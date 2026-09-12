"""Synthetic tests for TPS-FOLD-D1 failure-only diagnostic artifacts."""

from types import SimpleNamespace

import numpy as np
from rasterio.transform import Affine


def _failure_registration(shape=(32, 40)):
    from src.klt_tps_registration import analyze_tps_dense_flow

    height, width = shape
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0] = -2 * np.indices(shape)[1]
    analysis = analyze_tps_dense_flow(flow)
    points = np.asarray([[6, 6], [30, 6], [6, 24], [30, 24]], dtype=float)
    displacement = np.tile([1.0, -0.5], (len(points), 1))
    failure_diagnostics = {
        "stage": "tps_geometry_gate",
        "accepted_point_count": len(points),
        "control_displacement": {
            "dx": {"count": len(points), "min": 1.0, "max": 1.0},
            "dy": {"count": len(points), "min": -0.5, "max": -0.5},
            "magnitude": {"count": len(points), "min": 1.118, "max": 1.118},
        },
        "control_support": {
            "control_bbox": [6.0, 6.0, 30.0, 24.0],
            "control_hull_available": True,
            "control_hull_vertices_xy": [[6.0, 6.0], [30.0, 6.0], [30.0, 24.0], [6.0, 24.0]],
            "control_hull_pixels": 475,
            "control_hull_fraction": 475 / (height * width),
            "target_overlap_window": [2, 30, 3, 37],
            "target_overlap_bbox_pixels": 28 * 34,
            "target_overlap_bbox_fraction": (28 * 34) / (height * width),
        },
        "field_displacement": analysis["summary"],
        "jacobian": {
            "min": analysis["summary"]["jacobian_min"],
            "p01": analysis["summary"]["jacobian_p01"],
            "p05": analysis["summary"]["jacobian_p05"],
            "median": analysis["summary"]["jacobian_median"],
            "p95": analysis["summary"]["jacobian_p95"],
            "p99": analysis["summary"]["jacobian_p99"],
            "max": analysis["summary"]["jacobian_max"],
        },
        "fold_support": {"fold_pixels_total": analysis["summary"]["fold_pixels"]},
    }
    band = np.arange(height * width, dtype=np.float32).reshape(shape)
    return {
        "status": "fail",
        "connected": False,
        "registration_backend": "klt_tps",
        "registration_band_name": "B12",
        "validation_band_name": "B14",
        "quality": {"quality": "fail", "rmse": None, "p95": None, "median": None},
        "failure": {"code": "registration_connectivity_failed", "reason": "TPS geometry rejected"},
        "diagnostics": {
            "registration_blocked": True,
            "klt_tps": {
                "available": False,
                "failure_reason": "KLT/TPS geometry rejected: TPS flow contains fold pixels: 1200",
                "failure_diagnostics": failure_diagnostics,
            },
        },
        "klt_tps": {
            "available": False,
            "failure_reason": "KLT/TPS geometry rejected: TPS flow contains fold pixels: 1200",
            "accepted_point_count": len(points),
            "initial_corner_count": len(points),
            "reference_points_overlap_xy": points,
            "moving_points_overlap_xy": points + [1.0, -0.5],
            "control_points_moving_xy": points,
            "source_points_moving_xy": points + [1.0, -0.5],
            "displacement_xy": displacement,
            "forward_backward_error": np.zeros(len(points)),
            "failure_diagnostics": failure_diagnostics,
            "_failure_diagnostic_arrays": {
                "flow": flow,
                "jacobian_determinant": analysis["jacobian_determinant"],
                "fold_mask": analysis["fold_mask"],
            },
        },
    }, band


def _scene_data(band):
    return {
        "arrays": [np.stack([band, band + 1]), np.stack([band + 2, band + 3])],
        "transforms": [Affine.identity(), Affine.identity()],
        "nodata_values": [None, None],
        "crs": "EPSG:3857",
    }


def test_klt_tps_failure_writer_writes_control_csv(tmp_path):
    from scripts.diagnose_registration_pair import write_klt_tps_failure_artifacts

    registration, band = _failure_registration()
    paths = write_klt_tps_failure_artifacts(
        registration, _scene_data(band), tmp_path, registration_band_idx=0,
    )

    assert (tmp_path / "klt_tps_control_points.csv").exists()
    assert paths["klt_tps_control_points"].endswith("klt_tps_control_points.csv")


def test_klt_tps_failure_writer_writes_pregate_flow_and_fold_mask(tmp_path):
    from scripts.diagnose_registration_pair import write_klt_tps_failure_artifacts

    registration, band = _failure_registration()
    paths = write_klt_tps_failure_artifacts(
        registration, _scene_data(band), tmp_path, registration_band_idx=0,
    )

    assert (tmp_path / "klt_tps_pregate_displacement.tif").exists()
    assert (tmp_path / "klt_tps_fold_mask.tif").exists()
    assert paths["klt_tps_pregate_displacement"].endswith("klt_tps_pregate_displacement.tif")
    assert paths["klt_tps_fold_mask"].endswith("klt_tps_fold_mask.tif")


def test_klt_tps_failure_writer_writes_failure_map_and_field_pngs(tmp_path):
    from scripts.diagnose_registration_pair import write_klt_tps_failure_artifacts

    registration, band = _failure_registration()
    paths = write_klt_tps_failure_artifacts(
        registration, _scene_data(band), tmp_path, registration_band_idx=0,
    )

    for name in (
        "klt_tps_pregate_displacement_magnitude.png",
        "klt_tps_jacobian_determinant.png",
        "klt_tps_failure_map.png",
    ):
        assert (tmp_path / name).exists()
        assert str(tmp_path / name) in paths.values()


def test_klt_tps_failure_payload_keeps_compact_statistics_only(tmp_path):
    from scripts.diagnose_registration_pair import build_diagnostic_payload

    registration, _ = _failure_registration()
    build_diagnostic_payload(registration, ["ref", "moving"], tmp_path)
    text = (tmp_path / "registration_diagnostics.json").read_text(encoding="utf-8")

    assert "_failure_diagnostic_arrays" not in text
    assert "jacobian_determinant" not in text
    assert "fold_mask" not in text
    assert "fold_pixels_total" in text


def test_blocked_klt_tps_result_can_write_failure_diagnostics(tmp_path):
    from scripts.diagnose_registration_pair import write_klt_tps_failure_artifacts

    registration, band = _failure_registration()
    paths = write_klt_tps_failure_artifacts(
        registration, _scene_data(band), tmp_path, registration_band_idx=0,
    )

    assert paths
    assert (tmp_path / "klt_tps_failure_map.png").exists()


def test_normal_artifact_writer_still_rejects_blocked_klt_tps_result(tmp_path):
    import pytest
    from scripts.diagnose_registration_pair import write_diagnostic_artifacts

    registration, band = _failure_registration()
    with pytest.raises(ValueError, match="blocked/disconnected"):
        write_diagnostic_artifacts(
            registration, _scene_data(band), ["ref", "moving"], tmp_path,
        )


def test_blocked_klt_tps_main_path_writes_failure_diagnostics(tmp_path, monkeypatch):
    from scripts import diagnose_registration_pair

    registration, band = _failure_registration()
    config = SimpleNamespace(
        scenes=[{"id": "ref"}, {"id": "moving"}],
        control_scene="ref",
        output_root=str(tmp_path),
        registration_band="B12",
        registration_params={"required_quality": "pass"},
    )

    class FakePipeline:
        registration_band_idx = 0
        common_bands = ["B12", "B14"]

        def __init__(self, pipeline_config):
            self.config = pipeline_config

        def load_scenes(self):
            return _scene_data(band) | {"scene_ids": ["ref", "moving"]}

        def detect_overlaps(self, scene_data):
            return [{"idx_i": 0, "idx_j": 1}]

        def register_scenes(self, scene_data, overlaps, **kwargs):
            return registration

    monkeypatch.setattr(diagnose_registration_pair, "load_config", lambda _: config)
    monkeypatch.setattr(diagnose_registration_pair, "MultibandPipeline", FakePipeline)

    result = diagnose_registration_pair.main([
        "--config", "ignored.yaml", "--scene-i", "0", "--scene-j", "1",
        "--output-dir", str(tmp_path), "--validation-band", "B14",
    ])

    assert result == 1
    payload = (tmp_path / "registration_diagnostics.json").read_text(encoding="utf-8")
    assert "klt_tps_failure_map.png" in payload
    assert "fold_pixels_total" in payload
    assert "_failure_diagnostic_arrays" not in payload
