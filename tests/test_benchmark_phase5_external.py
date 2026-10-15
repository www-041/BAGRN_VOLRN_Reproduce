"""Synthetic output parser and subprocess-isolation contract tests."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import subprocess

from affine import Affine
import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_phase5.inputs import prepare_pair
from benchmark_v1_phase5.protocol import PairManifest, load_config
from benchmark_v1_phase5.external import (
    DirectTranslation, parse_arosics_result, parse_karios_csv,
    run_external,
)


def _pair(tmp_path, *, resize_side=64):
    paths = []
    for name, transform in (("i", Affine(14, 0, 100, 0, -14, 300)),
                            ("j", Affine(14, 0, 128, 0, -14, 286))):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                           count=1, dtype="float32", crs="EPSG:32650",
                           transform=transform, nodata=0) as dst:
            data = np.random.default_rng(9).random((64, 64), dtype=np.float32)
            dst.write(data, 1)
            dst.write_mask(np.ones(data.shape, dtype="uint8") * 255)
        paths.append(path)
    i, j = paths
    manifest = PairManifest("00_01", "i", "j", i, j, i.stat().st_size, j.stat().st_size,
                            hashlib.sha256(i.read_bytes()).hexdigest(),
                            hashlib.sha256(j.read_bytes()).hexdigest())
    config = replace(load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                     max_view_side=resize_side)
    return prepare_pair(manifest, config), config


def test_karios_csv_restores_reference_to_monitored_native_correspondences(tmp_path):
    pair, _ = _pair(tmp_path)
    path = tmp_path / "KLT_matcher.csv"
    path.write_text("x0;y0;dx;dy;score\n10;12;2.5;-1.25;0.9\n",
                    encoding="utf-8")
    matches = parse_karios_csv(path, pair)
    assert matches.method == "KARIOS"
    assert matches.i_view_xy[0].tolist() == [10, 12]
    assert matches.j_view_xy[0].tolist() == [12.5, 10.75]
    np.testing.assert_allclose(matches.j_rc - matches.i_rc, [[-2.25, .5]], atol=1e-6)


@pytest.mark.parametrize("csv_text", [
    "x0;y0;dx;dx\n1;2;3;4\n",
    "x0;y0;dx\n1;2;3\n",
    "x0;y0;dx;dy\n1;2;NaN;4\n",
])
def test_karios_duplicate_missing_and_nonfinite_columns_fail(tmp_path, csv_text):
    pair, _ = _pair(tmp_path)
    path = tmp_path / "bad.csv"
    path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_karios_csv(path, pair)


def test_karios_mask_invalid_points_are_filtered_and_empty_is_valid(tmp_path):
    pair, _ = _pair(tmp_path)
    pair.view.mask_i[10, 10] = False
    path = tmp_path / "points.csv"
    path.write_text("x0;y0;dx;dy\n10;10;0;0\n20;20;0;0\n", encoding="utf-8")
    matches = parse_karios_csv(path, pair)
    assert len(matches.i_rc) == 1
    assert matches.i_view_xy[0].tolist() == [20, 20]


def test_arosics_raw_correction_sign_and_affine_offset_are_converted():
    # For a target feature displacement (+2 row,+3 col), COREG's geocoding
    # correction is (-3 x,-2 y); the output is full native scene-i -> scene-j.
    pair = _simple_pair()
    parsed = parse_arosics_result({"success": True, "x_shift_px": -3.0,
                                   "y_shift_px": -2.0,
                                   "shift_convention": "target_geocoding_correction",
                                   "sign_verified": True,
                                   "library_version": "synthetic"}, pair)
    assert isinstance(parsed, DirectTranslation)
    assert parsed.status == "SUCCESS"
    np.testing.assert_allclose(parsed.delta_row_col, (1.0, 1.0), atol=1e-8)
    assert parsed.raw_shift_xy == (-3.0, -2.0)


def test_arosics_view_pixel_shift_uses_scaled_staged_affine():
    # A 0.5x view means one view pixel spans two native pixels.
    pair = _simple_pair(resize_x=0.5, resize_y=0.5)
    parsed = parse_arosics_result({"success": True, "x_shift_px": -3.0,
                                   "y_shift_px": -2.0,
                                   "shift_convention": "target_geocoding_correction",
                                   "sign_verified": True,
                                   "library_version": "synthetic"}, pair)
    np.testing.assert_allclose(parsed.delta_row_col, (3.0, 4.0), atol=1e-8)


def _simple_pair(*, resize_x=1.0, resize_y=1.0):
    from benchmark_v1_phase5.inputs import MatchView, PairInput, ViewProvenance
    overlap = Affine(14, 0, 128, 0, -14, 286)
    view = MatchView(np.ones((64, 64), dtype=np.float32),
                     np.ones((64, 64), dtype=np.float32),
                     np.ones((64, 64), dtype=bool), np.ones((64, 64), dtype=bool),
                     np.ones((64, 64), dtype=bool),
                     ViewProvenance(Affine(14, 0, 100, 0, -14, 300),
                                    Affine(14, 0, 128, 0, -14, 286), overlap,
                                    resize_x, resize_y, (2, 98), (0, 1), (0, 1),
                                    "bilinear", "nearest"))
    return PairInput("00_01", "READY", view,
                     Affine(14, 0, 100, 0, -14, 300),
                     Affine(14, 0, 128, 0, -14, 286), 4096, 4096)


@pytest.mark.parametrize("payload", [
    {"success": False, "x_shift_px": None, "y_shift_px": None,
     "shift_convention": "target_geocoding_correction", "sign_verified": True},
    {"success": True, "x_shift_px": 1, "y_shift_px": 2,
     "shift_convention": "unknown", "sign_verified": True},
    {"success": True, "x_shift_px": 1, "y_shift_px": 2,
     "shift_convention": "target_geocoding_correction", "sign_verified": False},
    {"success": True, "x_shift_px": float("nan"), "y_shift_px": 0,
     "shift_convention": "target_geocoding_correction", "sign_verified": True},
])
def test_arosics_failure_unknown_sign_and_nonfinite_are_rejected(payload):
    with pytest.raises(ValueError):
        parse_arosics_result(payload, _simple_pair())


def test_external_missing_environment_and_nonoverlap_are_distinct(tmp_path):
    pair, config = _pair(tmp_path)
    configured = replace(config, external_python=(("KARIOS", str(tmp_path / "missing-python.exe")),
                                                  ("AROSICS_GLOBAL", None)))
    result = run_external("KARIOS", pair, configured, tmp_path / "missing-python.exe",
                          tmp_path / "out")
    assert result.status == "ENV_UNAVAILABLE"
    assert not (tmp_path / "out").exists()
    empty = replace(pair, status="NO_VALID_OVERLAP", view=None)
    result = run_external("KARIOS", empty, configured, tmp_path / "missing-python.exe",
                          tmp_path / "out")
    assert result.status == "NO_VALID_OVERLAP"


def test_external_python_must_match_explicit_isolated_config(tmp_path):
    pair, config = _pair(tmp_path)
    executable = tmp_path / "isolated-python.exe"
    executable.write_bytes(b"synthetic executable marker")
    result = run_external("KARIOS", pair, config, executable, tmp_path / "out")
    assert result.status == "ENV_UNAVAILABLE"
    assert result.reason == "EXTERNAL_ENV_NOT_CONFIGURED"
    configured = replace(config, external_python=(("KARIOS", str(tmp_path / "other-python.exe")),
                                                   ("AROSICS_GLOBAL", None)))
    with pytest.raises(ValueError, match="does not match frozen config"):
        run_external("KARIOS", pair, configured, executable, tmp_path / "out")
    registration_python = (ROOT / ".venv-registration" / "Scripts" / "python.exe").resolve()
    project_config = replace(config, external_python=(("KARIOS", str(registration_python)),
                                                       ("AROSICS_GLOBAL", None)))
    rejected = run_external("KARIOS", pair, project_config, registration_python,
                            tmp_path / "registration_env")
    assert rejected.status == "ENV_UNAVAILABLE"
    assert rejected.reason == "EXTERNAL_ENV_MUST_NOT_USE_REGISTRATION_ENV"


def test_external_subprocess_receives_only_staged_view_and_returns_parsed_result(
        tmp_path, monkeypatch):
    import benchmark_v1_phase5.external as external
    pair, config = _pair(tmp_path)
    executable = tmp_path / "isolated-python.exe"
    executable.write_bytes(b"synthetic executable marker")
    config = replace(config, external_python=(("KARIOS", str(executable)),
                                               ("AROSICS_GLOBAL", None)))

    def fake_run(command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["method"] == "KARIOS"
        assert Path(request["reference_path"]).parent == request_path.parent
        assert Path(request["target_path"]).parent == request_path.parent
        assert request["parameters"]["input_scale"] == "uint8_round_0_255"
        matches = request_path.parent / "synthetic_matches.csv"
        matches.write_text("x0;y0;dx;dy\n10;10;0;0\n", encoding="utf-8")
        response_path.write_text(json.dumps({"status": "SUCCESS",
                                             "matches_csv": str(matches)}),
                                 encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "worker ok", "")

    monkeypatch.setattr(external.subprocess, "run", fake_run)
    result = run_external("KARIOS", pair, config, executable, tmp_path / "external_run")
    assert result.status == "SUCCESS"
    assert result.matches.method == "KARIOS"
    assert len(result.matches.i_rc) == 1
    assert result.provenance["external_python"] == str(executable.resolve())


@pytest.mark.parametrize("outcome", ["timeout", "no_response"])
def test_attempted_external_failures_record_interpreter(outcome, tmp_path, monkeypatch):
    import benchmark_v1_phase5.external as external
    pair, config = _pair(tmp_path)
    executable = tmp_path / "isolated-python.exe"
    executable.write_bytes(b"synthetic executable marker")
    config = replace(config, external_python=(("KARIOS", str(executable)),
                                               ("AROSICS_GLOBAL", None)))

    def fake_run(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1, "", "worker failed")

    monkeypatch.setattr(external.subprocess, "run", fake_run)
    result = run_external("KARIOS", pair, config, executable, tmp_path / outcome)
    expected_status = "ALGORITHM_FAIL" if outcome == "timeout" else "ENV_UNAVAILABLE"
    assert result.status == expected_status
    assert result.provenance["external_python"] == str(executable.resolve())


@pytest.mark.parametrize("response_text, expected_prefix", [
    ("not-json", "INVALID_WORKER_RESPONSE"),
    ("[]", "INVALID_WORKER_RESPONSE"),
    ('{"status":"SUCCESS"}', "INVALID_WORKER_RESULT"),
])
def test_malformed_external_worker_responses_become_algorithm_failures(
        tmp_path, monkeypatch, response_text, expected_prefix):
    import benchmark_v1_phase5.external as external
    pair, config = _pair(tmp_path)
    executable = tmp_path / "isolated-python.exe"
    executable.write_bytes(b"synthetic executable marker")
    config = replace(config, external_python=(("KARIOS", str(executable)),
                                               ("AROSICS_GLOBAL", None)))

    def fake_run(command, **kwargs):
        response_path = Path(command[command.index("--response") + 1])
        response_path.write_text(response_text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "worker", "")

    monkeypatch.setattr(external.subprocess, "run", fake_run)
    result = run_external("KARIOS", pair, config, executable, tmp_path / "malformed")
    assert result.status == "ALGORITHM_FAIL"
    assert result.reason.startswith(expected_prefix)
    assert result.provenance["external_python"] == str(executable.resolve())
