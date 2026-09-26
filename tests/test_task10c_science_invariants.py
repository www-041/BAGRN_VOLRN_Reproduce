"""Synthetic Task10C invariants that do not depend on visual mosaics."""

import json

import numpy as np
from rasterio.transform import Affine

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric
from tests.multiscene_sift.test_fixed_radiometric_runner import _inputs


def _local_radiometry_case():
    transform = Affine(1, 0, 0, 0, -1, 16)
    bounds = (0.0, 0.0, 16.0, 16.0)
    yy, xx = np.mgrid[0:16, 0:16]
    base = (100.0 + 2.0 * xx + 3.0 * yy)[None, ...]
    scene1 = base.copy()
    scene1[:, :8, :] *= 1.25
    scene1[:, 8:, :] *= 0.75
    scene2 = base.copy()
    scene2[:, :8, :] *= 0.85
    scene2[:, 8:, :] *= 1.15
    arrays = [base, scene1, scene2]
    overlaps = [
        {
            "idx_i": i,
            "idx_j": j,
            "window_i": (0, 16, 0, 16),
            "window_j": (0, 16, 0, 16),
        }
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return arrays, [transform] * 3, [bounds] * 3, [None] * 3, overlaps


def _local_pair_mae(arrays):
    values = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for rows in (slice(0, 8), slice(8, 16)):
            values.append(float(np.mean(np.abs(arrays[i][0, rows] - arrays[j][0, rows]))))
    return float(np.mean(values))


def test_bagrn_volrn_improves_known_spatially_varying_mismatch():
    arrays, transforms, bounds, nodata, overlaps = _local_radiometry_case()
    bagrn, _, _ = bagrn_normalize(arrays, nodata, overlaps, control_idx=0)
    volrn, coefficients, diagnostics = volrn_normalize(
        bagrn,
        transforms,
        bounds,
        nodata,
        block_size_pixels=4,
        lambda_param=0.5,
        rho=1.0,
        max_iter=200,
        tol=1e-4,
        return_diagnostics=True,
    )

    assert np.isfinite(coefficients).all()
    assert diagnostics["all_converged"]
    assert any(
        abs(float(a) - 1.0) > 1e-6 or abs(float(b)) > 1e-6
        for a, b in coefficients.reshape(-1, 2)
    )
    assert _local_pair_mae(volrn) < _local_pair_mae(bagrn)


def test_nonconverged_runner_result_is_not_scientific_pass(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    result = run_fixed_geometry_radiometric(
        source,
        global_dir,
        grid,
        tmp_path / "nonconverged",
        method="BAGRN_VOLRN",
        geometry_run="sift_mst",
        block_size_pixels=8,
        max_iter=1,
    )

    assert result["science_status"] == "COMPLETED_NONCONVERGED"
    assert result["science_status"] != "PASS_CONVERGED"


def test_all_radiometric_methods_emit_six_metric_schema(tmp_path):
    source, global_dir, grid = _inputs(tmp_path)
    for method in ("RAW", "BAGRN", "BAGRN_VOLRN"):
        output = tmp_path / method
        run_fixed_geometry_radiometric(
            source,
            global_dir,
            grid,
            output,
            method=method,
            geometry_run="sift_mst",
            block_size_pixels=8,
            max_iter=4,
        )
        summary = json.loads((output / "radiometric_summary.json").read_text(encoding="utf-8"))
        assert set(summary["paper_metrics"]) == {"ADM", "ADSD", "CD", "GL", "RDOA", "Ave"}
        assert "additional_diagnostics" in summary
