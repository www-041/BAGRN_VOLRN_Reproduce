import json

from src.multiscene_sift.radiometric_audit import build_radiometric_audit


def _write_run(root, geometry, method, mae, rmse, bias, seam):
    run = root / geometry / method
    run.mkdir(parents=True)
    (run / "radiometric_summary.json").write_text(
        json.dumps({
            "geometry_run": geometry,
            "radiometric_method": method,
            "overlap_metrics": {"summary": {
                "mae": mae, "rmse": rmse, "bias": bias, "seam_gradient": seam,
            }},
        }),
        encoding="utf-8",
    )


def test_build_radiometric_audit_writes_summary_report_and_figures(tmp_path):
    for geometry in ("efficient_loftr_translation_l2", "sift_mst"):
        for method, value in (("RAW", 10.0), ("BAGRN", 2.0), ("BAGRN_VOLRN", 1.0)):
            _write_run(tmp_path / "runs", geometry, method, value, value + 1, value - 0.5, value / 10)

    result = build_radiometric_audit(tmp_path / "runs", tmp_path / "audit")

    assert (tmp_path / "audit" / "radiometric_audit_summary.json").is_file()
    assert (tmp_path / "audit" / "radiometric_audit.md").is_file()
    assert (tmp_path / "audit" / "histogram_changes.png").is_file()
    assert (tmp_path / "audit" / "seam_diagnostics.png").is_file()
    assert (tmp_path / "audit" / "RAW_seam.png").is_file()
    assert (tmp_path / "audit" / "BAGRN_seam.png").is_file()
    assert (tmp_path / "audit" / "BAGRN_VOLRN_seam.png").is_file()
    assert len(result["rows"]) == 6
    assert result["factual_answers"]["bagrn_mae_below_raw"] is True
