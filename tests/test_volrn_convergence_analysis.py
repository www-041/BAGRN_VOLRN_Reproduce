import json


def test_convergence_analysis_classifies_monotone_dual_decay_as_slow(tmp_path):
    from scripts.analyze_volrn_convergence import analyze_volrn_history

    history = {
        "schema_version": 1,
        "n_bands": 1,
        "bands": [{
            "band": 0,
            "history": [
                {
                    "iteration": 1,
                    "objective": 10.0,
                    "primal_residual": 1.0,
                    "dual_residual": 100.0,
                    "x_update_norm": 1.0,
                    "z_update_norm": 1.0,
                    "dual_update_norm": 1.0,
                    "cg_iterations": 2,
                    "cg_residual": 1e-6,
                    "relative_change": 1e-2,
                },
                {
                    "iteration": 2,
                    "objective": 5.0,
                    "primal_residual": 0.5,
                    "dual_residual": 10.0,
                    "x_update_norm": 0.1,
                    "z_update_norm": 0.1,
                    "dual_update_norm": 0.5,
                    "cg_iterations": 2,
                    "cg_residual": 1e-6,
                    "relative_change": 1e-3,
                },
            ],
        }],
    }
    input_path = tmp_path / "history.json"
    input_path.write_text(json.dumps(history), encoding="utf-8")

    summary = analyze_volrn_history(input_path, tmp_path / "diagnostic")

    assert summary["classification"] == "slow convergence"
    assert summary["iterations"] == 2
    assert summary["dual_trend"] == "decreasing"
    assert (tmp_path / "diagnostic" / "objective_curve.png").exists()
    assert (tmp_path / "diagnostic" / "primal_residual_curve.png").exists()
    assert (tmp_path / "diagnostic" / "dual_residual_curve.png").exists()
    assert (tmp_path / "diagnostic" / "update_norm_curve.png").exists()
    assert (tmp_path / "diagnostic" / "convergence_summary.json").exists()
    assert (tmp_path / "diagnostic" / "2026-09-27-task10e-convergence-analysis.md").exists()
