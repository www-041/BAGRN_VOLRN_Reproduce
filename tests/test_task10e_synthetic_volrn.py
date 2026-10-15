import subprocess
import sys


def test_task10e_synthetic_cases_generate_convergence_report(tmp_path):
    from scripts.run_task10e_synthetic_volrn import run_synthetic_diagnostics

    report_path = tmp_path / "synthetic_volrn_convergence_report.md"
    result = run_synthetic_diagnostics(report_path)

    assert result["global_gain"]["converged"]
    assert result["global_gain"]["after_pair_mae"] < result["global_gain"]["before_pair_mae"]
    assert result["local_gain"]["converged"]
    assert result["local_gain"]["coefficient_a_range"] > 0.05
    assert result["local_gain"]["after_pair_mae"] < result["local_gain"]["before_pair_mae"]
    assert result["identity"]["converged"]
    assert result["identity"]["max_coefficient_deviation"] < 1e-8
    assert report_path.exists()


def test_synthetic_cli_runs_from_repository_root(tmp_path):
    output = tmp_path / "synthetic_volrn_convergence_report.md"
    completed = subprocess.run(
        [sys.executable, "scripts/run_task10e_synthetic_volrn.py", str(output)],
        cwd=".", capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
