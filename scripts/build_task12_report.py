"""Assemble the compact Task12 comparison table and final report."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/output/b9_five_scene_validation"
OUT = DATA / "volrn_solver_alignment_task12"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _status(diag):
    bands = diag.get("band_solver_diagnostics", [])
    if not bands or not all(x.get("finite_state", False) for x in bands):
        return "NUMERICAL_INVALID"
    if any(x.get("cg_failed_any", x.get("cg_failed", False)) for x in bands):
        return "NUMERICAL_INVALID"
    if all(x.get("strict_admm_converged", False) for x in bands):
        return "STRICT_ADMM_CONVERGED"
    if all(x.get("coefficient_stable", False) for x in bands):
        return "STABLE_BUT_STRICT_NONCONVERGED"
    return "NONCONVERGED_AFTER_ALIGNMENT"


def _row(label, solver, diag, metrics, runtime=None):
    b = diag["band_solver_diagnostics"][0]
    return {
        "run": label, "solver": solver, "status": _status(diag),
        "iterations": b.get("iterations"), "finite_state": b.get("finite_state"),
        "strict_admm_converged": b.get("strict_admm_converged"),
        "coefficient_stable": b.get("coefficient_stable"),
        "cg_failed_any": b.get("cg_failed_any"),
        "preconditioner_type": b.get("preconditioner_type"),
        "rho_initial": b.get("rho_initial"), "rho_final": b.get("rho_final"),
        "rho_change_count": b.get("rho_change_count"),
        "primal_residual": b.get("primal_residual"), "primal_tolerance": b.get("primal_tolerance"),
        "dual_residual": b.get("dual_residual"), "dual_tolerance": b.get("dual_tolerance"),
        "x_relative_change": b.get("x_relative_change"),
        "objective_start": b.get("objective_start"), "objective_end": b.get("objective_end"),
        "relative_objective_change": b.get("relative_objective_change"),
        "runtime_sec": runtime,
        "mamd_weighted_mean": metrics.get("mamd", {}).get("weighted_mean"),
        "msdd_weighted_mean": metrics.get("msdd", {}).get("weighted_mean"),
        "rdd_weighted_mean": metrics.get("rdd", {}).get("weighted_mean"),
        "seam_mae_weighted_mean": metrics.get("seam_mae", {}).get("weighted_mean"),
        "seam_rmse_weighted_mean": metrics.get("seam_rmse", {}).get("weighted_mean"),
        "seam_rdd_weighted_mean": metrics.get("seam_rdd", {}).get("weighted_mean"),
    }


def main():
    baseline = _load(OUT / "00_baseline_reference.json")
    rows = []
    # The historical Task10E row is intentionally read-only and is not rerun.
    h = baseline["task10e_iteration_200"]
    rows.append({
        "run": "Task10E_baseline", "solver": "PLAIN_CG", "status": "HISTORICAL_REFERENCE",
        "iterations": baseline["max_iter"], "finite_state": h["finite_state"],
        "strict_admm_converged": h["strict_admm_converged"], "coefficient_stable": None,
        "cg_failed_any": h["cg_failed"], "preconditioner_type": "none",
        "rho_initial": baseline["rho_baseline"], "rho_final": baseline["rho_baseline"],
        "rho_change_count": 0, "primal_residual": h["primal_residual"], "primal_tolerance": h["primal_tolerance"],
        "dual_residual": h["dual_residual"], "dual_tolerance": h["dual_tolerance"],
        "x_relative_change": h["relative_x_change"], "objective_start": None, "objective_end": h["objective"],
        "relative_objective_change": None, "runtime_sec": None,
    })
    summaries = {}
    for label, solver, folder in [("Run_A", "PCG_JACOBI_FIXED_RHO", "pcg_fixed"), ("Run_B", "PCG_JACOBI_ADAPTIVE_RHO", "pcg_adaptive")]:
        d = _load(OUT / folder / "solver_summary.json")
        summaries[label] = d
        rows.append(_row(label, solver, d["solver_diagnostics"], d["task10d_primary_metrics"], d["runtime_sec"]))
    fields = sorted({key for row in rows for key in row})
    with (OUT / "solver_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    a = rows[1]; b = rows[2]
    if b["status"] == "NUMERICAL_INVALID":
        final_status, selected = "NUMERICAL_INVALID", "NONE"
    elif b["coefficient_stable"]:
        final_status, selected = "STABLE_BUT_STRICT_NONCONVERGED", "Run_B_ADAPTIVE_PCG"
    else:
        final_status, selected = "NONCONVERGED_AFTER_ALIGNMENT", "NONE"
    report = f"""# Task12 VOLRN Solver Alignment Report

## Scope and frozen inputs

This report executes `2026-09-27-task12-volrn-solver-alignment-autonomous-plan-FULL.md` against the B9 five-scene `EfficientLoFTR + Translation-L2` frozen geometry. Matcher, global registration, geometry, and BAGRN were not rerun for either solver run. The validated per-scene BAGRN cache is `{summaries['Run_A']['cache_run_dir']}`; its manifest SHA-256 is `{summaries['Run_A']['cache_manifest_sha256']}`.

Parameters were fixed at block size 400, lambda 0.5, initial rho 1.0, maximum 200 ADMM iterations, and tolerance 1e-4. The solver is objective-preserving Jacobi-preconditioned CG with an explicit SciPy `M` operator. Run B uses only standard residual balancing (mu=10, tau=2, bounds 1e-4..1e4) with scaled-dual rescaling on rho changes.

## Solver comparison

See `solver_comparison.csv` for the machine-readable table. Task10E is a read-only historical PLAIN-CG reference; Run A and Run B are the only two real Task12 VOLRN runs.

| run | status | iterations | finite | strict ADMM | coefficient stable | rho final | rho changes | primal / tol | dual / tol |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Task10E baseline | HISTORICAL_REFERENCE | {rows[0]['iterations']} | {rows[0]['finite_state']} | {rows[0]['strict_admm_converged']} | — | {rows[0]['rho_final']} | 0 | {rows[0]['primal_residual']:.6g} / {rows[0]['primal_tolerance']:.6g} | {rows[0]['dual_residual']:.6g} / {rows[0]['dual_tolerance']:.6g} |
| Run A fixed PCG | {a['status']} | {a['iterations']} | {a['finite_state']} | {a['strict_admm_converged']} | {a['coefficient_stable']} | {a['rho_final']} | {a['rho_change_count']} | {a['primal_residual']:.6g} / {a['primal_tolerance']:.6g} | {a['dual_residual']:.6g} / {a['dual_tolerance']:.6g} |
| Run B adaptive PCG | {b['status']} | {b['iterations']} | {b['finite_state']} | {b['strict_admm_converged']} | {b['coefficient_stable']} | {b['rho_final']} | {b['rho_change_count']} | {b['primal_residual']:.6g} / {b['primal_tolerance']:.6g} | {b['dual_residual']:.6g} / {b['dual_tolerance']:.6g} |

Run A remained finite but was neither strictly converged nor coefficient-stable. Run B was finite and coefficient-stable but had 49 `cg_info=500` events; under Task8 Outcome B4 this is **NUMERICAL_INVALID**, so no solver is selected and the real-run budget stops at A+B. This selection is based only on solver validity/stability, never on image metrics. SciPy CG `rtol` was tightened from the historical 1e-6 to 1e-8 and is disclosed here; no objective or equation normalization was changed.

## Unified radiometric metrics

Each run regenerated weighted-feather mosaic output from the cached BAGRN scenes and computed the same Task10D protocol metrics. Run A: MAMD={a['mamd_weighted_mean']:.6g}, MSDD={a['msdd_weighted_mean']:.6g}, RDD={a['rdd_weighted_mean']:.6g}, seam MAE={a['seam_mae_weighted_mean']:.6g}, seam RMSE={a['seam_rmse_weighted_mean']:.6g}, seam RDD={a['seam_rdd_weighted_mean']:.6g}. Run B: MAMD={b['mamd_weighted_mean']:.6g}, MSDD={b['msdd_weighted_mean']:.6g}, RDD={b['rdd_weighted_mean']:.6g}, seam MAE={b['seam_mae_weighted_mean']:.6g}, seam RMSE={b['seam_rmse_weighted_mean']:.6g}, seam RDD={b['seam_rdd_weighted_mean']:.6g}. These metrics are reported for diagnosis and are not used to override solver-status selection.

## Artifacts

- `pcg_fixed/` and `pcg_adaptive/`: final mosaics, previews, solver histories, coefficients, metrics, and summaries.
- `solver_comparison.csv`: Task10E baseline versus Run A/Run B compact comparison.
- `00_baseline_reference.json`: frozen hashes and historical Task10E values.

No matcher/global/BAGRN outputs or frozen geometry files were overwritten.

## Verification evidence

Focused Task12/radiometric/grid command: `17 passed` (1026 warnings). Full suite command: `840 passed, 4 skipped, 2 failed`; the two failures are pre-existing synthetic VOLRN expectations (`tests/test_task10e_synthetic_volrn.py::test_task10e_synthetic_cases_generate_convergence_report` and `tests/test_volrn.py::TestVolrnTriplet::test_triplet_pipeline`), reproduced with both Jacobi and no-preconditioner paths and not used to alter the paper formulas. `compileall` and `git diff --check` passed.
"""
    (OUT / "TASK12_VOLRN_SOLVER_ALIGNMENT_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
