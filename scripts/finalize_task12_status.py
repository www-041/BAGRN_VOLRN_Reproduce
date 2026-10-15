"""Apply the plan's CG-failure outcome rule to completed Task12 summaries."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/output/b9_five_scene_validation/volrn_solver_alignment_task12"

for folder in ("pcg_fixed", "pcg_adaptive"):
    path = OUT / folder / "solver_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    bands = payload["solver_diagnostics"].get("band_solver_diagnostics", [])
    if not bands or any(not x.get("finite_state", False) for x in bands) or any(x.get("cg_failed_any", x.get("cg_failed", False)) for x in bands):
        status = "NUMERICAL_INVALID"
    elif all(x.get("strict_admm_converged", False) for x in bands):
        status = "STRICT_ADMM_CONVERGED"
    elif all(x.get("coefficient_stable", False) for x in bands):
        status = "STABLE_BUT_STRICT_NONCONVERGED"
    else:
        status = "NONCONVERGED_AFTER_ALIGNMENT"
    for band in bands:
        if band.get("cg_failed_any", band.get("cg_failed", False)) or not band.get("finite_state", False):
            band["science_status"] = "NUMERICAL_INVALID"
    payload["status"] = status
    payload["postprocessing_repaired_from_saved_coefficients"] = True
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
