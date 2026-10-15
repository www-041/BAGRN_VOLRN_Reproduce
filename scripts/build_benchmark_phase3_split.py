"""Freeze the four-pair DZ01V trial checkpoint split without editing source reviews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("00_01", "02_03", "05_07", "08_10")
SCENE_INDICES = (0, 1, 2, 3, 5, 7, 8, 10)
CELL_M = 10_000
MARGIN_M = 500
ORIGIN = (0, 0)
VERSION = "benchmark_v1_trial4_grid10km_margin500m_v1"
ROLE_FORMULA = "DEVELOPMENT if (ix + 2*iy) % 3 == 0 else TEST"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(x: float, y: float) -> tuple[int, int]:
    return math.floor((x - ORIGIN[0]) / CELL_M), math.floor((y - ORIGIN[1]) / CELL_M)


def _safe(x: float, y: float, ix: int, iy: int) -> bool:
    dx = x - (ORIGIN[0] + ix * CELL_M)
    dy = y - (ORIGIN[1] + iy * CELL_M)
    return MARGIN_M < dx < CELL_M - MARGIN_M and MARGIN_M < dy < CELL_M - MARGIN_M


def assign_candidate(row: dict) -> dict:
    """Use both effective projected positions; never infer approval from split."""
    adjusted = str(row.get("human_adjusted", "")).lower() in {"true", "1", "yes"}
    prefix = "adjusted_" if adjusted else ""
    coords = [float(row[prefix + key]) for key in ("x_i", "y_i", "x_j", "y_j")]
    if not all(math.isfinite(value) for value in coords):
        raise ValueError(f"Nonfinite coordinates: {row['candidate_id']}")
    xi, yi, xj, yj = coords
    cell_i, cell_j = _cell(xi, yi), _cell(xj, yj)
    reason = "SAFE_CORE"
    split = "UNASSIGNED"
    cell_id = ""
    if row.get("human_review_status") != "APPROVED":
        reason = "FIRST_REVIEW_NOT_APPROVED"
    elif cell_i != cell_j:
        reason = "DIFFERENT_CELLS"
    elif not (_safe(xi, yi, *cell_i) and _safe(xj, yj, *cell_j)):
        reason = "BUFFER_ZONE"
    else:
        ix, iy = cell_i
        cell_id = f"g{ix}_{iy}"
        split = "DEVELOPMENT" if (ix + 2 * iy) % 3 == 0 else "TEST"
    result = dict(row)  # full immutable source row, including human history, in the derived CSV
    result.update({
        "original_candidate_grid_cell_id": row.get("grid_cell_id", ""),
        "grid_cell_id": cell_id,
        "effective_x_i": xi, "effective_y_i": yi, "effective_x_j": xj, "effective_y_j": yj,
        "split": split, "reason": reason, "source_review_status": row.get("human_review_status", ""),
        "source_split": row.get("split", ""), "source_second_reviewer": row.get("second_reviewer", ""),
        "second_review_status": "PENDING_SECOND_REVIEW" if split == "TEST" else "NOT_APPLICABLE",
        "formal_test_eligible": False,
    })
    return result


def collect_assignments(root: Path) -> tuple[list[dict], dict[str, str]]:
    base = root / "docs/benchmark/v1/phase2/auto_candidates"
    rows, hashes = [], {}
    seen = set()
    for pair in PAIRS:
        path = base / f"{pair}.csv"
        hashes[pair] = sha256(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            source = list(csv.DictReader(stream))
        if len(source) != 40:
            raise ValueError(f"Expected 40 source candidates for {pair}; found {len(source)}")
        for row in source:
            if row["pair_id"] != pair or row["candidate_id"] in seen:
                raise ValueError(f"Candidate identity mismatch/duplicate in {path}")
            if row["human_review_status"] != "APPROVED" or row["split"] != "UNASSIGNED":
                raise ValueError(f"Unexpected source review/split state: {row['candidate_id']}")
            seen.add(row["candidate_id"])
            rows.append(assign_candidate(row))
    return rows, hashes


def build_regions(bounds: tuple[float, float, float, float]) -> dict:
    """Global safe-core polygons over the trial data envelope, for GUI suggestions."""
    xmin, ymin, xmax, ymax = bounds
    features = []
    for ix in range(math.floor(xmin / CELL_M), math.floor(xmax / CELL_M) + 1):
        for iy in range(math.floor(ymin / CELL_M), math.floor(ymax / CELL_M) + 1):
            role = "DEVELOPMENT" if (ix + 2 * iy) % 3 == 0 else "TEST"
            geom = box(ix * CELL_M + MARGIN_M, iy * CELL_M + MARGIN_M,
                       (ix + 1) * CELL_M - MARGIN_M, (iy + 1) * CELL_M - MARGIN_M)
            features.append({"type": "Feature", "geometry": mapping(geom),
                             "properties": {"split_id": f"g{ix}_{iy}", "role": role,
                                            "ix": ix, "iy": iy}})
    return {"type": "FeatureCollection", "metadata": {
        "status": "FROZEN", "version": VERSION, "freeze_scope": "TRIAL4_ONLY",
        "crs": "EPSG:32650", "origin_m": list(ORIGIN), "cell_size_m": CELL_M,
        "edge_margin_m": MARGIN_M, "index_formula": "ix=floor((x-origin_x)/10000); iy=floor((y-origin_y)/10000)",
        "role_formula": ROLE_FORMULA, "boundary_rule": "strict safe-core interior for BOTH image positions",
        "random_seed": "NONE", "full_15_pair_freeze": False}, "features": features}


def audit_assignments(assignments: list[dict]) -> dict:
    by_pair = defaultdict(list)
    cells = defaultdict(set)
    invalid = []
    points = {"DEVELOPMENT": [], "TEST": []}
    for row in assignments:
        pair, role = row["pair_id"], row["split"]
        by_pair[pair].append(row)
        if role not in points:
            continue
        cell_id = row["grid_cell_id"]
        cells[cell_id].add(role)
        try:
            ix, iy = _cell(float(row["effective_x_i"]), float(row["effective_y_i"]))
            xj, yj = float(row["effective_x_j"]), float(row["effective_y_j"])
            expected_role = "DEVELOPMENT" if (ix + 2 * iy) % 3 == 0 else "TEST"
            good = (cell_id == f"g{ix}_{iy}" and _cell(xj, yj) == (ix, iy)
                    and _safe(float(row["effective_x_i"]), float(row["effective_y_i"]), ix, iy)
                    and _safe(xj, yj, ix, iy) and role == expected_role)
        except (ValueError, KeyError, TypeError):
            good = False
        if not good:
            invalid.append(row["candidate_id"])
        for suffix in ("i", "j"):
            points[role].append((row["candidate_id"], float(row[f"effective_x_{suffix}"]),
                                 float(row[f"effective_y_{suffix}"])))
    conflicts = []
    minimum = None
    for aid, ax, ay in points["DEVELOPMENT"]:
        for bid, bx, by in points["TEST"]:
            distance = math.hypot(ax - bx, ay - by)
            minimum = distance if minimum is None else min(minimum, distance)
            if distance < 2 * MARGIN_M - 1e-9:
                conflicts.append({"development": aid, "test": bid, "distance_m": distance})
    def summarize(rows):
        counts = Counter(row["split"] for row in rows)
        role_cells = {}
        for role in ("DEVELOPMENT", "TEST"):
            groups = Counter(row["grid_cell_id"] for row in rows if row["split"] == role)
            n = sum(groups.values())
            role_cells[role] = {"unique_cells": len(groups), "counts_by_cell": dict(sorted(groups.items())),
                                "max_cell_share": max(groups.values()) / n if n else None,
                                "hhi": sum((v / n) ** 2 for v in groups.values()) if n else None,
                                "concentration_flag": bool(n and (len(groups) < 3 or max(groups.values()) / n > 0.5))}
        return {"counts": {role: counts[role] for role in ("DEVELOPMENT", "TEST", "UNASSIGNED")},
                "unique_assigned_cells": len({row["grid_cell_id"] for row in rows if row["split"] in points}),
                "role_cells": role_cells}
    pair_summary = {pair: summarize(rows) for pair, rows in sorted(by_pair.items())}
    pending = sum(row["split"] == "TEST" and not row.get("source_second_reviewer") for row in assignments)
    erroneous_test = [row["candidate_id"] for row in assignments if row["split"] == "TEST" and row.get("formal_test_eligible") is not False]
    duplicate_ids = [item for item, count in Counter(row["candidate_id"] for row in assignments).items() if count > 1]
    return {"leakage_pass": not (invalid or conflicts or duplicate_ids or erroneous_test or any(len(v) > 1 for v in cells.values())),
            "per_pair": pair_summary, "overall": summarize(assignments),
            "unassigned_reason_counts": dict(Counter(row["reason"] for row in assignments if row["split"] == "UNASSIGNED")),
            "assigned_point_safe_core_violations": invalid,
            "same_cell_opposite_role_conflicts": [cell for cell, roles in cells.items() if len(roles) > 1],
            "opposite_role_point_conflicts": conflicts, "minimum_opposite_role_point_distance_m": minimum,
            "duplicate_candidate_ids": duplicate_ids, "test_eligibility_violations": erroneous_test,
            "test_pending_second_review": pending,
            "formal_test_eligible_count": sum(row["split"] == "TEST" and row.get("formal_test_eligible") is True for row in assignments)}


def _identity(root: Path) -> tuple[list[dict], list[int], tuple[float, float, float, float]]:
    import rasterio
    import yaml
    with (root / "configs/b9_13scene.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    input_root = Path(config["dataset"]["input_root"])
    if not input_root.is_dir():
        # Historical YAML has a mojibake absolute path; the explicitly established
        # sibling research-data location is resolved without editing the YAML.
        input_root = root.parent / "BAGRN_VOLRN_Reproduce/data/input/B9_reference_13scenes_full"
    if not input_root.is_dir():
        raise FileNotFoundError(f"B9 input root not found: {input_root}")
    with (root / "docs/benchmark/v1/phase2/input_identity_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        manifest = {int(row["scene_index"]): row for row in csv.DictReader(stream)}
    records, bounds = [], []
    for index in SCENE_INDICES:
        item = manifest[index]
        matches = list(input_root.rglob(f"{item['scene_id']}_B9.TIF"))
        if len(matches) != 1:
            raise ValueError(f"B9 identity ambiguous/missing for index {index}: {len(matches)} matches")
        path = matches[0]
        current = sha256(path)
        if current != item["recorded_source_sha256_task15"] or path.stat().st_size != int(item["source_file_size_bytes"]):
            raise ValueError(f"B9 version mismatch: scene {index}")
        with rasterio.open(path) as src:
            if src.crs.to_epsg() != 32650:
                raise ValueError(f"Unexpected CRS for {path}: {src.crs}")
            b = src.bounds
            bounds.append((b.left, b.bottom, b.right, b.top))
        records.append({"scene_index": index, "scene_id": item["scene_id"], "path": str(path),
                        "size_bytes": path.stat().st_size, "sha256": current,
                        "matches_task15_recorded_sha256": True})
    remaining = sorted(set(manifest) - set(SCENE_INDICES))
    return records, remaining, (min(b[0] for b in bounds), min(b[1] for b in bounds),
                                max(b[2] for b in bounds), max(b[3] for b in bounds))


def _plot(path: Path, rows: list[dict], regions: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    colors = {"DEVELOPMENT": "#2878b5", "TEST": "#d4572a", "UNASSIGNED": "#888888"}
    for feature in regions["features"]:
        geom = feature["geometry"]["coordinates"][0]
        x0, y0 = geom[0]
        x1, y1 = geom[2]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec=colors[feature["properties"]["role"]], lw=0.3, alpha=0.3))
    for role in colors:
        selected = [row for row in rows if row["split"] == role]
        ax.scatter([float(r["effective_x_i"]) for r in selected],
                   [float(r["effective_y_i"]) for r in selected], c=colors[role],
                   s=20, alpha=0.82, label=f"{role} ({len(selected)})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("EPSG:32650 Easting (m)")
    ax.set_ylabel("EPSG:32650 Northing (m)")
    ax.set_title("Benchmark V1: four-pair trial spatial split\nTEST = second review pending; grey = buffer/cell mismatch")
    ax.legend(loc="best")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_outputs(root: Path, output: Path) -> dict:
    assignments, source_hashes = collect_assignments(root)
    identity, remaining, bounds = _identity(root)
    audit = audit_assignments(assignments)
    if not audit["leakage_pass"] or audit["overall"]["counts"] != {"DEVELOPMENT": 43, "TEST": 84, "UNASSIGNED": 33}:
        raise ValueError("Split preflight failed; no outputs written")
    regions = build_regions(bounds)
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = output / "checkpoint_split_assignments.csv"
    fields = list(assignments[0])
    with assignment_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)
    region_path = output / "spatial_split_regions.geojson"
    region_path.write_text(json.dumps(regions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_path = output / "spatial_split_map.png"
    _plot(plot_path, assignments, regions)
    pair_lines = "\n".join(
        f"| {pair} | {s['counts']['DEVELOPMENT']} | {s['counts']['TEST']} | {s['counts']['UNASSIGNED']} | "
        f"{s['unique_assigned_cells']} | {s['role_cells']['DEVELOPMENT']['unique_cells']} | {s['role_cells']['TEST']['unique_cells']} |"
        for pair, s in audit["per_pair"].items())
    protocol = f"""# Benchmark V1 four-pair spatial split protocol

Status: **FROZEN for four trial pairs only**. Full 15-pair Benchmark is **NOT FROZEN**.

- Native projected CRS: EPSG:32650; fixed origin: `(0, 0)` metres; cell size: 10,000 m.
- Index formula: `ix = floor((x - 0)/10000)`, `iy = floor((y - 0)/10000)`; global cell ID `g{{ix}}_{{iy}}`.
- Role formula: `{ROLE_FORMULA}`. No random seed is used.
- Each edge has a 500 m excluded margin. A point is safe only if `500 < x - ix*10000 < 9500` and `500 < y - iy*10000 < 9500` (strict inequalities).
- Both images' effective projected coordinates (human-adjusted if present) must occupy the same cell safe interior; otherwise UNASSIGNED. Geometry is EPSG:32650, not candidate-local 8x8 grid.
- Opposite-role safe interiors are separated by at least 1,000 m. All assigned left/right point coordinates are tested for this rule, including across pairs.
- This deterministic global rule applies unchanged to the remaining 11 pairs. Do not tune the 10 km cell, 500 m margin, origin or role formula against matcher accuracy. For future pairs, extend GeoJSON coverage with the **same** formula before use; the present polygon file covers the four-pair data envelope only.
- TEST assignment is a candidate role only. A distinct second reviewer must independently visually recheck before formal TEST scoring. No second reviewer or approval is fabricated here.
- The 160 source candidate CSV rows and all first-review history are preserved untouched. The derived assignment CSV retains every source column plus split diagnostics; its `split` column is the Phase3 proposed role, while `source_split` keeps the prior value.
- Concentration warning: fewer than 3 role-specific cells or one cell holding >50% of that role's points. Warnings are descriptive, not grounds to change the frozen rule.

| Pair | DEV | TEST candidates | UNASSIGNED | all assigned cells | DEV cells | TEST cells |
|---|---:|---:|---:|---:|---:|---:|
{pair_lines}

Formal TEST scoring eligibility remains zero until independent second review and separate truth export. No matcher or radiometric experiment was run.
"""
    protocol_path = output / "spatial_split_protocol.md"
    protocol_path.write_text(protocol, encoding="utf-8")
    for pair, prior in source_hashes.items():
        if sha256(root / "docs/benchmark/v1/phase2/auto_candidates" / f"{pair}.csv") != prior:
            raise ValueError(f"Source candidate file changed during build: {pair}")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "UNKNOWN"
    audit.update({"version": VERSION, "freeze_scope": "TRIAL4_ONLY", "crs": "EPSG:32650",
                  "origin_m": list(ORIGIN), "cell_size_m": CELL_M, "edge_margin_m": MARGIN_M,
                  "role_formula": ROLE_FORMULA, "random_seed": None, "git_head": head,
                  "builder_sha256": sha256(Path(__file__)), "source_candidate_sha256": source_hashes,
                  "input_identity_manifest_sha256": sha256(root / "docs/benchmark/v1/phase2/input_identity_manifest.csv"),
                  "b9_verified_inputs": identity, "remaining_scene_indices_not_verified": remaining,
                  "output_sha256": {p.name: sha256(p) for p in (assignment_path, region_path, plot_path, protocol_path)}})
    audit_path = output / "split_leakage_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readiness = f"""# Benchmark V1 Phase3 readiness

**Four-pair trial split frozen; full 15-pair Benchmark not frozen.** All 160 first-review-approved source candidates remain intact.

- DEVELOPMENT: 43; TEST candidates: 84; UNASSIGNED: 33 (all in excluded 500 m buffer zones). Assigned-only shares: 33.9% / 66.1%. No quota-driven reallocation.
- Formal TEST scoring eligible: **0**. Distinct second-reviewer checks outstanding: **{audit['test_pending_second_review']}**. A split is not final approval.
- Leakage audit: **{'PASS' if audit['leakage_pass'] else 'FAIL'}**; assigned-point safe-core violations {len(audit['assigned_point_safe_core_violations'])}; opposite-role cross-pair proximity conflicts {len(audit['opposite_role_point_conflicts'])}; minimum opposite-role left/right projected-point distance {audit['minimum_opposite_role_point_distance_m']:.3f} m.
- Overall distinct assigned global cells: {audit['overall']['unique_assigned_cells']}; role-specific DEV/TEST cells: {audit['overall']['role_cells']['DEVELOPMENT']['unique_cells']}/{audit['overall']['role_cells']['TEST']['unique_cells']}. Cell distributions and concentration flags are in `split_leakage_audit.json`.
- The eight trial-scene B9 SHA256 and sizes match the Task15-recorded source identities. Remaining scene indices {remaining} are **not yet reverified** and must be checked before full 13-scene formal freeze.
- Original B1–B16 multiband identities are not frozen by this B9-only check. TEST second review, additional 11-pair annotations, full input version check and final truth export remain required.
- Spatial map: `spatial_split_map.png`; grid is a preselected evaluation design, not tuned on matcher results.

| Pair | DEV | TEST candidates | UNASSIGNED | all assigned cells | DEV cells | TEST cells |
|---|---:|---:|---:|---:|---:|---:|
{pair_lines}
"""
    (output / "benchmark_split_readiness.md").write_text(readiness, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmark/v1/phase3")
    args = parser.parse_args()
    result = write_outputs(ROOT, args.output)
    print(json.dumps({"leakage_pass": result["leakage_pass"], "per_pair": result["per_pair"],
                      "test_pending_second_review": result["test_pending_second_review"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
