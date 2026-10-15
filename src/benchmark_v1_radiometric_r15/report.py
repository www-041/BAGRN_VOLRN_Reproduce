from __future__ import annotations

import json
import re
import hashlib
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape

from .inputs import ALLOWED_PAIRS
from .spatial import audit_candidate_sets
from .storage import load_candidate_collection


ROLE_COLORS = {"DEVELOPMENT": "#2468a2", "TEST": "#d07b19", "UNASSIGNED": "#777777"}


def _counts(features, key):
    return dict(sorted(Counter(str(f["properties"].get(key, "UNKNOWN")) for f in features).items()))


def _sampling_cell(props):
    if props.get("sampling_grid_cell_id"):
        return str(props["sampling_grid_cell_id"])
    match = re.search(r"_R15_(\d{2})_(\d{2})_", str(props.get("candidate_id", "")))
    return f"r{match.group(1)}_c{match.group(2)}" if match else "UNKNOWN"


def build_aggregate_outputs(run_dir: str | Path, docs_output: str | Path) -> Path:
    run_dir, target = Path(run_dir), Path(docs_output)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("run manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = set(manifest.get("generated_pairs", []))
    if generated != ALLOWED_PAIRS:
        raise ValueError(f"aggregate requires exactly the four frozen pairs; found {sorted(generated)}")
    allowed_outputs = {"candidate_quality_report.md", "candidate_spatial_distribution.png",
                       "radiometric_r1_5_auto_roi_usage.md", "cross_pair_leakage_audit.json",
                       "report_manifest.json"}
    if target.exists() and any(p.name not in allowed_outputs for p in target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-R1.5 aggregate files in: {target}")
    collections, pair_quality = [], {}
    for pair in sorted(ALLOWED_PAIRS):
        path = run_dir / f"{pair}.geojson"
        quality_path = run_dir / f"{pair}.quality.json"
        if not path.is_file() or not quality_path.is_file():
            raise ValueError(f"missing candidate or quality artifact for {pair}")
        collection = load_candidate_collection(path, pair)
        features = collection["features"]
        if any(f["properties"].get("review_status") != "UNVERIFIED" or
               f["properties"].get("eligible_for_test") for f in features):
            raise ValueError(f"{pair} contains automatically approved or human state in candidate source")
        collections.append(collection)
        pair_quality[pair] = json.loads(quality_path.read_text(encoding="utf-8"))
    features = [f for c in collections for f in c["features"]]
    leakage = audit_candidate_sets(collections)
    if leakage["role_conflicts"]:
        raise ValueError("candidate roles disagree with frozen Phase3 formula")
    target.mkdir(parents=True, exist_ok=True)
    _write_report(target / "candidate_quality_report.md", manifest, pair_quality, collections, leakage,
                  _read_temporal_inventory(run_dir))
    _write_map(target / "candidate_spatial_distribution.png", features)
    _write_usage(target / "radiometric_r1_5_auto_roi_usage.md", manifest["run_id"])
    (target / "cross_pair_leakage_audit.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "report_manifest.json").write_text(json.dumps({
        "run_id": manifest["run_id"],
        "source_run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate_count": len(features)}, indent=2), encoding="utf-8")
    return target


def _read_temporal_inventory(run_dir):
    inventory = next((parent / "docs/benchmark/v1/temporal_pair_inventory.csv"
                      for parent in run_dir.parents
                      if (parent / "docs/benchmark/v1/temporal_pair_inventory.csv").is_file()), None)
    if inventory is None:
        return {}
    with inventory.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["pair_id"]: row for row in csv.DictReader(stream)}


def _write_report(path, manifest, pair_quality, collections, leakage, temporal_rows):
    features = [f for collection in collections for f in collection["features"]]
    lines = ["# DZ01V Radiometric ROI candidate quality report — R1.5",
             "", f"Run ID: `{manifest['run_id']}`", "",
             "All records are automatic proposals only. `review_status=UNVERIFIED`; no human truth or approval is created.",
             "Units remain UNKNOWN. Per-band median/MAD and relative-shape values are diagnostic-only and are not used as label truth. "
             "No radiometric normalization or formal evaluation was run.", "",
             f"Total automatic candidates: {len(features)}", f"Label suggestions: `{_counts(features, 'suggested_label')}`",
             f"Confidence suggestions: `{_counts(features, 'suggestion_confidence')}`",
             f"Spatial roles: `{_counts(features, 'role')}`", "APPROVED: 0", "",
             "## Per-pair counts and filters", "", "| Pair | Candidates | Labels | Roles | Examined windows | Quality accepted before cap | Rejected by reason |", "|---|---:|---|---|---:|---:|---|"]
    rejected_all = []
    collection_by_pair = {str(c["features"][0]["properties"]["pair_id"]): c for c in collections if c["features"]}
    for pair, collection in zip(sorted(ALLOWED_PAIRS), collections):
        rows = collection["features"]
        diag = pair_quality[pair].get("diagnostics", {})
        rejected = diag.get("rejected_windows", [])
        rejected_all.extend(rejected)
        reasons = dict(sorted(Counter(r.get("reason", "UNKNOWN") for r in rejected).items()))
        lines.append(f"| {pair} | {len(rows)} | {_counts(rows, 'suggested_label')} | {_counts(rows, 'role')} | "
                     f"{diag.get('windows_examined', 0)} | {diag.get('quality_accepted_before_cap', 0)} | `{reasons}` |")
    lines += ["", "## Input coverage and quality", "",
              "| Pair | Days apart | Input quality | Common V bands | Wavelength centers known | Missing/unusable bands | B9 support (min/mean) | Units |",
              "|---|---:|---|---:|---:|---|---|---|"]
    input_headers = manifest.get("pair_input_headers", {})
    for pair in sorted(ALLOWED_PAIRS):
        temporal = temporal_rows.get(pair, {})
        scenes = input_headers.get(pair, {})
        bands_by_scene = list(scenes.values())
        if len(bands_by_scene) >= 2:
            common = sorted(b for b in set(bands_by_scene[0]) & set(bands_by_scene[1])
                            if bands_by_scene[0][b].get("usable") and bands_by_scene[1][b].get("usable"))
            missing = sorted(set(f"B{i}" for i in range(1,17)) - set(common))
        else:
            common, missing = [], [f"B{i}" for i in range(1,17)]
        rows = collection_by_pair.get(pair, {"features": []})["features"]
        supports = [float(f["properties"].get("valid_support_fraction", 0)) for f in rows]
        support_text = (f"{min(supports):.3f}/{sum(supports)/len(supports):.3f}" if supports else "NO_CANDIDATE")
        quality = f"{temporal.get('quality_i','UNKNOWN')}/{temporal.get('quality_j','UNKNOWN')}"
        band_support = rows[0]["properties"].get("support_by_scene_band", {}) if rows else {}
        known_wavelengths = sorted(b for b, v in band_support.items() if isinstance(v, dict)
                                   and v.get("wavelength_center_nm_i") is not None
                                   and v.get("wavelength_center_nm_j") is not None)
        grid_matches = sum(bool(v.get("scene_i_grid_matches_b9")) for v in band_support.values()
                           if isinstance(v, dict) and v.get("available"))
        lines.append(f"| {pair} | {temporal.get('days_apart','UNKNOWN')} | {quality} | {len(common)} | "
                     f"{len(known_wavelengths)} | {','.join(missing) if missing else 'none'} | {support_text} | UNKNOWN |")
        if known_wavelengths:
            centers = ", ".join(f"{band}={band_support[band]['wavelength_center_nm_i']}/{band_support[band]['wavelength_center_nm_j']} nm"
                                 for band in known_wavelengths)
            lines.append(f"- {pair} metadata centers (scene i/j): {centers}; in-scene bands matching the B9 grid: {grid_matches}/{len(common)}.")
    lines += ["", "Band wavelengths are only reported where input metadata supplies them; physical units remain UNKNOWN. "
              "Cross-scene geometry and cross-band co-registration are not independently verified.", ""]
    lines += ["", "## Spatial/texture/edge selection audit", "",
              "Fixed 32 x 32 native B9 windows (448 m at 14 m), deterministic spatial strata, B9 common-valid fraction >= 0.95, "
              "strong-edge fraction <= 0.10 after the fixed 2-pixel edge buffer. Thresholds were not relaxed to reach a quota.",
              "", "Fixed windows and the edge filter can under-sample narrow or fragmented features, boundary-rich landforms, "
              "low-texture terrain, shadowed/low-contrast areas, and small targets. A low candidate count is reported as such, not repaired by weakening gates.",
              "", f"Cross-pair audit: `{leakage}`", "",
              "### Per-subregion window counts", "",
              "Counts below distinguish all examined/rejected fixed windows from selected accepted candidates. Rejected candidates remain excluded; they are not labels.", ""]
    cell_stats = defaultdict(Counter)
    edge_stats = defaultdict(Counter)
    texture_values = []
    accepted_pool = []
    for pair in sorted(ALLOWED_PAIRS):
        diag = pair_quality[pair].get("diagnostics", {})
        rejected = diag.get("rejected_windows", [])
        accepted = diag.get("accepted_windows", [])
        if not accepted:
            accepted = [{"grid_cell_id": _sampling_cell(f["properties"]),
                         "strong_edge_fraction": f["properties"].get("strong_edge_fraction", 0),
                         "texture_strength": f["properties"].get("texture_strength", 0)}
                        for f in collection_by_pair.get(pair, {"features": []})["features"]]
        accepted_pool.extend(accepted)
        for row in rejected:
            cell = row.get("grid_cell_id", "UNKNOWN")
            cell_stats[cell]["rejected"] += 1
            cell_stats[cell][row.get("reason", "UNKNOWN")] += 1
            edge_stats["<=0.10" if float(row.get("edge_fraction", 0)) <= 0.10 else ">0.10"]["rejected"] += 1
            texture_values.append(float(row.get("texture_strength", 0)))
    for row in accepted_pool:
        cell_stats[row.get("grid_cell_id", "UNKNOWN")]["accepted"] += 1
        edge_stats["<=0.10" if float(row.get("strong_edge_fraction", 0)) <= 0.10 else ">0.10"]["accepted"] += 1
        texture_values.append(float(row.get("texture_strength", 0)))
    for cell, counts in sorted(cell_stats.items()):
        lines.append(f"- `{cell}`: rejected windows={counts['rejected']}; quality-accepted windows={counts['accepted']}; "
                     f"exclusion details={dict(sorted((k, v) for k, v in counts.items() if k not in {'accepted','rejected'}))}")
    if texture_values:
        q1, q2 = __import__("numpy").quantile(texture_values, [1/3, 2/3])
        texture_bin = lambda x: "LOW" if x <= q1 else ("MEDIUM" if x <= q2 else "HIGH")
        texture_stats = defaultdict(Counter)
        for row in rejected_all:
            texture_stats[texture_bin(float(row.get("texture_strength", 0)))]["rejected"] += 1
        for row in accepted_pool:
            texture_stats[texture_bin(float(row.get("texture_strength", 0)))]["accepted"] += 1
    else:
        q1 = q2 = 0.0
        texture_stats = defaultdict(Counter)
    lines += ["", f"Edge strata at the fixed 0.10 gate: `{dict((k,dict(v)) for k,v in sorted(edge_stats.items()))}`", "",
              f"Texture strata use pooled candidate-window texture tertiles (cutpoints {q1:.3f}, {q2:.3f}): "
              f"`{dict((k,dict(v)) for k,v in sorted(texture_stats.items()))}`.", "",
              f"Quality support summary: selected candidate support min={min((float(f['properties'].get('valid_support_fraction',0)) for f in features), default=float('nan')):.3f}; "
              f"rejected windows below 0.95={sum(float(r.get('valid_fraction',0)) < .95 for r in rejected_all)}; "
              f"rejected windows above 0.10 edge={sum(float(r.get('edge_fraction',0)) > .10 for r in rejected_all)}.", "",
              "### Role/suggestion caveats", "",
              "Unsafe or crossing-grid geometries remain UNASSIGNED. Geometry edits invalidate support/role/leakage status and require a full four-pair re-audit.",
              "No candidate is HIGH confidence. No REAL_CHANGE label was suggested because independent cross-scene geometry and cross-band co-registration are unverified. "
              "No grayscale brightness cue is interpreted as STABLE, CLOUD, or SHADOW.", "",
              "## Input identity and protection", "",
              f"Protected small-file SHA256 records captured: {len(manifest.get('protected_file_sha256', {}))}.",
              "Large imagery is not full-file hashed by this tool; its path/size/mtime and GeoTIFF header fingerprint are recorded in the run manifest. "
              "B9 may include a previously recorded Task15 hash; auxiliary-band hashes remain NOT_FROZEN unless present in the manifest.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_map(path, features):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    pair_ids = sorted(ALLOWED_PAIRS)
    bounds = [shape(feature["geometry"]).bounds for feature in features]
    if bounds:
        minx, miny = min(b[0] for b in bounds), min(b[1] for b in bounds)
        maxx, maxy = max(b[2] for b in bounds), max(b[3] for b in bounds)
        limits = (minx - 1000, maxx + 1000, miny - 1000, maxy + 1000)
    else:
        limits = (0, 10000, 0, 10000)
    for ax, pair in zip(axes.flat, pair_ids):
        pair_features = [feature for feature in features if feature["properties"].get("pair_id") == pair]
        drawn_cells = set()
        for feature in pair_features:
            geom = shape(feature["geometry"])
            props = feature["properties"]
            color = ROLE_COLORS.get(props.get("role", "UNASSIGNED"), "#777777")
            cell = str(props.get("grid_cell_id", ""))
            if cell.startswith("g") and cell not in drawn_cells:
                try:
                    ix, iy = (int(n) for n in cell[1:].split("_", 1))
                    x0, y0 = ix * 10_000, iy * 10_000
                    ax.add_patch(MplPolygon([(x0,y0),(x0+10000,y0),(x0+10000,y0+10000),(x0,y0+10000)],
                                            closed=True, fill=False, edgecolor="#777777", linewidth=.8, linestyle=":"))
                    ax.add_patch(MplPolygon([(x0+500,y0+500),(x0+9500,y0+500),(x0+9500,y0+9500),(x0+500,y0+9500)],
                                            closed=True, fill=False, edgecolor="#777777", linewidth=.6, linestyle="--"))
                    drawn_cells.add(cell)
                except ValueError:
                    pass
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for polygon in geoms:
                ax.add_patch(MplPolygon(list(polygon.exterior.coords), closed=True,
                                        facecolor=color, edgecolor=color, alpha=.55, linewidth=1.4))
                ax.scatter([polygon.centroid.x], [polygon.centroid.y], color=color, s=13, zorder=3)
        ax.set_title(f"{pair} — {len(pair_features)} candidates")
        ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel("Easting (EPSG:32650, m)"); ax.set_ylabel("Northing (m)")
        ax.grid(True, alpha=.18)
        if not pair_features:
            ax.text(.5, .5, "No windows passed frozen gates", transform=ax.transAxes,
                    ha="center", va="center", color="darkred")
    fig.suptitle("R1.5 candidate ROI distribution — role colors; dotted cells and dashed 500 m safety interiors")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=r, markersize=9)
               for r, c in ROLE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_usage(path, run_id):
    root = Path(".")
    out = f"data\\output\\benchmark_v1_radiometric_r1_5\\auto_roi_candidates\\{run_id}"
    text = f'''# R1.5 automatic ROI review — usage

Run ID: `{run_id}`. The candidate GeoJSON/CSV files are immutable and separate from the R1 review store.
All suggestions begin `UNVERIFIED`; human review snapshot and JSONL history are stored under
`data/output/benchmark_v1_radiometric_r1_5/reviews/`. Units remain UNKNOWN.

Generate the first pair (already completed for this run):

```powershell
.\\.venv-registration\\Scripts\\python.exe scripts\\benchmark\\generate_radiometric_r1_5_auto_rois.py --pair 00_01 --max-candidates 15 --run-id {run_id}
```

Append the remaining pairs with the same config and thresholds:

```powershell
.\\.venv-registration\\Scripts\\python.exe scripts\\benchmark\\generate_radiometric_r1_5_auto_rois.py --pairs 02_03 05_07 08_10 --max-candidates 15 --run-id {run_id}
```

Open a pair for first human review (replace the reviewer placeholder with a real person):

```powershell
.\\.venv-registration\\Scripts\\python.exe scripts\\benchmark\\annotate_radiometric_rois.py --pair 00_01 --reviewer "真实审核人" --auto-candidates {out}\\00_01.geojson
```

Independent TEST review uses a different real person and the explicit second-review mode:

```powershell
.\\.venv-registration\\Scripts\\python.exe scripts\\benchmark\\annotate_radiometric_rois.py --pair 00_01 --reviewer "第二审核人" --second-review --auto-candidates {out}\\00_01.geojson
```

Review steps: inspect both original-resolution image windows; select one candidate; use zoom controls or the band selector; optionally edit EPSG:32650 WKT and apply (this clears prior approval and recalculates per-scene/per-band mask support and frozen role); choose a human label/status; enter visual evidence; apply. An approval is never inferred from the score. TEST decisions require a distinct reviewer; disagreements remain unresolved. Save events are append-only and can be resumed by reopening the same candidate file.

The 32×32 window and strong-edge screen can under-sample narrow/boundary-rich, fragmented, low-texture, shadowed or low-contrast terrain. Do not weaken thresholds to fill a quota. Empty/short candidate sets are valid outputs.
'''
    path.write_text(text, encoding="utf-8")
