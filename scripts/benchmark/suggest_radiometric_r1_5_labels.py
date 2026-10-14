from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_radiometric_r15.inputs import ALLOWED_PAIRS, load_pair_inputs  # noqa: E402
from benchmark_v1_radiometric_r15.machine_suggestions import (  # noqa: E402
    SUGGESTION_FIELDS,
    build_machine_suggestion,
    sha256_file,
    select_human_review_sample,
    write_machine_suggestion_artifacts,
)
from benchmark_v1_radiometric_r15.storage import load_candidate_collection  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create separate conservative R1.5 machine-label suggestions (not truth).")
    parser.add_argument("--candidate-run-id", default="r15_20261012_04")
    parser.add_argument("--run-id", required=True, help="new machine-only run ID; existing output is never overwritten")
    parser.add_argument("--pairs", nargs="+", choices=sorted(ALLOWED_PAIRS), default=sorted(ALLOWED_PAIRS))
    return parser


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if len(set(args.pairs)) != len(args.pairs):
        print("duplicate pairs are not allowed", file=sys.stderr); return 2
    candidate_root = ROOT / "data/output/benchmark_v1_radiometric_r1_5/auto_roi_candidates" / args.candidate_run_id
    output_root = ROOT / "data/output/benchmark_v1_radiometric_r1_5/machine_label_suggestions" / args.run_id
    report_root = ROOT / "docs/benchmark/v1/radiometric_r1_5/machine_suggestions" / args.run_id
    if output_root.exists() or report_root.exists():
        print(f"refusing to overwrite existing machine run: {output_root} or {report_root}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "machine_run_id": args.run_id,
        "candidate_run_id": args.candidate_run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "R1.5_CONSERVATIVE_MULTIBAND_TEXTURE_EVIDENCE_V1",
        "scope": "four existing pair candidate sets only; no R1/R1.5 candidate generation",
        "physical_units": "UNKNOWN_RAW_DN",
        "geometry_verified": False,
        "semantic_classifier": "NONE; conservative UNKNOWN recommendations until independent references exist",
        "missing_auxiliary_data": ["DEM", "CLOUD_MASK_OR_QA", "INDEPENDENT_CHANGE_REFERENCE", "VALIDATED_LANDCOVER_REFERENCE"],
        "pairs": {},
        "protected_inputs": [],
    }
    report_lines = ["# R1.5 machine suggestions", "", f"Machine run: `{args.run_id}`",
                    f"Candidate run: `{args.candidate_run_id}`", "",
                    "All entries are machine-only suggestions (`SUGGESTED`), not human review or ground truth. "
                    "Because pixel units are UNKNOWN, there is no suitable DEM/cloud-quality mask, geometry is not independently verified, "
                    "and no independent change/land-cover reference is available, raw multi-band and texture evidence is not converted into semantic classes. "
                    "In particular, no reflectance indices are computed and `CONFIRMED_CHANGE` is never emitted.", ""]
    all_features = []
    try:
        for pair_id in args.pairs:
            candidate_path = candidate_root / f"{pair_id}.geojson"
            if not candidate_path.is_file():
                raise FileNotFoundError(f"frozen candidate file missing: {candidate_path}")
            candidate_sha = sha256_file(candidate_path)
            collection = load_candidate_collection(candidate_path, pair_id)
            pair_inputs = load_pair_inputs(pair_id, ROOT)
            missing_for_pair = list(manifest["missing_auxiliary_data"])
            for scene_name, scene in zip(("scene_i", "scene_j"), pair_inputs.scenes):
                missing_bands = [name for name, band in scene.bands.items() if not band.usable]
                if missing_bands:
                    missing_for_pair.append(f"{scene_name}_MISSING_OR_UNUSABLE_BANDS:{','.join(missing_bands)}")
            suggestions = [build_machine_suggestion(feature, run_id=args.run_id,
                           pair_inputs=pair_inputs, missing_auxiliary=missing_for_pair)
                           for feature in collection["features"]]
            target_json, target_csv = write_machine_suggestion_artifacts(
                output_root, pair_id, suggestions,
                source_candidate_sha256=candidate_sha, run_id=args.run_id)
            after_sha = sha256_file(candidate_path)
            if after_sha != candidate_sha:
                raise RuntimeError(f"source candidate changed during processing: {candidate_path}")
            for scene in pair_inputs.scenes:
                for band in scene.bands.values():
                    if band.usable and band.path:
                        manifest["protected_inputs"].append({
                            "scene_id": scene.scene_id, "band": band.band,
                            "path": str(band.path),
                            "recorded_task15_source_sha256": band.manifest_sha256 or "UNKNOWN_NOT_RECORDED",
                            "current_header_fingerprint": band.header_fingerprint.get("fingerprint_sha256"),
                            "file_size_bytes": band.header_fingerprint.get("file_size_bytes"),
                            "mtime_ns": band.header_fingerprint.get("mtime_ns"),
                            "note": "recorded source SHA is inherited metadata; current full-file bytes were not re-hashed"})
            per_field = {}; per_field_confidence = {}
            for field in SUGGESTION_FIELDS:
                per_field[field] = dict(Counter(feature["properties"][field]["label"] for feature in suggestions))
                per_field_confidence[field] = dict(Counter(feature["properties"][field]["confidence"] for feature in suggestions))
            manifest["pairs"][pair_id] = {
                "candidate_count": len(collection["features"]),
                "candidate_source_sha256": candidate_sha,
                "candidate_sha256_after_generation": after_sha,
                "suggestion_geojson": str(target_json.relative_to(ROOT)),
                "suggestion_csv": str(target_csv.relative_to(ROOT)),
                "suggestion_geojson_sha256": sha256_file(target_json),
                "suggestion_csv_sha256": sha256_file(target_csv),
                "label_distribution": per_field,
                "confidence_distribution": per_field_confidence,
            }
            all_features.extend(suggestions)
            report_lines.extend([f"## {pair_id}", "", f"Candidate count: {len(suggestions)}", "",
                "| Field | Machine label distribution |", "|---|---|"])
            for field, counts in per_field.items():
                labels = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
                confidence = ", ".join(f"{level}: {count}" for level, count in sorted(per_field_confidence[field].items()))
                report_lines.append(f"| `{field}` | {labels}; confidence {confidence} |")
            report_lines.append("")
        manifest["candidate_count_total"] = len(all_features)
        manifest["candidate_identity_unique"] = len({(f["properties"]["pair_id"], f["properties"]["candidate_id"])
                                                       for f in all_features}) == len(all_features)
        manifest["review_decisions_created"] = 0
        manifest["human_events_created"] = 0
        manifest["semantic_extension_recommendation"] = "PAUSE_UNTIL_RESEARCHER_REVIEW"
        sample_rows = select_human_review_sample(all_features, max_samples=12)
        sample_path = report_root / "human_review_sample.csv"
        with sample_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]) if sample_rows else
                ["pair_id", "candidate_id", "role", "center_x_epsg32650", "center_y_epsg32650",
                 "all_machine_labels_unknown", "selection_basis"])
            writer.writeheader(); writer.writerows(sample_rows)
        manifest["human_review_sample_count"] = len(sample_rows)
        manifest["human_review_sample_basis"] = "pair and frozen-role coverage only; no algorithm score or machine label used"
        manifest["human_review_sample_sha256"] = sha256_file(sample_path)
        manifest_path = output_root / "machine_suggestion_manifest.json"
        _write_new(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        report_lines.extend(["## Human agreement", "",
            "No human labels are generated here. Agreement remains `NOT_EVALUATED` until eligible human review records exist. "
            "The provided comparison interface is descriptive agreement only, not classifier accuracy.", "",
            "## Bias and confusion checks", "",
            "The generator does not use inter-date brightness similarity to assign `NO_CHANGE`; raw-DN median differences are retained as evidence only. "
            "No land-cover/shadow/change confusion rate can be estimated because no semantic labels were emitted or independently reviewed. "
            "All four fields are UNKNOWN, so this run demonstrates insufficient current semantic evidence rather than validated classification.", "",
            "## Sampling / limitations", "",
            "Review representatives across all four pairs and across DEVELOPMENT/TEST/UNASSIGNED before relying on any suggestion. "
            f"A deterministic pair/role-balanced sample of {len(sample_rows)} candidates is listed in `human_review_sample.csv`. "
            "Because these recommendations are conservatively UNKNOWN, inspect whether the measured evidence could support any semantic label; "
            "do not treat a low uncertainty or candidate score as class confidence. Fixed candidate windows and strong-edge filters can under-sample "
            "narrow boundaries, low-texture areas, shadowed terrain, and homogeneous regions.", ""])
        _write_new(report_root / "machine_suggestion_quality_report.md", "\n".join(report_lines))
        print(f"R1.5_MACHINE_SUGGESTIONS_READY run={args.run_id} pairs={len(args.pairs)} "
              f"candidates={len(all_features)} output={output_root}")
        return 0
    except Exception as exc:
        print(f"R1.5_MACHINE_SUGGESTIONS_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
