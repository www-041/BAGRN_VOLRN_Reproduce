from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_radiometric_r1.inventory import (  # noqa: E402
    build_pair_inventory,
    select_representative_pairs,
    write_csv,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the read-only R1 temporal pair inventory.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    docs = args.root / "docs/benchmark/v1"
    out = docs / "radiometric_r1"
    pair_path = docs / "pair_candidates.csv"
    temporal_path = docs / "temporal_pair_inventory.csv"
    scene_path = docs / "scene_inventory.csv"
    rows = build_pair_inventory(pair_path, temporal_path, scene_path)
    selected = select_representative_pairs(rows, target=10)
    write_csv(out / "radiometric_pair_inventory.csv", rows)
    bins = Counter(row["temporal_bin"] for row in rows)
    cross = sum(bins[k] for k in ("T1", "T2", "T3"))
    report = [
        "# R1 temporal stratification report",
        "",
        "This report joins existing frozen manifests. It does not recompute registration, radiometric values, or image overlap.",
        "",
        f"- Scene combinations: {len(rows)} (13 choose 2).",
        f"- B9 effective-overlap pairs: {len(rows) - bins['NO_VALID_OVERLAP']}.",
        f"- No-effective-overlap pairs: {bins['NO_VALID_OVERLAP']} (`NO_VALID_OVERLAP`, not algorithm FAIL).",
        f"- Same-date T0: {bins['T0']}; cross-date T1 (<7d): {bins['T1']}; T2 (7–<30d): {bins['T2']}; T3 (>=30d): {bins['T3']}.",
        f"- Cross-date effective overlap pairs: {cross}.",
        f"- Source SHA256: pair candidates `{sha256(pair_path)}`, temporal inventory `{sha256(temporal_path)}`, scene inventory `{sha256(scene_path)}`.",
        "- Stable-landcover and real-change truth remain UNKNOWN/UNVERIFIED; valid support is not stable-ground truth.",
        "",
        "## Pair counts by time stratum",
        "",
        "| Stratum | Count | Meaning |",
        "|---|---:|---|",
        f"| T0 | {bins['T0']} | Same date |",
        f"| T1 | {bins['T1']} | Different date, <7 days |",
        f"| T2 | {bins['T2']} | 7 to <30 days |",
        f"| T3 | {bins['T3']} | >=30 days |",
        f"| NO_VALID_OVERLAP | {bins['NO_VALID_OVERLAP']} | No shared valid B9 pixels |",
    ]
    (out / "temporal_stratification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    selection = [
        "# R1 radiometric pair selection — draft",
        "",
        "This is a pre-method, unverified draft. It uses temporal coverage, existing effective-overlap support, metadata quality flags and deterministic pair-ID tie-breaking only. No Ours/VOLRN/Matcher performance was read. Researcher confirmation is required before freezing.",
        "",
        "| Pair | Bin | Days | Valid B9 overlap pixels | Quality i/j | Role | Reason |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in selected:
        reason = "required representative pair" if row["pair_id"] in {"00_01", "04_11", "02_10", "00_10"} else "stratified by quality/support; no algorithm score"
        selection.append(
            f"| {row['pair_id']} | {row['temporal_bin']} | {row.get('days_apart','')} | {row.get('valid_overlap_pixels','')} | {row.get('quality_i','UNKNOWN')}/{row.get('quality_j','UNKNOWN')} | UNVERIFIED | {reason} |"
        )
    selection += ["", "All selected pairs still require human ROI review. A shortlist does not establish stable ground or actual land change."]
    (out / "radiometric_pair_selection_draft.md").write_text("\n".join(selection) + "\n", encoding="utf-8")
    print(f"R1_INVENTORY_COMPLETE total={len(rows)} overlaps={len(rows)-bins['NO_VALID_OVERLAP']} no_overlap={bins['NO_VALID_OVERLAP']} shortlist={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
