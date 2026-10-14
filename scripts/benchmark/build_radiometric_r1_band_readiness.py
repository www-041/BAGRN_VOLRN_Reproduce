from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_radiometric_r1.band_readiness import summarize_v_bands  # noqa: E402
from src.benchmark_v1_radiometric_r1.inventory import write_csv  # noqa: E402


def main() -> int:
    docs = ROOT / "docs/benchmark/v1"
    r1 = docs / "radiometric_r1"
    rows, summary = summarize_v_bands(
        docs / "v_band_inventory.csv", docs / "scene_inventory.csv",
        ROOT / "data/output/final_pipeline/b9_13scene_formal_v2/scene_manifest.csv",
    )
    write_csv(r1 / "v_band_quality_inventory.csv", rows)
    risk_scenes = ", ".join(summary["quality_risk_scenes"]) or "None"
    report = [
        "# V-band radiometric readiness (metadata-only)", "",
        "This report reads the existing V-band CSV and frozen B9 scene identity manifest only. It does not hash full rasters, read image pixels, apply MTL coefficients, resample, or modify any source file.", "",
        f"- Scenes: {summary['scene_count']}; metadata rows: {summary['band_rows']}; expected bands per scene: {summary['expected_band_count_per_scene']}; complete scenes: {summary['complete_scene_count']}.",
        f"- Missing bands: `{summary['missing_bands'] or 'none'}`.",
        f"- Center wavelength present: {summary['wavelength_metadata_rows']} / {summary['band_rows']} records.",
        f"- Data unit labels in inventory: `{summary['data_units']}`; physical unit remains UNKNOWN unless the source metadata explicitly establishes it.",
        f"- QUALITY_STATUS=RISK scenes: {summary['quality_risk_scene_count']} ({risk_scenes}).",
        f"- Within-scene grid metadata discrepancies: {summary['within_scene_grid_mismatch_scenes'] or 'none'}.",
        f"- B9 grid identical across scenes: {summary['cross_scene_b9_grid_uniform']} (distinct metadata signatures: {summary['cross_scene_b9_grid_signature_count']}).",
        f"- Existing SHA256 available: {summary['sha256_available_rows']} records; missing: {summary['sha256_missing_rows']} records. {summary['sha256_scope_note']}", "",
        "## Interpretation limits", "",
        "The inventory contains wavelength center/bandwidth values from metadata, but this does not prove that MTL calibration coefficients were applied to stored pixels. Existing raster scale/offset and pixel type are retained as reported; no calibration was performed. Similar per-scene Affine/dimensions can support map-grid planning but do not prove detector-level or subpixel inter-band co-registration. A calibration/orthorectification provenance record and spatially independent inter-band check are still needed before physical reflectance metrics, SAM, or SAC.", "",
        "The three flagged RISK scenes remain in the candidate population and must be reported as a quality stratum, not removed after seeing method results.",
    ]
    (r1 / "v_band_radiometric_readiness.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"R1_BAND_READINESS_READY scenes={summary['scene_count']} records={summary['band_rows']} complete={summary['complete_scene_count']} hashes={summary['sha256_available_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
