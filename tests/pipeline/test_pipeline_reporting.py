import json
from pathlib import Path

from src.pipeline.final_report import write_final_report


def test_final_report_is_self_contained_and_writes_required_artifacts(tmp_path: Path):
    result = write_final_report(tmp_path, metrics={"scientific_decision": "MIXED_SCALE_REVIEW"})
    assert result["scientific_decision"] == "MIXED_SCALE_REVIEW"
    assert (tmp_path / "12_report" / "FINAL_PIPELINE_REPORT.md").is_file()
    assert (tmp_path / "12_report" / "final_summary.json").is_file()
    assert (tmp_path / "12_report" / "artifact_manifest.json").is_file()
    assert (tmp_path / "12_report" / "figure_selection.json").is_file()
    text = (tmp_path / "12_report" / "FINAL_PIPELINE_REPORT.md").read_text(encoding="utf-8")
    assert "1. Stage07原bug是什么？" in text
    assert "23. 是否允许进入下一scale benchmark？" in text
    assert json.loads((tmp_path / "12_report" / "final_summary.json").read_text(encoding="utf-8"))["scientific_decision"] == "MIXED_SCALE_REVIEW"
