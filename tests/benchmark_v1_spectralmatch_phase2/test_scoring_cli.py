import json
import subprocess
import sys
from pathlib import Path


def test_cli_keeps_b3_explicitly_not_run_and_writes_json(tmp_path):
    root = Path(__file__).resolve().parents[2]
    submission_path = tmp_path / "submission.json"
    output_path = tmp_path / "score.json"
    submission_path.write_text(json.dumps({"scope": "FORMAL_FULL"}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark/evaluate_unified_v1_phase2_1.py"),
         "--submission", str(submission_path), "--regime", "B3", "--output", str(output_path)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "NOT_RUN_B3"
    assert result["formal_ranking_eligible"] is False


def test_cli_rejects_crop_scope_for_formal_ranking(tmp_path):
    root = Path(__file__).resolve().parents[2]
    submission_path = tmp_path / "crop_submission.json"
    output_path = tmp_path / "crop_score.json"
    submission_path.write_text(json.dumps({"scope": "DIAGNOSTIC_CROP_ONLY"}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark/evaluate_unified_v1_phase2_1.py"),
         "--submission", str(submission_path), "--regime", "B1", "--scope", "DIAGNOSTIC_CROP_ONLY",
         "--output", str(output_path)], cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "SCOPE_REJECTED"
    assert result["formal_ranking_eligible"] is False
