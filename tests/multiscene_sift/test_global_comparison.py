"""Tests for the fixed 4 matcher x 2 Global comparison contract."""

from __future__ import annotations

from src.multiscene_sift.global_comparison import build_global_comparison_rows


def test_global_comparison_always_preserves_eight_expected_rows(tmp_path):
    ledger = {
        "matchers": {
            "sift": {"rerun": "FAILED", "validation": "PENDING", "mst": "PENDING", "translation_l2": "PENDING"},
            "loftr": {"rerun": "PENDING", "validation": "PENDING", "mst": "PENDING", "translation_l2": "PENDING"},
            "efficient_loftr": {"rerun": "PENDING", "validation": "PENDING", "mst": "PENDING", "translation_l2": "PENDING"},
            "lightglue_disk": {"rerun": "PENDING", "validation": "PENDING", "mst": "PENDING", "translation_l2": "PENDING"},
        }
    }

    rows = build_global_comparison_rows(tmp_path, ledger)

    assert len(rows) == 8
    assert {(row["matcher"], row["global_method"]) for row in rows} == {
        (matcher, method)
        for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk")
        for method in ("MST", "Translation-L2")
    }
    assert all(row["status"] in {"RERUN_FAILED", "RERUN_NOT_STARTED"} for row in rows)
