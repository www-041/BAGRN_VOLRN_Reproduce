"""Strict method-submission grid and support validation."""
from __future__ import annotations

def validate_grid(expected_affine, actual_affine, *, expected_crs="EPSG:32650", actual_crs="EPSG:32650", expected_shape=None, actual_shape=None):
    if expected_crs!=actual_crs or tuple(expected_affine)!=tuple(actual_affine): return "INVALID_GRID"
    if expected_shape is not None and tuple(expected_shape)!=tuple(actual_shape): return "INVALID_GRID"
    return "PASS"

def validate_submission(submission, frozen):
    errors=[]
    if submission.get("input_freeze_sha256")!=frozen.get("input_freeze_sha256"): errors.append("INPUT_HASH_MISMATCH")
    if submission.get("geometry_freeze_sha256")!=frozen.get("geometry_freeze_sha256"): errors.append("GEOMETRY_HASH_MISMATCH")
    if submission.get("band")!=frozen.get("band"): errors.append("INVALID_BAND")
    if not submission.get("method_id"): errors.append("MISSING_METHOD_ID")
    return {"status":"PASS" if not errors else errors[0],"errors":errors,"eligible":not errors}
