"""Small contracts shared by the PIF sensitivity runner and worker."""

import hashlib
import json

PIF_METHODS = frozenset({"flood_from_match_points", "entire"})


def validate_pif_method(value: str) -> str:
    method = str(value)
    if method not in PIF_METHODS:
        raise ValueError(f"unsupported PIF method: {method!r}; choose one of {sorted(PIF_METHODS)}")
    return method


def assert_support_identity(expected_hash: str, actual_hash: str) -> None:
    if not expected_hash or not actual_hash or expected_hash != actual_hash:
        raise ValueError(f"evaluation support changed: expected {expected_hash!r}, got {actual_hash!r}")


def frozen_input_support_identity(mask_hashes: dict[str, str], crop_window: dict | None) -> str:
    """Identity of frozen scene masks and diagnostic crop window, not a new mask."""
    if set(mask_hashes) != {"i", "j"} or not all(mask_hashes.values()):
        raise ValueError("both frozen input mask hashes are required")
    payload = {"frozen_scene_mask_sha256": mask_hashes, "crop_window": crop_window}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluation_support_identity(mask_i, mask_j) -> tuple[str, int]:
    """Hash the exact pairwise valid intersection; no support is regenerated."""
    import numpy as np

    a, b = np.asarray(mask_i) > 0, np.asarray(mask_j) > 0
    if a.shape != b.shape:
        raise ValueError("pair masks must share the frozen grid")
    support = np.ascontiguousarray(a & b, dtype=np.uint8)
    return hashlib.sha256(support.tobytes()).hexdigest(), int(support.sum())
