"""Verify Windows-origin freeze locks against explicitly mapped local roots."""
from __future__ import annotations

import hashlib
import json
import ntpath
from pathlib import Path, PurePosixPath, PureWindowsPath


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_parts(path):
    raw = str(path)
    windows = bool(ntpath.splitdrive(raw)[0]) or raw.startswith("\\\\")
    pure = PureWindowsPath(raw) if windows else PurePosixPath(raw)
    parts = tuple(part for part in pure.parts if part not in (pure.anchor, "\\", "/"))
    return windows, pure, parts


def _map_path(locked_path, mappings):
    source_windows, source_pure, source_parts = _source_parts(locked_path)
    if not source_pure.is_absolute():
        return None, "UNSAFE_PATH", "locked path is not absolute"
    if ".." in source_parts:
        return None, "UNSAFE_PATH", "locked path contains parent traversal"
    matches = []
    for mapping in mappings:
        source_root = str(mapping.get("source_root", ""))
        target_root = mapping.get("target_root")
        if not source_root or not target_root:
            continue
        root_windows, root_pure, root_parts = _source_parts(source_root)
        if root_windows != source_windows or not root_pure.is_absolute() or ".." in root_parts:
            continue
        if len(source_parts) < len(root_parts):
            continue
        equal = all(a.casefold() == b.casefold() for a, b in zip(source_parts, root_parts))
        if equal:
            matches.append((mapping, source_parts[len(root_parts):]))
    if not matches:
        return None, "PATH_MAPPING_INCOMPLETE", "no declared root maps this locked path"
    if len(matches) != 1:
        return None, "PATH_MAPPING_AMBIGUOUS", "multiple declared roots match this locked path"
    mapping, relative_parts = matches[0]
    root = Path(mapping["target_root"]).expanduser().resolve()
    target = root.joinpath(*relative_parts).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None, "UNSAFE_PATH", "mapped path escapes its target root"
    return target, None, None


def verify_portable_lock(lock_path, mappings, *, strict=True, audit_path=None):
    """Check original lock hashes using a caller-supplied Windows→local map.

    The lock is opened read-only and never rewritten. Ambiguous, incomplete,
    traversal, absent, and hash-mismatched records all fail closed.
    """
    path = Path(lock_path).resolve()
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result = {"status": "FREEZE_LOCK_INVALID", "error": f"{type(exc).__name__}:{exc}",
                  "locked_file_count": 0, "matched_file_count": 0, "files": []}
        if audit_path:
            Path(audit_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    entries = lock.get("files")
    if lock.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        return {"status": "FREEZE_LOCK_INVALID", "locked_file_count": 0,
                "matched_file_count": 0, "files": [], "error": "INVALID_LOCK_SCHEMA"}

    rows = []
    for item in entries:
        original = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        mapped, error, message = _map_path(original, mappings or [])
        if error:
            rows.append({"locked_path": original, "expected_sha256": expected,
                         "mapped_path": None, "actual_sha256": None,
                         "status": "UNSAFE_PATH" if error == "UNSAFE_PATH" else error,
                         "detail": message})
        elif not mapped.is_file():
            rows.append({"locked_path": original, "expected_sha256": expected,
                         "mapped_path": str(mapped), "actual_sha256": None,
                         "status": "MISSING_FILE"})
        else:
            actual = _digest(mapped)
            rows.append({"locked_path": original, "expected_sha256": expected,
                         "mapped_path": str(mapped), "actual_sha256": actual,
                         "status": "MATCH" if actual == expected else "HASH_MISMATCH"})
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if counts.get("UNSAFE_PATH"):
        status = "FREEZE_PATH_MAPPING_INVALID"
    elif counts.get("PATH_MAPPING_AMBIGUOUS"):
        status = "FREEZE_PATH_MAPPING_AMBIGUOUS"
    elif counts.get("PATH_MAPPING_INCOMPLETE"):
        status = "FREEZE_PATH_MAPPING_INCOMPLETE"
    elif counts.get("MISSING_FILE"):
        status = "FREEZE_MAPPED_FILE_MISSING"
    elif counts.get("HASH_MISMATCH"):
        status = "FREEZE_HASH_MISMATCH"
    else:
        status = "PASS"
    result = {"status": status, "valid": status == "PASS",
              "locked_file_count": len(entries), "matched_file_count": counts.get("MATCH", 0),
              "counts_by_status": counts, "files": rows,
              "strict": bool(strict), "lock_sha256": _digest(path)}
    if audit_path:
        Path(audit_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def as_freeze_verification(result):
    """Adapt a portable audit into the frozen evaluator's read-only interface."""
    from .freeze import FreezeVerification

    errors = tuple(f"{row.get('status')}:{row.get('locked_path')}" for row in result.get("files", [])
                   if row.get("status") != "MATCH")
    if not result.get("valid") and not errors:
        errors = (str(result.get("status", "PORTABLE_FREEZE_INVALID")),)
    return FreezeVerification(bool(result.get("valid")), errors)
