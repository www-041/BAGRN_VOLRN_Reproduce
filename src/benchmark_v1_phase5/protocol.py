"""Immutable protocol constants and a B9-only bridge to the Phase4 freeze."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import yaml


if TYPE_CHECKING:
    from benchmark_phase4_freeze import FrozenTrial


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from benchmark_phase4_freeze import PAIRS, FrozenInputError, sha256_file, verify_b9_identity


_FROZEN = {
    "pair_ids": list(PAIRS),
    "method_ids": ["SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR", "EfficientLoFTR", "LightGlue+DISK"],
    "crs": "EPSG:32650", "native_pixel_m": 14,
    "stretch_percentiles": [2, 98], "max_view_side": 1600,
    "seed": 20261011, "ransac_hypotheses": 1000, "ransac_radius_native_px": 2.0,
    "min_matches": 12, "min_inliers": 8, "min_inlier_fraction": 0.10,
    "coverage_grid": [4, 4], "min_coverage_cells": 4,
    "min_coverage_rows": 2, "min_coverage_columns": 2,
    "smoke_seconds": 900, "smoke_rss_gib": 12,
    "external_python": {"KARIOS": None, "AROSICS_GLOBAL": None},
    "method_settings": {
        "SIFT": {"nfeatures": 8000, "ratio": 0.75},
        "KARIOS": {"large_shift": False},
        "AROSICS_GLOBAL": {"center_window": [256, 256], "max_iter": 5, "max_shift": 5},
        "LoFTR": {"weights": "outdoor", "confidence": 0.2, "max_matches": 5000},
        "EfficientLoFTR": {"resolution": "full", "precision": "fp32", "confidence": 0.0},
        "LightGlue+DISK": {"features": 2048, "extractor_resize": 1024, "confidence": 0.0},
    },
}


@dataclass(frozen=True)
class Phase5Config:
    config_sha256: str
    pair_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    crs: str
    native_pixel_m: int
    stretch_percentiles: tuple[int, int]
    max_view_side: int
    seed: int
    ransac_hypotheses: int
    ransac_radius_native_px: float
    min_matches: int
    min_inliers: int
    min_inlier_fraction: float
    coverage_grid: tuple[int, int]
    min_coverage_cells: int
    min_coverage_rows: int
    min_coverage_columns: int
    smoke_seconds: int
    smoke_rss_gib: int
    external_python: tuple[tuple[str, str | None], ...]
    method_settings: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]


def load_config(path: Path) -> Phase5Config:
    """Accept exactly the approved protocol; extra keys cannot carry truth."""
    config_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(config_bytes.decode("utf-8"))
    if raw != _FROZEN or type(raw) is not dict:
        raise ValueError("Phase5 config differs from the frozen truth-free protocol")
    settings = tuple((name, tuple((key, tuple(value) if isinstance(value, list) else value)
                                   for key, value in options.items()))
                     for name, options in raw["method_settings"].items())
    return Phase5Config(
        hashlib.sha256(config_bytes).hexdigest(),
        tuple(raw["pair_ids"]), tuple(raw["method_ids"]), raw["crs"], raw["native_pixel_m"],
        tuple(raw["stretch_percentiles"]), raw["max_view_side"], raw["seed"],
        raw["ransac_hypotheses"], raw["ransac_radius_native_px"], raw["min_matches"],
        raw["min_inliers"], raw["min_inlier_fraction"], tuple(raw["coverage_grid"]),
        raw["min_coverage_cells"], raw["min_coverage_rows"], raw["min_coverage_columns"],
        raw["smoke_seconds"], raw["smoke_rss_gib"],
        tuple(raw["external_python"].items()), settings)


@dataclass(frozen=True)
class PairManifest:
    pair_id: str
    scene_i: str
    scene_j: str
    path_i: Path
    path_j: Path
    size_i: int
    size_j: int
    sha256_i: str
    sha256_j: str


def build_pair_manifest(snapshot: FrozenTrial, pair_id: str) -> PairManifest:
    """Copy only audited B9 header/identity fields; never inspect candidates."""
    if pair_id not in PAIRS or tuple(snapshot.trial_pairs) != PAIRS:
        raise ValueError("Pair is outside the frozen four-pair trial")
    verify_b9_identity(snapshot)
    by_index = {int(record["scene_index"]): record
                for record in snapshot.audit["b9_verified_inputs"]}
    i, j = (int(value) for value in pair_id.split("_"))
    if i not in by_index or j not in by_index:
        raise FrozenInputError("Pair lacks frozen B9 identities")
    a, b = by_index[i], by_index[j]
    return PairManifest(pair_id, str(a["scene_id"]), str(b["scene_id"]),
                        Path(a["path"]), Path(b["path"]),
                        int(a["size_bytes"]), int(b["size_bytes"]),
                        str(a["sha256"]), str(b["sha256"]))


def verify_manifest_bytes(manifest: PairManifest) -> None:
    """Catch B9 mutation between manifest construction and raster opening."""
    for path, size, expected in ((manifest.path_i, manifest.size_i, manifest.sha256_i),
                                 (manifest.path_j, manifest.size_j, manifest.sha256_j)):
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != expected:
            raise FrozenInputError(f"B9 SHA256/size mismatch: {path.name}")
