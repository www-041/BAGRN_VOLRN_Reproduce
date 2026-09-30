"""Scene-count-independent composition of the frozen Task13 seam/label stages.

This module is intentionally an adapter, not a second seam or labeling
algorithm.  Pair processing is delegated to :func:`process_pair`; ownership,
global labels, geometry tie resolution, and cosine blending are delegated to
the frozen Task13 modules.  The array entry points make the protocol easy to
test, while callers that read GeoTIFFs can supply one canonical window at a
time around the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from .multiscene_label import (
    LABEL_UNRESOLVED,
    MultiSceneLabelResult,
    aggregate_multiscene_labels,
    build_pairwise_preference_field,
)
from .pipeline import PairResult, _corridor, process_pair
from .config import SeamLocalRuntimeConfig
from .source_side import SourceSideResult, resolve_source_sides
from .tie_resolution import TieResolutionResult, resolve_unresolved_geometry


@dataclass(frozen=True)
class OverlapPair:
    scene_i: int
    scene_j: int
    shared_valid_pixel_count: int | None = None
    registration_edge_accepted: bool | None = None

    @property
    def pair_id(self) -> str:
        return f"{self.scene_i:02d}_{self.scene_j:02d}"


@dataclass(frozen=True)
class AdapterPairResult:
    pair_id: str
    scene_i: int
    scene_j: int
    process: PairResult
    initial_source_side: SourceSideResult | None
    refined_source_side: SourceSideResult | None
    source_side_status: str
    status: str
    shared_valid_pixel_count: int
    registration_edge_accepted: bool | None

    @property
    def initial_source_side_status(self) -> str:
        return self.initial_source_side.status if self.initial_source_side is not None else "UNRESOLVED"

    @property
    def refined_source_side_status(self) -> str:
        return self.refined_source_side.status if self.refined_source_side is not None else "UNRESOLVED"


@dataclass(frozen=True)
class AdapterResult:
    scene_count: int
    pair_results: dict[str, AdapterPairResult]
    labels: np.ndarray
    label_methods: np.ndarray
    v1_mosaic: np.ndarray
    v2_mosaic: np.ndarray
    corrected_scenes: np.ndarray
    multi_label_weights: np.ndarray
    diagnostics: dict[str, Any]
    v1_labels: np.ndarray | None = None
    v1_label_methods: np.ndarray | None = None
    v2_labels: np.ndarray | None = None
    v2_label_methods: np.ndarray | None = None
    v1_weights: np.ndarray | None = None
    v2_weights: np.ndarray | None = None
    label_diagnostics: dict[str, Any] | None = None

    @property
    def shared_label_map(self) -> bool:
        return False


def normalize_overlap_pairs(
    overlap_pairs: Iterable[OverlapPair | Mapping[str, Any] | Sequence[int]],
) -> list[OverlapPair]:
    """Normalize spatial-overlap records without conflating registration edges."""

    result: list[OverlapPair] = []
    seen: set[tuple[int, int]] = set()
    for item in overlap_pairs:
        if isinstance(item, OverlapPair):
            row = item
        elif isinstance(item, Mapping):
            i = item.get("scene_i", item.get("idx_i", item.get("i")))
            j = item.get("scene_j", item.get("idx_j", item.get("j")))
            if i is None or j is None:
                raise ValueError("overlap record requires scene_i and scene_j")
            shared = item.get("shared_valid_pixel_count", item.get("shared_valid_pixels"))
            accepted = item.get("registration_edge_accepted")
            row = OverlapPair(int(i), int(j), None if shared is None else int(shared),
                              None if accepted is None else bool(accepted))
        else:
            if len(item) != 2:
                raise ValueError("a tuple overlap pair must contain two scene ids")
            row = OverlapPair(int(item[0]), int(item[1]))
        if row.scene_i == row.scene_j or min(row.scene_i, row.scene_j) < 0:
            raise ValueError("overlap pair scene ids must be distinct and nonnegative")
        if row.scene_i > row.scene_j:
            row = OverlapPair(row.scene_j, row.scene_i, row.shared_valid_pixel_count,
                              row.registration_edge_accepted)
        if (row.scene_i, row.scene_j) not in seen:
            seen.add((row.scene_i, row.scene_j))
            result.append(row)
    return result


def _crop_to_full_seam(pair: PairResult, full_shape: tuple[int, int], seam: Any | None = None):
    seam = pair.refined_seam if seam is None else seam
    if seam is None or pair.crop_origin is None:
        return None
    path = seam.row_col_path.copy()
    path[:, 0] += pair.crop_origin[0]
    path[:, 1] += pair.crop_origin[1]
    return replace(seam, row_col_path=path)


def _expand_seam_to_full(pair: PairResult, full_shape: tuple[int, int], seam: Any | None = None):
    """Extend a crop seam to the canonical line domain for field construction."""

    seam = _crop_to_full_seam(pair, full_shape, seam)
    if seam is None or pair.crop_origin is None:
        return None
    path = seam.row_col_path
    origin_r, origin_c = pair.crop_origin
    if seam.orientation == "vertical":
        rows = np.arange(full_shape[0], dtype=np.int64)
        local_rows = rows - origin_r
        clipped = np.clip(local_rows, 0, path.shape[0] - 1)
        cols = path[clipped, 1]
        expanded_path = np.column_stack((rows, cols))
    else:
        cols = np.arange(full_shape[1], dtype=np.int64)
        local_cols = cols - origin_c
        clipped = np.clip(local_cols, 0, path.shape[0] - 1)
        rows = path[clipped, 0]
        expanded_path = np.column_stack((rows, cols))
    return replace(seam, row_col_path=expanded_path)


def _seam_domain(pair: PairResult, full_shape: tuple[int, int]) -> np.ndarray:
    """Return the original crop seam's valid line domain (no extrapolation evidence)."""

    domain = np.zeros(full_shape, dtype=bool)
    if pair.refined_seam is None or pair.crop_origin is None:
        return domain
    r0, c0 = pair.crop_origin
    if pair.refined_seam.orientation == "vertical":
        r1 = min(full_shape[0], r0 + len(pair.refined_seam.row_col_path))
        domain[r0:r1, :] = True
    else:
        c1 = min(full_shape[1], c0 + len(pair.refined_seam.row_col_path))
        domain[:, c0:c1] = True
    return domain


def _source_side(
    seam: Any,
    va: np.ndarray,
    vb: np.ndarray,
    footprint_a: Any,
    footprint_b: Any,
) -> SourceSideResult:
    """Run the frozen strict ownership resolver on the pair crop."""

    overlap = va & vb
    return resolve_source_sides(seam, overlap, va, vb, footprint_a, footprint_b)


def _taper_for_seam(seam: Any, shape: tuple[int, int], half_width: int = 128) -> np.ndarray:
    corridor = _corridor(seam, shape, half_width)
    if seam.orientation == "vertical":
        centers = seam.row_col_path[:, 1][:, None]
        coordinates = np.arange(shape[1])[None, :]
    else:
        centers = seam.row_col_path[:, 0][None, :]
        coordinates = np.arange(shape[0])[:, None]
    distance = np.abs(coordinates - centers)
    return np.where(corridor, 0.5 * (1.0 + np.cos(np.pi * distance / half_width)), 0.0)


def _aggregate_corrections(
    scenes: np.ndarray,
    masks: np.ndarray,
    pair_results: Mapping[str, AdapterPairResult],
) -> np.ndarray:
    """Aggregate pair deltas by commutative weighted mean, never cumulatively."""

    corrected = np.asarray(scenes, dtype=np.float64).copy()
    sums = np.zeros_like(corrected, dtype=np.float64)
    weights = np.zeros_like(corrected, dtype=np.float64)
    for item in pair_results.values():
        pair = item.process
        if pair.initial_seam is None or pair.crop_origin is None:
            continue
        if pair.corrected_a is None or pair.corrected_b is None:
            continue
        origin_r, origin_c = pair.crop_origin
        h, w = pair.corrected_a.shape
        sl = (slice(origin_r, origin_r + h), slice(origin_c, origin_c + w))
        va = masks[item.scene_i][sl]
        vb = masks[item.scene_j][sl]
        taper = _taper_for_seam(pair.initial_seam, (h, w))
        for scene_index, original, local, valid in (
            (item.scene_i, scenes[item.scene_i][sl], pair.corrected_a, va),
            (item.scene_j, scenes[item.scene_j][sl], pair.corrected_b, vb),
        ):
            active = valid & np.isfinite(original) & np.isfinite(local) & (taper > 0)
            delta = np.where(active, local - original, 0.0)
            weight = np.where(active, taper, 0.0)
            sums[scene_index][sl] += delta * weight
            weights[scene_index][sl] += weight
    active = weights > 0
    corrected[active] += sums[active] / weights[active]
    corrected[~masks] = np.nan
    return corrected


def aggregate_labels_with_ties(
    masks: np.ndarray,
    pairwise_fields: Mapping[tuple[int, int], Any],
    raw_edt: np.ndarray,
    *,
    p95_edt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compose Task13B aggregation with the frozen Task13B.1 tie hierarchy."""

    masks = np.asarray(masks, dtype=bool)
    raw = np.asarray(raw_edt)
    if raw.shape != masks.shape:
        raise ValueError("raw_edt and masks must have the same scene-first shape")
    base: MultiSceneLabelResult = aggregate_multiscene_labels(
        masks, pairwise_fields, interiority=raw, interiority_p95=p95_edt
    )
    tie_raw = raw.astype(np.float32, copy=False)
    if p95_edt is None:
        p95 = np.asarray([
            float(np.percentile(tie_raw[i][masks[i]], 95)) if np.any(masks[i]) else 1.0
            for i in range(masks.shape[0])
        ], dtype=np.float32)
    else:
        p95 = np.asarray(p95_edt, dtype=np.float32)
    tie: TieResolutionResult = resolve_unresolved_geometry(
        base.labels, masks, base.scores, tie_raw, p95
    )
    methods = np.zeros(base.labels.shape, dtype=np.uint8)
    unique = (base.labels != LABEL_UNRESOLVED) & (base.diagnostics["top_score_tie_pixels"] == 0)
    # The map below is diagnostic only; the label values remain authoritative.
    methods[unique] = 1
    methods[base.diagnostics["unresolved_mask"] & (tie.labels != LABEL_UNRESOLVED)] = 2
    methods[tie.methods == 3] = 3
    methods[tie.methods == 4] = 4
    diagnostics = dict(base.diagnostics)
    diagnostics.update({
        "labels_before_tie": base.labels,
        "tie_methods": tie.methods,
        "unresolved_before_tie": int(base.diagnostics["unresolved_pixels"]),
        # Pixels outside the union are intentionally nodata, not unresolved
        # labeling failures.  The Task13 reports count unresolved only inside
        # the candidate support.
        "unresolved_pixels": int(np.count_nonzero((tie.labels == LABEL_UNRESOLVED) & masks.any(axis=0))),
        "resolved_by_unclipped_interiority": int(tie.resolved_by_unclipped),
        "resolved_by_raw_edt": int(tie.resolved_by_raw_edt),
        "changed_old_resolved_labels": int(tie.changed_old_resolved_pixels),
        "all_clipped_equal_count": int(tie.all_clipped_equal_count),
        "invalid_label_pixels": int(np.count_nonzero((tie.labels < 0) & ~masks.any(axis=0))),
        "label_pixels_per_scene": {
            str(i): int(np.count_nonzero(tie.labels == i)) for i in range(masks.shape[0])
        },
        "label_method_pixels": {
            str(i): int(np.count_nonzero(methods == i)) for i in range(5)
        },
    })
    return tie.labels, methods, diagnostics


def _cosine_weights(labels: np.ndarray, masks: np.ndarray, width: int = 64) -> np.ndarray:
    weights = np.zeros(masks.shape, dtype=np.float32)
    for i in range(masks.shape[0]):
        region = labels == i
        inside = distance_transform_edt(region)
        outside = distance_transform_edt(~region)
        signed = inside - outside
        active = masks[i] & (signed > -width)
        weights[i][active] = np.where(
            signed[active] >= width, 1.0,
            0.5 * (1.0 + np.cos(np.pi * np.abs(signed[active]) / width)),
        )
    return weights


def blend_labeled_scenes(
    scenes: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    tile_size: int = 1024,
    halo: int = 128,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Blend all scenes simultaneously using one hard-label map and weights."""

    del halo  # The full-grid EDT is computed once; tile writes are equivalent.
    values = np.asarray(scenes, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or values.shape != masks.shape or labels.shape != values.shape[1:]:
        raise ValueError("scenes, masks and labels have incompatible shapes")
    q = _cosine_weights(labels, masks) if weights is None else np.asarray(weights, dtype=np.float32)
    if q.shape != masks.shape:
        raise ValueError("weights shape mismatch")
    out = np.full(labels.shape, np.nan, dtype=np.float64)
    for r0 in range(0, labels.shape[0], tile_size):
        for c0 in range(0, labels.shape[1], tile_size):
            rs = slice(r0, min(labels.shape[0], r0 + tile_size))
            cs = slice(c0, min(labels.shape[1], c0 + tile_size))
            a = values[:, rs, cs]
            w = q[:, rs, cs]
            valid = masks[:, rs, cs] & np.isfinite(a) & (w > 0)
            den = np.sum(np.where(valid, w, 0.0), axis=0)
            num = np.sum(np.where(valid, a * w, 0.0), axis=0)
            target = np.full(den.shape, np.nan, dtype=np.float64)
            np.divide(num, den, out=target, where=den > 0)
            out[rs, cs] = target
    return out


def run_multiscene_adapter(
    bagrn_scene_paths: Sequence[Any] | np.ndarray,
    valid_mask_paths: Sequence[Any] | np.ndarray,
    overlap_pairs: Iterable[OverlapPair | Mapping[str, Any] | Sequence[int]],
    *,
    output_dir: str | Path | None = None,
    canonical_grid: Mapping[str, Any] | None = None,
    frozen_protocol: Mapping[str, Any] | None = None,
    footprints: Sequence[Any] | None = None,
    runtime_config: SeamLocalRuntimeConfig | None = None,
) -> AdapterResult:
    """Run the same pair→ownership→label→correction→blend graph for any N.

    The primary tested form accepts in-memory scene arrays and boolean masks.
    File orchestration deliberately remains outside this pure adapter so a
    GeoTIFF runner can stream windows without changing the frozen algorithms.
    """

    scenes = np.asarray(bagrn_scene_paths) if isinstance(bagrn_scene_paths, np.ndarray) else None
    masks = np.asarray(valid_mask_paths, dtype=bool) if isinstance(valid_mask_paths, np.ndarray) else None
    if scenes is None or masks is None:
        # This path is intentionally limited to the small replay gate.  The
        # 13-scene runner uses its windowed provider instead of loading a
        # canonical stack into RAM.
        try:
            import rasterio
            scene_paths = [Path(path) for path in bagrn_scene_paths]  # type: ignore[arg-type]
            mask_paths = [Path(path) for path in valid_mask_paths]  # type: ignore[arg-type]
            if len(scene_paths) != len(mask_paths):
                raise ValueError("scene and mask path counts differ")
            arrays, mask_arrays = [], []
            for scene_path, mask_path in zip(scene_paths, mask_paths):
                with rasterio.open(scene_path) as src, rasterio.open(mask_path) as mask_src:
                    arrays.append(src.read(1).astype(np.float32))
                    if scene_path.resolve() == mask_path.resolve():
                        mask_arrays.append(mask_src.read_masks(1) > 0)
                    else:
                        mask_arrays.append(mask_src.read(1) > 0)
            scenes = np.asarray(arrays)
            masks = np.asarray(mask_arrays, dtype=bool)
        except ImportError as exc:
            raise TypeError("the adapter needs ndarray inputs or rasterio-readable paths") from exc
    if scenes.ndim != 3 or masks.shape != scenes.shape or scenes.shape[0] < 1:
        raise ValueError("scene arrays must have shape (N, rows, cols) and match masks")
    if not np.isfinite(scenes[masks]).all():
        raise ValueError("valid scene pixels must be finite")
    n_scenes = scenes.shape[0]
    pairs = normalize_overlap_pairs(overlap_pairs)
    if any(max(p.scene_i, p.scene_j) >= n_scenes for p in pairs):
        raise ValueError("overlap pair references a scene outside the input")
    if footprints is None:
        footprints = [None] * n_scenes
    if len(footprints) != n_scenes:
        raise ValueError("one footprint is required per scene")

    pair_results: dict[str, AdapterPairResult] = {}
    v1_fields: dict[tuple[int, int], Any] = {}
    v2_fields: dict[tuple[int, int], Any] = {}
    for pair in pairs:
        shared = masks[pair.scene_i] & masks[pair.scene_j]
        shared_count = int(np.count_nonzero(shared))
        if pair.shared_valid_pixel_count is not None:
            shared_count = int(pair.shared_valid_pixel_count)
        if not np.any(shared):
            continue
        processed = process_pair(scenes[pair.scene_i], scenes[pair.scene_j],
                                 masks[pair.scene_i], masks[pair.scene_j],
                                 runtime_config=runtime_config)
        initial_side = refined_side = None
        source_status = "REQUIRES_MULTISCENE_LABELING"
        if processed.initial_seam is not None and processed.crop_origin is not None:
            origin = processed.crop_origin
            h, w = processed.initial_seam.row_col_path.shape[0], processed.initial_seam.row_col_path.shape[1]
            # process_pair stores crop-local arrays; ownership uses the same crop.
            local_shape = processed.corrected_a.shape if processed.corrected_a is not None else processed.v1_mosaic.shape
            outer_h, outer_w = local_shape
            sl = (slice(origin[0], origin[0] + outer_h), slice(origin[1], origin[1] + outer_w))
            va, vb = masks[pair.scene_i][sl], masks[pair.scene_j][sl]
            initial_side = _source_side(processed.initial_seam, va, vb,
                                        footprints[pair.scene_i], footprints[pair.scene_j])
            if processed.refined_seam is not None:
                refined_side = _source_side(processed.refined_seam, va, vb,
                                            footprints[pair.scene_i], footprints[pair.scene_j])
            initial_resolved = initial_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
            refined_resolved = refined_side is not None and refined_side.status in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
            source_status = "PASS" if initial_resolved and refined_resolved else "REQUIRES_MULTISCENE_LABELING"
            if initial_resolved:
                full_seam = _expand_seam_to_full(processed, masks.shape[1:], processed.initial_seam)
                field = build_pairwise_preference_field(
                    masks[pair.scene_i], masks[pair.scene_j], full_seam,
                    initial_side.side_1_source, scene_a=pair.scene_i, scene_b=pair.scene_j,
                )
                # The extrapolated seam only supplies the required canonical
                # line domain; it is not evidence outside this pair's crop.
                pair_support = masks[pair.scene_i] & masks[pair.scene_j]
                initial_domain = _seam_domain(replace(processed, refined_seam=processed.initial_seam), masks.shape[1:])
                pair_support &= initial_domain
                v1_fields[(pair.scene_i, pair.scene_j)] = replace(
                    field,
                    available=field.available & pair_support,
                    vote=np.where(field.available & pair_support, field.vote, 0.0).astype(np.float32),
                    confidence=np.where(field.available & pair_support, field.confidence, 0.0).astype(np.float32),
                )
            if refined_resolved:
                full_seam = _expand_seam_to_full(processed, masks.shape[1:], processed.refined_seam)
                field = build_pairwise_preference_field(
                    masks[pair.scene_i], masks[pair.scene_j], full_seam,
                    refined_side.side_1_source, scene_a=pair.scene_i, scene_b=pair.scene_j,
                )
                pair_support = masks[pair.scene_i] & masks[pair.scene_j] & _seam_domain(processed, masks.shape[1:])
                v2_fields[(pair.scene_i, pair.scene_j)] = replace(
                    field,
                    available=field.available & pair_support,
                    vote=np.where(field.available & pair_support, field.vote, 0.0).astype(np.float32),
                    confidence=np.where(field.available & pair_support, field.confidence, 0.0).astype(np.float32),
                )
        status = "PASS" if processed.status == "PASS" else processed.status
        pair_results[pair.pair_id] = AdapterPairResult(
            pair.pair_id, pair.scene_i, pair.scene_j, processed, initial_side,
            refined_side, source_status, status, shared_count, pair.registration_edge_accepted,
        )

    raw_edt = np.asarray([distance_transform_edt(mask) for mask in masks], dtype=np.float64)
    v1_labels, v1_methods, v1_diag = aggregate_labels_with_ties(masks, v1_fields, raw_edt)
    v2_labels, v2_methods, v2_diag = aggregate_labels_with_ties(masks, v2_fields, raw_edt)
    corrected = _aggregate_corrections(scenes, masks, pair_results)
    v1_weights = _cosine_weights(v1_labels, masks)
    v2_weights = _cosine_weights(v2_labels, masks)
    v1 = blend_labeled_scenes(scenes, masks, v1_labels, weights=v1_weights)
    v2 = blend_labeled_scenes(corrected, masks, v2_labels, weights=v2_weights)
    diagnostics = dict(v2_diag)
    diagnostics.update({
        "scene_count": n_scenes,
        "union_valid_pixels": int(np.count_nonzero(masks.any(axis=0))),
        "sum_scene_valid_pixels": int(np.count_nonzero(masks)),
        "coverage_max": int(np.max(np.count_nonzero(masks, axis=0))),
        "mosaic_overlap_edges": len(pairs),
        "pairwise_seam_processed_edges": len(pair_results),
        "v1_label_status": "PASS" if not np.any((v1_labels == LABEL_UNRESOLVED) & masks.any(axis=0)) else "UNRESOLVED",
        "v2_label_status": "PASS" if not np.any((v2_labels == LABEL_UNRESOLVED) & masks.any(axis=0)) else "UNRESOLVED",
        "local_correction_status": "PASS" if all(row.process.status == "PASS" for row in pair_results.values()) else "PARTIAL",
        "shared_label_map": False,
        "v1_label_diagnostics": v1_diag,
        "v2_label_diagnostics": v2_diag,
        "source_side_status_counts": {
            status: sum(row.source_side_status == status for row in pair_results.values())
            for status in ("PASS", "REQUIRES_MULTISCENE_LABELING", "UNSUPPORTED_TOPOLOGY",
                           "LOCAL_MOMENT_UNAVAILABLE", "UNSTABLE_LOCAL_GAIN",
                           "AMBIGUOUS_SOURCE_SIDE", "NUMERICAL_INVALID")
        },
        "multiscene_pixels": int(np.count_nonzero(np.count_nonzero(masks, axis=0) >= 3)),
    })
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return AdapterResult(
        n_scenes, pair_results, v2_labels, v2_methods, v1, v2, corrected,
        v2_weights, diagnostics, v1_labels, v1_methods, v2_labels,
        v2_methods, v1_weights, v2_weights, {"v1": v1_diag, "v2": v2_diag},
    )


__all__ = [
    "AdapterPairResult", "AdapterResult", "LABEL_UNRESOLVED", "OverlapPair",
    "aggregate_labels_with_ties", "blend_labeled_scenes", "normalize_overlap_pairs",
    "run_multiscene_adapter",
]
