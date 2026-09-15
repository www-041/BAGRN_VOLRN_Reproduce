"""Geometry-only diagnostics for two-image registration evidence.

This module deliberately does not feed measurements back into the formal
Global + RBF registration pipeline.
"""

from collections import Counter

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from src.coregistration import phase_correlation, structural_image


def masked_ncc(ref, tgt, valid_mask):
    """Return absolute mean-centered NCC on valid pixels, or ``None``."""
    ref = np.asarray(ref, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(ref)
        & np.isfinite(tgt)
    )
    if int(valid.sum()) < 100:
        return None

    r = ref[valid].astype(np.float64, copy=True)
    t = tgt[valid].astype(np.float64, copy=True)
    r -= r.mean()
    t -= t.mean()
    denom = np.linalg.norm(r) * np.linalg.norm(t)
    if denom <= 1e-12:
        return None
    return float(abs(np.dot(r, t) / denom))


def compute_grid_relationship(ref_transform, tgt_transform):
    """Describe the target pixel-grid origin/phase in reference pixels."""
    target_center_world = tgt_transform * (0.5, 0.5)
    target_center_ref_edge = (~ref_transform) * target_center_world
    target_col = float(target_center_ref_edge[0] - 0.5)
    target_row = float(target_center_ref_edge[1] - 0.5)
    nearest_col = round(target_col)
    nearest_row = round(target_row)

    return {
        'ref_pixel_size': {
            'x': float(np.hypot(ref_transform.a, ref_transform.d)),
            'y': float(np.hypot(ref_transform.b, ref_transform.e)),
        },
        'tgt_pixel_size': {
            'x': float(np.hypot(tgt_transform.a, tgt_transform.d)),
            'y': float(np.hypot(tgt_transform.b, tgt_transform.e)),
        },
        'target_origin_in_ref_pixels': {
            'col': target_col,
            'row': target_row,
        },
        'nearest_integer_offset_pixels': {
            'col': int(nearest_col),
            'row': int(nearest_row),
        },
        'fractional_phase_pixels': {
            'col': float(target_col - nearest_col),
            'row': float(target_row - nearest_row),
        },
    }


def _summary_values(values, names):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    result = {name: None for name in names}
    if values.size == 0:
        return result
    percentiles = {
        'p10': 10,
        'median': 50,
        'p90': 90,
    }
    for name, percentile in percentiles.items():
        if name in result:
            result[name] = float(np.percentile(values, percentile))
    if 'mean' in result:
        result['mean'] = float(np.mean(values))
    if 'min' in result:
        result['min'] = float(np.min(values))
    if 'max' in result:
        result['max'] = float(np.max(values))
    return result


def summarize_candidate_rows(rows):
    """Summarize candidate distributions while retaining rejection counts."""
    rows = list(rows or [])
    measured_dx = []
    measured_dy = []
    measured_magnitude = []
    confidence = []
    reject_reasons = Counter()

    for row in rows:
        reason = row.get('reject_reason')
        if reason is not None:
            reject_reasons[str(reason)] += 1

        dx = row.get('shift_dx_pixels')
        dy = row.get('shift_dy_pixels')
        magnitude = row.get('shift_magnitude_pixels')
        try:
            dx = float(dx)
            dy = float(dy)
        except (TypeError, ValueError):
            dx = dy = None
        if dx is not None and dy is not None and np.isfinite(dx) and np.isfinite(dy):
            measured_dx.append(dx)
            measured_dy.append(dy)
            try:
                magnitude = float(magnitude)
            except (TypeError, ValueError):
                magnitude = float(np.hypot(dx, dy))
            if np.isfinite(magnitude):
                measured_magnitude.append(magnitude)

        try:
            confidence_value = float(row.get('confidence'))
        except (TypeError, ValueError):
            confidence_value = None
        if confidence_value is not None and np.isfinite(confidence_value):
            confidence.append(confidence_value)

    return {
        'total_rows': int(len(rows)),
        'measured_rows': int(len(measured_dx)),
        'reject_reason_counts': dict(reject_reasons),
        'confidence': _summary_values(
            confidence, ['min', 'median', 'mean', 'p90', 'max']),
        'shift_dx_pixels': _summary_values(
            measured_dx, ['median', 'p10', 'p90']),
        'shift_dy_pixels': _summary_values(
            measured_dy, ['median', 'p10', 'p90']),
        'shift_magnitude_pixels': _summary_values(
            measured_magnitude, ['median', 'p90', 'max']),
    }


def _raster_bounds(arr, transform):
    height, width = arr.shape
    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), max(xs), min(ys), max(ys)


def _window_for_bounds(transform, bounds, shape):
    left, right, bottom, top = bounds
    height, width = shape
    corners = [
        (~transform) * (left, top),
        (~transform) * (right, top),
        (~transform) * (left, bottom),
        (~transform) * (right, bottom),
    ]
    cols = [point[0] for point in corners]
    rows = [point[1] for point in corners]
    row_start = max(0, min(height, int(round(min(rows)))))
    row_end = max(0, min(height, int(round(max(rows)))))
    col_start = max(0, min(width, int(round(min(cols)))))
    col_end = max(0, min(width, int(round(max(cols)))))
    return row_start, row_end, col_start, col_end


def _overlap_windows(arr_ref, tr_ref, arr_tgt, tr_tgt):
    ref_left, ref_right, ref_bottom, ref_top = _raster_bounds(arr_ref, tr_ref)
    tgt_left, tgt_right, tgt_bottom, tgt_top = _raster_bounds(arr_tgt, tr_tgt)
    overlap = (
        max(ref_left, tgt_left),
        min(ref_right, tgt_right),
        max(ref_bottom, tgt_bottom),
        min(ref_top, tgt_top),
    )
    left, right, bottom, top = overlap
    if left >= right or bottom >= top:
        return None
    return (
        _window_for_bounds(tr_ref, overlap, arr_ref.shape),
        _window_for_bounds(tr_tgt, overlap, arr_tgt.shape),
    )


def _empty_raw_result(failure_reason=None):
    screening = {
        'total': 0,
        'low_valid': 0,
        'low_texture': 0,
        'low_conf': 0,
        'large_shift': 0,
        'accepted': 0,
    }
    return {
        'available': False,
        'failure_reason': failure_reason,
        'overlap_windows': {'ref': None, 'tgt': None},
        'screening': screening,
        'candidates': [],
        'summary': summarize_candidate_rows([]),
    }


def collect_raw_grid_candidate_diagnostics(
    arr_ref,
    tr_ref,
    arr_tgt,
    tr_tgt,
    nodata_ref=0,
    nodata_tgt=0,
    block_size=512,
    max_global_shift=40,
    confidence_threshold=0.5,
    min_valid_ratio=0.30,
):
    """Replicate raw-grid sampling while retaining every candidate row."""
    arr_ref = np.asarray(arr_ref)
    arr_tgt = np.asarray(arr_tgt)
    windows = _overlap_windows(arr_ref, tr_ref, arr_tgt, tr_tgt)
    if windows is None:
        return _empty_raw_result('no geographic overlap')

    ref_window, tgt_window = windows
    r1_start, r1_end, c1_start, c1_end = ref_window
    r2_start, r2_end, c2_start, c2_end = tgt_window
    patch_ref = arr_ref[r1_start:r1_end, c1_start:c1_end]
    patch_tgt = arr_tgt[r2_start:r2_end, c2_start:c2_end]
    if patch_ref.size < block_size * block_size:
        result = _empty_raw_result('overlap smaller than block size')
        result['overlap_windows'] = {
            'ref': list(ref_window), 'tgt': list(tgt_window),
        }
        return result

    if nodata_ref is None:
        valid_ref_patch = np.isfinite(patch_ref)
    else:
        valid_ref_patch = np.isfinite(patch_ref) & (patch_ref != nodata_ref)
    if nodata_tgt is None:
        valid_tgt_patch = np.isfinite(patch_tgt)
    else:
        valid_tgt_patch = np.isfinite(patch_tgt) & (patch_tgt != nodata_tgt)

    struct_ref = structural_image(patch_ref, valid_ref_patch)
    struct_tgt = structural_image(patch_tgt, valid_tgt_patch)
    global_texture_std = (
        float(np.std(struct_ref[valid_ref_patch]))
        if valid_ref_patch.sum() > 0 else 1.0
    )
    texture_threshold = max(global_texture_std * 0.10, 1e-4)
    ph, pw = patch_ref.shape
    stride = max(block_size // 2, 1)
    screening = {
        'total': 0,
        'low_valid': 0,
        'low_texture': 0,
        'low_conf': 0,
        'large_shift': 0,
        'accepted': 0,
    }
    candidates = []
    candidate_index = 0

    for br in range(0, ph - block_size + 1, stride):
        for bc in range(0, pw - block_size + 1, stride):
            br2 = min(br + block_size, ph)
            bc2 = min(bc + block_size, pw)
            screening['total'] += 1
            v_ref = valid_ref_patch[br:br2, bc:bc2]
            v_tgt = valid_tgt_patch[br:br2, bc:bc2]
            joint = v_ref & v_tgt
            valid_count = int(joint.sum())
            valid_ratio = float(valid_count / (block_size * block_size))
            ref_x = c1_start + bc + block_size // 2
            ref_y = r1_start + br + block_size // 2
            tgt_x = c2_start + bc + block_size // 2
            tgt_y = r2_start + br + block_size // 2
            row = {
                'candidate_index': int(candidate_index),
                'block_row_offset': int(br),
                'block_col_offset': int(bc),
                'ref_x': float(ref_x),
                'ref_y': float(ref_y),
                'tgt_x': float(tgt_x),
                'tgt_y': float(tgt_y),
                'valid_count': valid_count,
                'valid_ratio': valid_ratio,
                'texture_std': None,
                'texture_threshold': float(texture_threshold),
                'shift_dx_pixels': None,
                'shift_dy_pixels': None,
                'shift_magnitude_pixels': None,
                'confidence': None,
                'reject_reason': None,
            }
            candidate_index += 1

            if valid_ratio < min_valid_ratio:
                screening['low_valid'] += 1
                row['reject_reason'] = 'low_valid'
                candidates.append(row)
                continue

            blk_struct = struct_ref[br:br2, bc:bc2]
            texture_std = float(np.std(blk_struct[joint]))
            row['texture_std'] = texture_std
            if texture_std < texture_threshold:
                screening['low_texture'] += 1
                row['reject_reason'] = 'low_texture'
                candidates.append(row)
                continue

            sy, sx, conf = phase_correlation(
                struct_ref[br:br2, bc:bc2],
                struct_tgt[br:br2, bc:bc2],
                valid_ref=v_ref,
                valid_tgt=v_tgt,
            )
            row['shift_dx_pixels'] = float(sx)
            row['shift_dy_pixels'] = float(sy)
            row['shift_magnitude_pixels'] = float(np.hypot(sx, sy))
            row['confidence'] = float(conf)

            if conf <= confidence_threshold:
                screening['low_conf'] += 1
                row['reject_reason'] = 'low_conf'
            elif abs(sy) >= max_global_shift or abs(sx) >= max_global_shift:
                screening['large_shift'] += 1
                row['reject_reason'] = 'large_shift'
            else:
                screening['accepted'] += 1
                row['reject_reason'] = 'accepted'
            candidates.append(row)

    return {
        'available': True,
        'failure_reason': None,
        'overlap_windows': {
            'ref': list(ref_window), 'tgt': list(tgt_window),
        },
        'screening': screening,
        'candidates': candidates,
        'summary': summarize_candidate_rows(candidates),
    }


def _empty_common_grid_result(failure_reason):
    return {
        'available': False,
        'failure_reason': failure_reason,
        'ref_overlap': None,
        'tgt_on_ref_grid': None,
        'valid_ref': None,
        'valid_tgt': None,
        'common_valid': None,
        'ref_window': None,
        'tgt_window': None,
        'overlap_transform': None,
        'reprojected_target': False,
    }


def _valid_mask(array, nodata):
    valid = np.isfinite(array)
    if nodata is not None:
        valid &= array != nodata
    return valid


def build_reference_common_grid_overlap(
    arr_ref,
    tr_ref,
    arr_tgt,
    tr_tgt,
    crs_ref,
    crs_tgt,
    nodata_ref=0,
    nodata_tgt=0,
):
    """Map target to the reference overlap grid using metadata only."""
    if crs_ref != crs_tgt:
        return _empty_common_grid_result('CRS mismatch')

    arr_ref = np.asarray(arr_ref)
    arr_tgt = np.asarray(arr_tgt)
    windows = _overlap_windows(arr_ref, tr_ref, arr_tgt, tr_tgt)
    if windows is None:
        return _empty_common_grid_result('no geographic overlap')

    ref_window, tgt_window = windows
    r0, r1, c0, c1 = ref_window
    ref_overlap = arr_ref[r0:r1, c0:c1]
    overlap_transform = tr_ref * Affine.translation(c0, r0)
    valid_ref = _valid_mask(ref_overlap, nodata_ref)

    same_grid = tr_ref == tr_tgt
    if same_grid:
        tr0, tr1, tc0, tc1 = tgt_window
        tgt_on_ref_grid = arr_tgt[tr0:tr1, tc0:tc1].astype(
            np.float64, copy=True)
        valid_tgt = _valid_mask(tgt_on_ref_grid, nodata_tgt)
        common_valid = valid_ref & valid_tgt
        return {
            'available': True,
            'failure_reason': None,
            'ref_overlap': ref_overlap.astype(np.float64, copy=True),
            'tgt_on_ref_grid': tgt_on_ref_grid,
            'valid_ref': valid_ref,
            'valid_tgt': valid_tgt,
            'common_valid': common_valid,
            'ref_window': list(ref_window),
            'tgt_window': list(tgt_window),
            'overlap_transform': overlap_transform,
            'reprojected_target': False,
        }

    dst_shape = ref_overlap.shape
    tgt_valid_source = _valid_mask(arr_tgt, nodata_tgt)
    # Use a finite sentinel for invalid data so valid zero-valued pixels are
    # never mistaken for NoData during bilinear resampling.
    sentinel = np.finfo(np.float64).max
    source_data = arr_tgt.astype(np.float64, copy=True)
    source_data[~tgt_valid_source] = sentinel
    tgt_on_ref_grid = np.full(dst_shape, np.nan, dtype=np.float64)
    reproject(
        source=source_data,
        destination=tgt_on_ref_grid,
        src_transform=tr_tgt,
        src_crs=crs_tgt,
        src_nodata=sentinel,
        dst_transform=overlap_transform,
        dst_crs=crs_ref,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    valid_tgt_float = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=tgt_valid_source.astype(np.uint8),
        destination=valid_tgt_float,
        src_transform=tr_tgt,
        src_crs=crs_tgt,
        src_nodata=0,
        dst_transform=overlap_transform,
        dst_crs=crs_ref,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    valid_tgt = (valid_tgt_float > 0.5) & np.isfinite(tgt_on_ref_grid)
    common_valid = valid_ref & valid_tgt
    return {
        'available': True,
        'failure_reason': None,
        'ref_overlap': ref_overlap.astype(np.float64, copy=True),
        'tgt_on_ref_grid': tgt_on_ref_grid,
        'valid_ref': valid_ref,
        'valid_tgt': valid_tgt,
        'common_valid': common_valid,
        'ref_window': list(ref_window),
        'tgt_window': list(tgt_window),
        'overlap_transform': overlap_transform,
        'reprojected_target': True,
    }


def _add_common_grid_summary_metrics(summary, candidates):
    for field, output_key in [
        ('zero_shift_ncc', 'zero_shift_ncc'),
        ('best_shift_ncc', 'best_shift_ncc'),
        ('ncc_gain', 'ncc_gain'),
    ]:
        values = []
        for row in candidates:
            value = row.get(field)
            if value is not None and np.isfinite(value):
                values.append(float(value))
        summary[output_key] = {
            'median': float(np.median(values)) if values else None,
            'p90': float(np.percentile(values, 90)) if values else None,
            'max': float(np.max(values)) if values else None,
        }
    return summary


def collect_common_grid_candidate_diagnostics(
    ref_overlap,
    tgt_on_ref_grid,
    common_valid,
    block_size=512,
    confidence_threshold=0.5,
    max_residual_shift=40,
    min_valid_ratio=0.30,
):
    """Run the raw block evidence logic after both arrays share one grid."""
    ref_overlap = np.asarray(ref_overlap)
    tgt_on_ref_grid = np.asarray(tgt_on_ref_grid)
    common_valid = np.asarray(common_valid, dtype=bool)
    if (
        ref_overlap.ndim != 2
        or tgt_on_ref_grid.shape != ref_overlap.shape
        or common_valid.shape != ref_overlap.shape
    ):
        return {
            'available': False,
            'failure_reason': 'common-grid arrays have incompatible shapes',
            'screening': {
                'total': 0, 'low_valid': 0, 'low_texture': 0,
                'low_conf': 0, 'large_shift': 0, 'accepted': 0,
            },
            'candidates': [],
                'summary': _add_common_grid_summary_metrics(
                summarize_candidate_rows([]), []),
        }

    valid_ref = common_valid & np.isfinite(ref_overlap)
    valid_tgt = common_valid & np.isfinite(tgt_on_ref_grid)
    struct_ref = structural_image(ref_overlap, valid_ref)
    struct_tgt = structural_image(tgt_on_ref_grid, valid_tgt)
    global_texture_std = (
        float(np.std(struct_ref[valid_ref]))
        if valid_ref.sum() > 0 else 1.0
    )
    texture_threshold = max(global_texture_std * 0.10, 1e-4)
    ph, pw = ref_overlap.shape
    stride = max(block_size // 2, 1)
    screening = {
        'total': 0,
        'low_valid': 0,
        'low_texture': 0,
        'low_conf': 0,
        'large_shift': 0,
        'accepted': 0,
    }
    candidates = []
    candidate_index = 0

    for br in range(0, ph - block_size + 1, stride):
        for bc in range(0, pw - block_size + 1, stride):
            br2 = min(br + block_size, ph)
            bc2 = min(bc + block_size, pw)
            screening['total'] += 1
            joint = valid_ref[br:br2, bc:bc2] & valid_tgt[br:br2, bc:bc2]
            valid_count = int(joint.sum())
            valid_ratio = float(valid_count / (block_size * block_size))
            row = {
                'candidate_index': int(candidate_index),
                'block_row_offset': int(br),
                'block_col_offset': int(bc),
                'ref_x': float(bc + block_size // 2),
                'ref_y': float(br + block_size // 2),
                'tgt_x': float(bc + block_size // 2),
                'tgt_y': float(br + block_size // 2),
                'valid_count': valid_count,
                'valid_ratio': valid_ratio,
                'texture_std': None,
                'texture_threshold': float(texture_threshold),
                'shift_dx_pixels': None,
                'shift_dy_pixels': None,
                'shift_magnitude_pixels': None,
                'confidence': None,
                'zero_shift_ncc': None,
                'best_shift_ncc': None,
                'ncc_gain': None,
                'reject_reason': None,
            }
            candidate_index += 1

            if valid_ratio < min_valid_ratio:
                screening['low_valid'] += 1
                row['reject_reason'] = 'low_valid'
                candidates.append(row)
                continue

            blk_ref = struct_ref[br:br2, bc:bc2]
            blk_tgt = struct_tgt[br:br2, bc:bc2]
            texture_std = float(np.std(blk_ref[joint]))
            row['texture_std'] = texture_std
            if texture_std < texture_threshold:
                screening['low_texture'] += 1
                row['reject_reason'] = 'low_texture'
                candidates.append(row)
                continue

            row['zero_shift_ncc'] = masked_ncc(blk_ref, blk_tgt, joint)
            sy, sx, conf = phase_correlation(
                blk_ref, blk_tgt,
                valid_ref=joint,
                valid_tgt=joint,
            )
            row['shift_dx_pixels'] = float(sx)
            row['shift_dy_pixels'] = float(sy)
            row['shift_magnitude_pixels'] = float(np.hypot(sx, sy))
            row['confidence'] = float(conf)
            row['best_shift_ncc'] = float(conf)
            if row['zero_shift_ncc'] is not None:
                row['ncc_gain'] = float(conf - row['zero_shift_ncc'])

            if conf <= confidence_threshold:
                screening['low_conf'] += 1
                row['reject_reason'] = 'low_conf'
            elif abs(sy) >= max_residual_shift or abs(sx) >= max_residual_shift:
                screening['large_shift'] += 1
                row['reject_reason'] = 'large_shift'
            else:
                screening['accepted'] += 1
                row['reject_reason'] = 'accepted'
            candidates.append(row)

    summary = summarize_candidate_rows(candidates)
    return {
        'available': True,
        'failure_reason': None,
        'screening': screening,
        'candidates': candidates,
        'summary': _add_common_grid_summary_metrics(summary, candidates),
    }
