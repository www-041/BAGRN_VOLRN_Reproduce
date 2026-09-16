"""
Two-image B14 pipeline: co-register → BAGRN → VOLRN → mosaic
"""
import os, sys, time, json, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

OUTPUT = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\two_image_float'
os.makedirs(OUTPUT, exist_ok=True)

IMG2 = r'data/input/flat/DZ01V_L2_E113.6_N36.3_20260616031133_01_T1/DZ01V_L2_E113.6_N36.3_20260616031133_01_T1_{}.TIF'
IMG1 = r'data/input/flat/DZ01V_L2_E113.4_N36.6_20260810030932_01_T1/DZ01V_L2_E113.4_N36.6_20260810030932_01_T1_{}.TIF'
BANDS = ['B14']


def compute_overlap(arr1, tr1, arr2, tr2):
    rows1, cols1 = arr1.shape
    rows2, cols2 = arr2.shape
    b1 = (tr1.c, tr1.f + tr1.e * rows1, tr1.c + tr1.a * cols1, tr1.f)
    b2 = (tr2.c, tr2.f + tr2.e * rows2, tr2.c + tr2.a * cols2, tr2.f)
    return get_overlap_window(b1, tr1, b2, tr2)


def compute_bounds(tr, shape):
    h, w = shape
    return (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)


def save_csv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def get_band_output_dir(band):
    """Create and return the output directory used by one band."""
    out_dir = os.path.join(OUTPUT, band)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _json_safe(value):
    """Convert NumPy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _residual_arrays(matches, global_dx, global_dy):
    """Return block residual components relative to the global shift."""
    if not matches:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    dx = np.asarray([m['shift_dx'] for m in matches], dtype=np.float64) - float(global_dx)
    dy = np.asarray([m['shift_dy'] for m in matches], dtype=np.float64) - float(global_dy)
    finite = np.isfinite(dx) & np.isfinite(dy)
    return dx[finite], dy[finite]


def _residual_inlier_mask(residual_dx, residual_dy):
    """Apply the same component-wise MAD rule used by global block screening."""
    if len(residual_dx) == 0:
        return np.zeros(0, dtype=bool)
    med_dx = np.median(residual_dx)
    med_dy = np.median(residual_dy)
    mad_dx = np.median(np.abs(residual_dx - med_dx))
    mad_dy = np.median(np.abs(residual_dy - med_dy))
    return (
        (np.abs(residual_dx - med_dx) < max(3 * mad_dx, 0.3))
        & (np.abs(residual_dy - med_dy) < max(3 * mad_dy, 0.3))
    )


def _summarize_residuals(residual_dx, residual_dy):
    """Summarize Euclidean and signed component registration residuals."""
    if len(residual_dx) == 0:
        return {
            'count': 0,
            'rmse_pixels': None,
            'mean_magnitude_pixels': None,
            'median_magnitude_pixels': None,
            'max_magnitude_pixels': None,
            'p95_magnitude_pixels': None,
            'mean_residual_dx_pixels': None,
            'mean_residual_dy_pixels': None,
            'median_residual_dx_pixels': None,
            'median_residual_dy_pixels': None,
        }

    errors = np.hypot(residual_dx, residual_dy)
    return {
        'count': int(len(errors)),
        'rmse_pixels': float(np.sqrt(np.mean(errors ** 2))),
        'mean_magnitude_pixels': float(np.mean(errors)),
        'median_magnitude_pixels': float(np.median(errors)),
        'max_magnitude_pixels': float(np.max(errors)),
        'p95_magnitude_pixels': float(np.percentile(errors, 95)),
        'mean_residual_dx_pixels': float(np.mean(residual_dx)),
        'mean_residual_dy_pixels': float(np.mean(residual_dy)),
        'median_residual_dx_pixels': float(np.median(residual_dx)),
        'median_residual_dy_pixels': float(np.median(residual_dy)),
    }


def _spatial_coverage(
    matches, reference_shape, overlap_window=None, grid_rows=4, grid_cols=4
):
    """Describe match-point spread in the reference image coordinate system."""
    height, width = reference_shape
    if overlap_window is None:
        r0, r1, c0, c1 = 0, height, 0, width
        coverage_domain = 'full_reference'
        domain_window = None
    else:
        r0, r1, c0, c1 = [int(value) for value in overlap_window]
        coverage_domain = 'overlap'
        domain_window = [r0, r1, c0, c1]
    domain_height = max(r1 - r0, 1)
    domain_width = max(c1 - c0, 1)
    empty = {
        'reference_shape': [int(height), int(width)],
        'coverage_domain': coverage_domain,
        'domain_window': domain_window,
        'bbox_pixels': None,
        'row_span_pixels': 0.0,
        'col_span_pixels': 0.0,
        'grid_rows': int(grid_rows),
        'grid_cols': int(grid_cols),
        'grid_cells_covered': 0,
        'grid_coverage_ratio': 0.0,
        'row_bands_covered': 0,
        'col_bands_covered': 0,
    }
    if not matches:
        return empty

    points = np.asarray([[m['ref_x'], m['ref_y']] for m in matches], dtype=np.float64)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    x_rel = (points[:, 0] - c0) / domain_width
    y_rel = (points[:, 1] - r0) / domain_height
    x_bins = np.clip((x_rel * grid_cols).astype(int), 0, grid_cols - 1)
    y_bins = np.clip((y_rel * grid_rows).astype(int), 0, grid_rows - 1)
    cells = {(int(row), int(col)) for row, col in zip(y_bins, x_bins)}

    return {
        'reference_shape': [int(height), int(width)],
        'coverage_domain': coverage_domain,
        'domain_window': domain_window,
        'bbox_pixels': {
            'min_x': float(min_x), 'min_y': float(min_y),
            'max_x': float(max_x), 'max_y': float(max_y),
        },
        'row_span_pixels': float(max_y - min_y),
        'col_span_pixels': float(max_x - min_x),
        'grid_rows': int(grid_rows),
        'grid_cols': int(grid_cols),
        'grid_cells_covered': int(len(cells)),
        'grid_coverage_ratio': float(len(cells) / (grid_rows * grid_cols)),
        'row_bands_covered': int(len(set(y_bins.tolist()))),
        'col_bands_covered': int(len(set(x_bins.tolist()))),
    }


def build_registration_match_rows(matches, global_dx, global_dy):
    """Add residual and inlier fields to block matches for CSV export."""
    residual_dx, residual_dy = _residual_arrays(matches, global_dx, global_dy)
    inliers = _residual_inlier_mask(residual_dx, residual_dy)
    rows = []
    for index, (match, dx, dy, inlier) in enumerate(
        zip(matches, residual_dx, residual_dy, inliers)
    ):
        rows.append({
            'match_index': int(index),
            'ref_x': float(match['ref_x']),
            'ref_y': float(match['ref_y']),
            'tgt_x': float(match['tgt_x']),
            'tgt_y': float(match['tgt_y']),
            'shift_dx_pixels': float(match['shift_dx']),
            'shift_dy_pixels': float(match['shift_dy']),
            'confidence': float(match['confidence']),
            'residual_dx_pixels': float(dx),
            'residual_dy_pixels': float(dy),
            'residual_magnitude_pixels': float(np.hypot(dx, dy)),
            'inlier': int(bool(inlier)),
        })
    return rows


def build_post_warp_match_rows(matches):
    """Build CSV rows whose displacement fields are residuals to zero."""
    rows = []
    for index, match in enumerate(matches):
        dx = float(match['shift_dx'])
        dy = float(match['shift_dy'])
        rows.append({
            'match_index': int(index),
            'ref_x': float(match['ref_x']),
            'ref_y': float(match['ref_y']),
            'tgt_x': float(match['tgt_x']),
            'tgt_y': float(match['tgt_y']),
            'residual_dx_pixels': dx,
            'residual_dy_pixels': dy,
            'residual_magnitude_pixels': float(np.hypot(dx, dy)),
            'confidence': float(match['confidence']),
        })
    return rows


def select_translation_or_rbf(local_cv_summary, min_p95_improvement=0.10):
    """Select only a model that the two-image baseline can actually apply."""
    translation_p95 = local_cv_summary.get('translation', {}).get('p95')
    rbf_p95 = local_cv_summary.get('rbf', {}).get('p95')
    if translation_p95 is not None and not np.isfinite(translation_p95):
        translation_p95 = None
    if rbf_p95 is not None and not np.isfinite(rbf_p95):
        rbf_p95 = None

    improvement_pixels = None
    improvement_ratio = None
    if translation_p95 is not None and rbf_p95 is not None:
        improvement_pixels = float(translation_p95 - rbf_p95)
        if translation_p95 > 0:
            improvement_ratio = float(improvement_pixels / translation_p95)

    if improvement_ratio is not None and improvement_ratio >= min_p95_improvement:
        selected_model = 'rbf'
        reason = 'RBF P95 improvement meets the minimum requirement.'
    else:
        selected_model = 'translation'
        reason = 'RBF P95 improvement does not meet the minimum requirement.'

    return {
        'selected_model': selected_model,
        'translation_p95': float(translation_p95) if translation_p95 is not None else None,
        'rbf_p95': float(rbf_p95) if rbf_p95 is not None else None,
        'p95_improvement_pixels': improvement_pixels,
        'p95_improvement_ratio': improvement_ratio,
        'min_required_improvement_ratio': float(min_p95_improvement),
        'reason': reason,
    }


def format_translation_estimate(lag_y, lag_x, conf, translation_stats):
    """Format a measured translation or an explicitly unavailable fallback."""
    available = bool((translation_stats or {}).get('available', True))
    if not available:
        reason = (translation_stats or {}).get(
            'failure_reason', 'translation estimate unavailable')
        return [
            'Translation estimate: UNAVAILABLE',
            f'Applied fallback shift: dy={lag_y:.4f}, dx={lag_x:.4f}',
            f'Reason: {reason}',
        ]
    return [
        f'Translation: dy={lag_y:.4f}, dx={lag_x:.4f}, confidence={conf:.4f}'
    ]


def apply_registration_warps(
    arr2_orig, global_dx, global_dy, local_dx_field, local_dy_field,
    nodata, use_local, warp_fn=None,
):
    """Apply comparable global-only and final warps from the original target.

    The global-only result is the baseline for post-warp diagnostics.  When
    local registration is disabled, the final result is that same baseline;
    otherwise both results are independently sampled from ``arr2_orig``.
    """
    if warp_fn is None:
        from src.coregistration import warp_with_displacement_field
        warp_fn = warp_with_displacement_field

    zero_dx = np.zeros_like(arr2_orig, dtype=np.float64)
    zero_dy = np.zeros_like(arr2_orig, dtype=np.float64)
    global_warp = warp_fn(
        arr2_orig, global_dx, global_dy, zero_dx, zero_dy, nodata)
    if use_local:
        final_warp = warp_fn(
            arr2_orig, global_dx, global_dy,
            local_dx_field, local_dy_field, nodata)
    else:
        final_warp = global_warp
    return global_warp, final_warp


def evaluate_post_warp_registration(
    arr_ref,
    tr_ref,
    arr_tgt_warped,
    tr_tgt,
    nodata_ref,
    nodata_tgt,
    reference_overlap_window=None,
    block_size=512,
    max_residual_shift=5.0,
    confidence_threshold=0.5,
):
    """Measure remaining displacement after a warp by rematching blocks.

    The rematched shifts are residual-to-zero measurements.  They are not
    refit against a second global model and are explicitly same-data
    diagnostics rather than independent holdout validation.
    """
    from src.coregistration import collect_block_matches

    matches, screening = collect_block_matches(
        arr_ref,
        tr_ref,
        arr_tgt_warped,
        tr_tgt,
        nodata_ref,
        nodata_tgt,
        block_size=block_size,
        max_global_shift=max_residual_shift,
        confidence_threshold=confidence_threshold,
    )
    matches = list(matches or [])
    screening = dict(screening or {})
    candidate_blocks = int(screening.get('total', len(matches)))
    accepted_matches = len(matches)
    confidence_values = np.asarray(
        [m['confidence'] for m in matches], dtype=np.float64)
    finite_confidence = confidence_values[np.isfinite(confidence_values)]
    mean_confidence = (
        float(np.mean(finite_confidence))
        if finite_confidence.size else None
    )

    if matches:
        residual_dx = np.asarray(
            [m['shift_dx'] for m in matches], dtype=np.float64)
        residual_dy = np.asarray(
            [m['shift_dy'] for m in matches], dtype=np.float64)
    else:
        residual_dx = np.empty(0, dtype=np.float64)
        residual_dy = np.empty(0, dtype=np.float64)

    result = {
        'available': bool(matches),
        'metric_scope': 'same-data post-warp diagnostic; not independent holdout',
        'candidate_blocks': candidate_blocks,
        'accepted_matches': int(accepted_matches),
        'mean_confidence': mean_confidence,
        'residual_to_zero': _summarize_residuals(residual_dx, residual_dy),
        'spatial_coverage': _spatial_coverage(
            matches, arr_ref.shape, overlap_window=reference_overlap_window),
        'screening': _json_safe(screening),
        'matches': _json_safe(matches),
    }
    if not matches:
        result['failure_reason'] = 'no accepted post-warp block matches'
    return result


def compare_post_warp_stages(global_post, final_post):
    """Compare post-warp residual-to-zero diagnostics.

    Improvement is defined as ``global_only - final``.  Missing stage
    metrics remain unavailable instead of being treated as zero error.
    """
    metric_names = {
        'median': 'median_magnitude_pixels',
        'rmse': 'rmse_pixels',
        'p95': 'p95_magnitude_pixels',
    }

    def metric_value(stage, key):
        if not stage or not stage.get('available'):
            return None
        value = stage.get('residual_to_zero', {}).get(key)
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    global_values = {
        name: metric_value(global_post, key)
        for name, key in metric_names.items()
    }
    final_values = {
        name: metric_value(final_post, key)
        for name, key in metric_names.items()
    }
    improvements = {
        name: (
            global_values[name] - final_values[name]
            if global_values[name] is not None and final_values[name] is not None
            else None
        )
        for name in metric_names
    }

    def ratio(name):
        improvement = improvements[name]
        baseline = global_values[name]
        if improvement is None or baseline in (None, 0.0):
            return None
        return float(improvement / baseline)

    return {
        'median_improvement_pixels': improvements['median'],
        'rmse_improvement_pixels': improvements['rmse'],
        'p95_improvement_pixels': improvements['p95'],
        'rmse_improvement_ratio': ratio('rmse'),
        'p95_improvement_ratio': ratio('p95'),
        'interpretation': 'positive means final is better',
    }


def build_registration_metrics(
    matches, screening, global_dx, global_dy, phase_confidence,
    reference_shape, reference_transform, block_stats=None,
    reference_overlap_window=None,
):
    """Build only initial model-fit diagnostics from block-match results."""
    residual_dx, residual_dy = _residual_arrays(matches, global_dx, global_dy)
    inliers = _residual_inlier_mask(residual_dx, residual_dy)
    inlier_matches = [m for m, keep in zip(matches, inliers) if keep]

    dx_map = reference_transform.a * global_dx + reference_transform.b * global_dy
    dy_map = reference_transform.d * global_dx + reference_transform.e * global_dy
    magnitude = float(np.hypot(global_dx, global_dy))
    direction = float(np.degrees(np.arctan2(global_dy, global_dx))) if magnitude else 0.0

    accepted = len(matches)
    candidate_blocks = int(screening.get('total', accepted))
    return {
        'offset': {
            'dx_pixels': float(global_dx),
            'dy_pixels': float(global_dy),
            'magnitude_pixels': magnitude,
            'direction_image_degrees': direction,
            'dx_map_units': float(dx_map),
            'dy_map_units': float(dy_map),
            'note': 'Offset is the correction applied to the target image; image +y points downward.',
        },
        'phase_confidence': float(phase_confidence),
        'matching': {
            'candidate_blocks': candidate_blocks,
            'accepted_matches': int(accepted),
            'inlier_matches': int(inliers.sum()),
            'accepted_match_ratio': float(accepted / candidate_blocks) if candidate_blocks else 0.0,
            'inlier_ratio': float(inliers.sum() / accepted) if accepted else 0.0,
            'screening': _json_safe(screening),
        },
        'residual_relative_to_global_model': _summarize_residuals(
            residual_dx, residual_dy),
        'inlier_residual_relative_to_global_model': _summarize_residuals(
            residual_dx[inliers], residual_dy[inliers]),
        'spatial_coverage': _spatial_coverage(
            matches, reference_shape, overlap_window=reference_overlap_window),
        'inlier_spatial_coverage': _spatial_coverage(
            inlier_matches, reference_shape, overlap_window=reference_overlap_window),
        'phase_block_stats': _json_safe(block_stats or {}),
    }


def build_registration_schema(
    initial_model_fit, local, global_post, final_post, comparison,
):
    """Assemble the versioned schema separating fit and post-warp metrics."""
    return {
        'schema_version': 2,
        'metric_scope_note': (
            'post_warp metrics are same-data diagnostics, not independent HOLDOUT'
        ),
        'initial_model_fit': initial_model_fit,
        'local': local,
        'post_warp': {
            'global_only': global_post,
            'final': final_post,
            'comparison': comparison,
        },
    }


def process_band(band):
    print(f"\n{'='*70}")
    print(f"  Processing {band}")
    print(f"{'='*70}")

    # ---- Load ----
    arr1, tr1, crs1, nd1 = read_geotiff(IMG1.format(band))
    arr2_orig, tr2_orig, crs2, nd2 = read_geotiff(IMG2.format(band))
    while arr1.ndim > 2: arr1 = arr1[0]
    while arr2_orig.ndim > 2: arr2_orig = arr2_orig[0]
    out_dir = get_band_output_dir(band)
    initial_overlap = compute_overlap(arr1, tr1, arr2_orig, tr2_orig)
    ref_overlap_window = initial_overlap[0] if initial_overlap is not None else None

    print(f"  Image 1 (ref): {arr1.shape}, nodata={nd1}")
    print(f"  Image 2 (tgt): {arr2_orig.shape}, nodata={nd2}")

    if crs1 != crs2:
        raise ValueError(f"CRS不一致: {crs1} vs {crs2}")
    crs = crs1

    # ================================================================
    # Step 1: Co-registration (existing logic, abbreviated)
    # ================================================================
    print("\n  --- Co-registration ---")
    from src.coregistration import (
        collect_block_matches, compute_model_residual_stats,
        fit_affine_ransac, compute_shifts_from_overlap,
        build_local_residual_controls, fit_local_rbf,
        spatial_cross_validate, warp_with_displacement_field,
    )

    matches, screening = collect_block_matches(arr1, tr1, arr2_orig, tr2_orig, nd1, nd2)
    lag_y, lag_x, conf, translation_stats = compute_shifts_from_overlap(
        arr1, tr1, arr2_orig, tr2_orig, nd1, nd2)
    for translation_line in format_translation_estimate(
        lag_y, lag_x, conf, translation_stats):
        print(f"  {translation_line}")

    ctrl = build_local_residual_controls(matches, lag_x, lag_y, confidence_threshold=0.75)
    print(f"  Local control points: {ctrl['n_valid']}")

    use_local = False
    local_rbf_dx = local_rbf_dy = local_coord_range = None
    best_smoothing = None
    local_cv_summary = {}
    smoothing_candidates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    smoothing_cv = []
    selection_diagnostics = select_translation_or_rbf({})

    if ctrl['n_valid'] >= 30:
        local_cv_summary, _ = spatial_cross_validate(
            ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
            lag_x, lag_y, matches, rbf_smoothing=0.1)
        if local_cv_summary:
            selection_diagnostics = select_translation_or_rbf(local_cv_summary)
            if selection_diagnostics['selected_model'] == 'rbf':
                for sm in smoothing_candidates:
                    try:
                        summary_sm, _ = spatial_cross_validate(
                            ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
                            lag_x, lag_y, matches, rbf_smoothing=sm)
                        rbf_stats = summary_sm.get('rbf', {})
                        smoothing_cv.append({
                            'smoothing': float(sm),
                            'p95': rbf_stats.get('p95'),
                            'rmse': rbf_stats.get('rmse'),
                        })
                    except Exception:
                        smoothing_cv.append({
                            'smoothing': float(sm), 'p95': None, 'rmse': None,
                        })

                valid_smoothing = [
                    item for item in smoothing_cv
                    if item['p95'] is not None and np.isfinite(item['p95'])
                ]
                if valid_smoothing:
                    best_entry = min(valid_smoothing, key=lambda item: item['p95'])
                    best_smoothing = best_entry['smoothing']
                    try:
                        local_rbf_dx, local_rbf_dy, cmin, cmax = fit_local_rbf(
                            ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
                            smoothing=best_smoothing,
                            neighbors=min(20, ctrl['n_valid']))
                        local_coord_range = (cmin[0], cmin[1], cmax[0], cmax[1])
                    except Exception:
                        best_smoothing = None

                if local_rbf_dx is not None:
                    use_local = True
                    print(f"  => LOCAL RBF (smoothing={best_smoothing})")

    local_model = 'rbf' if use_local else 'translation'
    initial_model_fit = build_registration_metrics(
        matches,
        screening,
        global_dx=lag_x,
        global_dy=lag_y,
        phase_confidence=conf,
        reference_shape=arr1.shape,
        reference_transform=tr1,
        block_stats=translation_stats,
        reference_overlap_window=ref_overlap_window,
    )
    registration_rows = build_registration_match_rows(matches, lag_x, lag_y)
    save_csv(
        os.path.join(out_dir, f'registration_{band}_matches.csv'),
        registration_rows,
        [
            'match_index', 'ref_x', 'ref_y', 'tgt_x', 'tgt_y',
            'shift_dx_pixels', 'shift_dy_pixels', 'confidence',
            'residual_dx_pixels', 'residual_dy_pixels',
            'residual_magnitude_pixels', 'inlier',
        ],
    )
    residual = initial_model_fit['residual_relative_to_global_model']
    matching = initial_model_fit['matching']
    coverage = initial_model_fit['spatial_coverage']
    offset = initial_model_fit['offset']
    print(
        f"  Registration metrics: magnitude={offset['magnitude_pixels']:.4f}px, "
        f"direction={offset['direction_image_degrees']:.2f}deg"
    )
    print(
        f"  Matches: candidates={matching['candidate_blocks']}, "
        f"accepted={matching['accepted_matches']}, "
        f"inliers={matching['inlier_matches']}, "
        f"inlier_ratio={matching['inlier_ratio']:.3f}"
    )
    print(
        f"  Residual (accepted blocks): RMSE={residual['rmse_pixels']}, "
        f"mean={residual['mean_magnitude_pixels']}, "
        f"median={residual['median_magnitude_pixels']}, "
        f"P95={residual['p95_magnitude_pixels']}, "
        f"max={residual['max_magnitude_pixels']} px"
    )
    print(
        f"  Match coverage: {coverage['grid_cells_covered']}/"
        f"{coverage['grid_rows'] * coverage['grid_cols']} grid cells, "
        f"row_span={coverage['row_span_pixels']:.1f}px, "
        f"col_span={coverage['col_span_pixels']:.1f}px"
    )

    # Apply transform
    h2, w2 = arr2_orig.shape
    local_dx_field = np.zeros((h2, w2), dtype=np.float64)
    local_dy_field = np.zeros((h2, w2), dtype=np.float64)

    if use_local and ctrl['n_valid'] >= 3:
        from scipy.spatial import Delaunay
        hull = Delaunay(ctrl['points_xy'])
        yy, xx = np.mgrid[0:h2, 0:w2]
        test_points = np.column_stack([xx.ravel(), yy.ravel()])
        chunk = 50000
        for i in range(0, len(test_points), chunk):
            cp = test_points[i:i+chunk]
            in_hull = hull.find_simplex(cp) >= 0
            tx = (cp[:, 0] - local_coord_range[0]) / max(local_coord_range[2] - local_coord_range[0], 1e-10)
            ty = (cp[:, 1] - local_coord_range[1]) / max(local_coord_range[3] - local_coord_range[1], 1e-10)
            pts_norm = np.column_stack([tx, ty])
            cdx = np.clip(local_rbf_dx(pts_norm), -2.5, 2.5)
            cdy = np.clip(local_rbf_dy(pts_norm), -2.5, 2.5)
            cdx[~in_hull] = 0
            cdy[~in_hull] = 0
            local_dx_field.ravel()[i:i+chunk] = cdx
            local_dy_field.ravel()[i:i+chunk] = cdy
    arr2_global, arr2_coreg = apply_registration_warps(
        arr2_orig,
        global_dx=lag_x,
        global_dy=lag_y,
        local_dx_field=local_dx_field,
        local_dy_field=local_dy_field,
        nodata=nd2,
        use_local=use_local and ctrl['n_valid'] >= 3,
        warp_fn=warp_with_displacement_field,
    )

    tr2_coreg = tr2_orig

    post_warp_kwargs = {
        'reference_overlap_window': ref_overlap_window,
        'block_size': 512,
        'max_residual_shift': 5.0,
        'confidence_threshold': 0.5,
    }
    global_post = evaluate_post_warp_registration(
        arr1, tr1, arr2_global, tr2_orig, nd1, nd2, **post_warp_kwargs)
    final_post = evaluate_post_warp_registration(
        arr1, tr1, arr2_coreg, tr2_coreg, nd1, nd2, **post_warp_kwargs)
    post_comparison = compare_post_warp_stages(global_post, final_post)
    local_diagnostics = {
        'model_used': str(local_model),
        'control_points': int(ctrl['n_valid']),
        'selection': _json_safe(selection_diagnostics or {}),
        'cross_validation': _json_safe(local_cv_summary or {}),
        'smoothing_candidates': _json_safe(smoothing_cv or []),
        'rbf_smoothing': float(best_smoothing) if best_smoothing is not None else None,
    }
    registration_metrics = build_registration_schema(
        initial_model_fit,
        local_diagnostics,
        global_post,
        final_post,
        post_comparison,
    )
    post_headers = [
        'match_index', 'ref_x', 'ref_y', 'tgt_x', 'tgt_y',
        'residual_dx_pixels', 'residual_dy_pixels',
        'residual_magnitude_pixels', 'confidence',
    ]
    save_csv(
        os.path.join(out_dir, f'registration_{band}_global_postwarp_matches.csv'),
        build_post_warp_match_rows(global_post['matches']),
        post_headers,
    )
    save_csv(
        os.path.join(out_dir, f'registration_{band}_final_postwarp_matches.csv'),
        build_post_warp_match_rows(final_post['matches']),
        post_headers,
    )
    with open(os.path.join(out_dir, f'registration_{band}_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(_json_safe(registration_metrics), f, indent=2, ensure_ascii=False)

    def _post_summary(stage):
        residual_zero = stage['residual_to_zero']
        return (
            f"n={stage['accepted_matches']}, "
            f"conf={stage['mean_confidence']}, "
            f"median={residual_zero['median_magnitude_pixels']}, "
            f"RMSE={residual_zero['rmse_pixels']}, "
            f"P95={residual_zero['p95_magnitude_pixels']}"
        )

    print("  Post-warp registration (same-data diagnostic, not HOLDOUT)")
    print(f"    Global-only : {_post_summary(global_post)}")
    print(f"    Final       : model={local_model}, {_post_summary(final_post)}")
    print(
        "    Improvement : "
        f"median={post_comparison['median_improvement_pixels']}, "
        f"RMSE={post_comparison['rmse_improvement_pixels']}, "
        f"P95={post_comparison['p95_improvement_pixels']} "
        "(positive = better)"
    )

    # ================================================================
    # Step 2: BAGRN + VOLRN + Mosaic
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  B14 radiometric normalization and mosaic")
    print(f"{'='*70}")

    arrs_registered = [arr1[np.newaxis, :, :], arr2_coreg[np.newaxis, :, :]]
    nodatas = [nd1, nd2]
    transforms_all = [tr1, tr2_coreg]
    bounds_all = [compute_bounds(tr1, arr1.shape), compute_bounds(tr2_coreg, arr2_coreg.shape)]

    win2 = compute_overlap(arr1, tr1, arr2_coreg, tr2_coreg)
    overlaps = []
    if win2 is not None:
        wi2, wj2 = win2
        n = (wi2[1] - wi2[0]) * (wi2[3] - wi2[2])
        if n > 100:
            overlaps = [{'idx_i': 0, 'idx_j': 1, 'window_i': wi2, 'window_j': wj2}]
    print(f"  Overlap pairs: {len(overlaps)}")

    # --- Original metrics ---
    original_metrics = compute_all(arrs_registered, arrs_registered, nodatas, overlaps, [0])

    # --- BAGRN ---
    t_bagrn = time.time()
    bagrn_result, bagrn_coeffs, bagrn_info = bagrn_normalize(
        arrs_registered, nodatas, overlaps, control_idx=0)
    t_bagrn = time.time() - t_bagrn
    bagrn_metrics = compute_all(arrs_registered, bagrn_result, nodatas, overlaps, [0])

    # --- VOLRN ---
    t_volrn = time.time()
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result, transforms_all, bounds_all, nodatas,
        block_size_pixels=200, lambda_param=0.1, rho=1.0,
        max_iter=200, tol=1e-4, verbose=False)
    t_volrn = time.time() - t_volrn
    volrn_metrics = compute_all(arrs_registered, volrn_result, nodatas, overlaps, [0])

    # --- Print table ---
    def _v(m, key):
        return m.get(key, float('nan'))

    print(f"\n{'='*70}")
    print(f"  B14 radiometric normalization and mosaic")
    print(f"{'='*70}")
    print(f"  {'Metric':<12s} {'Registered / Before normalization':>34s} {'BAGRN':>12s} {'BAGRN + VOLRN':>16s}")
    print(f"  {'-'*48}")
    for key, label in [
        ("adm", "ADM"),
        ("adsd", "ADSD"),
        ("cd", "CD"),
        ("gl", "GL"),
        ("rdoa", "RDOA"),
        ("ave", "Ave"),
    ]:
        v0 = _v(original_metrics, key)
        v1 = _v(bagrn_metrics, key)
        v2 = _v(volrn_metrics, key)
        print(f"  {label:<12s} {v0:12.6f} {v1:12.6f} {v2:12.6f}")
    print(f"\n  BAGRN time: {t_bagrn:.1f}s")
    print(f"  VOLRN time: {t_volrn:.1f}s")

    # --- Mosaics ---
    path_bagrn = os.path.join(out_dir, f'mosaic_{band}_bagrn.tif')
    t0 = time.time()
    create_mosaic(bagrn_result, transforms_all, crs, nodatas, path_bagrn)
    # NaN/Inf check
    _check_mosaic(path_bagrn)
    print(f"\n  BAGRN mosaic: {path_bagrn} ({os.path.getsize(path_bagrn)/1024/1024:.1f} MB, {time.time()-t0:.1f}s)")

    path_volrn = os.path.join(out_dir, f'mosaic_{band}_volrn.tif')
    t0 = time.time()
    create_mosaic(volrn_result, transforms_all, crs, nodatas, path_volrn)
    _check_mosaic(path_volrn)
    print(f"  VOLRN mosaic: {path_volrn} ({os.path.getsize(path_volrn)/1024/1024:.1f} MB, {time.time()-t0:.1f}s)")

    return {
        'quality_status': 'DONE',
        'metrics': {
            'original': original_metrics,
            'bagrn': bagrn_metrics,
            'volrn': volrn_metrics,
        },
        'registration': registration_metrics,
    }


def _check_mosaic(path):
    """检查镶嵌图中是否存在 NaN 或 Inf，若存在则抛出错误。"""
    from src.io_utils import read_geotiff
    arr, _, _, nd = read_geotiff(path)
    if arr.ndim > 2:
        arr = arr[0]
    valid = np.isfinite(arr) & (arr != nd)
    nan_count = np.sum(~np.isfinite(arr))
    inf_count = np.sum(np.isinf(arr))
    if nan_count > 0 or inf_count > 0:
        raise RuntimeError(f"Mosaic contains NaN={nan_count}, Inf={inf_count}: {path}")


if __name__ == '__main__':
    t_start = time.time()
    all_results = {}
    for band in BANDS:
        try:
            r = process_band(band)
            if r:
                all_results[band] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")

    with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nTotal: {time.time()-t_start:.1f}s")
    print(f"Output: {OUTPUT}")
