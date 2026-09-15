"""Fast, geometry-only evidence diagnostics for a two-image registration pair."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.two_image_pipeline import IMG1, IMG2, OUTPUT
from src.io_utils import read_geotiff
from src.registration_diagnostics import (
    build_edge_overlay,
    build_reference_common_grid_overlap,
    collect_common_grid_candidate_diagnostics,
    collect_raw_grid_candidate_diagnostics,
    compare_raw_and_common_grid,
    compute_grid_relationship,
    choose_diagnostic_chip_centers,
    summarize_candidate_rows,
)


RAW_CSV_FIELDS = [
    'candidate_index',
    'block_row_offset',
    'block_col_offset',
    'ref_x',
    'ref_y',
    'tgt_x',
    'tgt_y',
    'valid_count',
    'valid_ratio',
    'texture_std',
    'texture_threshold',
    'shift_dx_pixels',
    'shift_dy_pixels',
    'shift_magnitude_pixels',
    'confidence',
    'reject_reason',
]
COMMON_CSV_FIELDS = RAW_CSV_FIELDS + [
    'zero_shift_ncc',
    'best_shift_ncc',
    'ncc_gain',
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Diagnose two-image registration evidence without changing the baseline.'
    )
    parser.add_argument('--band', default='B14')
    parser.add_argument('--ref')
    parser.add_argument('--tgt')
    parser.add_argument('--output')
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--max-shift', type=float, default=40.0)
    return parser.parse_args(argv)


def _json_safe(value):
    if isinstance(value, Affine):
        return {
            'a': float(value.a), 'b': float(value.b), 'c': float(value.c),
            'd': float(value.d), 'e': float(value.e), 'f': float(value.f),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False)


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _resolve_paths(args):
    if bool(args.ref) != bool(args.tgt):
        raise ValueError('--ref and --tgt must be provided together')
    if args.ref and args.tgt:
        ref_path = args.ref
        tgt_path = args.tgt
    else:
        ref_path = IMG1.format(args.band)
        tgt_path = IMG2.format(args.band)
    output_dir = args.output or os.path.join(
        OUTPUT, args.band, 'registration_diagnostics')
    return ref_path, tgt_path, output_dir


def _as_2d(array):
    array = np.asarray(array)
    while array.ndim > 2:
        array = array[0]
    return array


def _unavailable_common_result(reason):
    empty_screening = {
        'total': 0,
        'low_valid': 0,
        'low_texture': 0,
        'low_conf': 0,
        'large_shift': 0,
        'accepted': 0,
    }
    return {
        'available': False,
        'failure_reason': reason,
        'screening': empty_screening,
        'candidates': [],
        'summary': summarize_candidate_rows([]),
    }


def _summary_value(result, path):
    current = result
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _print_stage_summary(name, result, common=False):
    summary = result.get('summary', {})
    confidence = _summary_value(summary, ('confidence', 'median'))
    magnitude = _summary_value(summary, ('shift_magnitude_pixels', 'median'))
    print(f'  available={result.get("available")}, '
          f'measured={summary.get("measured_rows", 0)}, '
          f'accepted={result.get("screening", {}).get("accepted", 0)}, '
          f'median confidence={confidence}, '
          f'median shift magnitude={magnitude}')
    if common:
        print(
            f'  common median zero-shift NCC='
            f'{_summary_value(summary, ("zero_shift_ncc", "median"))}, '
            f'median NCC gain='
            f'{_summary_value(summary, ("ncc_gain", "median"))}'
        )


def _robust_stretch(array, valid):
    values = np.asarray(array, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    stretched = np.zeros(values.shape, dtype=np.float64)
    if not np.any(valid):
        return stretched
    p2, p98 = np.percentile(values[valid], [2, 98])
    if p98 <= p2 + 1e-12:
        p2 = float(np.min(values[valid]))
        p98 = float(np.max(values[valid]))
    if p98 <= p2 + 1e-12:
        stretched[valid] = 0.5
        return stretched
    stretched[valid] = np.clip(
        (values[valid] - p2) / (p98 - p2), 0.0, 1.0)
    return stretched


def save_common_grid_chips(common_grid, output_dir, chip_size=256, max_chips=9):
    """Save deterministic reference, target, and edge-overlay chip PNGs."""
    if not common_grid.get('available'):
        return 0
    common_valid = common_grid.get('common_valid')
    ref_overlap = common_grid.get('ref_overlap')
    tgt_on_ref_grid = common_grid.get('tgt_on_ref_grid')
    if common_valid is None or ref_overlap is None or tgt_on_ref_grid is None:
        return 0

    from matplotlib import image as mpimg

    chips_dir = os.path.join(output_dir, 'chips')
    os.makedirs(chips_dir, exist_ok=True)
    centers = choose_diagnostic_chip_centers(
        common_valid, chip_size=chip_size, max_chips=max_chips)
    height, width = common_valid.shape
    written = 0
    for index, (center_row, center_col) in enumerate(centers):
        row_start = max(0, min(height - chip_size, center_row - chip_size // 2))
        col_start = max(0, min(width - chip_size, center_col - chip_size // 2))
        row_end = row_start + chip_size
        col_end = col_start + chip_size
        valid_chip = common_valid[row_start:row_end, col_start:col_end]
        ref_chip = ref_overlap[row_start:row_end, col_start:col_end]
        tgt_chip = tgt_on_ref_grid[row_start:row_end, col_start:col_end]
        stem = f'chip_{index:02d}'
        mpimg.imsave(
            os.path.join(chips_dir, f'{stem}_ref.png'),
            _robust_stretch(ref_chip, valid_chip), cmap='gray', vmin=0, vmax=1)
        mpimg.imsave(
            os.path.join(chips_dir, f'{stem}_tgt_common.png'),
            _robust_stretch(tgt_chip, valid_chip), cmap='gray', vmin=0, vmax=1)
        mpimg.imsave(
            os.path.join(chips_dir, f'{stem}_edge_overlay.png'),
            build_edge_overlay(ref_chip, tgt_chip, valid_chip))
        written += 1
    return written


def build_diagnostic_report(
    grid_metadata, raw_result, common_result, comparison, generated_files
):
    """Build a factual report explaining the raw/common-grid evidence."""
    raw_screening = raw_result.get('screening', {})
    common_screening = common_result.get('screening', {})
    raw_summary = raw_result.get('summary', {})
    common_summary = common_result.get('summary', {})
    phase = grid_metadata.get('grid_relationship', {}).get(
        'fractional_phase_pixels', {})
    lines = [
        'Two-image registration evidence diagnostic',
        '',
        '1. Input',
        f"Reference: {grid_metadata.get('reference_path')}",
        f"Target: {grid_metadata.get('target_path')}",
        '',
        '2. GeoTIFF grid relationship',
        f"Fractional phase (col, row): {phase.get('col')}, {phase.get('row')}",
        '',
        '3. Raw-grid matcher',
        f"Measured rows: {raw_summary.get('measured_rows', 0)}",
        f"Accepted rows: {raw_screening.get('accepted', 0)}",
        '',
        '4. Common-grid matcher',
        f"Measured rows: {common_summary.get('measured_rows', 0)}",
        f"Accepted rows: {common_screening.get('accepted', 0)}",
        '',
        '5. Zero-shift vs best-shift',
        f"Median zero-shift NCC: "
        f"{_summary_value(common_summary, ('zero_shift_ncc', 'median'))}",
        f"Median NCC gain: "
        f"{_summary_value(common_summary, ('ncc_gain', 'median'))}",
        '',
        '6. Evidence comparison',
    ]
    hints = comparison.get('interpretation_hints', [])
    lines.extend([f"Evidence hint: {hint}" for hint in hints])
    if not hints:
        lines.append('No conservative evidence hint was triggered.')
    lines.extend([
        '',
        '7. Why a good-looking mosaic can coexist with accepted=0',
        'accepted=0 means no block passed the current matcher acceptance rule; '
        'it does not by itself prove that the georeferenced images are spatially misaligned.',
        'The mosaic path uses each scene CRS/Affine to reproject data into a '
        'shared output grid before blending. Therefore a visually well-aligned '
        'mosaic can coexist with a raw-array block matcher that accepts no blocks, '
        'especially when source pixel grids have different origins/phases.',
        '',
        '8. What this diagnostic does NOT prove',
        'These are same-data diagnostics, not independent HOLDOUT accuracy.',
        'They do not establish performance on unseen imagery or select a new matcher.',
        '',
        '9. Generated files',
    ])
    lines.extend([f'- {item}' for item in generated_files])
    return '\n'.join(lines) + '\n'


def run_diagnostic(args):
    ref_path, tgt_path, output_dir = _resolve_paths(args)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'chips'), exist_ok=True)

    arr_ref, tr_ref, crs_ref, nodata_ref = read_geotiff(ref_path)
    arr_tgt, tr_tgt, crs_tgt, nodata_tgt = read_geotiff(tgt_path)
    arr_ref = _as_2d(arr_ref)
    arr_tgt = _as_2d(arr_tgt)
    if crs_ref != crs_tgt:
        raise ValueError(f'CRS mismatch: {crs_ref} vs {crs_tgt}')

    grid_relationship = compute_grid_relationship(tr_ref, tr_tgt)
    raw_result = collect_raw_grid_candidate_diagnostics(
        arr_ref,
        tr_ref,
        arr_tgt,
        tr_tgt,
        nodata_ref=nodata_ref,
        nodata_tgt=nodata_tgt,
        block_size=args.block_size,
        max_global_shift=args.max_shift,
        confidence_threshold=args.confidence_threshold,
    )
    common_grid = build_reference_common_grid_overlap(
        arr_ref,
        tr_ref,
        arr_tgt,
        tr_tgt,
        crs_ref,
        crs_tgt,
        nodata_ref=nodata_ref,
        nodata_tgt=nodata_tgt,
    )
    if common_grid['available']:
        common_result = collect_common_grid_candidate_diagnostics(
            common_grid['ref_overlap'],
            common_grid['tgt_on_ref_grid'],
            common_grid['common_valid'],
            block_size=args.block_size,
            confidence_threshold=args.confidence_threshold,
            max_residual_shift=args.max_shift,
        )
    else:
        common_result = _unavailable_common_result(
            common_grid.get('failure_reason', 'common grid unavailable'))
    comparison = compare_raw_and_common_grid(
        raw_result, common_result, grid_relationship)
    chip_count = save_common_grid_chips(common_grid, output_dir)

    grid_metadata = {
        'band': args.band,
        'reference_path': ref_path,
        'target_path': tgt_path,
        'reference_shape': list(arr_ref.shape),
        'target_shape': list(arr_tgt.shape),
        'reference_crs': str(crs_ref),
        'target_crs': str(crs_tgt),
        'nodata_ref': nodata_ref,
        'nodata_tgt': nodata_tgt,
        'block_size': args.block_size,
        'confidence_threshold': args.confidence_threshold,
        'max_shift': args.max_shift,
        'grid_relationship': grid_relationship,
        'raw_overlap_windows': raw_result.get('overlap_windows'),
        'common_grid': {
            'available': common_grid.get('available'),
            'failure_reason': common_grid.get('failure_reason'),
            'ref_window': common_grid.get('ref_window'),
            'tgt_window': common_grid.get('tgt_window'),
            'overlap_transform': common_grid.get('overlap_transform'),
            'reprojected_target': common_grid.get('reprojected_target'),
            'chip_count': chip_count,
        },
    }
    _write_json(os.path.join(output_dir, 'grid_metadata.json'), grid_metadata)
    _write_json(
        os.path.join(output_dir, 'raw_grid_summary.json'),
        {key: value for key, value in raw_result.items() if key != 'candidates'},
    )
    _write_json(
        os.path.join(output_dir, 'common_grid_summary.json'),
        {key: value for key, value in common_result.items() if key != 'candidates'},
    )
    _write_json(os.path.join(output_dir, 'grid_comparison.json'), comparison)
    _write_csv(
        os.path.join(output_dir, 'raw_grid_candidates.csv'),
        raw_result.get('candidates', []),
        RAW_CSV_FIELDS,
    )
    _write_csv(
        os.path.join(output_dir, 'common_grid_candidates.csv'),
        common_result.get('candidates', []),
        COMMON_CSV_FIELDS,
    )
    generated_files = [
        'grid_metadata.json',
        'raw_grid_candidates.csv',
        'raw_grid_summary.json',
        'common_grid_candidates.csv',
        'common_grid_summary.json',
        'grid_comparison.json',
        'diagnostic_report.txt',
        'chips/',
    ]
    report = build_diagnostic_report(
        grid_metadata,
        raw_result,
        common_result,
        comparison,
        generated_files,
    )
    with open(os.path.join(output_dir, 'diagnostic_report.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)

    print('=== Grid relationship ===')
    print(f'  ref pixel size={grid_relationship["ref_pixel_size"]}')
    print(f'  tgt pixel size={grid_relationship["tgt_pixel_size"]}')
    print(f'  fractional grid phase={grid_relationship["fractional_phase_pixels"]}')
    print('=== Raw-grid matcher ===')
    _print_stage_summary('raw', raw_result)
    print('=== Common-grid matcher ===')
    _print_stage_summary('common', common_result, common=True)
    print('=== Evidence hints ===')
    for hint in comparison.get('interpretation_hints', []):
        print(f'  {hint}')
    if not comparison.get('interpretation_hints'):
        print('  No conservative evidence hint triggered.')

    return {
        'output_dir': output_dir,
        'grid_relationship': grid_relationship,
        'raw': raw_result,
        'common_grid': common_result,
        'comparison': comparison,
    }


def main(argv=None):
    run_diagnostic(parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
