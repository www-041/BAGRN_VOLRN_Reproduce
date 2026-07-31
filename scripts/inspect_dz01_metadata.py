"""
DZ01 Metadata Inspector — print sensor and band information from MTL files.

Usage:
  python scripts/inspect_dz01_metadata.py --vnir-mtl <path> --swir-mtl <path>
  python scripts/inspect_dz01_metadata.py --vnir-mtl <path>
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dz01_metadata import (
    parse_dz01_mtl,
    extract_band_metadata,
    validate_sensor_metadata,
    VNIR_BAND_TABLE,
    SWIR_BAND_TABLE,
)


def main():
    parser = argparse.ArgumentParser(description="DZ01 Metadata Inspector")
    parser.add_argument("--vnir-mtl", type=str, default=None, help="VNIR MTL file path")
    parser.add_argument("--swir-mtl", type=str, default=None, help="SWIR MTL file path")
    args = parser.parse_args()

    print("=" * 60)
    print("DZ01 Metadata Inspector")
    print("=" * 60)

    for label, mtl_path in [("VNIR", args.vnir_mtl), ("SWIR", args.swir_mtl)]:
        if mtl_path is None:
            print(f"\n{label}: (not provided)")
            continue

        print(f"\n{'─' * 60}")
        print(f"{label}: {mtl_path}")
        print(f"{'─' * 60}")

        try:
            metadata = parse_dz01_mtl(mtl_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        sensor_id = metadata["sensor_id"]
        bands = metadata["bands"]
        scene = metadata["scene"]
        warnings = metadata["warnings"]

        print(f"  sensor_id: {sensor_id}")
        print(f"  spacecraft_id: {metadata['spacecraft_id']}")
        print(f"  product_id: {metadata['product_id']}")
        print(f"  date_acquired: {metadata['date_acquired']}")
        print(f"  scene_center_time: {metadata['scene_center_time']}")
        print(f"  bands: {len(bands)}")
        print(f"  resolution_vi: {metadata.get('resolution_vi')} m")
        print(f"  resolution_pan: {metadata.get('resolution_pan')} m")

        # Wavelength range
        wl_centers = [
            b["wavelength_center_nm"]
            for b in bands.values()
            if b.get("wavelength_center_nm") is not None
        ]
        if wl_centers:
            print(f"  wavelength_range: {min(wl_centers)}-{max(wl_centers)} nm")

        # Scene geometry (only print if not None)
        print(f"  sun_azimuth: {scene.get('SUN_AZIMUTH')}")
        print(f"  sun_zenith: {scene.get('SUN_ZENITH')}")
        print(f"  sun_elevation: {scene.get('SUN_ELEVATION')}")
        print(f"  sat_azimuth: {scene.get('SAT_AZIMUTH')}")
        print(f"  sat_zenith: {scene.get('SAT_ZENITH')}")
        print(f"  cloud_cover: {scene.get('CLOUD_COVER')}")
        print(f"  roll_angle: {scene.get('ROLL_ANGLE')}")
        print(f"  pitch_angle: {scene.get('PITCH_ANGLE')}")
        print(f"  yaw_angle: {scene.get('YAW_ANGLE')}")

        # Band table
        print(f"\n  Band table:")
        print(f"  {'Band':<6} {'Center(nm)':<12} {'Min(nm)':<10} {'Max(nm)':<10}")
        for band_name in sorted(bands.keys()):
            b = bands[band_name]
            center = b.get("wavelength_center_nm", "?")
            bmin = b.get("wavelength_min_nm", "?")
            bmax = b.get("wavelength_max_nm", "?")
            print(f"  {band_name:<6} {center:<12} {bmin:<10} {bmax:<10}")

        if warnings:
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    - {w}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    if args.vnir_mtl:
        vnir_meta = parse_dz01_mtl(args.vnir_mtl)
        vnir_bands = len(vnir_meta["bands"])
        vnir_wl = [
            b["wavelength_center_nm"]
            for b in vnir_meta["bands"].values()
            if b.get("wavelength_center_nm") is not None
        ]
        print(f"VNIR:")
        print(f"  sensor_id: {vnir_meta['sensor_id']}")
        print(f"  bands: {vnir_bands}")
        print(f"  selected_for_stage2: B01-B14")
        if vnir_wl:
            print(f"  wavelength_range_selected: {min(vnir_wl[:14])}-{max(vnir_wl[:14])} nm")
        print(f"  resolution: {vnir_meta.get('resolution_vi')} m")
    else:
        print("VNIR: (not provided)")

    if args.swir_mtl:
        swir_meta = parse_dz01_mtl(args.swir_mtl)
        swir_bands = len(swir_meta["bands"])
        swir_wl = [
            b["wavelength_center_nm"]
            for b in swir_meta["bands"].values()
            if b.get("wavelength_center_nm") is not None
        ]
        print(f"SWIR:")
        print(f"  sensor_id: {swir_meta['sensor_id']}")
        print(f"  bands: {swir_bands}")
        if swir_wl:
            print(f"  wavelength_range: {min(swir_wl)}-{max(swir_wl)} nm")
        print(f"  resolution: {swir_meta.get('resolution_vi')} m")
    else:
        print("SWIR: (not provided)")

    print(f"\nCurrent Stage 2 data source: VNIR only")
    print(f"SWIR included in Stage 2: false")


if __name__ == "__main__":
    main()
