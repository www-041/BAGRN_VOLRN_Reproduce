#!/usr/bin/env python3
"""
Preflight check for mosaic series experiments.

Usage:
    python -m scripts.preflight_mosaic_series --config configs/dz01_mosaic_series_b14.yaml --scene-count 2,4,6
    OR
    cd /path/to/BAGRN_VOLRN_Reproduce && python scripts/preflight_mosaic_series.py --config configs/dz01_mosaic_series_b14.yaml --scene-count 2,4,6
"""

import os
import sys
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preflight import preflight_mosaic_series

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="Preflight check for mosaic series")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--scene-count", required=True, help="Comma-separated scene counts (e.g., 2,4,6)")
    
    args = parser.parse_args()
    
    scene_counts = [int(x.strip()) for x in args.scene_count.split(",")]
    
    try:
        results = preflight_mosaic_series(args.config, scene_counts)
        print("\n" + "=" * 60)
        print("PREFLIGHT SUMMARY")
        print("=" * 60)
        print(f"Config: {results['config']}")
        print(f"Band: {results['band']}")
        print(f"Total scenes: {results['total_scenes']}")
        print(f"Scene counts tested: {results['scene_counts']}")
        print("\nChecks:")
        for check, passed in results['checks'].items():
            if isinstance(passed, dict):
                for n, ok in passed.items():
                    status = "✓ PASS" if ok else "✗ FAIL"
                    print(f"  {check} (N={n}): {status}")
            else:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {check}: {status}")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ PREFLIGHT FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
