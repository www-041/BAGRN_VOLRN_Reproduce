#!/usr/bin/env python3
"""Registration-only diagnostic for manual validation.

Usage:
    python scripts/diagnose_registration_pair.py \
        --config configs/dz01_mosaic_series_b14.yaml \
        --scene-i 0 --scene-j 1

This script runs ONLY registration (no BAGRN/VOLRN/mosaic) to diagnose
geometric alignment quality between two scenes.
"""

import os
import sys
import argparse
import logging
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiment_config import load_config
from src.multiband_pipeline import MultibandPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Registration-only diagnostic")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scene-i", type=int, default=0, help="First scene index")
    parser.add_argument("--scene-j", type=int, default=1, help="Second scene index")
    parser.add_argument("--output-dir", default=None, help="Output directory for diagnostics")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("Registration Diagnostic (N=2 subset)")
    logger.info("Config: %s", args.config)
    logger.info("Scene pair: %d-%d", args.scene_i, args.scene_j)
    logger.info("=" * 60)
    
    # Subset to 2 scenes
    scene_i = config.scenes[args.scene_i]
    scene_j = config.scenes[args.scene_j]
    
    config.scenes = [scene_i, scene_j]
    config.output_root = args.output_dir or os.path.join(config.output_root, "registration_diagnostic")
    
    # Create pipeline with minimal settings
    pipeline = MultibandPipeline(config)
    
    # Load data
    scene_data = pipeline.load_scenes()
    
    # Detect overlaps
    overlaps = pipeline.detect_overlaps(scene_data)
    logger.info("Detected %d overlap pair(s)", len(overlaps))
    
    if not overlaps:
        logger.error("No overlap detected between scenes")
        return 1
    
    # Run registration only
    registration = pipeline.register_scenes(scene_data, overlaps)
    
    logger.info("=" * 60)
    logger.info("Registration Results:")
    logger.info("  Connected: %s", registration.get("connected", False))
    logger.info("  Global shifts: %s", registration.get("global_shifts", "N/A"))
    
    if "quality" in registration:
        q = registration["quality"]
        logger.info("  Quality: %s", q.get("quality", "unknown"))
        logger.info("  RMSE: %.3f px", q.get("rmse", 0))
        logger.info("  P95: %.3f px", q.get("p95", 0))
        logger.info("  Confidence: %.3f", q.get("confidence", 0))
    
    logger.info("=" * 60)
    logger.info("Diagnostic complete. No BAGRN/VOLRN/mosaic was run.")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
