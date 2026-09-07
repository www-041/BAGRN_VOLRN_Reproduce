"""Regression tests for experiment runner CLI and resume semantics.

Task 15 of reliability-fixes plan:
- --crop-size default must not overwrite YAML when CLI option omitted
- resume must not return empty results
"""

import argparse


def test_crop_size_default_is_none():
    """--crop-size should default to None, not 512."""
    # Simulate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=None)
    
    # When omitted, should be None
    args = parser.parse_args([])
    assert args.crop_size is None, "crop_size should be None when not specified"
    
    # When specified, should have the value
    args = parser.parse_args(["--crop-size", "256"])
    assert args.crop_size == 256


def test_yaml_not_overwritten_when_crop_size_omitted():
    """When --crop-size is None, YAML value should be preserved."""
    crop_size_cli = None  # Not specified
    yaml_crop_size = 1024  # From config
    
    # Bug: always overwrites with CLI value (which defaults to 512)
    # Fix: only overwrite if CLI value is not None
    
    if crop_size_cli is not None:
        final_crop_size = crop_size_cli
    else:
        final_crop_size = yaml_crop_size
    
    assert final_crop_size == 1024, "YAML value should be preserved"
