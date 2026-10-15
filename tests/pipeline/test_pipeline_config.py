from pathlib import Path

import pytest

from src.pipeline.config import PipelineConfig, load_pipeline_config


def _write_config(tmp_path: Path, **overrides) -> Path:
    text = """
dataset:
  input_root: /data/b9
  band: B9
  expected_scene_count: 13
registration:
  matcher: efficient_loftr
  match_max_side: 1024
  checkpoint: /models/eloftr_outdoor.ckpt
  global_method: translation_l2
canonical_grid:
  crs: EPSG:32650
  resolution_m: 14.0
radiometric:
  method: bagrn
  control_scene_index: 0
seam:
  intensity_weight: 0.5
  gradient_weight: 0.5
  normalization: p95
  coarse_factor: 4
  refine_half_width: 64
local_correction:
  corridor_half_width: 128
  segment_length: 256
  min_valid_pixels: 4096
  percentile_low: 1
  percentile_high: 99
  gain_min: 0.5
  gain_max: 2.0
labeling:
  preference_distance_scale: 64
  tie_tolerance: 1.0e-6
  hierarchy: [pairwise_score, clipped_normalized_interiority, unclipped_normalized_interiority, raw_edt]
blend:
  method: multilabel_cosine
  half_width: 64
streaming:
  tile_size: 1024
  halo: 128
outputs:
  root: /tmp/task15
  write_geotiff: true
  write_quicklooks: true
  write_metrics: true
  write_report: true
"""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pipeline_config_returns_immutable_validated_config(tmp_path):
    config = load_pipeline_config(_write_config(tmp_path))
    assert isinstance(config, PipelineConfig)
    assert config.dataset.expected_scene_count == 13
    with pytest.raises(AttributeError):
        config.dataset = config.dataset


def test_missing_matcher_checkpoint_is_rejected(tmp_path):
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("  checkpoint: /models/eloftr_outdoor.ckpt\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        load_pipeline_config(path)


def test_unknown_label_hierarchy_is_rejected(tmp_path):
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("raw_edt]", "raw_edt, unknown_rule]"), encoding="utf-8")
    with pytest.raises(ValueError, match="hierarchy"):
        load_pipeline_config(path)

