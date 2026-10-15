"""Small synthetic checks for the read-only Benchmark V1 inventories."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


spec = spec_from_file_location("benchmark_v1_inputs", Path(__file__).resolve().parents[1] / "scripts/build_benchmark_v1_inputs.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_mask_count_excludes_nodata_and_bbox_only_overlap():
    with MemoryFile() as ma, MemoryFile() as mb:
        profile = dict(driver="GTiff", width=3, height=3, count=1, dtype="uint8", crs="EPSG:32650", nodata=0)
        with ma.open(**profile, transform=from_origin(0, 42, 14, 14)) as a, mb.open(
            **profile, transform=from_origin(14, 42, 14, 14)
        ) as b:
            a.write(np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.uint8), 1)
            b.write(np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8), 1)
            assert module.count_shared_valid(a, b, chunk_rows=1) == (3, 6)


def test_selection_ignores_matcher_outcome():
    pairs = [dict(pair_id=f"00_0{i}", idx_i=0, idx_j=i,
                  valid_overlap_pixels=1_000_000+i, time_bin="SAME_DATE",
                  quality_i="PASS", quality_j="PASS", registration_status_task15="OK") for i in range(1, 4)]
    chosen = module.choose_pairs(pairs, count=2)
    for pair in pairs:
        pair["registration_status_task15"] = "FAIL"
    assert chosen == module.choose_pairs(pairs, count=2)


def test_temporal_bins_and_subpixel_checkpoint_template_contract():
    assert module.time_bin(.5, True) == "SAME_DATE"
    assert module.time_bin(6.9, False) == "CROSS_DATE_LT_7D"
    assert module.time_bin(7, False) == "CROSS_DATE_7_29D"
    assert module.time_bin(30, False) == "CROSS_DATE_GE_30D"
