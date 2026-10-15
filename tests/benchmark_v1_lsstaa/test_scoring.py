import hashlib

import numpy as np
import rasterio
from affine import Affine

from src.benchmark_v1_lsstaa.scoring import score_pair, support_digest


def test_frozen_pair_scoring_keeps_support_and_all_seven_metrics(tmp_path):
    profile = dict(driver="GTiff", width=3, height=2, count=1, dtype="float32",
                   crs="EPSG:32650", transform=Affine(14, 0, 0, 0, -14, 28))
    arrays = [np.array([[1, 2, 0], [4, 5, 6]], np.float32),
              np.array([[2, 4, 0], [5, 6, 7]], np.float32)]
    masks = [np.array([[1, 1, 0], [1, 1, 1]], np.uint8),
             np.array([[1, 1, 0], [1, 1, 1]], np.uint8)]
    paths = []
    for i in range(2):
        data, mask = tmp_path / f"d{i}.tif", tmp_path / f"m{i}.tif"
        with rasterio.open(data, "w", **profile) as ds:
            ds.write(arrays[i], 1)
        with rasterio.open(mask, "w", **{**profile, "dtype": "uint8", "nodata": 0}) as ds:
            ds.write(masks[i], 1)
        paths.append((data, mask))
    support = masks[0].astype(bool) & masks[1].astype(bool)
    window = dict(window_id="w", pair_id="00_01", row_off="0", col_off="0",
                  height="2", width="3", valid_count=str(support.sum()),
                  support_sha256=support_digest(support))
    result = score_pair(paths[0], paths[1], [window])
    assert result["support_count"] == 5
    assert result["windows"][0]["support_sha256_match"]
    assert result["b1"]["mae"] == 1.2
    assert all(k in result["b1"] for k in ("mae", "rmse", "bias", "p95ae", "mamd", "msdd", "rdd"))
