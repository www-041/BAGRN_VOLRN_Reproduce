"""Read-only B1/B2 diagnostic on the original frozen support and windows."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.benchmark_v1_unified.radiometric import (
    compute_pair_direct_metrics, compute_pair_distribution_metrics,
)


def support_digest(support: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(support, bitorder="little").tobytes()).hexdigest()


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    support = np.ones(a.shape, dtype=bool)
    direct = asdict(compute_pair_direct_metrics(a, b, support))
    distribution = asdict(compute_pair_distribution_metrics(a, b, support))
    return {**direct, **{k: distribution[k] for k in
                       ("mean_a", "mean_b", "std_a", "std_b", "mamd", "msdd", "rdd")}}


def score_pair(left: tuple[Path, Path], right: tuple[Path, Path], windows: list[dict]) -> dict:
    with (rasterio.open(left[0]) as a, rasterio.open(left[1]) as am,
          rasterio.open(right[0]) as b, rasterio.open(right[1]) as bm):
        grid = (a.width, a.height, a.transform, a.crs)
        for ds in (am, b, bm):
            if (ds.width, ds.height, ds.transform, ds.crs) != grid:
                raise ValueError("SCORING_GRID_MISMATCH")
        left_values, right_values = [], []
        for row in range(0, a.height, 512):
            for col in range(0, a.width, 512):
                w = Window(col, row, min(512, a.width-col), min(512, a.height-row))
                support = (am.read(1, window=w) != 0) & (bm.read(1, window=w) != 0)
                if np.any(support):
                    av = a.read(1, window=w)[support]
                    bv = b.read(1, window=w)[support]
                    if not np.isfinite(av).all() or not np.isfinite(bv).all():
                        raise ValueError("NONFINITE_FROZEN_SUPPORT")
                    left_values.append(av)
                    right_values.append(bv)
        if not left_values:
            return {"status": "NO_VALID_OVERLAP", "support_count": 0, "b1": None, "windows": []}
        x, y = np.concatenate(left_values), np.concatenate(right_values)
        b1 = _metrics(x, y)
        window_results = []
        for record in windows:
            w = Window(int(record["col_off"]), int(record["row_off"]),
                       int(record["width"]), int(record["height"]))
            support = (am.read(1, window=w) != 0) & (bm.read(1, window=w) != 0)
            digest = support_digest(support)
            if digest != record["support_sha256"] or int(support.sum()) != int(record["valid_count"]):
                raise ValueError(f"FROZEN_WINDOW_SUPPORT_MISMATCH:{record['window_id']}")
            av, bv = a.read(1, window=w)[support], b.read(1, window=w)[support]
            if not np.isfinite(av).all() or not np.isfinite(bv).all():
                raise ValueError(f"NONFINITE_WINDOW_SUPPORT:{record['window_id']}")
            window_results.append({"window_id": record["window_id"],
                                   "support_count": int(support.sum()),
                                   "support_sha256_match": True,
                                   "metrics": _metrics(av, bv)})
        return {"status": "EVALUABLE", "support_count": int(len(x)), "b1": b1,
                "windows": window_results}
