"""Post-run read-only pixel, support and frozen-window audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    freeze = root.parent / "data/output/benchmark_v1_unified/freeze_20261013_v3"
    status = json.loads((args.run / "run_status.json").read_text(encoding="utf-8"))
    pair_scores = json.loads((args.run / "b1_b2_scores.json").read_text(encoding="utf-8"))
    output = {"status": "PASS", "scenes": [], "pair_windows": {}}
    for i, scene in enumerate(status["scene_outputs"]):
        input_data = freeze / "aligned_inputs" / f"scene_{i:03d}_B9_input.tif"
        input_mask = freeze / "aligned_inputs" / f"scene_{i:03d}_input_mask.tif"
        gain = status["model"]["gains_b9"][i]
        offset = status["model"]["offsets_b9"][i]
        max_abs = 0.0
        valid_count = 0
        with rasterio.open(input_data) as a, rasterio.open(input_mask) as am, rasterio.open(scene["data_path"]) as b, rasterio.open(scene["mask_path"]) as bm:
            if any((ds.width, ds.height, ds.transform, ds.crs) != (a.width, a.height, a.transform, a.crs) for ds in (am, b, bm)):
                raise ValueError(f"GRID_MISMATCH:{i}")
            if b.dtypes[0] != "float32" or bm.dtypes[0] != "uint8":
                raise ValueError(f"OUTPUT_DTYPE_MISMATCH:{i}")
            for row in range(0, a.height, 512):
                for col in range(0, a.width, 512):
                    w = Window(col, row, min(512, a.width-col), min(512, a.height-row))
                    valid = am.read(1, window=w) != 0
                    mask = bm.read(1, window=w) != 0
                    if not np.array_equal(valid, mask):
                        raise ValueError(f"MASK_MISMATCH:{i}")
                    x = a.read(1, window=w)
                    y = b.read(1, window=w)
                    if not np.isnan(y[~valid]).all() or not np.isfinite(y[valid]).all():
                        raise ValueError(f"NODATA_MISMATCH:{i}")
                    expected = (x[valid].astype(np.float64) * gain + offset).astype(np.float32)
                    if not np.array_equal(y[valid], expected):
                        raise ValueError(f"AFFINE_VALUE_MISMATCH:{i}")
                    valid_count += int(valid.sum())
            output["scenes"].append({"index": i, "valid_pixels": valid_count,
                                      "pixelwise_float32_affine_exact": True,
                                      "mask_exact": True, "grid_exact": True,
                                      "invalid_output_nan": True, "max_abs_value_error": max_abs})
    for pair_id, pair in pair_scores.items():
        raw, corrected = pair["raw"], pair["corrected"]
        if raw["support_count"] != corrected["support_count"] or len(raw["windows"]) != pair["frozen_window_count"] or len(corrected["windows"]) != pair["frozen_window_count"]:
            raise ValueError(f"PAIR_SUPPORT_MISMATCH:{pair_id}")
        if not all(w["support_sha256_match"] for w in raw["windows"] + corrected["windows"]):
            raise ValueError(f"WINDOW_HASH_MISMATCH:{pair_id}")
        output["pair_windows"][pair_id] = {"frozen_window_count": pair["frozen_window_count"],
                                           "all_support_hashes_match": True,
                                           "support_count": raw["support_count"]}
    target = args.run / "output_audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "scenes": len(output["scenes"]),
                      "windows": sum(x["frozen_window_count"] for x in output["pair_windows"].values())}))


if __name__ == "__main__":
    main()
