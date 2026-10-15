"""Read-only check of four suggestion CSVs against native B9 masks and Affine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from benchmark_candidate_core import CandidateReviewStore
from benchmark_phase2_core import pixel_to_projected
from build_benchmark_v1_inputs import overlap_window
from generate_benchmark_v1_candidates import DEFAULT_DIR, TRIAL_PAIRS, _identity_rows


def validate(pair_id: str) -> dict:
    i, j = (int(part) for part in pair_id.split("_"))
    identities = _identity_rows()
    left, right = identities[i], identities[j]
    store = CandidateReviewStore(DEFAULT_DIR / f"{pair_id}.csv", pair_id,
                                 left["scene_id"], right["scene_id"])
    with rasterio.open(left["source_path"]) as a, rasterio.open(right["source_path"]) as b:
        wa, wb = overlap_window(a, b)
        quadrants = set()
        cells = set()
        for row in store.rows:
            if row["recommendation_status"] != "SUGGESTED":
                raise ValueError("Unexpected recommendation status")
            for ds, suffix in ((a, "i"), (b, "j")):
                rr, cc = float(row[f"row_{suffix}"]), float(row[f"col_{suffix}"])
                ir, ic = int(round(rr)), int(round(cc))
                if not (0 <= ir < ds.height and 0 <= ic < ds.width):
                    raise ValueError("Candidate outside source bounds")
                if ds.read_masks(1, window=Window(ic, ir, 1, 1))[0, 0] == 0:
                    raise ValueError("Candidate on NoData")
                x, y = pixel_to_projected(ds.transform, rr, cc)
                if not np.allclose([x, y], [float(row[f"x_{suffix}"]), float(row[f"y_{suffix}"])], atol=1e-6):
                    raise ValueError("Candidate projected coordinate mismatch")
            local_r = float(row["row_i"]) - wa.row_off
            local_c = float(row["col_i"]) - wa.col_off
            quadrants.add((int(local_r >= wa.height/2), int(local_c >= wa.width/2)))
            cells.add(row["grid_cell_id"])
        ids = [r["candidate_id"] for r in store.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate ID")
    return {"pair_id": pair_id, "candidates": len(store.rows),
            "occupied_grid_cells": len(cells), "quadrants": len(quadrants),
            "review_statuses": {state: sum(r["human_review_status"] == state for r in store.rows)
                                for state in sorted({r["human_review_status"] for r in store.rows})},
            "native_mask_and_affine_check": "PASS"}


if __name__ == "__main__":
    print(json.dumps([validate(pair) for pair in TRIAL_PAIRS], ensure_ascii=False, indent=2))
