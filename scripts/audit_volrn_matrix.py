"""Audit the VOLRN normal-equation matrix without changing its scale."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def _scale_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _build_from_block_details(block_details: list[dict], band: int, rho: float):
    blocks = sorted((item for item in block_details if int(item.get("band", 0)) == band), key=lambda item: int(item["block_id"]))
    if not blocks:
        raise ValueError(f"no block details for band {band}")
    block_by_id = {int(item["block_id"]): item for item in blocks}
    groups: dict[tuple[int, int], list[int]] = {}
    for item in blocks:
        groups.setdefault((int(item["grid_m"]), int(item["grid_n"])), []).append(int(item["block_id"]))
    pairs = []
    for block_ids in groups.values():
        for first, second in itertools.combinations(sorted(block_ids), 2):
            pairs.append((first, second))

    n_blocks = len(blocks)
    b_rows, b_cols, b_data = [], [], []
    for pair_idx, (first, second) in enumerate(pairs):
        left = block_by_id[first]
        right = block_by_id[second]
        row_mu = 2 * pair_idx
        b_rows.extend([row_mu] * 4)
        b_cols.extend([2 * first, 2 * first + 1, 2 * second, 2 * second + 1])
        b_data.extend([float(left["mu"]), 1.0, -float(right["mu"]), -1.0])
        row_sigma = row_mu + 1
        b_rows.extend([row_sigma, row_sigma])
        b_cols.extend([2 * first, 2 * second])
        b_data.extend([float(left["sigma"]), -float(right["sigma"])])
    B = sparse.csr_matrix((b_data, (b_rows, b_cols)), shape=(2 * len(pairs), 2 * n_blocks))

    a_rows, a_cols, a_data = [], [], []
    b_vec = np.zeros(2 * n_blocks)
    for item in blocks:
        block_id = int(item["block_id"])
        row_mu = 2 * block_id
        a_rows.extend([row_mu, row_mu])
        a_cols.extend([2 * block_id, 2 * block_id + 1])
        a_data.extend([float(item["mu"]), 1.0])
        b_vec[row_mu] = float(item["mu"])
        row_sigma = row_mu + 1
        a_rows.append(row_sigma)
        a_cols.append(2 * block_id)
        a_data.append(float(item["sigma"]))
        b_vec[row_sigma] = float(item["sigma"])
    A = sparse.csr_matrix((a_data, (a_rows, a_cols)), shape=(2 * n_blocks, 2 * n_blocks))
    M = (B.T @ B + rho * (A.T @ A)).tocsr()
    return blocks, pairs, B, A, M, b_vec


def audit_volrn_matrix(summary_path: str | Path, output_path: str | Path, band: int = 0) -> dict:
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = summary["volrn"]["diagnostics"]
    rho = float(diagnostics.get("rho", 1.0))
    blocks, pairs, B, A, M, _ = _build_from_block_details(diagnostics["block_details"], band, rho)
    dense_m = M.toarray()
    row_norms = np.linalg.norm(dense_m, axis=1)
    column_norms = np.linalg.norm(dense_m, axis=0)
    result = {
        "schema_version": 1,
        "source_summary": str(summary_path),
        "band": int(band),
        "num_blocks": len(blocks),
        "num_pairs": len(pairs),
        "num_variables": int(M.shape[1]),
        "matrix_shape": [int(M.shape[0]), int(M.shape[1])],
        "matrix_density": float(M.nnz / (M.shape[0] * M.shape[1])),
        "B_density": float(B.nnz / (B.shape[0] * B.shape[1])) if B.shape[0] else 0.0,
        "A_density": float(A.nnz / (A.shape[0] * A.shape[1])),
        "condition_estimate": float(np.linalg.cond(dense_m)),
        "row_scale": _scale_stats(row_norms),
        "column_scale": _scale_stats(column_norms),
        "rho": rho,
        "normalization_applied": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--band", type=int, default=0)
    args = parser.parse_args()
    audit_volrn_matrix(args.summary, args.output, band=args.band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
