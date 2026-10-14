from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_radiometric_r1.synthetic import generate_case  # noqa: E402


def _synthetic_source(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    base = 100.0 + 0.35 * x + 0.6 * y + 12.0 * np.sin(x / 7.0) * np.cos(y / 9.0)
    return np.stack((base, 0.7 * base + 25, 1.3 * base - 10), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bounded synthetic R1 cases; no normalization algorithm is run.")
    parser.add_argument("--size", type=int, default=None, help="square raster size (default from frozen YAML, max 512)")
    parser.add_argument("--seed", type=int, default=None, help="override the frozen default seed for an explicitly new run")
    args = parser.parse_args()
    config_path = ROOT / "docs/benchmark/v1/radiometric_r1/synthetic_cases.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    size = int(args.size or config["default_size"])
    if not 8 <= size <= int(config["max_size"]):
        parser.error(f"--size must be in [8,{config['max_size']}] to bound memory")
    seed = int(args.seed if args.seed is not None else config["seed"])
    output_dir = ROOT / "data/output/benchmark_v1_radiometric_r1/synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _synthetic_source(size)
    valid = np.ones((size, size), dtype=bool)
    valid[: min(2, size), : min(3, size)] = False

    for case_id, raw_case in config["cases"].items():
        case = dict(raw_case)
        if case_id == "S4":
            fractions = case.pop("change_region_fraction")
            h0, h1, w0, w1 = fractions
            case["change_region"] = [int(h0 * size), int(h1 * size), int(w0 * size), int(w1 * size)]
        case["case"] = case_id
        input_source = source if case_id == "S3" else source[0]
        result = generate_case(input_source, valid, case, seed)
        path = output_dir / f"{case_id}_seed{seed}_size{size}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing synthetic output: {path}")
        arrays = {
            "case_id": np.asarray(case_id), "seed": np.asarray(seed),
            "clean_target": result.clean_target, "observed": result.observed,
            "valid_mask": result.valid_mask, "change_mask": result.change_mask,
            "change_amplitude": result.change_amplitude, "saturation_mask": result.saturation_mask,
            "source_numeric_units": np.asarray("SYNTHETIC_ARBITRARY_UNITS"),
        }
        arrays.update({f"truth_{key}": value for key, value in result.perturbation.items()})
        np.savez_compressed(path, **arrays)
        print(f"{case_id}: {path.relative_to(ROOT)} shape={result.observed.shape} support={int(result.valid_mask.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
