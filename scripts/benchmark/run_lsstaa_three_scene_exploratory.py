"""Fixed 00/01/02 diagnostic; never writes into the frozen Benchmark."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import psutil
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.freeze import verify_lock
from src.benchmark_v1_unified.structure import compute_sobel_structure, compute_range_retention
from src.benchmark_v1_lsstaa.raster import load_three_scene_estimation
from src.benchmark_v1_lsstaa.experiment import derive_model, write_corrected_scene
from src.benchmark_v1_lsstaa.scoring import score_pair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_manifest_check(freeze: Path) -> int:
    rows = json.loads((freeze / "code_manifest.json").read_text(encoding="utf-8"))["files"]
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"FROZEN_CODE_MISMATCH:{row['path']}")
    return len(rows)


def compare_masks(frozen: Path, output: Path) -> int:
    with rasterio.open(frozen) as original, rasterio.open(output) as result:
        if (original.width, original.height, original.transform, original.crs) != (result.width, result.height, result.transform, result.crs):
            raise ValueError("OUTPUT_MASK_GRID_MISMATCH")
        count = 0
        for row in range(0, original.height, 512):
            for col in range(0, original.width, 512):
                w = Window(col, row, min(512, original.width-col), min(512, original.height-row))
                a = original.read(1, window=w) != 0
                b = result.read(1, window=w) != 0
                if not np.array_equal(a, b):
                    raise ValueError("OUTPUT_MASK_CHANGED")
                count += int(a.sum())
    return count


def structure_diagnostic(raw: tuple[Path, Path], corrected: tuple[Path, Path], windows: list[dict]) -> dict:
    scores = []
    with rasterio.open(raw[0]) as a, rasterio.open(raw[1]) as mask, rasterio.open(corrected[0]) as b:
        for record in windows:
            w = Window(int(record["col_off"]), int(record["row_off"]), int(record["width"]), int(record["height"]))
            support = mask.read(1, window=w) != 0
            original = a.read(1, window=w)
            output = b.read(1, window=w)
            structure = compute_sobel_structure(original, output, support)
            dynamic = compute_range_retention(original, output, support)
            scores.append({"window_id": record["window_id"], "structure": asdict(structure),
                           "range": asdict(dynamic)})
    return {"status": "DIAGNOSTIC_WINDOW_STRUCTURE", "windows": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT.parent / "data/output/benchmark_v1_unified/freeze_20261013_v3")
    parser.add_argument("--task15-root", type=Path, default=ROOT.parent / "data/output/final_pipeline/b9_13scene_formal_v2")
    parser.add_argument("--source-root", type=Path, default=Path("D:/科研/地质一号/文献/BAGRN_VOLRN_Reproduce/data/input/B9_reference_13scenes_full"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/output/benchmark_v1_lsstaa/three_scene_exploratory_20261015")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite exploratory output: {args.output}")
    report = {"official_numerical_status": "OFFICIAL_REFERENCE_MISMATCH",
              "foundation_status": "PASS", "three_scene_status": "RUNNING",
              "formal_13scene_status": "NOT_RUN", "method": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED",
              "fixed_scene_indices": [0, 1, 2], "fixed_bands": ["B9", "B4", "B7", "B14"],
              "factor": 10, "graph_day_threshold": 14, "stages_seconds": {}}
    started = time.perf_counter()
    proc = psutil.Process()
    samples = []
    stop = threading.Event()
    def sample_rss() -> None:
        while not stop.wait(.2):
            try:
                samples.append(proc.memory_info().rss)
            except psutil.Error:
                break
    watcher = threading.Thread(target=sample_rss, daemon=True)
    watcher.start()
    args.output.mkdir(parents=True)
    try:
        t = time.perf_counter()
        lock = verify_lock(args.freeze)
        if not lock.valid:
            raise ValueError(f"FREEZE_LOCK_FAILED:{lock.errors}")
        report["freeze_lock_entries"] = len(json.loads((args.freeze / "freeze_lock.json").read_text(encoding="utf-8"))["files"])
        report["code_manifest_entries"] = code_manifest_check(args.freeze)
        memory = psutil.virtual_memory().available
        import shutil
        disk = shutil.disk_usage(args.output).free
        report["preflight"] = {"available_ram_bytes": memory, "free_disk_bytes": disk}
        if memory < 1_000_000_000 or disk < 5_000_000_000:
            raise RuntimeError("RESOURCE_PREFLIGHT_FAILED")
        report["stages_seconds"]["preflight"] = time.perf_counter()-t
        t = time.perf_counter()
        stack, days, adaptation = load_three_scene_estimation(args.freeze, args.source_root, args.task15_root)
        report["adaptation"] = adaptation
        report["absolute_day_offsets"] = days
        report["stages_seconds"]["load_estimation_stack"] = time.perf_counter()-t
        t = time.perf_counter()
        model = derive_model(stack, days)
        report["model"] = model
        del stack
        report["stages_seconds"]["pif_and_joint_solve"] = time.perf_counter()-t
        print(json.dumps({"model": model, "days": days}), flush=True)
        frozen = [
            (args.freeze / "aligned_inputs" / f"scene_{i:03d}_B9_input.tif",
             args.freeze / "aligned_inputs" / f"scene_{i:03d}_input_mask.tif")
            for i in range(3)
        ]
        corrected = []
        for i in range(3):
            t = time.perf_counter()
            corrected.append(write_corrected_scene(*frozen[i], args.output / "corrected", i,
                                                   model["gains_b9"][i], model["offsets_b9"][i]))
            valid_count = compare_masks(frozen[i][1], corrected[i][1])
            report.setdefault("scene_outputs", []).append({"scene_index": i,
                "data_path": str(corrected[i][0]), "mask_path": str(corrected[i][1]),
                "valid_pixels": valid_count, "data_sha256": sha256_file(corrected[i][0]),
                "mask_sha256": sha256_file(corrected[i][1])})
            report["stages_seconds"][f"write_scene_{i:02d}"] = time.perf_counter()-t
        all_windows = list(csv.DictReader((args.freeze / "fixed_windows.csv").open(encoding="utf-8-sig", newline="")))
        scores = {}
        structures = {}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            key = f"{i:02d}_{j:02d}"
            records = [r for r in all_windows if r["pair_id"] == key]
            t = time.perf_counter()
            raw_score = score_pair(frozen[i], frozen[j], records)
            normalized_score = score_pair(corrected[i], corrected[j], records)
            if raw_score["support_count"] != normalized_score["support_count"]:
                raise ValueError(f"CHANGED_PAIR_SUPPORT:{key}")
            scores[key] = {"raw": raw_score, "corrected": normalized_score,
                           "frozen_window_count": len(records)}
            structures[key] = {str(scene): structure_diagnostic(frozen[scene], corrected[scene], records)
                               for scene in (i, j)}
            report["stages_seconds"][f"score_{key}"] = time.perf_counter()-t
            print(json.dumps({"pair": key, "raw": raw_score["b1"],
                              "corrected": normalized_score["b1"], "windows": len(records)}), flush=True)
        (args.output / "b1_b2_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        (args.output / "structure_diagnostics.json").write_text(json.dumps(structures, indent=2), encoding="utf-8")
        report["three_scene_status"] = "SUCCESS_DIAGNOSTIC"
    except Exception as exc:
        report["three_scene_status"] = "FAILED_" + type(exc).__name__.upper()
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        stop.set()
        watcher.join(timeout=2)
        report["rss_sample_count"] = len(samples)
        report["peak_rss_bytes"] = max(samples, default=None)
        report["elapsed_seconds"] = time.perf_counter()-started
        (args.output / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["three_scene_status"] == "SUCCESS_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
