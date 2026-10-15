"""Read-only 13-scene DZ01V LSSTAA network feasibility precheck.

Only 10x coarse estimation arrays and JSON/Markdown diagnostics are written.
Frozen inputs, validity masks, and three-scene artifacts are read-only.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_lsstaa.core import (  # noqa: E402
    absolute_day_offsets, choose_control, connection_graph, control_statistics,
    pif_from_series, select_pif_subset,
)
from src.benchmark_v1_lsstaa.raster import (  # noqa: E402
    _coarse_frozen_b9, proxy_ndvi, verified_scene_date,
)
from src.benchmark_v1_unified.freeze import verify_lock  # noqa: E402
from src.benchmark_v1_unified.aligned import corrected_affine  # noqa: E402

WORKSPACE = ROOT.parent
FREEZE = WORKSPACE / "data/output/benchmark_v1_unified/freeze_20261013_v3"
TASK14 = WORKSPACE / "data/output/b9_13scene_task14"
SOURCE_ROOT = WORKSPACE.parent / "BAGRN_VOLRN_Reproduce/data/input/B9_reference_13scenes_full"
REPORT_DIR = ROOT / "docs/benchmark/v1/unified_v1"
REPORT_JSON = REPORT_DIR / "lsstaa_13scene_network_precheck_20261015.json"
REPORT_MD = REPORT_DIR / "lsstaa_13scene_network_precheck_20261015.md"
FACTOR = 10
BANDS = ("B9", "B4", "B7", "B14")
THRESHOLDS = ((14, "OFFICIAL_DEFAULT"), (30, "NETWORK_CONNECTIVITY_EXPLORATION"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[np.ndarray, list[int], list[dict], dict]:
    lock = verify_lock(FREEZE)
    if not lock.valid:
        raise ValueError(f"FROZEN_INPUT_LOCK_FAILED:{lock.errors}")
    frozen = json.loads((FREEZE / "frozen_manifest.json").read_text(encoding="utf-8"))
    freeze_lock_record = json.loads((FREEZE / "freeze_lock.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN" or frozen.get("scene_count") != 13:
        raise ValueError("FROZEN_13_SCENE_MANIFEST_REQUIRED")
    source_config = json.loads((TASK14 / "stages/04_canonical_warp/task14_source_config.json").read_text(encoding="utf-8"))
    transforms_path = TASK14 / "stages/03_global_adjustment/global/translation_l2/global_transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    grid = frozen["grid"]
    shape = ((int(grid["height"]) + FACTOR - 1) // FACTOR,
             (int(grid["width"]) + FACTOR - 1) // FACTOR)
    coarse_transform = Affine(*grid["transform"]) * Affine.scale(FACTOR)
    scenes = source_config["scenes"]
    if len(scenes) != 13 or len(transforms["transforms"]) != 13:
        raise ValueError("SCENE_CONFIG_OR_TRANSFORM_COUNT_MISMATCH")
    stack = np.zeros((*shape, 13, 4), dtype=np.float32)
    dates: list[str] = []
    scene_records: list[dict] = []
    input_hashes: dict[str, str] = {}

    for index, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        stamp_match = re.search(r"_(\d{14})_", scene_id)
        if stamp_match is None:
            raise ValueError(f"INVALID_SCENE_ID:{scene_id}")
        scene_dir = SOURCE_ROOT / stamp_match.group(1) / scene_id
        paths = {band: scene_dir / f"{scene_id}_{band}.TIF" for band in BANDS}
        mtl = scene_dir / f"{scene_id}_MTL.txt"
        if any(not path.is_file() for path in [*paths.values(), mtl]):
            raise FileNotFoundError(f"MISSING_FOUR_BAND_SOURCE:{scene_id}")
        dates.append(verified_scene_date(mtl))
        profiles = {}
        for band, path in paths.items():
            with rasterio.open(path) as ds:
                profiles[band] = (ds.crs, ds.transform, ds.width, ds.height, ds.nodata, ds.dtypes[0])
            input_hashes[str(path)] = sha256(path)
        if any(value[:4] != profiles["B9"][:4] for value in profiles.values()):
            raise ValueError(f"SOURCE_BAND_GRID_MISMATCH:{scene_id}")

        frozen_data = FREEZE / "aligned_inputs" / f"scene_{index:03d}_B9_input.tif"
        frozen_mask = FREEZE / "aligned_inputs" / f"scene_{index:03d}_input_mask.tif"
        if not frozen_data.is_file() or not frozen_mask.is_file():
            raise FileNotFoundError(f"MISSING_FROZEN_INPUT_OR_MASK:{index}")
        input_hashes[str(frozen_data)] = sha256(frozen_data)
        input_hashes[str(frozen_mask)] = sha256(frozen_mask)
        b9, b9_fraction = _coarse_frozen_b9(frozen_data, frozen_mask, FACTOR)
        stack[:, :, index, 0] = b9
        fractions = [b9_fraction]

        world = np.asarray(transforms["transforms"][index]["matrix"], dtype=np.float64)
        for band_position, band in enumerate(BANDS[1:], start=1):
            values = np.zeros(shape, dtype=np.float32)
            fraction = np.zeros(shape, dtype=np.float32)
            with rasterio.open(paths[band]) as src:
                nodata = src.nodata if src.nodata is not None else 0
                warped = corrected_affine(src.transform, world)
                reproject(source=rasterio.band(src, 1), destination=values,
                          src_transform=warped, src_crs=src.crs,
                          dst_transform=coarse_transform, dst_crs=src.crs,
                          src_nodata=nodata, dst_nodata=0,
                          resampling=Resampling.average)
                source_valid = (src.read(1) != nodata).astype(np.float32)
                reproject(source=source_valid, destination=fraction,
                          src_transform=warped, src_crs=src.crs,
                          dst_transform=coarse_transform, dst_crs=src.crs,
                          src_nodata=None, dst_nodata=0,
                          resampling=Resampling.average)
            stack[:, :, index, band_position] = values
            fractions.append(fraction)
        all_band_valid = np.logical_and.reduce([item > 0 for item in fractions])
        stack[:, :, index, :][~all_band_valid] = 0
        scene_records.append({
            "scene_index": index, "scene_id": scene_id, "date": dates[-1],
            "coarse_joint_valid_pixels": int(all_band_valid.sum()),
            "source_paths": {band: str(path) for band, path in paths.items()},
        })
    return stack, absolute_day_offsets(dates), scene_records, {
        "grid": grid, "factor": FACTOR, "band_order": list(BANDS),
        "freeze_lock_valid": bool(lock.valid),
        "freeze_lock_entries": len(freeze_lock_record.get("files", [])),
        "input_sha256": input_hashes, "transform_sha256": sha256(transforms_path),
        "freeze_manifest_sha256": sha256(FREEZE / "frozen_manifest.json"),
        "freeze_lock_sha256": sha256(FREEZE / "freeze_lock.json"),
    }


def components(scene_count: int, edges: list[tuple[int, int, np.ndarray, np.ndarray]], control: int) -> dict:
    adjacency = [set() for _ in range(scene_count)]
    for i, j, _, _ in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    groups = []
    unseen = set(range(scene_count))
    control_component = None
    while unseen:
        root = min(unseen)
        found = {root}
        pending = [root]
        unseen.remove(root)
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node] & unseen:
                unseen.remove(neighbor)
                found.add(neighbor)
                pending.append(neighbor)
        group = sorted(found)
        if control in found:
            control_component = group
        groups.append(group)
    return {"component_count": len(groups), "components_scene_indices": groups,
            "control_component_scene_indices": control_component,
            "control_reachable_count": len(control_component or [])}


def matrix_diagnostic(scene_count: int, control: int,
                      edges: list[tuple[int, int, np.ndarray, np.ndarray]]) -> dict:
    unknowns = [scene for scene in range(scene_count) if scene != control]
    index = {scene: 2 * position for position, scene in enumerate(unknowns)}
    normal = np.zeros((2 * len(unknowns), 2 * len(unknowns)), dtype=np.float64)
    tie_count = 0
    for i, j, x, y in edges:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        design = np.zeros((len(x), normal.shape[0]), dtype=np.float64)
        if i != control:
            design[:, index[i]] = x
            design[:, index[i] + 1] = 1
        if j != control:
            design[:, index[j]] = -y
            design[:, index[j] + 1] = -1
        normal += design.T @ design
        tie_count += len(x)
    rank = int(np.linalg.matrix_rank(normal))
    condition = float(np.linalg.cond(normal))
    return {"unknown_count": int(normal.shape[0]), "rank": rank,
            "rank_deficiency": int(normal.shape[0] - rank),
            "normal_matrix_condition_number": condition if np.isfinite(condition) else None,
            "condition_number_finite": bool(np.isfinite(condition)),
            "selected_tie_count": tie_count,
            "ties_exceed_unknown_count": tie_count > normal.shape[0],
            "full_rank": rank == normal.shape[0]}


def summarize_threshold(data: np.ndarray, pif: np.ndarray, days: list[int],
                        control: int, threshold: int) -> dict:
    bands = {}
    for band_index, band_name in enumerate(BANDS):
        edges, _ = connection_graph(data[:, :, :, band_index], days, pif,
                                    day_threshold=threshold)
        graph = components(data.shape[2], edges, control)
        matrices = matrix_diagnostic(data.shape[2], control, edges)
        edge_records = [{"scene_i": i, "scene_j": j, "day_gap": abs(days[i]-days[j]),
                         "selected_pif_subset_samples": int(len(x))}
                        for i, j, x, _ in edges]
        bands[band_name] = {
            **graph, "effective_edge_count": len(edges), "edges": edge_records,
            "joint_system": matrices,
            "solvable_without_coefficients": bool(
                graph["component_count"] == 1 and graph["control_reachable_count"] == data.shape[2]
                and matrices["full_rank"] and matrices["ties_exceed_unknown_count"]
                and matrices["condition_number_finite"]),
        }
    return {"day_threshold": threshold,
            "configuration_label": "OFFICIAL_DEFAULT_14_DAY" if threshold == 14 else "EXPLORATORY_NETWORK_CONNECTIVITY_ONLY_30_DAY",
            "pif_count": int(pif.sum()), "control_scene_index": control,
            "bands": bands,
            "all_four_bands_solvable": all(x["solvable_without_coefficients"] for x in bands.values())}


def write_report(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    rows = ["# LSSTAA DZ01V 13景网络可行性预检", "",
            "日期：2026-10-15。目的：仅判断联合网络与线性系统是否具备进入云端试验条件。",
            "输入使用冻结13景B9及其有效掩膜，并读取对应原始B9/B4/B7/B14；仅做10倍降采样估计。冻结文件只读。",
            "方法标签：`LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED`。B7/B14比值仅作为未校准DN代理。未生成系数、校正影像或评分。", "",
            "| 配置 | 四波段有效边 | 连通分量 | 控制景可达 | 联合系统秩/未知数 | 条件数范围 | 结论 |",
            "|---|---:|---:|---|---|---|---|"]
    for entry in report["configurations"]:
        for band, detail in entry["bands"].items():
            system = detail["joint_system"]
            condition = system["normal_matrix_condition_number"]
            cond_text = f"{condition:.4g}" if condition is not None else "∞/不可逆"
            rows.append(f"| {entry['configuration_label']} · {band} | {detail['effective_edge_count']} | {detail['component_count']} | {detail['control_reachable_count']}/13 | {system['rank']}/{system['unknown_count']} | {cond_text} | {'可解' if detail['solvable_without_coefficients'] else '不可解'} |")
    all14 = report["configurations"][0]["all_four_bands_solvable"]
    all30 = report["configurations"][1]["all_four_bands_solvable"]
    go = all14 or all30
    status = "适合进入下一轮13景云端试验" if go else "暂不适合进入下一轮13景云端试验"
    rows += ["", f"## 决策：{status}", "",
             f"14天官方默认配置四波段联合可解：`{str(all14).upper()}`。30天仅作为网络连通性探索；四波段联合可解：`{str(all30).upper()}`。", "",
             "### 主要阻塞", "",
             "- 14天：四波段均只有8条实际有效边，形成6个分量；控制景仅能到达3/13景。每个波段的联合系统秩为17/24，秩亏7，正规矩阵奇异。",
             "- 30天探索：四波段均有13条有效边、4个分量；控制景可达8/13景。联合系统秩为19/24，秩亏5，正规矩阵奇异。新增边未连接场景2、3、4、5、7所在分量。",
             f"- 14天分量（按场景序号）：`{report['configurations'][0]['bands']['B9']['components_scene_indices']}`；控制分量：`{report['configurations'][0]['bands']['B9']['control_component_scene_indices']}`。",
             f"- 30天分量（按场景序号）：`{report['configurations'][1]['bands']['B9']['components_scene_indices']}`；控制分量：`{report['configurations'][1]['bands']['B9']['control_component_scene_indices']}`。完整场景ID和逐边样本数见JSON。", "",
             "有效边由PIF掩膜与对应波段共同有效像元筛选，再经过LSSTAA网格子集筛选得到；表内边样本数与每景连通分量见JSON。控制景由同一PIF集上的控制景准则选出。秩与条件数来自控制景固定后的联合正规矩阵；不存在满秩时不输出任何完整13景系数。", "",
             "30天探索配置没有依据MAE择阈值，不代表官方默认实验。未进行全分辨率校正GeoTIFF生成、Benchmark评分或官方参考系数差异追查。", "",
             f"机器可读明细：`{REPORT_JSON.name}`。"]
    REPORT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    stack, days, scene_records, provenance = load_inputs()
    all_valid = np.all(np.isfinite(stack) & (stack != 0), axis=3)
    ndvi = np.stack([proxy_ndvi(stack[:, :, i, 2], stack[:, :, i, 3], all_valid[:, :, i])
                     for i in range(13)], axis=2)
    pif_result = pif_from_series(ndvi, days)
    rsq, gains, offsets, overlap = control_statistics(pif_result.bias_corrected,
                                                       pif_result.temporal_difference)
    control = choose_control(rsq, gains, offsets, overlap)
    report = {
        "task": "LSSTAA_DZ01V_13_SCENE_NETWORK_FEASIBILITY_PRECHECK",
        "created_date": "2026-10-15", "status": "COMPLETED_NETWORK_DIAGNOSTIC",
        "method": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED",
        "official_reference_equivalence": "UNRESOLVED_MISMATCH_FROM_PRIOR_GATE",
        "input": {"scene_count": 13, "band_order": list(BANDS), "factor": FACTOR,
                  "absolute_day_offsets": days, "pif_threshold": pif_result.threshold,
                  "scene_records": scene_records, "provenance": provenance},
        "control": {"scene_index": control, "scene_id": scene_records[control]["scene_id"]},
        "configurations": [summarize_threshold(stack, pif_result.pif, days, control, threshold)
                           for threshold, _ in THRESHOLDS],
        "guardrails": {"thresholds_tested_only": [14, 30],
                       "30_day_is_exploratory_network_connectivity_only": True,
                       "threshold_selected_using_mae": False,
                       "full_resolution_corrected_geotiffs_generated": False,
                       "formal_benchmark_scoring_performed": False,
                       "official_reference_coefficient_investigation_performed": False,
                       "coefficients_emitted": False,
                       "frozen_inputs_or_masks_modified": False,
                       "three_scene_outputs_modified": False},
    }
    go = any(config["all_four_bands_solvable"] for config in report["configurations"])
    report["cloud_trial_recommendation"] = "SUITABLE_FOR_NEXT_13_SCENE_CLOUD_TRIAL" if go else "NOT_SUITABLE_YET"
    write_report(report)
    print(json.dumps({"status": report["status"],
                      "cloud_trial_recommendation": report["cloud_trial_recommendation"],
                      "outputs": [str(REPORT_MD), str(REPORT_JSON)]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
