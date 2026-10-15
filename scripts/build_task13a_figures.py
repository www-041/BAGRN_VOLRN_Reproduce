"""Deterministic, read-only diagnostic figures from the frozen Task13A batch.

The small cost panels recompute the *published cost formula* on the displayed
512-pixel window, including its own P95 normalization. They are visualizations,
not the full-overlap optimization costs. Paths, coefficients, and V1/V2 mosaics
come from the saved batch artifacts; no pair metrics or outputs are revised.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from src.seam_local.cost import compute_seam_cost


OUTPUT = ROOT / "data/output/b9_five_scene_validation/seam_local_task13a"
SIZE = 512


def choose_examples(rows: list[dict]) -> dict[str, tuple[str, float]]:
    """Choose endpoints and the median-nearest PASS pair, breaking ties by ID."""
    eligible = sorted(
        (float(row["metric_actual_seam_v2_mae"]), row["pair_id"])
        for row in rows
        if row["status"] == "PASS"
        and row.get("v1_status", "PASS") == "PASS"
        and row.get("v2_status", "PASS") == "PASS"
        and row.get("metric_actual_seam_v2_mae")
        and math.isfinite(float(row["metric_actual_seam_v2_mae"]))
    )
    if len(eligible) < 3:
        raise ValueError("at least three finite V1/V2 PASS pairs are required")
    values = [value for value, _ in eligible]
    median = float(np.median(values))
    middle = min(eligible, key=lambda item: (abs(item[0] - median), item[1]))
    return {"best": (eligible[0][1], eligible[0][0]),
            "median": (middle[1], middle[0]),
            "worst": (eligible[-1][1], eligible[-1][0])}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_pixels(path: Path, transform: rasterio.Affine) -> np.ndarray:
    coordinates = json.loads(path.read_text(encoding="utf-8"))["features"][0]["geometry"]["coordinates"]
    rows, cols = rasterio.transform.rowcol(transform, [pt[0] for pt in coordinates],
                                          [pt[1] for pt in coordinates])
    return np.column_stack((rows, cols)).astype(np.int32)


def _window_for_path(path: np.ndarray, pair_origin: tuple[int, int],
                     pair_shape: tuple[int, int], full_shape: tuple[int, int]) -> tuple[int, int]:
    center_r, center_c = map(int, path[len(path) // 2])
    row0 = int(np.clip(center_r - SIZE // 2, max(0, pair_origin[0]),
                       min(full_shape[0], pair_origin[0] + pair_shape[0]) - SIZE))
    col0 = int(np.clip(center_c - SIZE // 2, max(0, pair_origin[1]),
                       min(full_shape[1], pair_origin[1] + pair_shape[1]) - SIZE))
    return row0, col0


def reconstruct_local_window(
    a: np.ndarray, b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray,
    full_path: np.ndarray, coefficient_rows: list[dict], origin: tuple[int, int],
    orientation: str, *, half_width: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved line-interpolated coefficients to a display window only."""
    ca, cb = np.asarray(a, float).copy(), np.asarray(b, float).copy()
    transverse_axis = 1 if orientation == "vertical" else 0
    line_axis = 1 - transverse_axis
    transverse = full_path[:, transverse_axis]
    arcs = np.zeros(len(full_path), dtype=float)
    arcs[1:] = np.cumsum(np.hypot(1.0, np.diff(transverse)))
    lines = full_path[:, line_axis]
    line_to_index = {int(line): k for k, line in enumerate(lines)}
    centers = np.array([float(row["center_arc"]) for row in coefficient_rows])
    matrix = np.array([[float(row[key]) for key in ("a_a", "b_a", "a_b", "b_b")]
                       for row in coefficient_rows])
    line_count = a.shape[line_axis]
    for local_line in range(line_count):
        global_line = origin[line_axis] + local_line
        k = line_to_index.get(global_line)
        if k is None:
            continue
        center = int(transverse[k]) - origin[transverse_axis]
        coefficients = [np.interp(arcs[k], centers, matrix[:, index]) for index in range(4)]
        lo = max(0, center - half_width)
        hi = min(a.shape[transverse_axis], center + half_width + 1)
        if lo >= hi:
            continue
        coords = np.arange(lo, hi)
        weights = .5 * (1.0 + np.cos(np.pi * np.abs(coords - center) / half_width))
        slot = (local_line, slice(lo, hi)) if orientation == "vertical" else (slice(lo, hi), local_line)
        for source, destination, valid, gain, offset in (
            (a, ca, valid_a, coefficients[0], coefficients[1]),
            (b, cb, valid_b, coefficients[2], coefficients[3]),
        ):
            active = valid[slot] & np.isfinite(source[slot])
            updated = destination[slot]
            updated[active] = source[slot][active] + weights[active] * (
                (gain - 1.0) * source[slot][active] + offset)
    return ca, cb


def _read_window(path: Path, row0: int, col0: int, *, pair_origin=(0, 0)) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as ds:
        window = Window(col0 - pair_origin[1], row0 - pair_origin[0], SIZE, SIZE)
        image = ds.read(1, window=window).astype(float)
        valid = ds.read_masks(1, window=window).astype(bool) & np.isfinite(image)
    if image.shape != (SIZE, SIZE):
        raise ValueError(f"display window outside {path}")
    return image, valid


def _full_grid_distance_crop(path: Path, row0: int, col0: int) -> np.ndarray:
    # The pair runner used canonical full-grid EDT; cropping before EDT would
    # change V0 weights at the display-window boundary.
    with rasterio.open(path) as ds:
        valid = ds.read_masks(1).astype(bool)
    distance = distance_transform_edt(valid)
    return distance[row0:row0 + SIZE, col0:col0 + SIZE].copy()


def _plot_pair(kind: str, pair_id: str, mae: float, row: dict, protocol: dict,
               output_path: Path) -> dict:
    pair_dir = OUTPUT / "pairs" / pair_id
    i, j = map(int, pair_id.split("_"))
    scene_a = ROOT / protocol["scenes"][i]["path"]
    scene_b = ROOT / protocol["scenes"][j]["path"]
    with rasterio.open(scene_a) as ds:
        full_shape, transform = ds.shape, ds.transform
    initial = _path_pixels(pair_dir / "seam_initial.geojson", transform)
    refined = _path_pixels(pair_dir / "seam_refined.geojson", transform)
    orientation = row["initial_seam_orientation"]
    pair_origin = tuple(json.loads(row["diagnostic_crop_origin"]))
    with rasterio.open(pair_dir / "v1_seam_only.tif") as ds:
        pair_shape = ds.shape
    row0, col0 = _window_for_path(initial, pair_origin, pair_shape, full_shape)
    a, valid_a = _read_window(scene_a, row0, col0)
    b, valid_b = _read_window(scene_b, row0, col0)
    v1, valid_v1 = _read_window(pair_dir / "v1_seam_only.tif", row0, col0,
                                pair_origin=pair_origin)
    v2, valid_v2 = _read_window(pair_dir / "v2_local_refined.tif", row0, col0,
                                pair_origin=pair_origin)
    with (pair_dir / "local_coefficients.csv").open(newline="", encoding="utf-8") as stream:
        coefficients = list(csv.DictReader(stream))
    ca, cb = reconstruct_local_window(a, b, valid_a, valid_b, initial,
                                      coefficients, (row0, col0), orientation)
    joint = valid_a & valid_b
    initial_cost = compute_seam_cost(a, b, joint)
    refined_cost = compute_seam_cost(ca, cb, joint)
    residual_change = (ca - cb) - (a - b)
    da = _full_grid_distance_crop(scene_a, row0, col0)
    db = _full_grid_distance_crop(scene_b, row0, col0)
    wa, wb = np.where(valid_a, da + 1e-6, 0.0), np.where(valid_b, db + 1e-6, 0.0)
    denom = wa + wb
    v0 = np.full((SIZE, SIZE), np.nan)
    np.divide(np.where(valid_a, a, 0) * wa + np.where(valid_b, b, 0) * wb,
              denom, out=v0, where=denom > 1e-12)
    union = valid_a | valid_b
    contrast = np.concatenate((a[valid_a], b[valid_b]))
    vmin, vmax = np.percentile(contrast, (2, 98))
    if vmax <= vmin:
        vmax = vmin + 1
    delta_scale = float(np.percentile(np.abs(residual_change[joint]), 95)) if joint.any() else 1.0
    delta_scale = max(delta_scale, 1e-6)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9.7), constrained_layout=True)
    figure.suptitle(f"Task13A {kind.upper()} pair {pair_id} | {row['status']} | V2 seam MAE {mae:.3f}\n"
                    f"B9, 512×512 display at row {row0}, col {col0} | "
                    f"shared intensity P2–P98 [{vmin:.1f}, {vmax:.1f}] | "
                    f"cost 0–1, local residual change ±{delta_scale:.1f}", fontsize=13)
    panels = [
        (a, valid_a, "BAGRN source A", "gray", vmin, vmax),
        (b, valid_b, "BAGRN source B", "gray", vmin, vmax),
        (initial_cost, joint, "Initial cost + saved seam", "magma", 0, 1),
        (residual_change, joint, "Local Δ(A−B) residual", "coolwarm", -delta_scale, delta_scale),
        (refined_cost, joint, "Refined cost + saved seam", "magma", 0, 1),
        (v0, union, "V0 weighted feather", "gray", vmin, vmax),
        (v1, valid_v1, "V1 seam only", "gray", vmin, vmax),
        (v2, valid_v2, "V2 local + refined seam", "gray", vmin, vmax),
    ]
    for axis, (image, mask, title, cmap, low, high) in zip(axes.flat, panels):
        axis.imshow(np.ma.masked_where(~mask, image), cmap=cmap, vmin=low, vmax=high,
                    interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis, path in ((axes[0, 2], initial), (axes[1, 0], refined)):
        axis.plot(path[:, 1] - col0, path[:, 0] - row0, color="cyan", linewidth=1.2)
        axis.set_xlim(-.5, SIZE - .5)
        axis.set_ylim(SIZE - .5, -.5)
    figure.savefig(output_path, dpi=145, facecolor="white")
    plt.close(figure)
    return {"pair_id": pair_id, "v2_actual_seam_mae": mae, "status": row["status"],
            "display_window_full_grid_row_col": [row0, col0], "display_window_size": SIZE,
            "shared_intensity_display_p2_p98": [float(vmin), float(vmax)],
            "residual_change_symmetric_p95": delta_scale,
            "png_sha256": _sha256(output_path)}


def main() -> None:
    metrics_path = OUTPUT / "pair_metrics.csv"
    protocol_path = OUTPUT / "protocol.json"
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
    row_by_id = {row["pair_id"]: row for row in rows}
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    examples = choose_examples(rows)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    selections = {}
    for kind in ("worst", "median", "best"):
        pair_id, mae = examples[kind]
        selections[kind] = _plot_pair(kind, pair_id, mae, row_by_id[pair_id], protocol,
                                       figure_dir / f"{kind}_pair.png")
        print(f"{kind}: {pair_id}, V2 seam MAE={mae:.6f}", flush=True)
    manifest = {
        "selection_rule": "Finite V1/V2 PASS; V2 actual-seam MAE descending/nearest six-pair median/ascending; pair ID breaks median ties",
        "median_rule": "numeric median of all eligible PASS V2 actual-seam MAEs",
        "source_pair_metrics_sha256": _sha256(metrics_path),
        "source_protocol_sha256": _sha256(protocol_path),
        "display_policy": "512x512 center of saved initial seam; same 2x4 panel order and range policy for all three; BAGRN P2-P98 shared within each figure; cost local P95 [0,1]; correction residual symmetric P95",
        "v0_policy": "exact canonical full-grid mask EDT weights matching the frozen pair runner",
        "local_policy": "saved segment coefficients and initial seam path only; 128px cosine taper applied to display window",
        "selections": selections,
    }
    (figure_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def check_only() -> None:
    """Check the frozen selection and saved PNG integrity without rerendering."""
    from PIL import Image

    metrics = OUTPUT / "pair_metrics.csv"
    protocol = OUTPUT / "protocol.json"
    figure_dir = OUTPUT / "figures"
    manifest = json.loads((figure_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_pair_metrics_sha256"] != _sha256(metrics):
        raise ValueError("figure selection metric provenance changed")
    if manifest["source_protocol_sha256"] != _sha256(protocol):
        raise ValueError("figure protocol provenance changed")
    with metrics.open(newline="", encoding="utf-8") as stream:
        expected = choose_examples(list(csv.DictReader(stream)))
    for kind, (pair_id, value) in expected.items():
        saved = manifest["selections"][kind]
        png = figure_dir / f"{kind}_pair.png"
        if saved["pair_id"] != pair_id or saved["v2_actual_seam_mae"] != value:
            raise ValueError(f"{kind} selection differs from the metric table")
        if _sha256(png) != saved["png_sha256"]:
            raise ValueError(f"{kind} PNG digest changed")
        with Image.open(png) as image:
            image.verify()
    print("verified three figures and frozen selection provenance")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-only"]:
        check_only()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: build_task13a_figures.py [--check-only]")
