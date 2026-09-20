"""Loop-closure diagnostics for non-tree edges in multi-scene registration.

Root-cause diagnosis only. This module never changes SIFT, LoFTR, RANSAC,
spanning-tree, or global-registration behaviour.  It reads the artefacts of an
existing run, re-runs exactly one problem pair to recover its inlier point
coordinates (pairwise summaries do not persist them), and compares the direct
pairwise transform with the MST-implied transform.

The point-level residual formula reproduces
``src.multiscene_sift.global_registration.global_consistency_diagnostics``
so results stay directly comparable with the numbers the runner reports.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.multiscene_sift.band_geometry import pixel_affine_to_world
from src.multiscene_sift.models import PairwiseRegistration, Scene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 1 — Consistency / spanning-tree loading and path finding
# ---------------------------------------------------------------------------

CONSISTENCY_COLUMNS = {
    "idx_i": int,
    "idx_j": int,
    "in_tree": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "n_points": int,
    "global_median_px": float,
    "global_rmse_px": float,
    "global_p90_px": float,
    "global_p95_px": float,
    "global_max_px": float,
}


def load_consistency_rows(consistency_csv: str | Path) -> list[dict]:
    """Read ``global_edge_consistency.csv`` into a list of typed dicts."""
    path = Path(consistency_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Consistency file not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Consistency CSV has no header: {path}")
        missing = set(CONSISTENCY_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Consistency CSV {path.name} is missing columns {sorted(missing)}; "
                "is this a run produced by the current runner?"
            )
        rows = []
        for raw in reader:
            row = {}
            for name, cast in CONSISTENCY_COLUMNS.items():
                try:
                    row[name] = cast(raw[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Bad value {raw.get(name)!r} for column {name!r} in {path}"
                    ) from exc
            rows.append(row)
    return rows


def find_worst_non_tree_edge(
    consistency_rows: list[dict],
) -> tuple[int, int] | None:
    """Return the non-tree edge with the largest ``global_p95_px``.

    Returns ``None`` when no non-tree edge exists.
    """
    non_tree = [r for r in consistency_rows if not r["in_tree"]]
    if not non_tree:
        return None
    worst = max(non_tree, key=lambda r: float(r["global_p95_px"]))
    return int(worst["idx_i"]), int(worst["idx_j"])


def edge_consistency_record(
    consistency_rows: list[dict], i: int, j: int
) -> dict | None:
    """Return the consistency record for the undirected edge ``(i, j)``."""
    for r in consistency_rows:
        if {int(r["idx_i"]), int(r["idx_j"])} == {int(i), int(j)}:
            return r
    return None


def load_spanning_tree(path: str | Path) -> dict:
    """Load ``spanning_tree.json`` (structure written by the runner)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Spanning tree not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "edges" not in data:
        raise ValueError(f"{p.name} has no 'edges' key")
    return data


def find_tree_path(tree_edges: list[dict], start: int, end: int) -> list[int] | None:
    """Find the unique path between two nodes of an (undirected) tree.

    The MST is treated as an undirected tree even though ``spanning_tree.json``
    records directed ``parent``/``child`` fields.  Returns ``None`` when the
    two nodes are in different components.
    """
    adj: dict[int, set[int]] = {}
    for edge in tree_edges:
        p, c = int(edge["parent"]), int(edge["child"])
        adj.setdefault(p, set()).add(c)
        adj.setdefault(c, set()).add(p)

    # BFS from start.
    prev: dict[int, int | None] = {start: None}
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node == end:
            break
        for neighbour in sorted(adj.get(node, ())):
            if neighbour not in prev:
                prev[neighbour] = node
                queue.append(neighbour)

    if end not in prev:
        return None
    path: list[int] = []
    node: int | None = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


# ---------------------------------------------------------------------------
# Task 2 — Direct vs MST-implied transforms
# ---------------------------------------------------------------------------


def load_global_transforms(path: str | Path) -> list[np.ndarray]:
    """Load ``global_transforms.json`` into ``G`` (list of 3×3 world matrices)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Global transforms not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    transforms = {int(t["scene"]): np.asarray(t["matrix"], dtype=np.float64)
                  for t in data["transforms"]}
    if not transforms:
        raise ValueError(f"{p.name} contains no transforms")
    n = max(transforms) + 1
    G = [transforms.get(k) for k in range(n)]
    if any(g is None for g in G):
        missing = [k for k in range(n) if transforms.get(k) is None]
        raise ValueError(f"{p.name} is missing transforms for scenes {missing}")
    return [np.asarray(g, dtype=np.float64) for g in G]


def load_pairwise_results(path: str | Path) -> list[dict]:
    """Load ``pairwise_summary.json`` results list."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pairwise summary not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data["results"])


def find_pair_row(
    pair_rows: list[dict], i: int, j: int
) -> tuple[str, dict] | None:
    """Locate the pairwise row for the undirected edge ``(i, j)``.

    Returns ``("fwd", row)`` when the stored row has ``idx_i=i, idx_j=j``
    (target is ``j``), ``("inv", row)`` for the reversed orientation, and
    ``None`` if the pair is absent.
    """
    for row in pair_rows:
        ii, jj = int(row["idx_i"]), int(row["idx_j"])
        if ii == i and jj == j:
            return "fwd", row
        if ii == j and jj == i:
            return "inv", row
    return None


def pair_direct_world_transform(
    pair_rows: list[dict], i: int, j: int, common_transform,
) -> np.ndarray | None:
    """World-coordinate affine mapping ``scene j -> scene i`` from saved data.

    ``pixel_matrix`` in the pairwise summary maps common-grid ``tgt_xy ->
    ref_xy`` (see ``fit_affine_ransac``), so a stored ``(i, j)`` row directly
    gives ``j -> i`` and a ``(j, i)`` row must be inverted.
    """
    found = find_pair_row(pair_rows, i, j)
    if found is None:
        return None
    orientation, row = found
    M = np.asarray(row["pixel_matrix"], dtype=np.float64)
    A = pixel_affine_to_world(M, common_transform)
    if orientation == "fwd":
        return A
    return np.linalg.inv(A)


def registration_as_pair_row(reg: PairwiseRegistration) -> dict:
    """Convert a reproduced ``PairwiseRegistration`` to summary-row form."""
    return {
        "idx_i": reg.idx_i,
        "idx_j": reg.idx_j,
        "pixel_matrix": reg.pair_pixel_matrix,
        "raw_matches": reg.raw_matches,
        "inliers": reg.inliers,
        "inlier_ratio": reg.inlier_ratio,
        "coverage": reg.coverage,
        "residual_rmse": reg.residual_rmse,
        "residual_p95": reg.residual_p95,
        "status": reg.status,
    }


def mst_implied_transform(
    G: list[np.ndarray], i: int, j: int
) -> np.ndarray:
    """World-coordinate affine mapping ``scene j -> scene i`` via the tree.

    ``G[k]`` maps scene-k coordinates into the reference frame
    (``G_ref = I``), so ``inv(G_i) @ G_j`` maps a scene-j point into
    scene-i coordinates when the composed transforms are consistent.
    """
    return np.linalg.inv(G[i]) @ G[j]


def _decompose_affine(
    matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    """Decompose a 3×3 affine into ``(scale_x, scale_y, rot_deg, shear_deg)``.

    Formula matches ``src.registration_benchmark.geometry._decompose_affine``
    so the difference decomposition uses the same convention as pair quality.
    """
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    scale_x = float(math.hypot(a, c))
    scale_y = float(math.hypot(b, d))
    rot1 = math.atan2(c, a)
    rot2 = math.atan2(-b, d)
    rot = math.atan2(math.sin(rot1) + math.sin(rot2),
                     math.cos(rot1) + math.cos(rot2))
    shear = math.atan2(-b * a - d * c, a * d - b * c)
    return scale_x, scale_y, math.degrees(rot), math.degrees(shear)


def compare_direct_and_mst_transforms(
    T_direct: np.ndarray,
    T_mst: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float | None = None,
) -> dict:
    """Compare two ``j -> i`` world transforms and decompose their difference.

    Difference matrix::

        D = inv(T_direct) @ T_mst

    which is identity when both closure paths agree.  ``D`` acts on scene-i
    coordinates.  Its translation (reported both in world units and pixels)
    captures the overall positional shift between the two paths; its rotation /
    scale / shear capture spatially varying disagreement.
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
    D = np.linalg.inv(T_direct) @ T_mst
    scale_x, scale_y, rot_deg, shear_deg = _decompose_affine(D)
    tx, ty = float(D[0, 2]), float(D[1, 2])
    return {
        "difference_matrix": D.tolist(),
        "difference_matrix_semantics": (
            "D = inv(T_direct) @ T_mst acting on scene-i coordinates; "
            "identity when direct and MST paths agree."
        ),
        "interpretation": (
            "translation_* and the linear terms are read off D, which mixes the "
            "coordinate frames the two transforms live in (pair common grid vs "
            "global reference). A large translation here usually reflects the "
            "frame anchor offset, not a geometric shift; trust the point-level "
            "residuals (04/05/06) and the overlays for geometry."
        ),
        "direct_matrix": np.asarray(T_direct).tolist(),
        "mst_matrix": np.asarray(T_mst).tolist(),
        "translation_x_px": tx / pixel_size_x,
        "translation_y_px": ty / pixel_size_y,
        "translation_magnitude_px": math.hypot(
            tx / pixel_size_x, ty / pixel_size_y
        ),
        "translation_x_world": tx,
        "translation_y_world": ty,
        "translation_magnitude_world": math.hypot(tx, ty),
        "rotation_deg": float(rot_deg),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "shear_deg": float(shear_deg),
    }


# ---------------------------------------------------------------------------
# Task 3 — Pair reproduction (recover inlier points)
# ---------------------------------------------------------------------------


def load_run_config(run_dir: str | Path) -> dict:
    """Load the run's ``run_config.json``."""
    path = Path(run_dir) / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: str | Path) -> dict:
    """Load the run's ``dataset_manifest.json``."""
    path = Path(run_dir) / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset_manifest.json not found in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenes_from_manifest(manifest: dict) -> list[Scene]:
    """Reconstruct :class:`Scene` objects from a saved dataset manifest.

    ``dataset_manifest.json`` stores band paths plus raster metadata, so the
    scenes can be rebuilt without re-running discovery (useful when the run
    dir is moved).  Each band is reopened to read its full geotransform.
    """
    scenes = []
    for entry in manifest["scenes"]:
        band_paths: dict[str, str] = {}
        transforms: dict[str, object] = {}
        shapes: dict[str, tuple[int, int]] = {}
        nodatas: dict[str, float | None] = {}
        bounds: dict[str, tuple[float, float, float, float]] = {}
        crs = None
        for band, info in entry["bands"].items():
            path = str(info["path"])
            with rasterio.open(path) as src:
                crs = src.crs
                transforms[band] = src.transform
                shapes[band] = src.shape
                nodatas[band] = src.nodata
                bounds[band] = src.bounds
            band_paths[band] = path
        scenes.append(Scene(
            index=int(entry["index"]),
            name=str(entry["name"]),
            directory=str(entry.get("directory", "")),
            band_paths=band_paths,
            crs=crs,
            transforms=transforms,
            shapes=shapes,
            nodata=nodatas,
            bounds=bounds,
        ))
    if not scenes:
        raise ValueError("Dataset manifest contains no scenes")
    return scenes


def load_scenes_for_run(
    run_dir: str | Path,
    input_root: str | None = None,
    bands: tuple[str, ...] | None = None,
) -> list[Scene]:
    """Load scenes for a run, preferring fresh discovery over the manifest.

    When ``input_root`` is provided, ``discover_five_scenes`` is reused so
    re-running the problem pair matches the original experiment as closely as
    possible.  Otherwise the saved ``dataset_manifest.json`` is used.
    """
    config = load_run_config(run_dir)
    if bands is None:
        cfg_bands = config.get("bands") or [config.get("registration_band", "B14")]
        bands = tuple(dict.fromkeys(cfg_bands))
    if input_root:
        from src.multiscene_sift.dataset import discover_five_scenes

        scene_names = config.get("scene_names")
        if not scene_names:
            raise ValueError("run_config.json has no scene_names")
        return list(discover_five_scenes(input_root, list(scene_names), bands=bands))
    return load_scenes_from_manifest(load_manifest(run_dir))


def reproduce_pair(
    scenes: list[Scene],
    i: int,
    j: int,
    config: dict,
) -> PairwiseRegistration:
    """Re-run exactly one pair using the run's saved matcher parameters.

    All SIFT/LoFTR/RANSAC settings come from ``run_config.json`` so the
    reprod product is as close as possible to the original run.
    """
    from src.multiscene_sift.pairwise import register_pair

    matcher = str(config.get("matcher", "sift")).lower()
    band = str(config.get("registration_band", "B14"))
    return register_pair(
        scenes[i],
        scenes[j],
        band=band,
        match_max_side=int(config.get("match_max_side", 1600)),
        ransac_threshold=float(config.get("ransac_threshold", 2.0)),
        random_seed=int(config.get("random_seed", 0)),
        matcher=matcher,
    )


def compare_pair_to_saved(
    saved_result: dict | None,
    reg: PairwiseRegistration,
) -> dict:
    """Compare a reproduced pair with the saved summary row.

    Status meaning:
      ``exact``    — core counts and residuals agree within float tolerance.
      ``close``    — core counts agree, residuals differ only slightly.
      ``mismatch`` — the reproduction materially differs from the saved run.
    """
    fields = (
        "raw_matches",
        "inliers",
        "inlier_ratio",
        "coverage",
        "residual_rmse",
        "residual_p95",
    )
    if saved_result is None:
        return {
            "status": "no_saved_row",
            "differences": None,
            "warning": "No saved pairwise row for this edge; cannot assess reproduction.",
        }
    differences = {}
    for key in fields:
        saved = saved_result.get(key)
        reproduced = getattr(reg, key)
        if saved is None:
            continue
        if isinstance(saved, (int, float)):
            differences[key] = {"saved": float(saved),
                                "reproduced": float(reproduced)}
        else:
            differences[key] = {"saved": saved, "reproduced": reproduced}

    same_core = (
        int(saved_result.get("inliers", -1)) == reg.inliers
        and int(saved_result.get("raw_matches", -1)) == reg.raw_matches
    )
    tol = 1e-3
    residual_close = (
        abs(float(saved_result.get("residual_rmse", 0.0)) - reg.residual_rmse) < tol
        and abs(float(saved_result.get("residual_p95", 0.0)) - reg.residual_p95) < tol
    )
    if same_core and residual_close:
        status = "exact"
    elif same_core:
        status = "close"
    else:
        status = "mismatch"
    return {
        "status": status,
        "differences": differences,
        "saved_status": saved_result.get("status"),
        "warning": (
            "Reproduced pair differs materially from saved run. "
            "Point-level conclusions may not exactly represent the original run."
            if status == "mismatch"
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Task 4 — Point-level closure residuals
# ---------------------------------------------------------------------------


def pair_point_residuals_px(
    reg: PairwiseRegistration,
    G: list[np.ndarray],
    pixel_size_x: float,
    pixel_size_y: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-inlier closure residuals, identical formula to the runner.

    For every RANSAC inlier ``k`` of pair ``(i, j)``::

        P_i_world = pair_common_transform * ref_xy[k]
        P_j_world = pair_common_transform * tgt_xy[k]
        e_k       = || G_i @ P_i_world - G_j @ P_j_world ||   (per axis / res)

    This mirrors ``global_registration.global_consistency_diagnostics``,
    returning ``(dx_px, dy_px, error_px)`` vectors.
    """
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
    ref_pts = np.asarray(reg.inlier_ref_xy, dtype=np.float64)
    tgt_pts = np.asarray(reg.inlier_tgt_xy, dtype=np.float64)
    if ref_pts.size == 0 or len(ref_pts) != len(tgt_pts):
        raise ValueError("Pair has no recoverable inlier points")
    common = reg.pair_common_transform
    ref_world = np.array([common * (px, py) for px, py in ref_pts])
    tgt_world = np.array([common * (px, py) for px, py in tgt_pts])

    Gi, Gj = G[reg.idx_i], G[reg.idx_j]
    ref_h = np.hstack([ref_world, np.ones((len(ref_world), 1))])
    tgt_h = np.hstack([tgt_world, np.ones((len(tgt_world), 1))])
    ref_trans = (Gi @ ref_h.T).T[:, :2]
    tgt_trans = (Gj @ tgt_h.T).T[:, :2]

    dx_world = ref_trans[:, 0] - tgt_trans[:, 0]
    dy_world = ref_trans[:, 1] - tgt_trans[:, 1]
    dx_px = dx_world / pixel_size_x
    dy_px = dy_world / pixel_size_y
    error_px = np.sqrt(dx_px**2 + dy_px**2)
    return dx_px, dy_px, error_px


def residual_statistics(
    dx_px: np.ndarray, dy_px: np.ndarray, error_px: np.ndarray
) -> dict:
    """Aggregate summary statistics of the per-point residuals (in px)."""
    e = np.asarray(error_px, dtype=np.float64)
    dx = np.asarray(dx_px, dtype=np.float64)
    dy = np.asarray(dy_px, dtype=np.float64)
    n = len(e)

    def _stats(values: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    return {
        "n_points": int(n),
        "dx": _stats(dx),
        "dy": _stats(dy),
        "error": {
            **_stats(e),
            "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)),
            "max": float(np.max(e)) if n else float("nan"),
        },
    }


def write_point_residuals_csv(
    reg: PairwiseRegistration,
    dx_px: np.ndarray,
    dy_px: np.ndarray,
    error_px: np.ndarray,
    path: str | Path,
) -> None:
    """Write the point-level residual table ``04_point_residuals.csv``.

    ``ref_x/ref_y`` are the (reproduced) inlier coordinates in the pair's
    common-grid pixel space; ``dx_px/dy_px`` are estimated per the runner's
    global-consistency formula.
    """
    ref_pts = np.asarray(reg.inlier_ref_xy, dtype=np.float64)
    tgt_pts = np.asarray(reg.inlier_tgt_xy, dtype=np.float64)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["point_id", "ref_x", "ref_y", "tgt_x", "tgt_y",
             "dx_px", "dy_px", "error_px"]
        )
        for k in range(len(ref_pts)):
            writer.writerow([
                k,
                f"{ref_pts[k, 0]:.6f}", f"{ref_pts[k, 1]:.6f}",
                f"{tgt_pts[k, 0]:.6f}", f"{tgt_pts[k, 1]:.6f}",
                f"{dx_px[k]:.6f}", f"{dy_px[k]:.6f}", f"{error_px[k]:.6f}",
            ])


# ---------------------------------------------------------------------------
# Task 5 — Residual pattern classification (heuristic, not a conclusion)
# ---------------------------------------------------------------------------

# Heuristic cut-offs used only to label the observed pattern.  They are not
# treated as scientific conclusions; every underlying statistic is reported.
SYSTEMATIC_COHERENCE_MIN = 0.90
SYSTEMATIC_CV_MAX = 0.15
OUTLIER_P95_MEDIAN_RATIO = 6.0
OUTLIER_MAX_P95_RATIO = 3.0
OUTLIER_COHERENCE_MAX = 0.60
SPATIAL_R2_MIN = 0.30


def _spatial_fit_r2(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """R² of ``z ~ c0 + c1*x + c2*y`` (least squares)."""
    X = np.column_stack([np.ones_like(x), x, y])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    pred = X @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def classify_error_pattern(
    ref_xy: np.ndarray,
    dx_px: np.ndarray,
    dy_px: np.ndarray,
) -> dict:
    """Classify whether the residual is a systematic shift / spatial / outlier.

    ``ref_xy`` are the inlier reference positions (common-grid pixels) used
    for the spatial-fit check.  All raw statistics are returned alongside the
    label so the human can override the heuristic.
    """
    dx = np.asarray(dx_px, dtype=np.float64)
    dy = np.asarray(dy_px, dtype=np.float64)
    if len(dx) == 0:
        raise ValueError("Cannot classify an empty residual set")
    e = np.hypot(dx, dy)
    v = np.column_stack([dx, dy])

    v_mean = v.mean(axis=0)
    mean_mag = float(np.mean(np.linalg.norm(v, axis=1)))
    direction_coherence = (
        float(np.linalg.norm(v_mean)) / mean_mag if mean_mag > 1e-12 else 0.0
    )

    mu_e = float(np.mean(e))
    cv = float(np.std(e) / mu_e) if mu_e > 1e-12 else 0.0

    pts = np.asarray(ref_xy, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    r2 = {
        "dx": _spatial_fit_r2(x, y, dx),
        "dy": _spatial_fit_r2(x, y, dy),
    }

    median = float(np.median(e))
    p95 = float(np.percentile(e, 95))
    emax = float(np.max(e))
    p95_over_median = p95 / median if median > 1e-9 else math.inf
    max_over_p95 = emax / p95 if p95 > 1e-9 else math.inf

    classification = _classify_pattern(
        direction_coherence, cv, r2, p95_over_median, max_over_p95
    )
    return {
        "classification": classification,
        "direction_coherence": round(direction_coherence, 6),
        "error_cv": round(cv, 6),
        "spatial_r2": {k: round(vr, 6) for k, vr in r2.items()},
        "mean_error_px": round(mu_e, 6),
        "median_error_px": round(median, 6),
        "p95_error_px": round(p95, 6),
        "max_error_px": round(emax, 6),
        "p95_over_median": round(p95_over_median, 6),
        "max_over_p95": round(max_over_p95, 6),
        "heuristic_note": (
            "Classification is a heuristic aid; inspect the arrow plot and "
            "crops before drawing conclusions."
        ),
    }


def _classify_pattern(
    coherence: float,
    cv: float,
    r2: dict[str, float],
    p95_over_median: float,
    max_over_p95: float,
) -> str:
    """Apply the documented heuristic ordering to a residual pattern."""
    if coherence >= SYSTEMATIC_COHERENCE_MIN and cv <= SYSTEMATIC_CV_MAX:
        return "SYSTEMATIC_SHIFT_LIKELY"
    if (
        (p95_over_median >= OUTLIER_P95_MEDIAN_RATIO
         or max_over_p95 >= OUTLIER_MAX_P95_RATIO)
        and coherence < OUTLIER_COHERENCE_MAX
    ):
        return "OUTLIER_DRIVEN"
    if max(r2.get("dx", 0.0), r2.get("dy", 0.0)) >= SPATIAL_R2_MIN:
        return "SPATIAL_VARIATION_LIKELY"
    return "NOT_YET_DETERMINED"


# ---------------------------------------------------------------------------
# Task 9 — Final diagnosis state (evidence-driven, never hard-coded)
# ---------------------------------------------------------------------------

DEFAULT_PHRASE = "NOT_YET_DETERMINED"


def decide_final_state(
    transform_diff: dict | None,
    pattern: dict,
    global_p95_px: float,
    consistent_p95_px: float = 2.0,
) -> dict:
    """Pick a final diagnosis state from hard evidence only.

    The state ``DIRECT_EDGE_SUSPECT`` is intentionally never produced here:
    telling direct from MST error requires the human to inspect the overlays.
    When the transform comparison is unavailable (``transform_diff is None``)
    the answer is deliberately undecided.
    """
    if transform_diff is None:
        return {
            "final_state": DEFAULT_PHRASE,
            "reason": (
                "Direct/MST transform comparison is unavailable (the problem pair "
                "could not be re-run); no state can be concluded yet."
            ),
            "evidence": {
                "transform_consistent": None,
                "global_p95_px": float(global_p95_px),
                "pattern_class": pattern.get("classification", "NOT_YET_DETERMINED"),
            },
        }
    pattern_class = pattern.get("classification", "NOT_YET_DETERMINED")
    t_px = float(transform_diff.get("translation_magnitude_px", 0.0))
    rot = abs(float(transform_diff.get("rotation_deg", 0.0)))
    scale_drift = (
        abs(float(transform_diff.get("scale_x", 1.0)) - 1.0) > 5e-3
        or abs(float(transform_diff.get("scale_y", 1.0)) - 1.0) > 5e-3
    )
    shear_out = abs(float(transform_diff.get("shear_deg", 0.0))) > 0.3
    deforms = rot > 0.3 or scale_drift or shear_out

    # Warning: the matrix translation diff is unreliable when the two transforms
    # live in different coordinate frames (their per-matrix world offsets absorb
    # the frame anchor, not the geometry); trust the point-level residuals.
    frame_warning = (
        t_px > 100.0
        and pattern_class == "SYSTEMATIC_SHIFT_LIKELY"
        and float(global_p95_px) <= t_px * 0.5
    )

    evidence = {
        "transform_consistent": t_px < 1.0 and not deforms,
        "global_p95_px": float(global_p95_px),
        "translation_magnitude_px": t_px,
        "matrix_rotation_deg": rot,
        "matrix_scale_drift": bool(scale_drift),
        "matrix_shear_deg": abs(float(transform_diff.get("shear_deg", 0.0))),
        "pattern_class": pattern_class,
        "coordinate_frame_warning": frame_warning,
    }

    if evidence["transform_consistent"]:
        if float(global_p95_px) > consistent_p95_px:
            state = "METRIC_IMPLEMENTATION_SUSPECT"
            reason = (
                "Direct and MST-implied transforms agree (difference ~ identity) yet "
                "the global consistency error remains large; the mismatch likely lives "
                "in the metric or coordinate-conversion implementation, not geometry."
            )
        else:
            state = "CLOSURE_CONSISTENT"
            reason = "Direct and MST-implied transforms agree and the closure residual is small."
    elif pattern_class == "SPATIAL_VARIATION_LIKELY":
        state = "AFFINE_MODEL_SUSPECT"
        reason = (
            "The point-level residual pattern varies with position; a single affine "
            "model may not describe the overlap (or the paths disagree spatially)."
        )
    elif pattern_class == "SYSTEMATIC_SHIFT_LIKELY":
        state = DEFAULT_PHRASE
        if frame_warning:
            reason = (
                "The clean point-level pattern is ~%.1f px systematic shift (coherence "
                "%.2f). The large matrix translation (%.1f px) is treated as a "
                "coordinate-frame anchor difference, not a geometric one; visual "
                "inspection of the direct vs MST overlays is required to decide which "
                "side is wrong." % (float(global_p95_px),
                                   pattern.get("direction_coherence", float("nan")),
                                   t_px)
            )
        else:
            reason = (
                "Both paths largely agree in form but differ by an overall translation "
                "~%.1f px. Visual inspection of the direct vs MST overlays is required "
                "to decide which side is wrong." % t_px
            )
    elif pattern_class == "OUTLIER_DRIVEN":
        state = DEFAULT_PHRASE
        reason = (
            "Residual is driven by a few outlier points (median small, tail large); "
            "drops / bad matches should be reviewed before geometry conclusions."
        )
    else:
        state = DEFAULT_PHRASE
        reason = (
            "Residual pattern and transform difference do not pin down a single cause; "
            "inspect the overlays, arrow plot, and crops."
        )
    return {"final_state": state, "reason": reason, "evidence": evidence}


# ---------------------------------------------------------------------------
# JSON/CSV writing helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task 6–8 — Visualization (lazily imported matplotlib keeps tests light)
# ---------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _stretch_array(arr: np.ndarray, valid: np.ndarray,
                   p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Percentile-stretch a band to uint8 for display."""
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    out = np.clip(arr, lo, hi)
    out = (out - lo) / (hi - lo) * 255.0
    out[~valid] = 0.0
    return out.astype(np.uint8)


def _scene_to_screen_pixels(
    common_pixels: np.ndarray,
    common_transform,
    dest_transform,
) -> np.ndarray:
    """Map common-grid pixel coordinates into a destination image's pixels."""
    inv = ~dest_transform
    world = [common_transform * (px, py) for px, py in common_pixels]
    return np.array([inv * (wx, wy) for wx, wy in world])


def plot_error_vectors(
    scene0: Scene,
    scene1: Scene,
    band: str,
    reg: PairwiseRegistration,
    dx_px: np.ndarray,
    dy_px: np.ndarray,
    out_path: str | Path,
    arrow_scale: float = 5.0,
) -> Path:
    """Plot per-inlier closure error arrows on a scene-0 B14 background.

    Arrows are drawn with the quoted ``arrow_scale`` visual magnification and
    are labelled with the true (unscaled) pixel magnitudes.
    """
    plt = _matplotlib()
    dx = np.asarray(dx_px, dtype=np.float64)
    dy = np.asarray(dy_px, dtype=np.float64)
    ref_pts = np.asarray(reg.inlier_ref_xy, dtype=np.float64)
    err = np.hypot(dx, dy)

    transform0 = scene0.transforms[band]
    screen = _scene_to_screen_pixels(ref_pts, reg.pair_common_transform, transform0)
    xs, ys = screen[:, 0], screen[:, 1]

    shape = scene0.shapes[band]
    pad = 100
    c0 = max(0, int(np.floor(xs.min())) - pad)
    c1 = min(shape[1], int(np.ceil(xs.max())) + pad)
    r0 = max(0, int(np.floor(ys.min())) - pad)
    r1 = min(shape[0], int(np.ceil(ys.max())) + pad)
    if c1 <= c0 or r1 <= r0:
        c0, r0, c1, r1 = 0, 0, shape[1], shape[0]

    with rasterio.open(scene0.band_paths[band]) as src:
        data = src.read(1, window=((r0, r1), (c0, c1)))
        nodata = src.nodata
    valid = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        valid = data != nodata
    display = _stretch_array(data, valid)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(display, cmap="gray", vmin=0, vmax=255)
    for (x0, y0, u, v, mag) in zip(
        xs - c0, ys - r0, dx * arrow_scale, -dy * arrow_scale, err
    ):
        ax.arrow(x0, y0, u, v, color="red", width=1.5, head_width=10,
                 length_includes_head=True, alpha=0.9)
    ax.set_xlim(0, c1 - c0)
    ax.set_ylim(r1 - r0, 0)
    ax.set_title(
        f"Scene {reg.idx_i} vs {reg.idx_j}: per-inlier closure error "
        f"(median {np.median(err):.2f} px, max {np.max(err):.2f} px)"
    )
    ax.text(
        0.01, 0.99, f"Visual arrow scale = {arrow_scale:g}x",
        transform=ax.transAxes, va="top", ha="left",
        color="yellow", fontsize=12,
        bbox=dict(facecolor="black", alpha=0.6, pad=2),
    )
    ax.axis("off")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual_histograms(
    dx_px: np.ndarray,
    dy_px: np.ndarray,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Histograms of the per-point dx and dy (in px)."""
    plt = _matplotlib()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx = np.asarray(dx_px, dtype=np.float64)
    dy = np.asarray(dy_px, dtype=np.float64)
    paths = []
    for name, arr in (("07_dx_histogram.png", dx), ("08_dy_histogram.png", dy)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(arr, bins=40, color="steelblue", edgecolor="white")
        ax.set_title(
            f"{name.startswith('07') and 'dx' or 'dy'} (px): "
            f"mean {np.mean(arr):.3f}, median {np.median(arr):.3f}, "
            f"std {np.std(arr):.3f}"
        )
        ax.set_xlabel(f"{name.startswith('07') and 'dx' or 'dy'} (px)")
        ax.set_ylabel("count")
        path = out_dir / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _affine_to_matrix(transform: Affine) -> np.ndarray:
    """Convert a rasterio ``Affine`` into a 3×3 world matrix."""
    return np.array([
        [transform.a, transform.b, transform.c],
        [transform.d, transform.e, transform.f],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _matrix_to_affine(matrix: np.ndarray) -> Affine:
    """Convert the top-left 2×3 of a 3×3 matrix back to a rasterio ``Affine``."""
    from rasterio.transform import Affine as RAffine

    return RAffine(*np.asarray(matrix, dtype=np.float64).flat[:6])


def _affine_footprint_bounds(
    transform: Affine, shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """World ``(left, bottom, right, top)`` covered by an affine + pixel shape."""
    rows, cols = int(shape[0]), int(shape[1])
    corners = [
        transform * (0, 0),
        transform * (cols, 0),
        transform * (0, rows),
        transform * (cols, rows),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _north_up_grid(
    bounds: tuple[float, float, float, float],
    res: float,
    max_side: int,
) -> tuple[Affine, int, int]:
    """North-up output grid covering *bounds* at (at most) *max_side* px."""
    left, bottom, right, top = bounds
    w = max(int(math.ceil((right - left) / res)), 1)
    h = max(int(math.ceil((top - bottom) / res)), 1)
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        res_eff = res * max(w, h) / max(new_w, new_h)
    else:
        new_w, new_h, res_eff = w, h, res
    transform = Affine(res_eff, 0.0, left, 0.0, -res_eff, top)
    return transform, new_w, new_h


def _reproject_band_to_grid(
    path: str,
    src_transform: Affine,
    src_crs,
    grid_transform: Affine,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Find read band *path* and warp it onto *grid*; returns (array, valid)."""
    from rasterio.warp import reproject, Resampling

    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        out = np.full(grid_shape, np.nan, dtype=np.float64)
        reproject(
            source=data,
            destination=out,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=grid_transform,
            dst_crs=src_crs,
            src_nodata=nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    valid = np.isfinite(out)
    return out, valid


def render_closure_overlays(
    scene0: Scene,
    scene1: Scene,
    band: str,
    src0: Affine,
    src1: Affine,
    grid_transform: Affine,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp both scenes onto one shared grid from pre-corrected geotransforms.

    ``src0``/``src1`` are the (already corrected) rasterio transforms of the
    two scenes.  Returns ``(s0, s1, valid0, valid1)`` stretched to uint8 with
    a shared stretch.
    """
    a0, v0 = _reproject_band_to_grid(
        scene0.band_paths[band], src0, scene0.crs, grid_transform, grid_shape
    )
    a1, v1 = _reproject_band_to_grid(
        scene1.band_paths[band], src1, scene1.crs, grid_transform, grid_shape
    )
    joint = np.concatenate([a0[v0], a1[v1]])
    if joint.size == 0:
        raise ValueError("No valid overlap pixels for overlay rendering")
    lo, hi = float(np.percentile(joint, 1.0)), float(np.percentile(joint, 99.0))
    if hi - lo < 1e-12:
        hi = lo + 1.0

    def _stretch(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.clip(a, lo, hi)
        out = (out - lo) / (hi - lo) * 255.0
        out[~v] = 0.0
        return out.astype(np.uint8)

    return _stretch(a0, v0), _stretch(a1, v1), v0, v1, a0, a1


def _ncc_between_overlays(
    a0: np.ndarray, a1: np.ndarray, v0: np.ndarray, v1: np.ndarray
) -> float:
    """Normalised cross-correlation of two warped bands over joint valid px.

    Computed on raw (unstretched) values.  ~1 when the two scenes align well
    under the assumed geometry; negative/NaN values mean no meaningful match.
    """
    joint = v0 & v1
    if int(joint.sum()) < 20:
        return float("nan")
    x = a0[joint].astype(np.float64)
    y = a1[joint].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _alpha_blend(
    s0: np.ndarray,
    s1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha blend two stretched overlays (single side keeps its value)."""
    out = np.zeros(s0.shape, dtype=np.float32)
    joint = v0 & v1
    if joint.any():
        out[joint] = alpha * s0[joint] + (1.0 - alpha) * s1[joint]
    only0 = v0 & ~v1
    only1 = v1 & ~v0
    out[only0] = s0[only0]
    out[only1] = s1[only1]
    return out.astype(np.uint8)


def plot_overlay_comparison(
    scene0: Scene,
    scene1: Scene,
    band: str,
    t_direct: np.ndarray | None,
    G: list[np.ndarray],
    out_dir: str | Path,
    max_side: int = 2048,
) -> dict:
    """Render and save the direct (09) and MST (10) overlays.

    Both overlays share one display frame based on the *G-corrected scene-0
    footprint*, one north-up pixel grid, one stretch and one alpha so they can
    be compared side by side:
      - Direct: scene 0 = G0∘T0, scene 1 = G0∘T_direct∘T1
      - MST:    scene 0 = G0∘T0, scene 1 = G1∘T1
    This keeps both figures in the identical geographic region even though the
    MST path has its own large world offsets.  Returns grid metadata plus the
    blended arrays (used by the crop step).
    """
    t0 = scene0.transforms[band]
    t1 = scene1.transforms[band]
    g0 = np.asarray(G[scene0.index], dtype=np.float64)
    g1 = np.asarray(G[scene1.index], dtype=np.float64)

    src0 = _matrix_to_affine(g0 @ _affine_to_matrix(t0))
    src1_direct = _matrix_to_affine(
        g0 @ np.asarray(t_direct, dtype=np.float64) @ _affine_to_matrix(t1)
    )
    src1_mst = _matrix_to_affine(g1 @ _affine_to_matrix(t1))

    bounds = _affine_footprint_bounds(src0, scene0.shapes[band])
    grid_transform, gw, gh = _north_up_grid(bounds, abs(t0.a), max_side)
    grid_shape = (gh, gw)

    s0_d, s1_d, v0_d, v1_d, raw0_d, raw1_d = render_closure_overlays(
        scene0, scene1, band, src0, src1_direct, grid_transform, grid_shape
    )
    blend_direct = _alpha_blend(s0_d, s1_d, v0_d, v1_d)
    ncc_direct = _ncc_between_overlays(raw0_d, raw1_d, v0_d, v1_d)

    s0_m, s1_m, v0_m, v1_m, raw0_m, raw1_m = render_closure_overlays(
        scene0, scene1, band, src0, src1_mst, grid_transform, grid_shape
    )
    blend_mst = _alpha_blend(s0_m, s1_m, v0_m, v1_m)
    ncc_mst = _ncc_between_overlays(raw0_m, raw1_m, v0_m, v1_m)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    path_direct = out_dir_path / "09_direct_0_1_overlay.png"
    path_mst = out_dir_path / "10_mst_0_4_1_overlay.png"

    plt = _matplotlib()
    for path, title, arr in (
        (path_direct, "Direct 0-1 registration (50/50)", blend_direct),
        (path_mst, "MST 0-4-1 global registration (50/50)", blend_mst),
    ):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    return {
        "direct_path": str(path_direct),
        "mst_path": str(path_mst),
        "grid_transform": str(grid_transform),
        "grid_width_px": int(gw),
        "grid_height_px": int(gh),
        "direct": blend_direct,
        "mst": blend_mst,
        "grid_transform_obj": grid_transform,
        "ncc_direct": float(ncc_direct),
        "ncc_mst": float(ncc_mst),
        "ncc_winner": (
            "direct" if float(ncc_direct) > float(ncc_mst)
            else "mst" if float(ncc_mst) > float(ncc_direct)
            else "tie"
        ),
    }


def plot_overlay_crops(
    overlay: dict,
    reg: PairwiseRegistration,
    out_dir: str | Path,
    crop_size: int = 512,
    n_crops: int = 3,
) -> list[dict]:
    """Save n Direct/MST crop pairs at inlier-x percentiles of the shared grid.

    Crops are plain slices of the same blended arrays, so each Direct/MST pair
    always shows the exact same geographic region, pixel size and stretch.
    """
    direct = overlay["direct"]
    mst = overlay["mst"]
    grid_transform = overlay["grid_transform_obj"]
    inv = ~grid_transform
    ref_pts = np.asarray(reg.inlier_ref_xy, dtype=np.float64)
    screen = _scene_to_screen_pixels(ref_pts, reg.pair_common_transform, grid_transform)
    x_px = screen[:, 0]

    fracs = [100.0 * (k + 0.5) / n_crops for k in range(n_crops)]
    centers = np.percentile(x_px, fracs) if len(x_px) else np.linspace(0, direct.shape[1], n_crops)
    centers = np.clip(centers, crop_size / 2, direct.shape[1] - crop_size / 2)

    crop_half = crop_size // 2
    saved = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    letters = "ABC"
    for order, cx in enumerate(centers):
        x0 = max(0, int(round(cx - crop_half)))
        x1 = min(direct.shape[1], x0 + crop_size)
        y0 = max(0, int(round(direct.shape[0] // 2 - crop_half)))
        y1 = min(direct.shape[0], y0 + crop_size)
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        letter = letters[order % 26]

        plt = _matplotlib()
        for kind, arr in (("direct", direct), ("mst", mst)):
            num = 2 * order + 11 if kind == "direct" else 2 * order + 12
            label = f"{num:02d}_crop_{letter}_{kind}.png"
            crop = arr[y0:y1, x0:x1]
            path = out_dir_path / label
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Crop {letter} ({kind}) ~ x={int(cx)}px")
            ax.axis("off")
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            saved.append({"label": label, "path": str(path), "kind": kind})
    return saved


def plot_loop_triangle(
    edge: tuple[int, int],
    path: list[int],
    records: dict[tuple[int, int], dict],
    quality: dict[tuple[int, int], float],
    out_path: str | Path,
) -> Path:
    """Draw the three-node loop-closure schematic (0, 1 and the MST hub)."""
    plt = _matplotlib()
    i, j = edge
    hub = path[-2] if len(path) >= 2 else None  # interior node of the MST path
    if hub is None or hub in (i, j):
        hub = sorted({i, j, path[0] if path else i})[0]

    pos = {hub if hub == path[0] else i: None}
    # Layout: hub on top, two endpoints on the bottom row.
    coordinates = {
        hub: (0.5, 0.9),
        i: (0.0, 0.1),
        j: (1.0, 0.1),
    }

    def _label(edge_key):
        rec = records.get(edge_key)
        if rec is None:
            return "(no data)"
        q = quality.get(edge_key)
        tree = "tree" if rec.get("in_tree") else "non-tree"
        return (
            f"{edge_key[0]}-{edge_key[1]}  [{tree}]\n"
            f"inliers={rec.get('n_points', '?')}  RMSE={rec.get('global_rmse_px', '?')} px\n"
            f"P95={rec.get('global_p95_px', '?')} px"
            f"{f'  Q={q:.2f}' if q is not None else ''}"
        )

    fig, ax = plt.subplots(figsize=(9, 7))
    for a, b, style in (
        (i, j, "r-"),
        (hub, i, "g-"),
        (hub, j, "b-"),
    ):
        (xa, ya), (xb, yb) = coordinates[a], coordinates[b]
        ax.plot([xa, xb], [ya, yb], style, linewidth=2)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my, _label((a, b)), ha="center", va="bottom",
                fontsize=10, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.4"))

    for node, (x, y) in coordinates.items():
        ax.plot(x, y, "ko", markersize=14)
        ax.text(x, y + 0.06, f"scene {node}", ha="center", fontsize=12, fontweight="bold")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.05)
    ax.axis("off")
    ax.set_title(
        f"Loop-closure triangle: scene {i} -- scene {j} (non-tree) "
        f"vs MST path {'->'.join(map(str, path))}"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Task 9 — Diagnosis summary
# ---------------------------------------------------------------------------


def build_diagnosis_summary(
    *,
    run_dir: str,
    output_dir: str,
    edge: tuple[int, int],
    mst_path: list[int],
    edge_record: dict,
    transform_compare: dict | None,
    pattern: dict | None,
    residual_stats: dict | None,
    reproduction: dict | None,
    final_state: dict,
    q5_explanation: str,
    visualization_files: dict | None = None,
    overlay_alignment: dict | None = None,
) -> dict:
    """Assemble the final ``18_diagnosis_summary.json`` payload."""
    i, j = edge
    summary = {
        "edge": {"idx_i": i, "idx_j": j},
        "mst_path": mst_path,
        "edge_global_consistency": edge_record,
        "final_state": final_state["final_state"],
        "final_state_reason": final_state["reason"],
        "evidence": final_state["evidence"],
        "questions": {
            "Q1_which_edge": (
                f"Edge {i}-{j} causes the current loop anomaly (worst non-tree "
                "edge by global P95)."
            ),
            "Q2_mst_path": (
                f"Scenes {i} and {j} are connected through the MST path "
                f"{' -> '.join(map(str, mst_path))}."
            ),
            "Q3_error_type": (
                pattern["classification"]
                if pattern else "unavailable"
            ),
            "Q4_transform_difference": (
                None
                if transform_compare is None
                else {
                    "translation_magnitude_px": transform_compare["translation_magnitude_px"],
                    "rotation_deg": transform_compare["rotation_deg"],
                    "scale_x": transform_compare["scale_x"],
                    "scale_y": transform_compare["scale_y"],
                    "shear_deg": transform_compare["shear_deg"],
                }
            ),
            "Q5_pairwise_rmse_sufficient": q5_explanation,
        },
        "transform_comparison": transform_compare,
        "error_pattern": pattern,
        "residual_statistics": residual_stats,
        "pair_reproduction": reproduction,
        "visualization_files": visualization_files or {},
        "overlay_alignment": overlay_alignment,
        "diagnostic_artifacts": {
            "problem_edge": f"{output_dir}/01_problem_edge.json",
            "transform_comparison": f"{output_dir}/02_transform_comparison.json",
            "pair_reproduction": f"{output_dir}/03_pair_reproduction.json",
            "point_residuals": f"{output_dir}/04_point_residuals.csv",
            "error_pattern": f"{output_dir}/05_error_pattern.json",
            "error_vectors": f"{output_dir}/06_error_vectors.png",
            "dx_histogram": f"{output_dir}/07_dx_histogram.png",
            "dy_histogram": f"{output_dir}/08_dy_histogram.png",
            "direct_overlay": f"{output_dir}/09_direct_0_1_overlay.png",
            "mst_overlay": f"{output_dir}/10_mst_0_4_1_overlay.png",
            "loop_triangle": f"{output_dir}/17_loop_triangle.png",
        },
        "run_dir": run_dir,
    }
    return summary


def write_summary_text(summary: dict, path: str | Path) -> Path:
    """Write a human-readable rendering of the diagnosis summary."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=== Loop-Closure Diagnosis Summary ===")
    edge = summary["edge"]
    lines.append("Q1. Which edge causes the loop error?")
    lines.append("   " + str(summary["questions"]["Q1_which_edge"]))
    lines.append("Q2. What is the MST path between these images?")
    lines.append("   " + str(summary["questions"]["Q2_mst_path"]))
    lines.append("Q3. What type is the ~error?")
    lines.append("   " + str(summary["questions"]["Q3_error_type"]))
    q4 = summary["questions"]["Q4_transform_difference"]
    if q4:
        lines.append("Q4. How do direct and MST-implied transforms differ?")
        keys = ("translation_magnitude_px", "rotation_deg", "scale_x",
                "scale_y", "shear_deg")
        for key in keys:
            if key in q4:
                lines.append(f"   {key}: {q4[key]}")
    lines.append("Q5. Can pairwise RMSE alone reveal the problem?")
    lines.append("   " + str(summary["questions"]["Q5_pairwise_rmse_sufficient"]))
    lines.append("")
    lines.append(f"Final state: {summary['final_state']}")
    lines.append(f"Reason: {summary['final_state_reason']}")
    lines.append("")
    lines.append("Diagnostic artifacts:")
    for name, path_ in summary.get("diagnostic_artifacts", {}).items():
        lines.append(f"   {name}: {path_}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out