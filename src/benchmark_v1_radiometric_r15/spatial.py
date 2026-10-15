from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from shapely.geometry import mapping, shape

from benchmark_v1_radiometric_r1.roi import classify_roi_role, to_projected


def assign_frozen_role(geometry) -> tuple[str, str, str]:
    """Delegate directly to Phase3's frozen EPSG:32650 10km/500m role rule."""
    geom = shape(geometry) if isinstance(geometry, dict) else geometry
    return classify_roi_role(geom, "EPSG:32650")


def _features(collections):
    for collection in collections:
        if isinstance(collection, dict) and collection.get("type") == "Feature":
            yield collection
            continue
        features = collection.get("features", []) if isinstance(collection, dict) else collection
        yield from features


def audit_candidate_sets(collections) -> dict[str, Any]:
    rows = list(_features(collections))
    ids: dict[str, list[str]] = {}
    for feature in rows:
        props = feature.get("properties", {})
        key = f"{props.get('pair_id', '')}:{props.get('candidate_id', '')}"
        ids.setdefault(key, []).append(key)
    duplicates = sorted(key for key, vals in ids.items() if len(vals) > 1)
    geometries = []
    role_conflicts = []
    unique_rows = {}
    for feature in rows:
        props = feature.get("properties", {})
        identity = (str(props.get("pair_id", "")), str(props.get("candidate_id", "")))
        unique_rows.setdefault(identity, feature)
    for feature in unique_rows.values():
        props = feature.get("properties", {})
        geom = shape(feature["geometry"])
        expected, cell_id, status = assign_frozen_role(geom)
        declared = str(props.get("role", expected))
        if declared != expected:
            role_conflicts.append({"pair_id": props.get("pair_id"), "candidate_id": props.get("candidate_id"),
                                   "declared_role": declared, "frozen_role": expected,
                                   "cell_id": cell_id, "spatial_status": status})
        geometries.append((str(props.get("pair_id", "")), str(props.get("candidate_id", "")), geom, expected))
    overlaps, opposite = [], []
    for i, left in enumerate(geometries):
        for right in geometries[i + 1:]:
            if left[0] == right[0]:
                continue
            if left[2].intersects(right[2]):
                overlaps.append({"pair_i": left[0], "candidate_i": left[1],
                                 "pair_j": right[0], "candidate_j": right[1],
                                 "intersection_area_m2": float(left[2].intersection(right[2]).area)})
            elif left[3] != right[3] and left[3] != "UNASSIGNED" and right[3] != "UNASSIGNED":
                distance = float(left[2].distance(right[2]))
                if distance < 1000.0:
                    opposite.append({"pair_i": left[0], "candidate_i": left[1],
                                     "pair_j": right[0], "candidate_j": right[1],
                                     "distance_m": distance})
    return {"candidate_count": len(rows), "duplicate_candidate_ids": duplicates,
            "overlaps": overlaps, "opposite_role_proximity": opposite,
            "role_conflicts": role_conflicts,
            "status": "PASS" if not (duplicates or overlaps or opposite or role_conflicts) else "CONFLICTS_REPORTED"}


def revalidate_geometry_edit(feature: dict[str, Any], geometry, *,
                             support_validator: Callable[[Any], dict[str, Any]],
                             all_candidates, all_pairs_available: bool = False,
                             preserve_frozen_role: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute mask support/role after edits and invalidate eligibility pending global re-audit."""
    edited = to_projected(geometry, "EPSG:32650")
    role, cell_id, spatial_status = assign_frozen_role(edited)
    support = support_validator(edited)
    b9_support = support.get("B9") if isinstance(support, dict) else None
    if isinstance(b9_support, dict):
        recalculated_b9_fraction = b9_support.get("joint_valid_fraction", b9_support.get("common_lower_bound"))
    else:
        recalculated_b9_fraction = None
    before = deepcopy(feature)
    updated = deepcopy(feature)
    updated["geometry"] = mapping(edited)
    props = updated.setdefault("properties", {})
    projected_sets = []
    for collection in all_candidates:
        if isinstance(collection, dict) and collection.get("type") == "Feature":
            projected_sets.append([updated if (collection.get("properties", {}).get("pair_id"),
                                                  collection.get("properties", {}).get("candidate_id")) ==
                                   (props.get("pair_id"), props.get("candidate_id")) else collection])
            continue
        features = collection.get("features", []) if isinstance(collection, dict) else collection
        projected_sets.append([updated if item is feature or
                               (item.get("properties", {}).get("pair_id"), item.get("properties", {}).get("candidate_id")) ==
                               (props.get("pair_id"), props.get("candidate_id")) else item for item in features])
    provisional_audit = audit_candidate_sets(projected_sets)
    pairwise_status = ("PASS" if provisional_audit["status"] == "PASS" else "CONFLICTS_REPORTED") if all_pairs_available else "PENDING_ALL_PAIR_REAUDIT"
    frozen_role = str(before.get("properties", {}).get("frozen_role", before.get("properties", {}).get("role", role)))
    frozen_cell = str(before.get("properties", {}).get("frozen_grid_cell_id", before.get("properties", {}).get("grid_cell_id", cell_id)))
    role_matches = role == frozen_role and cell_id == frozen_cell and spatial_status == "SAFE_SINGLE_CELL"
    try:
        support_ok = recalculated_b9_fraction is not None and float(recalculated_b9_fraction) >= 0.95
    except (TypeError, ValueError):
        support_ok = False
    if not preserve_frozen_role:
        edited_role, edited_cell = role, cell_id
        output_role, output_cell = role, cell_id
        spatial_edit_status = "NOT_APPLICABLE"
    else:
        edited_role, edited_cell = role, cell_id
        output_role, output_cell = frozen_role, frozen_cell
        if not role_matches:
            spatial_edit_status = "INELIGIBLE_FROZEN_ROLE_MISMATCH"
        elif not support_ok:
            spatial_edit_status = "INELIGIBLE_VALID_SUPPORT"
        elif not all_pairs_available:
            spatial_edit_status = "PENDING_PAIRWISE_REAUDIT"
        elif pairwise_status != "PASS":
            spatial_edit_status = "INELIGIBLE_PAIRWISE_LEAKAGE"
        else:
            spatial_edit_status = "SAFE_FROZEN_ROLE"
    props.update({"role": output_role, "grid_cell_id": output_cell,
                  "frozen_role": frozen_role, "frozen_grid_cell_id": frozen_cell,
                  "edited_geometry_role": edited_role, "edited_grid_cell_id": edited_cell,
                  "spatial_edit_eligibility": spatial_edit_status, "spatial_status": spatial_status,
                  "support_by_scene_band": support, "valid_support_recomputed": True,
                  "valid_support_fraction": recalculated_b9_fraction,
                  "relative_spectral_shape_similarity_diagnostic": None,
                  "multiband_feature_status": "INVALIDATED_BY_GEOMETRY_EDIT",
                  "pairwise_leakage_audit": pairwise_status,
                  "pairwise_leakage_audit_scope": "ALL_FOUR_PAIRS" if all_pairs_available else "PARTIAL_PENDING",
                  "full_pairwise_leakage_recomputed": bool(all_pairs_available),
                  "eligible_for_test": False, "review_status": "UNVERIFIED"})
    event = {"event_type": "GEOMETRY_EDIT", "pair_id": props.get("pair_id"),
             "candidate_id": props.get("candidate_id"), "before_geometry": before["geometry"],
             "after_geometry": updated["geometry"], "before_role": before.get("properties", {}).get("role"),
             "after_role": role, "support_by_scene_band": deepcopy(support),
             "frozen_role": frozen_role, "edited_geometry_role": edited_role,
             "spatial_edit_eligibility": spatial_edit_status,
             "provisional_pairwise_audit": provisional_audit,
             "pairwise_audit_status": pairwise_status,
             "pairwise_audit_scope": props["pairwise_leakage_audit_scope"]}
    return updated, event
