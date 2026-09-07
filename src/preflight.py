"""Metadata-only preflight checks for mosaic series.

Task 10 of dz01v mosaic readiness plan:
Verify scene metadata before running actual experiments.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import rasterio
from shapely.geometry import box
import networkx as nx

logger = logging.getLogger(__name__)


def inspect_scene_headers(scene: Dict[str, Any], band: str) -> Dict[str, Any]:
    """
    Inspect metadata headers for a scene's band file.
    
    Returns dict with:
    - scene_id, sensor, path
    - width, height, transform, crs, nodata, bounds
    - x_res, y_res
    """
    scene_id = scene["id"]
    sensor = scene.get("sensor", "unknown")
    path = scene["bands"].get(band)
    
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Scene {scene_id} band {band} not found: {path}")
    
    with rasterio.open(path) as src:
        info = {
            "scene_id": scene_id,
            "sensor": sensor,
            "path": path,
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": str(src.crs),
            "nodata": src.nodata,
            "bounds": src.bounds,
            "x_res": abs(src.transform.a),
            "y_res": abs(src.transform.e),
        }
    
    return info


def check_overlap(scene_i: Dict[str, Any], scene_j: Dict[str, Any]) -> bool:
    """Check if two scenes have geographic overlap (positive area)."""
    bounds_i = scene_i["bounds"]
    bounds_j = scene_j["bounds"]
    
    # Handle both tuples and rasterio Bounds objects
    if hasattr(bounds_i, 'left'):
        left_i, bottom_i, right_i, top_i = bounds_i.left, bounds_i.bottom, bounds_i.right, bounds_i.top
    else:
        left_i, bottom_i, right_i, top_i = bounds_i
    
    if hasattr(bounds_j, 'left'):
        left_j, bottom_j, right_j, top_j = bounds_j.left, bounds_j.bottom, bounds_j.right, bounds_j.top
    else:
        left_j, bottom_j, right_j, top_j = bounds_j
    
    box_i = box(left_i, bottom_i, right_i, top_i)
    box_j = box(left_j, bottom_j, right_j, top_j)
    
    intersection = box_i.intersection(box_j)
    return intersection.area > 0


def build_overlap_graph(scenes: List[Dict[str, Any]]) -> nx.Graph:
    """Build a graph where nodes are scenes and edges are overlaps."""
    G = nx.Graph()
    
    for scene in scenes:
        G.add_node(scene["scene_id"])
    
    for i, scene_i in enumerate(scenes):
        for j, scene_j in enumerate(scenes):
            if i < j:
                if check_overlap(scene_i, scene_j):
                    G.add_edge(scene_i["scene_id"], scene_j["scene_id"])
    
    return G


def preflight_mosaic_series(config_path: str, scene_counts: List[int]) -> Dict[str, Any]:
    """
    Run metadata-only preflight checks for mosaic series.
    
    Checks:
    1. All files exist
    2. All sensors are DZ01V
    3. CRS is identical across all scenes
    4. Resolution is identical across all scenes
    5. Overlap graph is connected for each N
    """
    from src.experiment_config import load_config
    
    config = load_config(config_path)
    band = config.registration_band
    
    logger.info("=" * 60)
    logger.info("Preflight check for mosaic series")
    logger.info("Config: %s", config_path)
    logger.info("Band: %s", band)
    logger.info("=" * 60)
    
    # Inspect all scenes
    scenes_info = []
    for scene in config.scenes:
        info = inspect_scene_headers(scene, band)
        scenes_info.append(info)
        logger.info("Scene %s: %dx%d, CRS=%s, res=(%.2f, %.2f)", 
                   info["scene_id"], info["width"], info["height"],
                   info["crs"], info["x_res"], info["y_res"])
    
    # Check all sensors are DZ01V
    for info in scenes_info:
        if info["sensor"] != "DZ01V":
            raise ValueError(f"Scene {info['scene_id']} has sensor {info['sensor']}, expected DZ01V")
    
    # Check CRS consistency
    crs_set = set(info["crs"] for info in scenes_info)
    if len(crs_set) > 1:
        raise ValueError(f"Inconsistent CRS across scenes: {crs_set}")
    
    # Check resolution consistency
    res_set = set((info["x_res"], info["y_res"]) for info in scenes_info)
    if len(res_set) > 1:
        raise ValueError(f"Inconsistent resolution across scenes: {res_set}")
    
    logger.info("✓ All sensors are DZ01V")
    logger.info("✓ CRS consistent: %s", list(crs_set)[0])
    logger.info("✓ Resolution consistent: %s", list(res_set)[0])
    
    # Check overlap graph connectivity for each N
    results = {
        "config": config_path,
        "band": band,
        "total_scenes": len(scenes_info),
        "scene_counts": scene_counts,
        "checks": {
            "all_files_exist": True,
            "all_sensors_dz01v": True,
            "crs_consistent": True,
            "resolution_consistent": True,
            "overlap_connected": {},
        },
        "scenes": scenes_info,
    }
    
    for n in scene_counts:
        if n > len(scenes_info):
            logger.warning("N=%d exceeds available scenes (%d)", n, len(scenes_info))
            results["checks"]["overlap_connected"][n] = False
            continue
        
        subset = scenes_info[:n]
        G = build_overlap_graph(subset)
        
        is_connected = nx.is_connected(G)
        results["checks"]["overlap_connected"][n] = is_connected
        
        if is_connected:
            logger.info("✓ N=%d: overlap graph connected (%d edges)", n, G.number_of_edges())
        else:
            logger.error("✗ N=%d: overlap graph NOT connected", n)
            raise ValueError(f"N={n} overlap graph is not connected")
    
    logger.info("=" * 60)
    logger.info("Preflight PASSED")
    logger.info("=" * 60)
    
    return results
