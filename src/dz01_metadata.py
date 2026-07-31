"""
DZ01 Metadata Parser — VNIR and SWIR MTL file parsing

Parses DZ01 satellite metadata files (MTL) for:
  - SENSOR_ID identification (VNIR vs SWIR)
  - Band wavelength definitions
  - Scene acquisition geometry
  - Per-band radiometric parameters

Key design rules:
  - VNIR (DZ01V): 16 bands (B01-B16), 14m resolution, 410-1019 nm
  - SWIR (DZ01S): 10 bands (B01-B10), 30m resolution, 1178-2468 nm
  - SENSOR_ID=VNIR must not load SWIR band table
  - SENSOR_ID=SWIR must not load VNIR band table
  - Missing fields → None + warning (never fabricated)
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===========================================================================
# Known band tables (fallback / validation reference)
# ===========================================================================

VNIR_BAND_TABLE: Dict[str, Dict[str, float]] = {
    "B01": {"min_nm": 410, "center_nm": 421.0, "max_nm": 432},
    "B02": {"min_nm": 445, "center_nm": 452.5, "max_nm": 460},
    "B03": {"min_nm": 480, "center_nm": 490.0, "max_nm": 500},
    "B04": {"min_nm": 521, "center_nm": 530.5, "max_nm": 540},
    "B05": {"min_nm": 539, "center_nm": 550.0, "max_nm": 561},
    "B06": {"min_nm": 561, "center_nm": 575.5, "max_nm": 590},
    "B07": {"min_nm": 619, "center_nm": 633.5, "max_nm": 648},
    "B08": {"min_nm": 668, "center_nm": 678.5, "max_nm": 689},
    "B09": {"min_nm": 697, "center_nm": 701.5, "max_nm": 706},
    "B10": {"min_nm": 718, "center_nm": 722.5, "max_nm": 727},
    "B11": {"min_nm": 739, "center_nm": 743.5, "max_nm": 748},
    "B12": {"min_nm": 759, "center_nm": 769.0, "max_nm": 779},
    "B13": {"min_nm": 799, "center_nm": 814.0, "max_nm": 829},
    "B14": {"min_nm": 841, "center_nm": 850.5, "max_nm": 860},
    "B15": {"min_nm": 940, "center_nm": 960.5, "max_nm": 981},
    "B16": {"min_nm": 982, "center_nm": 1000.5, "max_nm": 1019},
}

SWIR_BAND_TABLE: Dict[str, Dict[str, float]] = {
    "B01": {"min_nm": 1178, "center_nm": 1196.5, "max_nm": 1215},
    "B02": {"min_nm": 1581, "center_nm": 1600.0, "max_nm": 1619},
    "B03": {"min_nm": 1623, "center_nm": 1642.0, "max_nm": 1661},
    "B04": {"min_nm": 1737, "center_nm": 1757.5, "max_nm": 1778},
    "B05": {"min_nm": 1974, "center_nm": 1996.5, "max_nm": 2019},
    "B06": {"min_nm": 2128, "center_nm": 2147.0, "max_nm": 2166},
    "B07": {"min_nm": 2193, "center_nm": 2213.5, "max_nm": 2234},
    "B08": {"min_nm": 2243, "center_nm": 2265.0, "max_nm": 2287},
    "B09": {"min_nm": 2314, "center_nm": 2335.0, "max_nm": 2356},
    "B10": {"min_nm": 2390, "center_nm": 2429.0, "max_nm": 2468},
}

# Spectral region labels for display (neutral, non-mineral)
SPECTRAL_REGION_LABELS: Dict[str, str] = {
    "B01": "violet",
    "B02": "blue",
    "B03": "blue",
    "B04": "green",
    "B05": "green",
    "B06": "yellow-green",
    "B07": "red",
    "B08": "red",
    "B09": "red_to_nir_transition",
    "B10": "red_to_nir_transition",
    "B11": "red_to_nir_transition",
    "B12": "nir",
    "B13": "nir",
    "B14": "nir",
    "B15": "nir",
    "B16": "nir",
}


# ===========================================================================
# MTL Parser
# ===========================================================================

def parse_dz01_mtl(path: str) -> Dict[str, Any]:
    """
    Parse a DZ01 MTL metadata file.

    Parameters
    ----------
    path : str
        Path to the MTL text file.

    Returns
    -------
    dict
        Parsed metadata with keys:
        'raw': dict of all key=value pairs
        'sensor_id': 'VNIR' or 'SWIR'
        'spacecraft_id': 'DZ01'
        'product_id': str
        'date_acquired': str
        'scene_center_time': str
        'bands': dict of band_name -> band_metadata
        'scene': dict of scene-level metadata
        'warnings': list of str
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MTL file not found: {path}")

    warnings: List[str] = []
    raw: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("GROUP") or line.startswith("END_GROUP") or line.startswith("END"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                raw[key] = value

    # ---- Sensor identification ----
    sensor_id = raw.get("SENSOR_ID")
    if sensor_id is None:
        warnings.append("SENSOR_ID not found in MTL")
        sensor_id = "UNKNOWN"
    else:
        sensor_id = sensor_id.strip().upper()

    spacecraft_id = raw.get("SPACECRAFT_ID", "DZ01")
    product_id = raw.get("PRODUCT_ID", "")
    date_acquired = raw.get("DATE_ACQUIRED", "")
    scene_center_time = raw.get("SCENE_CENTER_TIME", "")

    # ---- Scene-level metadata ----
    scene: Dict[str, Any] = {}
    scene_fields = {
        "SUN_AZIMUTH": float,
        "SUN_ZENITH": float,
        "SUN_ELEVATION": float,
        "SAT_AZIMUTH": float,
        "SAT_ZENITH": float,
        "CLOUD_COVER": float,
        "ROLL_ANGLE": float,
        "PITCH_ANGLE": float,
        "YAW_ANGLE": float,
        "START_IMAGING_TIME": str,
        "END_IMAGING_TIME": str,
        "PROCESSING_SOFTWARE_VERSION": str,
    }
    for field, dtype in scene_fields.items():
        val = raw.get(field)
        if val is not None:
            try:
                scene[field] = dtype(val)
            except (ValueError, TypeError):
                scene[field] = val
                warnings.append(f"Cannot convert {field}={val} to {dtype.__name__}")
        else:
            scene[field] = None

    # ---- Band metadata ----
    bands: Dict[str, Dict[str, Any]] = {}

    # Determine expected band count from sensor
    if sensor_id == "VNIR":
        expected_band_numbers = list(range(1, 17))  # B01-B16
    elif sensor_id == "SWIR":
        expected_band_numbers = list(range(1, 11))  # B01-B10
    else:
        expected_band_numbers = list(range(1, 17))
        warnings.append(f"Unknown sensor '{sensor_id}', assuming up to 16 bands")

    for band_num in expected_band_numbers:
        band_name = f"B{band_num:02d}"

        # Try both BAND_N and BAND_NN forms
        min_wl = _get_field(raw, f"MEASURED_MIN_WAVELENGTH_BAND_{band_num}")
        center_wl = _get_field(raw, f"MEASURED_CENTER_WAVELENGTH_BAND_{band_num}")
        max_wl = _get_field(raw, f"MEASURED_MAX_WAVELENGTH_BAND_{band_num}")
        integration_time = _get_field(raw, f"INTEGRATION_TIME_BAND_{band_num}")
        integration_level = _get_field(raw, f"INTEGRATION_LEVEL_BAND_{band_num}")
        data_type = _get_field(raw, f"DATA_TYPE_BAND_{band_num}")
        radiance_mult = _get_field(raw, f"RADIANCE_MULT_BAND_{band_num}")
        radiance_add = _get_field(raw, f"RADIANCE_ADD_BAND_{band_num}")

        band_info: Dict[str, Any] = {
            "band_name": band_name,
            "band_number": band_num,
            "wavelength_min_nm": _to_float(min_wl, warnings, f"B{band_num:02d}.min_wl"),
            "wavelength_center_nm": _to_float(center_wl, warnings, f"B{band_num:02d}.center_wl"),
            "wavelength_max_nm": _to_float(max_wl, warnings, f"B{band_num:02d}.max_wl"),
            "integration_time": _to_float(integration_time, warnings, f"B{band_num:02d}.integration_time"),
            "integration_level": _to_float(integration_level, warnings, f"B{band_num:02d}.integration_level"),
            "data_type": data_type,
            "radiance_mult": _to_float(radiance_mult, warnings, f"B{band_num:02d}.radiance_mult"),
            "radiance_add": _to_float(radiance_add, warnings, f"B{band_num:02d}.radiance_add"),
        }
        bands[band_name] = band_info

    # ---- Resolution ----
    resolution_vi = _to_float(raw.get("GRID_CELL_SIZE_VI"), warnings, "resolution_vi")
    resolution_pan = _to_float(raw.get("GRID_CELL_SIZE_PAN"), warnings, "resolution_pan")

    # ---- Build result ----
    result = {
        "raw": raw,
        "sensor_id": sensor_id,
        "spacecraft_id": spacecraft_id,
        "product_id": product_id,
        "date_acquired": date_acquired,
        "scene_center_time": scene_center_time,
        "bands": bands,
        "scene": scene,
        "resolution_vi": resolution_vi,
        "resolution_pan": resolution_pan,
        "warnings": warnings,
        "mtl_path": path,
    }

    # Log warnings
    for w in warnings:
        logger.warning("MTL parse warning: %s", w)

    return result


def _get_field(raw: Dict[str, str], key: str) -> Optional[str]:
    """Get a field from raw dict, returning None if missing."""
    return raw.get(key)


def _to_float(val: Optional[str], warnings: List[str], field_name: str) -> Optional[float]:
    """Convert string to float, logging warning on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        warnings.append(f"Cannot convert {field_name}={val} to float")
        return None


# ===========================================================================
# Band metadata extraction
# ===========================================================================

def extract_band_metadata(
    metadata: Dict[str, Any],
    selected_bands: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract band metadata for selected bands.

    Parameters
    ----------
    metadata : dict
        Output of parse_dz01_mtl().
    selected_bands : list of str or None
        Bands to extract (e.g. ['B01','B02',...]). None = all bands.

    Returns
    -------
    dict
        band_name -> {wavelength_center_nm, region_label, ...}
    """
    sensor_id = metadata.get("sensor_id", "UNKNOWN")
    all_bands = metadata.get("bands", {})

    if selected_bands is None:
        selected_bands = sorted(all_bands.keys())

    result: Dict[str, Dict[str, Any]] = {}
    for band_name in selected_bands:
        band_info = all_bands.get(band_name)
        if band_info is None:
            logger.warning("Band %s not found in MTL metadata (sensor=%s)", band_name, sensor_id)
            continue

        region_label = SPECTRAL_REGION_LABELS.get(band_name, "unknown")
        qualified_name = f"{sensor_id}_{band_name}"

        result[band_name] = {
            "band_name": band_name,
            "qualified_band_name": qualified_name,
            "sensor_id": sensor_id,
            "wavelength_min_nm": band_info.get("wavelength_min_nm"),
            "wavelength_center_nm": band_info.get("wavelength_center_nm"),
            "wavelength_max_nm": band_info.get("wavelength_max_nm"),
            "region_label": region_label,
            "integration_time": band_info.get("integration_time"),
            "data_type": band_info.get("data_type"),
        }

    return result


def extract_scene_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract scene-level metadata.

    Returns
    -------
    dict
        Scene metadata with sensor_id, date, geometry, etc.
    """
    scene = metadata.get("scene", {})
    return {
        "sensor_id": metadata.get("sensor_id"),
        "spacecraft_id": metadata.get("spacecraft_id"),
        "product_id": metadata.get("product_id"),
        "date_acquired": metadata.get("date_acquired"),
        "scene_center_time": metadata.get("scene_center_time"),
        "sun_azimuth": scene.get("SUN_AZIMUTH"),
        "sun_zenith": scene.get("SUN_ZENITH"),
        "sun_elevation": scene.get("SUN_ELEVATION"),
        "sat_azimuth": scene.get("SAT_AZIMUTH"),
        "sat_zenith": scene.get("SAT_ZENITH"),
        "cloud_cover": scene.get("CLOUD_COVER"),
        "roll_angle": scene.get("ROLL_ANGLE"),
        "pitch_angle": scene.get("PITCH_ANGLE"),
        "yaw_angle": scene.get("YAW_ANGLE"),
        "start_imaging_time": scene.get("START_IMAGING_TIME"),
        "end_imaging_time": scene.get("END_IMAGING_TIME"),
        "processing_software": scene.get("PROCESSING_SOFTWARE_VERSION"),
    }


def validate_sensor_metadata(
    metadata: Dict[str, Any],
    expected_sensor: str,
) -> List[str]:
    """
    Validate that metadata matches expected sensor type.

    Returns
    -------
    list of str
        Error list (empty = valid).
    """
    errors: List[str] = []
    actual = metadata.get("sensor_id", "UNKNOWN")
    if actual != expected_sensor:
        errors.append(
            f"Sensor mismatch: expected={expected_sensor}, actual={actual} "
            f"(product={metadata.get('product_id', '?')})"
        )

    # Validate band count
    bands = metadata.get("bands", {})
    if expected_sensor == "VNIR" and len(bands) == 0:
        errors.append("VNIR metadata contains no bands")
    elif expected_sensor == "SWIR" and len(bands) == 0:
        errors.append("SWIR metadata contains no bands")

    return errors


def build_band_metadata_table(
    metadata: Dict[str, Any],
    selected_bands: List[str],
    metadata_status: str = "from_mtl",
) -> List[Dict[str, Any]]:
    """
    Build a flat table of band metadata for CSV output.

    Parameters
    ----------
    metadata : dict
        Output of parse_dz01_mtl().
    selected_bands : list of str
        Bands to include.
    metadata_status : str
        'from_mtl' if from this scene's MTL, 'reference_only' if from reference.

    Returns
    -------
    list of dict
        One dict per band with all fields.
    """
    sensor_id = metadata.get("sensor_id", "UNKNOWN")
    band_data = extract_band_metadata(metadata, selected_bands)

    table = []
    for band_name in selected_bands:
        info = band_data.get(band_name, {})
        table.append({
            "sensor_id": sensor_id,
            "band_name": band_name,
            "qualified_band_name": f"{sensor_id}_{band_name}",
            "wavelength_min_nm": info.get("wavelength_min_nm"),
            "wavelength_center_nm": info.get("wavelength_center_nm"),
            "wavelength_max_nm": info.get("wavelength_max_nm"),
            "region_label": info.get("region_label", "unknown"),
            "metadata_source": metadata.get("mtl_path", ""),
            "metadata_status": metadata_status,
        })

    return table


# ===========================================================================
# Cross-scene band consistency validation
# ===========================================================================

def validate_cross_scene_band_metadata(
    scene_metadata_list: List[Dict[str, Any]],
    selected_bands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare band definitions across multiple scenes for consistency.

    Checks MEASURED_MIN/CENTER/MAX_WAVELENGTH, DATA_TYPE, GRID_CELL_SIZE_VI,
    and PROCESSING_SOFTWARE_VERSION for each band across all scenes.

    Parameters
    ----------
    scene_metadata_list : list of dict
        Each dict is the output of parse_dz01_mtl() for one scene.
    selected_bands : list of str or None
        Bands to check. None = all bands from first scene.

    Returns
    -------
    dict with keys:
        'consistent': bool
        'band_consistency': dict of band_name -> {field: {scene_id: value, ...}, consistent: bool}
        'warnings': list of str
    """
    warnings: List[str] = []

    if not scene_metadata_list:
        return {"consistent": True, "band_consistency": {}, "warnings": ["No scenes provided"]}

    # Determine bands to check
    if selected_bands is None:
        first_bands = scene_metadata_list[0].get("bands", {})
        selected_bands = sorted(first_bands.keys())

    # Fields to compare per band
    band_fields = [
        "wavelength_min_nm",
        "wavelength_center_nm",
        "wavelength_max_nm",
        "data_type",
    ]
    # Scene-level fields
    scene_fields = ["resolution_vi", "processing_software"]

    band_consistency: Dict[str, Any] = {}
    all_consistent = True

    for band_name in selected_bands:
        field_values: Dict[str, Dict[str, Any]] = {}
        for field in band_fields:
            field_values[field] = {}

        for meta in scene_metadata_list:
            scene_id = meta.get("date_acquired", meta.get("product_id", "unknown"))
            bands = meta.get("bands", {})
            band_info = bands.get(band_name, {})
            for field in band_fields:
                field_values[field][scene_id] = band_info.get(field)

        # Check consistency per field
        band_ok = True
        for field in band_fields:
            values = list(field_values[field].values())
            non_none = [v for v in values if v is not None]
            if non_none and len(set(str(v) for v in non_none)) > 1:
                band_ok = False
                all_consistent = False
                warnings.append(
                    f"Band {band_name} {field} differs across scenes: "
                    f"{field_values[field]}"
                )

        band_consistency[band_name] = {
            "values": field_values,
            "consistent": band_ok,
        }

    # Check scene-level fields
    for field in scene_fields:
        values = {}
        for meta in scene_metadata_list:
            scene_id = meta.get("date_acquired", meta.get("product_id", "unknown"))
            if field == "resolution_vi":
                values[scene_id] = meta.get("resolution_vi")
            elif field == "processing_software":
                values[scene_id] = meta.get("scene", {}).get("PROCESSING_SOFTWARE_VERSION")

        non_none = [v for v in values.values() if v is not None]
        if non_none and len(set(str(v) for v in non_none)) > 1:
            all_consistent = False
            warnings.append(f"Scene field {field} differs across scenes: {values}")

    return {
        "consistent": all_consistent,
        "band_consistency": band_consistency,
        "warnings": warnings,
    }


# ===========================================================================
# Scene acquisition condition table
# ===========================================================================

def build_scene_acquisition_table(
    scene_metadata_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a table of acquisition conditions for each scene.

    Parameters
    ----------
    scene_metadata_list : list of dict
        Each dict is the output of parse_dz01_mtl().

    Returns
    -------
    list of dict
        One dict per scene with all acquisition parameters.
    """
    table = []
    for meta in scene_metadata_list:
        scene = meta.get("scene", {})
        row = {
            "scene_id": meta.get("date_acquired", "unknown"),
            "date_acquired": meta.get("date_acquired"),
            "scene_center_time": meta.get("scene_center_time"),
            "start_imaging_time": scene.get("START_IMAGING_TIME"),
            "end_imaging_time": scene.get("END_IMAGING_TIME"),
            "cloud_cover": scene.get("CLOUD_COVER"),
            "sun_azimuth": scene.get("SUN_AZIMUTH"),
            "sun_zenith": scene.get("SUN_ZENITH"),
            "sun_elevation": scene.get("SUN_ELEVATION"),
            "sat_azimuth": scene.get("SAT_AZIMUTH"),
            "sat_zenith": scene.get("SAT_ZENITH"),
            "roll_angle": scene.get("ROLL_ANGLE"),
            "pitch_angle": scene.get("PITCH_ANGLE"),
            "yaw_angle": scene.get("YAW_ANGLE"),
            "processing_software_version": scene.get("PROCESSING_SOFTWARE_VERSION"),
            "resolution_vi": meta.get("resolution_vi"),
            "sensor_id": meta.get("sensor_id"),
            "n_bands": len(meta.get("bands", {})),
        }
        table.append(row)
    return table


def analyze_acquisition_differences(
    scene_metadata_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze differences in acquisition conditions across scenes.

    Parameters
    ----------
    scene_metadata_list : list of dict
        Each dict is the output of parse_dz01_mtl().

    Returns
    -------
    dict with summary statistics and differences.
    """
    table = build_scene_acquisition_table(scene_metadata_list)

    if not table:
        return {"error": "No scenes provided"}

    # Numeric fields to analyze
    numeric_fields = [
        "cloud_cover", "sun_azimuth", "sun_zenith", "sun_elevation",
        "sat_azimuth", "sat_zenith", "roll_angle", "pitch_angle", "yaw_angle",
    ]

    analysis: Dict[str, Any] = {"scenes": table, "field_stats": {}}

    for field in numeric_fields:
        values = []
        scene_ids = []
        for row in table:
            val = row.get(field)
            if val is not None:
                try:
                    values.append(float(val))
                    scene_ids.append(row["scene_id"])
                except (ValueError, TypeError):
                    pass

        if values:
            import statistics
            median_val = statistics.median(values)
            analysis["field_stats"][field] = {
                "min": min(values),
                "max": max(values),
                "median": median_val,
                "range": max(values) - min(values),
                "values": dict(zip(scene_ids, values)),
            }

    return analysis


# ===========================================================================
# Band acquisition parameters (long-format table)
# ===========================================================================

def build_band_acquisition_table(
    scene_metadata_list: List[Dict[str, Any]],
    selected_bands: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a long-format table of per-band integration parameters.

    Parameters
    ----------
    scene_metadata_list : list of dict
        Each dict is the output of parse_dz01_mtl().
    selected_bands : list of str or None
        Bands to include. None = all bands from first scene.

    Returns
    -------
    list of dict
        One dict per (scene, band) combination.
    """
    if not scene_metadata_list:
        return []

    if selected_bands is None:
        selected_bands = sorted(scene_metadata_list[0].get("bands", {}).keys())

    table = []
    for meta in scene_metadata_list:
        scene_id = meta.get("date_acquired", "unknown")
        bands = meta.get("bands", {})
        for band_name in selected_bands:
            band_info = bands.get(band_name, {})
            center_wl = band_info.get("wavelength_center_nm")
            table.append({
                "scene_id": scene_id,
                "date_acquired": meta.get("date_acquired"),
                "band_name": band_name,
                "wavelength_center_nm": center_wl,
                "integration_time": band_info.get("integration_time"),
                "integration_level": band_info.get("integration_level"),
            })

    return table


# ===========================================================================
# Integration parameter anomaly detection
# ===========================================================================

def detect_integration_anomalies(
    scene_metadata_list: List[Dict[str, Any]],
    selected_bands: Optional[List[str]] = None,
    time_ratio_threshold: float = 1.5,
    time_ratio_low: float = 0.67,
) -> List[Dict[str, Any]]:
    """
    Detect integration time/level anomalies across scenes.

    For each band, computes the median integration_time and integration_level
    across all scenes, then flags scenes where the ratio exceeds thresholds.

    Parameters
    ----------
    scene_metadata_list : list of dict
        Each dict is the output of parse_dz01_mtl().
    selected_bands : list of str or None
        Bands to check. None = all bands from first scene.
    time_ratio_threshold : float
        Upper threshold for integration_time_ratio (default 1.5).
    time_ratio_low : float
        Lower threshold for integration_time_ratio (default 0.67).

    Returns
    -------
    list of dict
        One dict per anomaly with scene_id, band_name, parameter, value,
        cross_scene_median, ratio, status.
    """
    if not scene_metadata_list:
        return []

    if selected_bands is None:
        selected_bands = sorted(scene_metadata_list[0].get("bands", {}).keys())

    import statistics

    anomalies = []

    for band_name in selected_bands:
        # Collect integration_time and integration_level across scenes
        time_values = {}
        level_values = {}
        for meta in scene_metadata_list:
            scene_id = meta.get("date_acquired", "unknown")
            band_info = meta.get("bands", {}).get(band_name, {})
            it = band_info.get("integration_time")
            il = band_info.get("integration_level")
            if it is not None:
                time_values[scene_id] = float(it)
            if il is not None:
                level_values[scene_id] = float(il)

        # Compute medians
        time_median = statistics.median(time_values.values()) if time_values else None
        level_median = statistics.median(level_values.values()) if level_values else None

        # Check for anomalies
        for scene_id, it_val in time_values.items():
            if time_median is not None and time_median > 0:
                ratio = it_val / time_median
                if ratio > time_ratio_threshold or ratio < time_ratio_low:
                    anomalies.append({
                        "scene_id": scene_id,
                        "band_name": band_name,
                        "parameter": "integration_time",
                        "value": it_val,
                        "cross_scene_median": time_median,
                        "ratio": round(ratio, 4),
                        "status": "anomaly",
                    })

        for scene_id, il_val in level_values.items():
            if level_median is not None and level_median > 0:
                ratio = il_val / level_median
                if ratio > time_ratio_threshold or ratio < time_ratio_low:
                    anomalies.append({
                        "scene_id": scene_id,
                        "band_name": band_name,
                        "parameter": "integration_level",
                        "value": il_val,
                        "cross_scene_median": level_median,
                        "ratio": round(ratio, 4),
                        "status": "anomaly",
                    })

    return anomalies
