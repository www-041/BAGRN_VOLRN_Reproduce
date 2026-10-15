"""Runtime compatibility checks for the isolated Spectralmatch adapter."""

from __future__ import annotations

import platform
import re
from typing import Any


def _version_tuple(value: Any, *, gdal_numeric: bool = False) -> tuple[int, ...]:
    text = str(value).strip()
    if gdal_numeric and text.isdigit() and len(text) >= 6:
        number = int(text)
        return number // 1_000_000, (number // 10_000) % 100, (number // 100) % 100
    match = re.search(r"\d+(?:\.\d+){0,3}", text)
    if not match:
        raise ValueError(f"unparseable version: {value!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def _version_at_least(value: Any, minimum: tuple[int, ...], *, gdal_numeric: bool = False) -> bool:
    parsed = _version_tuple(value, gdal_numeric=gdal_numeric)
    return (parsed + (0,) * (len(minimum) - len(parsed))) >= minimum


def _collect_runtime() -> dict[str, str]:
    import rasterio
    from osgeo import gdal

    proj = getattr(rasterio, "__proj_version__", None)
    if not proj:
        try:
            import pyproj

            proj = pyproj.proj_version_str
        except ImportError:
            proj = "UNKNOWN"
    gdal_raw = gdal.VersionInfo()
    return {
        "python_version": platform.python_version(),
        "osgeo_gdal_version": gdal.VersionInfo("RELEASE_NAME") or gdal_raw,
        "osgeo_gdal_version_info": str(gdal_raw),
        "rasterio_gdal_version": str(rasterio.__gdal_version__),
        "proj_version": str(proj),
    }


def audit_runtime(*, runtime_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report runtime versions and whether the upstream environment is met.

    ``runtime_info`` is an explicit test/diagnostic seam. In normal use the
    values are collected from the active Python, GDAL, Rasterio and PROJ stack.
    """
    try:
        info = dict(runtime_info) if runtime_info is not None else _collect_runtime()
    except (ImportError, OSError) as exc:
        return {
            "status": "INCOMPATIBLE",
            "requirements": {},
            "gdal_consistent": False,
            "reasons": [f"runtime_dependency_unavailable: {exc}"],
        }
    try:
        py = _version_tuple(info["python_version"])
        python_ok = (3, 10) <= (py[0], py[1]) <= (3, 12)
        gdal_ok = _version_at_least(info["osgeo_gdal_version"], (3, 11), gdal_numeric=True)
        proj_ok = _version_at_least(info["proj_version"], (9, 3))
        gdal_consistent = _version_tuple(info["osgeo_gdal_version"], gdal_numeric=True)[:3] == _version_tuple(
            info["rasterio_gdal_version"]
        )[:3]
        checks = {
            "python_3_10_to_3_12": python_ok,
            "gdal_at_least_3_11": gdal_ok,
            "proj_at_least_9_3": proj_ok,
        }
        compatible = all(checks.values()) and gdal_consistent
        return {
            **info,
            "requirements": checks,
            "gdal_consistent": gdal_consistent,
            "status": "COMPATIBLE" if compatible else "INCOMPATIBLE",
            "reasons": [key for key, value in checks.items() if not value]
            + ([] if gdal_consistent else ["rasterio_gdal_mismatch"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            **info,
            "requirements": {},
            "gdal_consistent": False,
            "status": "INCOMPATIBLE",
            "reasons": [f"runtime_audit_error: {exc}"],
        }
