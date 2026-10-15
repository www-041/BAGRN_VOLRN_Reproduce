from src.benchmark_v1_spectralmatch_phase2 import environment
from src.benchmark_v1_spectralmatch_phase2.environment import audit_runtime


def test_matching_supported_runtime_is_compatible():
    result = audit_runtime(
        runtime_info={
            "python_version": "3.11.8",
            "osgeo_gdal_version": "3.11.5",
            "rasterio_gdal_version": "3.11.5",
            "proj_version": "9.5.1",
        }
    )

    assert result["status"] == "COMPATIBLE"
    assert result["gdal_consistent"] is True
    assert result["requirements"] == {
        "python_3_10_to_3_12": True,
        "gdal_at_least_3_11": True,
        "proj_at_least_9_3": True,
    }


def test_mismatched_rasterio_gdal_is_incompatible():
    result = audit_runtime(
        runtime_info={
            "python_version": "3.11.8",
            "osgeo_gdal_version": "3.11.5",
            "rasterio_gdal_version": "3.10.3",
            "proj_version": "9.5.1",
        }
    )

    assert result["status"] == "INCOMPATIBLE"
    assert result["gdal_consistent"] is False


def test_each_below_minimum_version_is_incompatible():
    cases = [
        ("3.9.18", "3.11.5", "3.11.5", "9.5.1", "python_3_10_to_3_12"),
        ("3.11.8", "3.10.3", "3.10.3", "9.5.1", "gdal_at_least_3_11"),
        ("3.11.8", "3.11.5", "3.11.5", "9.2.1", "proj_at_least_9_3"),
    ]
    for python, gdal, rio_gdal, proj, failed_requirement in cases:
        result = audit_runtime(
            runtime_info={
                "python_version": python,
                "osgeo_gdal_version": gdal,
                "rasterio_gdal_version": rio_gdal,
                "proj_version": proj,
            }
        )
        assert result["status"] == "INCOMPATIBLE"
        assert result["requirements"][failed_requirement] is False


def test_missing_runtime_dependency_is_reported_not_raised(monkeypatch):
    def missing_dependency():
        raise ModuleNotFoundError("rasterio is unavailable")

    monkeypatch.setattr(environment, "_collect_runtime", missing_dependency)
    result = audit_runtime()

    assert result["status"] == "INCOMPATIBLE"
    assert result["reasons"] == ["runtime_dependency_unavailable: rasterio is unavailable"]
