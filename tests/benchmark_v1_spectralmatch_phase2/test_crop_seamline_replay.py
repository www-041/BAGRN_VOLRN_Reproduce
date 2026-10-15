import os
import subprocess
from pathlib import Path


def test_saved_00_10_emp_selection_replays_official_seam_geometries():
    root = Path(__file__).resolve().parents[2]
    temp = Path(os.environ["TEMP"])
    python = temp / "spectralmatch_conda_phase2_2c35fef0" / "python.exe"
    upstream = temp / "spectralmatch_phase2_upstream_2c35fef0"
    stage = root / "data/output/benchmark_v1_unified/spectralmatch_phase2_20261013_retry5/00_10/worker_output/00_10/crop_smoke/stages"
    assert python.is_file()
    assert stage.joinpath("seamline_debug.gpkg").is_file()
    script = r"""
import sys, fiona
from pathlib import Path
from shapely.geometry import shape
sys.path.insert(0, str(Path(sys.argv[1])))
from spectralmatch.seamline.voronoi_center_seamline import _segment_emp
stage=Path(sys.argv[2]); debug=stage/'seamline_debug.gpkg'; saved=stage/'seamlines.gpkg'
emps={f['properties']['image']:shape(f['geometry']) for f in fiona.open(debug, layer='emp_outline')}
cuts=[shape(f['geometry']) for f in fiona.open(debug, layer='cutlines')]
outs={f['properties']['image']:shape(f['geometry']) for f in fiona.open(saved, layer='seamlines')}
assert set(emps)==set(outs) and len(emps)==2
for scene, emp in emps.items():
    selected=_segment_emp(emp, cuts)
    assert selected.symmetric_difference(outs[scene]).area < 1e-6
    assert selected.area < emp.area
    print(scene, selected.symmetric_difference(outs[scene]).area, emp.area-selected.area)
"""
    completed = subprocess.run(
        [str(python), "-c", script, str(upstream), str(stage)],
        cwd=root, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "scene_000_B9_frozen_mask_nan_Local 0.0 41272125.194227666" in completed.stdout
    assert "scene_010_B9_frozen_mask_nan_Local 0.0 38416013.194227666" in completed.stdout
