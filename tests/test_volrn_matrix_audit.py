import json


def test_matrix_audit_reports_shape_density_and_scales(tmp_path):
    from scripts.audit_volrn_matrix import audit_volrn_matrix

    summary = {
        "volrn": {
            "diagnostics": {
                "n_blocks": 2,
                "n_pairs": 1,
                "block_details": [
                    {"band": 0, "block_id": 0, "image_idx": 0, "grid_m": 0, "grid_n": 0, "mu": 10.0, "sigma": 2.0},
                    {"band": 0, "block_id": 1, "image_idx": 1, "grid_m": 0, "grid_n": 0, "mu": 12.0, "sigma": 3.0},
                ],
                "rho": 1.0,
            }
        }
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "matrix.json"

    result = audit_volrn_matrix(source, output)

    assert result["num_blocks"] == 2
    assert result["num_pairs"] == 1
    assert result["num_variables"] == 4
    assert 0.0 < result["matrix_density"] <= 1.0
    assert result["condition_estimate"] >= 1.0
    assert result["row_scale"]["max"] >= result["row_scale"]["min"]
    assert result["column_scale"]["max"] >= result["column_scale"]["min"]
    assert json.loads(output.read_text(encoding="utf-8"))["num_blocks"] == 2
