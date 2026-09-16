import numpy as np


def make_pair(i, j, dx, dy, confidence=0.9, n_blocks=20, rmse=0.2):
    return {
        "idx_i": i,
        "idx_j": j,
        "shift_dx": float(dx),
        "shift_dy": float(dy),
        "confidence": float(confidence),
        "n_blocks": int(n_blocks),
        "rmse": float(rmse),
        "matches": [],
    }


def test_network_adjustment_anchors_reference_and_solves_chain():
    from src.coregistration import multi_image_network_adjustment

    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
    ]

    result = multi_image_network_adjustment(pairs, n_images=4, reference_idx=0)

    assert result["n_edges"] == 3
    np.testing.assert_allclose(
        result["global_shifts"],
        np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(result["global_shifts"][0], [0.0, 0.0])


def test_network_adjustment_supports_nonzero_reference_index():
    from src.coregistration import multi_image_network_adjustment

    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
    ]

    result = multi_image_network_adjustment(pairs, n_images=4, reference_idx=1)

    np.testing.assert_allclose(
        result["global_shifts"],
        np.array([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(result["global_shifts"][1], [0.0, 0.0])
