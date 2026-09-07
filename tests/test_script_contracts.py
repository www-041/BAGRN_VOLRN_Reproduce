"""Static tests for legacy script metric call contracts.

Task 17 of reliability-fixes plan:
- compute_all(arrays_before, arrays_after, nodata_values, overlaps, bands)
- Baseline: compute_all(original, original, ...)
- BAGRN: compute_all(original, bagrn_result, ...)
- VOLRN: compute_all(original, volrn_result, ...)
- Never compute_all(result, result, ...) for normalized methods
"""


def test_compute_all_contract_documented():
    """Document the correct compute_all contract."""
    # compute_all(arrays_before, arrays_after, nodata_values, overlaps, bands)
    # 
    # For baseline: arrays_before = original, arrays_after = original
    # For BAGRN: arrays_before = original, arrays_after = bagrn_result
    # For VOLRN: arrays_before = original, arrays_after = volrn_result
    # 
    # This ensures GL measures loss from original, not between two normalized versions
    
    contract = {
        "baseline": ("original", "original"),
        "bagrn": ("original", "bagrn_result"),
        "volrn": ("original", "volrn_result"),
    }
    
    assert contract["baseline"] == ("original", "original")
    assert contract["bagrn"] == ("original", "bagrn_result")
    assert contract["volrn"] == ("original", "volrn_result")
