import pytest
import numpy as np
from definition_6fe3cf80afdc49ba91a5343da7d8564a import calculate_adjusted_rand_index

@pytest.mark.parametrize("labels_true, labels_pred, expected_result", [
    # Test case 1: Perfect agreement (identical labels)
    (np.array([0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2]), 1.0),
    # Test case 2: Perfect agreement (label permutation, ARI is invariant to label relabeling)
    (np.array([0, 0, 1, 1, 2]), np.array([1, 1, 0, 0, 3]), 1.0),
    # Test case 3: No agreement (random-like partitioning)
    (np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2]), 0.0),
    # Test case 4: Partial agreement
    (np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 2, 1, 2]), pytest.approx(0.5447154471544716)),
    # Test case 5: Edge case - Mismatched input lengths (should raise ValueError)
    (np.array([0, 1, 2]), np.array([0, 1]), ValueError),
])
def test_calculate_adjusted_rand_index(labels_true, labels_pred, expected_result):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            calculate_adjusted_rand_index(labels_true, labels_pred)
    else:
        result = calculate_adjusted_rand_index(labels_true, labels_pred)
        assert result == expected_result