import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.datasets import make_blobs

# Keep a placeholder definition_5c0b6f011241423c90554e5db7109d3e for the import of the module.
# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_5c0b6f011241423c90554e5db7109d3e import calculate_silhouette_score


# Helper function to generate mock scaled data and labels
def generate_mock_data(n_samples, n_features, n_centers=1, cluster_std=0.5, random_state=42):
    """
    Generates synthetic data and cluster labels suitable for silhouette score calculation.
    If n_samples < n_centers, n_centers is adjusted to n_samples to avoid make_blobs error.
    """
    if n_samples < n_centers:
        n_centers = n_samples # Adjust centers to avoid error in make_blobs
    
    # make_blobs requires n_samples > 0. If n_samples is 0, handle it.
    if n_samples == 0:
        return pd.DataFrame(columns=[f'feature_{i}' for i in range(n_features)]), np.array([])

    X, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                           centers=n_centers, cluster_std=cluster_std, random_state=random_state)
    scaled_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    return scaled_data, labels

# Define test cases for parametrization
# Each tuple contains (scaled_data, cluster_labels, expected_output_or_exception)
test_data_normal_case, test_labels_normal_case = generate_mock_data(n_samples=50, n_features=3, n_centers=3, cluster_std=0.8)
expected_normal_case = sk_silhouette_score(test_data_normal_case, test_labels_normal_case)

test_data_single_cluster, test_labels_single_cluster = generate_mock_data(n_samples=20, n_features=2, n_centers=1, cluster_std=0.5)

test_data_min_valid, test_labels_min_valid = generate_mock_data(n_samples=2, n_features=2, n_centers=2, cluster_std=0.1)
expected_min_valid = sk_silhouette_score(test_data_min_valid, test_labels_min_valid)

test_data_mismatched = pd.DataFrame(np.random.rand(10, 3), columns=['f1', 'f2', 'f3'])
test_labels_mismatched = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]) # 9 labels for 10 samples

test_data_invalid_type = [np.random.rand(3).tolist(), np.random.rand(3).tolist()] # List of lists
test_labels_invalid_type = np.array([0, 1])


@pytest.mark.parametrize(
    "scaled_data, cluster_labels, expected",
    [
        # Test Case 1: Normal operation with multiple well-separated clusters
        (test_data_normal_case, test_labels_normal_case, expected_normal_case),

        # Test Case 2: Single cluster (n_labels <= 1), should raise ValueError
        # sklearn.metrics.silhouette_score raises ValueError if n_labels <= 1
        (test_data_single_cluster, test_labels_single_cluster, ValueError),

        # Test Case 3: Minimum valid input - 2 samples, 2 clusters
        (test_data_min_valid, test_labels_min_valid, expected_min_valid),

        # Test Case 4: Mismatched number of samples between data and labels
        (test_data_mismatched, test_labels_mismatched, ValueError),

        # Test Case 5: Invalid scaled_data type (e.g., a list of lists instead of DataFrame/array)
        (test_data_invalid_type, test_labels_invalid_type, TypeError),
    ],
    ids=[
        "normal_operation",
        "single_cluster_value_error",
        "min_valid_input",
        "mismatched_lengths",
        "invalid_scaled_data_type"
    ]
)
def test_calculate_silhouette_score(scaled_data, cluster_labels, expected):
    """
    Tests the calculate_silhouette_score function with various valid and invalid inputs.
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            calculate_silhouette_score(scaled_data, cluster_labels)
    else:
        result = calculate_silhouette_score(scaled_data, cluster_labels)
        # Use pytest.approx for float comparison due to potential precision differences
        assert result == pytest.approx(expected)