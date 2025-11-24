import pytest
import pandas as pd
import numpy as np
# We need these imports to correctly predict the exception types,
# as the stub function will internally call sklearn and scipy.
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

# definition_f8f68a0138984572ac84600db697c086 block START
from definition_f8f68a0138984572ac84600db697c086 import perform_hierarchical_clustering
# definition_f8f68a0138984572ac84600db697c086 block END

# Sample data for testing
# 5 samples, 3 features
sample_scaled_data = pd.DataFrame(np.array([
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0],
    [5.1, 6.1, 7.1],
    [10.0, 11.0, 12.0]
]), columns=['F1', 'F2', 'F3'])

# Single sample data for edge case
single_sample_data = pd.DataFrame(np.array([[1.0, 2.0, 3.0]]), columns=['F1', 'F2', 'F3'])


@pytest.mark.parametrize(
    "scaled_data, n_clusters_hc, linkage_method, expected_labels_shape, expected_linkage_shape, expected_exception",
    [
        # Test Case 1: Basic functionality with 'ward' linkage and typical data
        # Expect labels for 5 samples, and linkage matrix for (5-1) merges
        (sample_scaled_data, 3, 'ward', (5,), (4, 4), None),

        # Test Case 2: Different linkage method ('single') with typical data
        # Expect labels for 5 samples, and linkage matrix for (5-1) merges
        (sample_scaled_data, 2, 'single', (5,), (4, 4), None),

        # Test Case 3: Edge case - single data point
        # AgglomerativeClustering.fit_predict would return [0], but linkage function raises ValueError
        # for a single observation. The function will raise ValueError before returning a tuple.
        (single_sample_data, 1, 'ward', None, None, ValueError),

        # Test Case 4: Invalid linkage method string
        # Both AgglomerativeClustering and linkage function will raise ValueError for unknown method.
        (sample_scaled_data, 2, 'invalid_method', None, None, ValueError),

        # Test Case 5: Invalid scaled_data type (e.g., None)
        # sklearn.cluster methods and scipy.cluster.hierarchy.linkage expect array-like input,
        # passing None will result in a TypeError.
        (None, 2, 'ward', None, None, TypeError),
    ]
)
def test_perform_hierarchical_clustering(
    scaled_data, n_clusters_hc, linkage_method,
    expected_labels_shape, expected_linkage_shape, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            perform_hierarchical_clustering(scaled_data, n_clusters_hc, linkage_method)
    else:
        labels, linkage_matrix = perform_hierarchical_clustering(scaled_data, n_clusters_hc, linkage_method)

        # Assert types
        assert isinstance(labels, np.ndarray)
        assert isinstance(linkage_matrix, np.ndarray)

        # Assert shapes
        assert labels.shape == expected_labels_shape
        assert linkage_matrix.shape == expected_linkage_shape

        # Assert content properties (e.g., number of unique clusters)
        # The number of unique labels should be less than or equal to n_clusters_hc
        # and at least 1 if there's data.
        if expected_labels_shape[0] > 0:
            assert 1 <= len(np.unique(labels)) <= n_clusters_hc

        # Linkage matrix dtype is typically float64
        assert linkage_matrix.dtype == np.float64