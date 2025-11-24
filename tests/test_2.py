import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans # Used for error type assertions and internal logic understanding

# Keep the definition_6da0945688a9453086744848c2035e26 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_6da0945688a9453086744848c2035e26 import perform_kmeans_clustering


@pytest.fixture
def sample_scaled_data():
    """Generates a synthetic pandas DataFrame to simulate scaled financial data."""
    # Using make_blobs to create data with some inherent cluster structure
    X, _ = make_blobs(n_samples=20, n_features=3, centers=4, cluster_std=0.8, random_state=42)
    return pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])


# Test Case 1: Basic functionality with valid inputs
def test_perform_kmeans_clustering_basic_functionality(sample_scaled_data):
    """
    Tests if the function performs k-Means clustering correctly with typical inputs,
    checking return types, shapes, and basic properties of labels and centroids.
    """
    n_clusters = 3
    random_state = 42
    
    labels, centroids = perform_kmeans_clustering(sample_scaled_data, n_clusters, random_state)
    
    # Assertions for cluster labels
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (sample_scaled_data.shape[0],)
    assert labels.dtype == np.int32 or labels.dtype == np.int64 # K-Means labels are typically int
    # Number of unique labels should be less than or equal to n_clusters (can be less if some clusters are empty)
    assert len(np.unique(labels)) <= n_clusters 
    assert np.all(labels >= 0) and np.all(labels < n_clusters) # Labels should be within [0, n_clusters-1]

    # Assertions for cluster centroids
    assert isinstance(centroids, np.ndarray)
    assert centroids.shape == (n_clusters, sample_scaled_data.shape[1])
    assert centroids.dtype == np.float32 or centroids.dtype == np.float64 # Centroids are float coordinates


# Test Case 2: Edge cases for n_clusters (k=1 and k=n_samples)
@pytest.mark.parametrize("n_clusters_input, expected_num_unique_labels", [
    (1, 1),  # Edge case: All samples in one cluster
    (20, 20) # Edge case: Each sample is its own cluster (n_samples for sample_scaled_data is 20)
])
def test_perform_kmeans_clustering_edge_n_clusters(sample_scaled_data, n_clusters_input, expected_num_unique_labels):
    """
    Tests edge cases where n_clusters is 1 or equal to the number of samples.
    """
    random_state = 42
    
    labels, centroids = perform_kmeans_clustering(sample_scaled_data, n_clusters_input, random_state)
    
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (sample_scaled_data.shape[0],)
    assert isinstance(centroids, np.ndarray)
    assert centroids.shape == (n_clusters_input, sample_scaled_data.shape[1])

    # Specific assertion for n_clusters = 1: all labels should be 0
    if n_clusters_input == 1:
        assert np.all(labels == 0)
    # Specific assertion for n_clusters = n_samples: each sample should be in its own unique cluster
    elif n_clusters_input == sample_scaled_data.shape[0]:
        assert len(np.unique(labels)) == expected_num_unique_labels
        # And conceptually, centroids should be very close to the data points themselves,
        # but direct equality check is complex due to floating point and assignment order.


# Test Case 3: Invalid n_clusters_input type
@pytest.mark.parametrize("invalid_n_clusters_type", [
    "3",       # String
    3.0,       # Float (KMeans expects int)
    None,      # NoneType
    [3],       # List
    np.array(3) # Numpy scalar
])
def test_perform_kmeans_clustering_invalid_n_clusters_type(sample_scaled_data, invalid_n_clusters_type):
    """
    Tests if the function raises a TypeError when n_clusters_input is not an integer.
    KMeans expects an integer for n_clusters.
    """
    random_state = 42
    with pytest.raises(TypeError, match="n_clusters should be an integer"): # Scikit-learn's KMeans init raises this
        perform_kmeans_clustering(sample_scaled_data, invalid_n_clusters_type, random_state)


# Test Case 4: Invalid n_clusters_input value (<= 0 or > n_samples)
@pytest.mark.parametrize("invalid_n_clusters_value", [
    0,   # n_clusters must be >= 1
    -1,  # n_clusters must be >= 1
    21   # n_clusters > n_samples (sample_scaled_data has 20 samples)
])
def test_perform_kmeans_clustering_invalid_n_clusters_value(sample_scaled_data, invalid_n_clusters_value):
    """
    Tests if the function raises a ValueError when n_clusters_input is invalid
    (e.g., non-positive or greater than the number of samples).
    """
    random_state = 42
    with pytest.raises(ValueError, match="n_clusters"): # Scikit-learn's KMeans init raises ValueError
        perform_kmeans_clustering(sample_scaled_data, invalid_n_clusters_value, random_state)


# Test Case 5: Invalid scaled_data type
@pytest.mark.parametrize("invalid_scaled_data", [
    None,
    [1, 2, 3, 4, 5, 6], # A list
    "not a dataframe", # A string
    123 # An integer
    # Note: A numpy array (e.g., np.array([[1, 2, 3], ...])) would likely be accepted by KMeans.fit
    # without an error, as KMeans can handle array-like inputs.
    # However, per the docstring, it *expects* a pandas.DataFrame, so passing non-DataFrame
    # inputs that would cause direct issues in KMeans.fit or internal DataFrame operations are covered.
])
def test_perform_kmeans_clustering_invalid_scaled_data_type(invalid_scaled_data):
    """
    Tests if the function raises an appropriate error when `scaled_data` is not a pandas.DataFrame
    or is otherwise unsuitable for k-Means clustering.
    """
    n_clusters = 3
    random_state = 42
    # The KMeans.fit method expects array-like input. If not convertible, it can raise
    # TypeError or ValueError. If the stub implementation tried to access DataFrame-specific
    # attributes before passing to KMeans, it could also raise an AttributeError.
    with pytest.raises((ValueError, TypeError, AttributeError)):
        perform_kmeans_clustering(invalid_scaled_data, n_clusters, random_state)