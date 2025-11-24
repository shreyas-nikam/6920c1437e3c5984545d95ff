import pytest
import pandas as pd
import numpy as np
# Assuming sklearn.datasets.make_blobs would be used internally by the function
from sklearn.datasets import make_blobs 

# Placeholder for your module import
# definition_cef0f8546fd24687ad64b91aa83cddd6
# Assume generate_financial_data is in this module
from definition_cef0f8546fd24687ad64b91aa83cddd6 import generate_financial_data

# Test Case 1: Standard functionality with typical inputs
def test_generate_financial_data_standard_case():
    n_samples = 100
    n_features = 3
    n_clusters = 4
    cluster_std = 0.8
    random_state = 42

    df, y_true = generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state)

    # Check return types
    assert isinstance(df, pd.DataFrame)
    assert isinstance(y_true, np.ndarray)

    # Check DataFrame and array shapes
    assert df.shape == (n_samples, n_features + 2) # +2 for Asset_ID and True_Cluster
    assert y_true.shape == (n_samples,)

    # Check column names
    expected_columns = ['Asset_ID'] + [f'Feature_{i+1}' for i in range(n_features)] + ['True_Cluster']
    assert list(df.columns) == expected_columns

    # Check Asset_ID format and uniqueness
    assert all(df['Asset_ID'].str.match(r'Asset_\d+'))
    assert df['Asset_ID'].nunique() == n_samples

    # Check True_Cluster consistency with y_true and value range
    assert np.array_equal(df['True_Cluster'].values, y_true)
    assert len(np.unique(y_true)) == n_clusters
    assert np.min(y_true) == 0
    assert np.max(y_true) == n_clusters - 1

    # Check feature column data types (should be float)
    for i in range(n_features):
        assert pd.api.types.is_float_dtype(df[f'Feature_{i+1}'])

# Test Case 2: Edge case - Zero samples (n_samples=0)
def test_generate_financial_data_zero_samples():
    n_samples = 0
    n_features = 3
    n_clusters = 1 # make_blobs requires n_clusters >= 1 even for n_samples=0
    cluster_std = 1.0
    random_state = 42

    df, y_true = generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state)

    # Check DataFrame and array shapes for zero samples
    assert df.shape == (0, n_features + 2)
    assert y_true.shape == (0,)

    # Columns should still be correctly defined, even if the DataFrame is empty
    expected_columns = ['Asset_ID'] + [f'Feature_{i+1}' for i in range(n_features)] + ['True_Cluster']
    assert list(df.columns) == expected_columns

# Test Case 3: Edge case - Single cluster (n_clusters=1)
def test_generate_financial_data_single_cluster():
    n_samples = 50
    n_features = 3
    n_clusters = 1
    cluster_std = 0.5
    random_state = 1

    df, y_true = generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state)

    # Check shapes
    assert df.shape == (n_samples, n_features + 2)
    assert y_true.shape == (n_samples,)

    # All cluster labels should be 0 for a single cluster
    assert np.all(y_true == 0)
    assert np.all(df['True_Cluster'] == 0)
    assert len(np.unique(y_true)) == 1

# Test Case 4: Dynamic n_features (e.g., 2 features) and cluster_std as a list
def test_generate_financial_data_dynamic_features_and_list_std():
    n_samples = 75
    n_features = 2 # Test with fewer than the "example" 3 features
    n_clusters = 3
    cluster_std = [0.5, 1.0, 0.7] # Provide a list of stds
    random_state = 10

    df, y_true = generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state)

    # Check shapes adapt to n_features
    assert df.shape == (n_samples, n_features + 2) # Asset_ID, F1, F2, True_Cluster
    assert y_true.shape == (n_samples,)

    # Check column names reflect the actual n_features
    expected_columns = ['Asset_ID'] + [f'Feature_{i+1}' for i in range(n_features)] + ['True_Cluster']
    assert list(df.columns) == expected_columns
    assert 'Feature_3' not in df.columns # Should not exist if n_features is 2

    # Check True_Cluster and y_true consistency and range
    assert np.array_equal(df['True_Cluster'].values, y_true)
    assert len(np.unique(y_true)) == n_clusters
    assert np.min(y_true) == 0
    assert np.max(y_true) == n_clusters - 1

# Test Case 5: Error handling - n_clusters greater than n_samples
def test_generate_financial_data_invalid_n_clusters():
    n_samples = 10
    n_features = 3
    n_clusters = 15 # More clusters than samples, which make_blobs cannot handle
    cluster_std = 1.0
    random_state = 5

    # Expecting a ValueError to be raised by the underlying make_blobs function
    with pytest.raises(ValueError) as excinfo:
        generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state)
    
    # Check that the error message indicates the problem with n_clusters vs n_samples
    assert "n_samples must be greater than or equal to n_clusters" in str(excinfo.value)