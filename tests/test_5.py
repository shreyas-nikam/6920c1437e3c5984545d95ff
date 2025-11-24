import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Keep the definition_03a74b27ba74489b86eec7521d374547 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_03a74b27ba74489b86eec7521d374547 import plot_dendrogram


# Helper function to create a dummy linkage matrix for tests
def create_dummy_linkage_matrix(n_samples=3):
    """Generates a simple linkage matrix for testing purposes."""
    if n_samples < 2:
        return np.array([])
    # A simplified linkage matrix structure: [cluster_id1, cluster_id2, distance, num_original_obs]
    if n_samples == 2:
        return np.array([[0., 1., 1.0, 2.]])
    # For n_samples > 2, create a cumulative sequence of merges
    linkage_rows = []
    for i in range(n_samples - 1):
        if i == 0:
            linkage_rows.append([0., 1., 1.0, 2.])
        else:
            # Merge the next original sample with the previous merged cluster
            linkage_rows.append([float(i + 1), float(n_samples + i - 1), 1.0 + (i * 0.5), float(i + 2)])
    return np.array(linkage_rows)


@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.gca') # get current axes
@patch('scipy.cluster.hierarchy.dendrogram')
def test_plot_dendrogram_valid_inputs(mock_dendrogram, mock_gca, mock_show, mock_figure):
    """
    Test Case 1: Ensures that `plot_dendrogram` correctly processes valid inputs,
    calls the underlying Matplotlib and Scipy functions, and displays the plot.
    """
    # Arrange
    num_samples = 4
    linkage_matrix = create_dummy_linkage_matrix(n_samples=num_samples)
    cutoff_distance = 2.0
    feature_data = pd.DataFrame({'Asset_ID': [f'Asset_{i}' for i in range(num_samples)], 'Feature_1': [1,2,3,4]})

    # Mock Matplotlib figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_figure.return_value = mock_fig
    mock_gca.return_value = mock_ax

    # Act
    plot_dendrogram(linkage_matrix, cutoff_distance, feature_data)

    # Assert
    mock_figure.assert_called_once()
    mock_gca.assert_called_once()
    mock_dendrogram.assert_called_once_with(
        linkage_matrix,
        ax=mock_ax, # Expecting `ax` to be passed for plotting on a specific axes
        leaf_rotation=90,
        leaf_font_size=8,
        labels=feature_data['Asset_ID'].values # Expecting Asset_ID to be used for labels
    )
    mock_ax.axhline.assert_called_once_with(y=cutoff_distance, color='red', linestyle='--')
    mock_ax.set_title.assert_called_once_with('Hierarchical Clustering Dendrogram')
    mock_ax.set_xlabel.assert_called_once_with('Asset Index or Asset_ID')
    mock_ax.set_ylabel.assert_called_once_with('Distance')
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.gca')
@patch('scipy.cluster.hierarchy.dendrogram')
def test_plot_dendrogram_minimal_data(mock_dendrogram, mock_gca, mock_show, mock_figure):
    """
    Test Case 2: Verifies behavior with the minimum number of samples (2) which results
    in a single merge in the linkage matrix.
    """
    # Arrange
    num_samples = 2
    linkage_matrix = create_dummy_linkage_matrix(n_samples=num_samples)
    cutoff_distance = 0.5
    feature_data = pd.DataFrame({'Asset_ID': [f'Asset_{i}' for i in range(num_samples)], 'Feature_1': [10, 20]})

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_figure.return_value = mock_fig
    mock_gca.return_value = mock_ax

    # Act
    plot_dendrogram(linkage_matrix, cutoff_distance, feature_data)

    # Assert
    mock_dendrogram.assert_called_once()
    mock_ax.axhline.assert_called_once_with(y=cutoff_distance, color='red', linestyle='--')
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.gca')
@patch('scipy.cluster.hierarchy.dendrogram')
def test_plot_dendrogram_extreme_cutoff_distances(mock_dendrogram, mock_gca, mock_show, mock_figure):
    """
    Test Case 3: Checks that `plot_dendrogram` handles extreme `cutoff_distance_input` values
    (e.g., zero or very high) without errors, ensuring the horizontal line is still drawn.
    """
    # Arrange
    num_samples = 3
    linkage_matrix = create_dummy_linkage_matrix(n_samples=num_samples)
    feature_data = pd.DataFrame({'Asset_ID': [f'Asset_{i}' for i in range(num_samples)], 'Feature_1': [1,2,3]})

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_figure.return_value = mock_fig
    mock_gca.return_value = mock_ax

    # Test with cutoff = 0.0
    plot_dendrogram(linkage_matrix, 0.0, feature_data)
    mock_ax.axhline.assert_called_once_with(y=0.0, color='red', linestyle='--')
    mock_ax.axhline.reset_mock() # Reset for the next assertion

    # Test with a very high cutoff (typically beyond any dendrogram merge height)
    plot_dendrogram(linkage_matrix, 1000.0, feature_data)
    mock_ax.axhline.assert_called_once_with(y=1000.0, color='red', linestyle='--')
    assert mock_show.call_count == 2 # show() should be called twice


@pytest.mark.parametrize("invalid_linkage, expected_error", [
    (None, TypeError),
    ("not a numpy array", TypeError),
    ([1, 2, 3], TypeError), # List instead of numpy array
    (np.array([[1, 2, 3]]), ValueError) # Malformed linkage matrix (wrong number of columns, expected 4)
])
def test_plot_dendrogram_invalid_linkage_matrix_type_or_format(invalid_linkage, expected_error):
    """
    Test Case 4: Ensures that `plot_dendrogram` raises appropriate errors (TypeError or ValueError)
    when `linkage_matrix` is of an invalid type or malformed.
    """
    # Arrange
    cutoff_distance = 1.0
    feature_data = pd.DataFrame({'Asset_ID': ['A', 'B'], 'Feature_1': [1,2]})

    # Act & Assert
    with pytest.raises(expected_error):
        plot_dendrogram(invalid_linkage, cutoff_distance, feature_data)


@pytest.mark.parametrize("invalid_cutoff, expected_error", [
    ("not a float", TypeError),
    ([1.0], TypeError),
    (None, TypeError),
    (np.array([5.0]), TypeError), # Numpy array of a single float, not a scalar float
])
def test_plot_dendrogram_invalid_cutoff_type(invalid_cutoff, expected_error):
    """
    Test Case 5: Ensures that `plot_dendrogram` raises a TypeError when `cutoff_distance_input`
    is not a scalar float or convertible to one.
    """
    # Arrange
    linkage_matrix = create_dummy_linkage_matrix(n_samples=3)
    feature_data = pd.DataFrame({'Asset_ID': ['A', 'B', 'C'], 'Feature_1': [1,2,3]})

    # Act & Assert
    with pytest.raises(expected_error):
        plot_dendrogram(linkage_matrix, invalid_cutoff, feature_data)