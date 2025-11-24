import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import MagicMock

# Block for your_module - DO NOT REMOVE OR REPLACE
from definition_8c97483909df4595b41a54a592325a4d import plot_kmeans_clusters
# End of your_module block


@pytest.fixture
def mock_plotly_express(mocker):
    """Mocks plotly.express module and plotly.graph_objects.Scatter."""
    # Mock px.scatter to return a mock figure object
    mock_fig = MagicMock(spec=go.Figure)
    plotly_express_scatter_mock = mocker.patch('plotly.express.scatter', return_value=mock_fig)
    
    # Mock go.Scatter constructor used for centroids
    plotly_go_scatter_mock = mocker.patch('plotly.graph_objects.Scatter', autospec=True)
    
    return plotly_express_scatter_mock, mock_fig, plotly_go_scatter_mock

@pytest.mark.parametrize(
    "test_id, original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y, expected_exception, expected_call_counts",
    [
        (
            "basic_functionality",
            pd.DataFrame({'Asset_ID': ['A1', 'A2'], 'Feature_1': [1, 2], 'Feature_2': [3, 4]}),
            pd.DataFrame({'Feature_1': [1.0, 2.0], 'Feature_2': [3.0, 4.0]}),
            np.array([0, 1]),
            np.array([[1.5, 3.5]]),
            'Feature_1',
            'Feature_2',
            None,
            {'px_scatter': 1, 'fig_add_trace': 1, 'fig_show': 1, 'go_scatter': 1}
        ),
        (
            "empty_data_no_centroids",
            pd.DataFrame({'Asset_ID': [], 'Feature_1': [], 'Feature_2': []}),
            pd.DataFrame({'Feature_1': [], 'Feature_2': []}), 
            np.array([]),
            np.array([]).reshape(0, 2), # Empty 2D array for centroids
            'Feature_1',
            'Feature_2',
            None,
            {'px_scatter': 1, 'fig_add_trace': 1, 'fig_show': 1, 'go_scatter': 1} # Plotly will still create and add an empty centroid trace
        ),
        (
            "single_data_point_with_centroid",
            pd.DataFrame({'Asset_ID': ['A1'], 'Feature_1': [1], 'Feature_2': [3]}),
            pd.DataFrame({'Feature_1': [1.0], 'Feature_2': [3.0]}),
            np.array([0]),
            np.array([[1.0, 3.0]]),
            'Feature_1',
            'Feature_2',
            None,
            {'px_scatter': 1, 'fig_add_trace': 1, 'fig_show': 1, 'go_scatter': 1}
        ),
        (
            "missing_feature_in_scaled_data",
            pd.DataFrame({'Asset_ID': ['A1'], 'Feature_1': [1], 'Feature_2': [3]}),
            pd.DataFrame({'Feature_1': [1.0], 'Feature_2': [3.0]}),
            np.array([0]),
            np.array([[1.0, 3.0]]),
            'NonExistentFeature',
            'Feature_2',
            KeyError, # Expected KeyError when accessing non-existent column
            {'px_scatter': 0, 'fig_add_trace': 0, 'fig_show': 0, 'go_scatter': 0}
        ),
        (
            "invalid_original_data_type",
            "not_a_dataframe", # Invalid type for original_data
            pd.DataFrame({'Feature_1': [1.0, 2.0], 'Feature_2': [3.0, 4.0]}),
            np.array([0, 1]),
            np.array([[1.5, 3.5]]),
            'Feature_1',
            'Feature_2',
            (AttributeError, TypeError), # Expected error when trying to call DataFrame methods
            {'px_scatter': 0, 'fig_add_trace': 0, 'fig_show': 0, 'go_scatter': 0}
        ),
    ]
)
def test_plot_kmeans_clusters(test_id, original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y, expected_exception, expected_call_counts, mock_plotly_express):
    px_scatter_mock, fig_mock, go_scatter_mock = mock_plotly_express

    if expected_exception:
        with pytest.raises(expected_exception):
            plot_kmeans_clusters(original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y)
    else:
        plot_kmeans_clusters(original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y)

        # Assert calls to plotly.express.scatter
        assert px_scatter_mock.call_count == expected_call_counts['px_scatter']
        # Assert calls to fig.add_trace (for centroids)
        assert fig_mock.add_trace.call_count == expected_call_counts['fig_add_trace']
        # Assert calls to fig.show
        assert fig_mock.show.call_count == expected_call_counts['fig_show']
        # Assert calls to plotly.graph_objects.Scatter (for centroid trace creation)
        assert go_scatter_mock.call_count == expected_call_counts['go_scatter']

        if expected_call_counts['px_scatter'] > 0:
            # Verify arguments passed to plotly.express.scatter
            call_args, call_kwargs = px_scatter_mock.call_args
            assert call_kwargs['x'] == feature_x
            assert call_kwargs['y'] == feature_y
            assert 'color' in call_kwargs
            assert 'hover_name' in call_kwargs
            assert 'title' in call_kwargs
            assert isinstance(call_kwargs['data_frame'], pd.DataFrame)
            assert 'Asset_ID' in call_kwargs['data_frame'].columns
            assert feature_x in call_kwargs['data_frame'].columns
            assert feature_y in call_kwargs['data_frame'].columns
            assert 'Cluster' in call_kwargs['data_frame'].columns # Check for the cluster labels column

        if expected_call_counts['fig_add_trace'] > 0:
            # Verify arguments passed to fig.add_trace for centroids
            call_args, call_kwargs = fig_mock.add_trace.call_args
            # Ensure it's adding a go.Scatter object (mocked instance)
            assert isinstance(call_args[0], MagicMock)
            assert call_args[0].__class__.__name__ == 'Scatter' # Should be a mocked go.Scatter instance
            # Check basic properties of the centroid trace
            assert call_args[0].mode == 'markers'
            assert call_args[0].marker.symbol == 'x'
            assert call_args[0].marker.size == 15
            assert call_args[0].name == 'Centroids'
            # Check if centroid coordinates are passed correctly
            assert np.array_equal(call_args[0].x, centroids[:, 0])
            assert np.array_equal(call_args[0].y, centroids[:, 1])

        if expected_call_counts['go_scatter'] > 0:
            # Verify arguments passed to plotly.graph_objects.Scatter constructor
            go_scatter_call_args, go_scatter_call_kwargs = go_scatter_mock.call_args
            assert go_scatter_call_kwargs['mode'] == 'markers'
            assert go_scatter_call_kwargs['marker']['symbol'] == 'x'
            assert go_scatter_call_kwargs['marker']['size'] == 15
            assert go_scatter_call_kwargs['name'] == 'Centroids'
            assert np.array_equal(go_scatter_call_kwargs['x'], centroids[:, 0])
            assert np.array_equal(go_scatter_call_kwargs['y'], centroids[:, 1])