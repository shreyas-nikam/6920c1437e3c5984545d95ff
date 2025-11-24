import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# definition_e00e33114d234c1c9482341dd27a12ec block
from definition_e00e33114d234c1c9482341dd27a12ec import scale_features
# End definition_e00e33114d234c1c9482341dd27a12ec block

@pytest.mark.parametrize(
    "input_data, expected_output_type, expected_error",
    [
        # Test Case 1: Standard numerical DataFrame with multiple columns
        (
            pd.DataFrame({'Feature_1': [1.0, 2.0, 3.0, 4.0, 5.0], 'Feature_2': [10.0, 20.0, 30.0, 40.0, 50.0]}),
            pd.DataFrame,
            None
        ),
        # Test Case 2: DataFrame with a single numerical column
        (
            pd.DataFrame({'Feature_A': [100.0, 200.0, 300.0, 400.0]}),
            pd.DataFrame,
            None
        ),
        # Test Case 3: Empty DataFrame (retains columns but no rows)
        (
            pd.DataFrame(columns=['Feature_X', 'Feature_Y']),
            pd.DataFrame,
            None
        ),
        # Test Case 4: DataFrame containing non-numerical columns (violates function contract)
        # StandardScaler will typically raise a ValueError if it encounters non-numeric data.
        (
            pd.DataFrame({'Numeric_Col': [1.0, 2.0, 3.0], 'Non_Numeric_Col': ['a', 'b', 'c']}),
            None, # Expecting an error, not a DataFrame
            ValueError
        ),
        # Test Case 5: Non-DataFrame input (violates function signature type hint)
        (
            [1, 2, 3, 4, 5], # Example: a list instead of a DataFrame
            None, # Expecting an error, not a DataFrame
            TypeError
        ),
    ]
)
def test_scale_features(input_data, expected_output_type, expected_error):
    if expected_error:
        # If an error is expected, assert that the correct exception is raised
        with pytest.raises(expected_error):
            scale_features(input_data)
    else:
        # If no error is expected, call the function and assert properties of the output
        result_df = scale_features(input_data)

        # Assert the output is a pandas DataFrame
        assert isinstance(result_df, expected_output_type)
        
        # Assert the shape and column names are retained
        assert result_df.shape == input_data.shape
        assert list(result_df.columns) == list(input_data.columns)

        if not input_data.empty:
            # For non-empty DataFrames, verify that features have been scaled
            # (mean close to 0, standard deviation close to 1)
            for col in result_df.columns:
                assert np.isclose(result_df[col].mean(), 0.0, atol=1e-9)
                assert np.isclose(result_df[col].std(), 1.0, atol=1e-9)
        else:
            # For empty DataFrames, ensure it remains empty (0 rows)
            assert result_df.empty