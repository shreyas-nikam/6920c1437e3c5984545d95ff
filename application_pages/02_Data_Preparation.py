
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

st.header("Section 3: Setup and Library Imports")
st.markdown("All required libraries have been successfully loaded. We are now ready to proceed with generating and analyzing our financial asset data.")

st.header("Section 4: Synthetic Financial Asset Data Generation")
st.markdown("""
To simulate a realistic scenario for financial asset grouping, we will generate a synthetic dataset. This dataset will represent `stock returns` or `bond features`, exhibiting some inherent cluster structure. We will use `sklearn.datasets.make_blobs` to create distinct groups of data points, which will serve as our "latent" asset classes.

Each asset will have a unique `Asset_ID` and a set of continuous numerical features:
*   `Feature_1`: Represents `Daily_Return_Volatility`.
*   `Feature_2`: Represents `Average_Daily_Return`.
*   `Feature_3`: Represents `Beta_to_Market`.
These features are chosen to reflect common characteristics used in financial analysis and portfolio management.
""")

# Function Definition
def generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state):
    """
    Generates a synthetic dataset of financial asset features.
    """
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, 
                           centers=n_clusters, cluster_std=cluster_std, random_state=random_state)
    
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['Asset_ID'] = [f'Asset_{i}' for i in range(n_samples)]
    df['True_Cluster'] = y_true
    
    # Reorder columns to have Asset_ID first
    df = df[['Asset_ID'] + [f'Feature_{i+1}' for i in range(n_features)] + ['True_Cluster']]
    
    return df, y_true

# Streamlit Usage: Generate and store data in session state
if 'financial_df' not in st.session_state:
    st.session_state.financial_df, st.session_state.y_true_labels = generate_financial_data(n_samples=100, n_features=3, n_clusters=4, cluster_std=0.8, random_state=42)

st.subheader("Generated Financial Data Sample:")
st.dataframe(st.session_state.financial_df.head())
st.write(f"Shape of generated data: {st.session_state.financial_df.shape}")

st.markdown("""
We have successfully generated a synthetic dataset consisting of 100 financial assets, each characterized by three distinct features (`Daily_Return_Volatility`, `Average_Daily_Return`, `Beta_to_Market`). The `True_Cluster` column represents the latent groups that `make_blobs` created, which we will use as a benchmark for some of our evaluation metrics. This dataset will serve as our input for exploring various clustering algorithms.
""")

st.header("Section 5: Data Preprocessing: Scaling Features")
st.markdown("""
Many clustering algorithms, particularly those based on distance metrics like k-Means and Hierarchical Clustering, are sensitive to the scale of the input features. Features with larger numerical ranges can disproportionately influence the distance calculations, leading to biased clustering results. To mitigate this, it's a standard practice to scale the features so that they all contribute equally to the distance computations.

We will use `StandardScaler` from `sklearn.preprocessing`, which transforms the data such that each feature has a mean of 0 and a standard deviation of 1 (unit variance). The formula for standardization for a data point $x$ and feature $j$ is:
$$ 
    z_j = \frac{x_j - \mu_j}{\sigma_j}
$$
where $\mu_j$ is the mean of feature $j$ and $\sigma_j$ is its standard deviation.
""")

# Function Definition
def scale_features(dataframe):
    """
    Scales numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    scaled_df = pd.DataFrame(scaled_data, columns=dataframe.columns)
    return scaled_df

# Streamlit Usage: Scale features and store in session state
if 'scaled_financial_df' not in st.session_state:
    feature_columns = ['Feature_1', 'Feature_2', 'Feature_3']
    st.session_state.scaled_financial_df = scale_features(st.session_state.financial_df[feature_columns])

st.subheader("Scaled Financial Data Sample:")
st.dataframe(st.session_state.scaled_financial_df.head())
st.write("Description of Scaled Features (Mean and Std Dev):")
st.dataframe(st.session_state.scaled_financial_df.describe().loc[['mean', 'std']])

st.markdown("""
The financial asset features have now been standardized, meaning each feature has a mean of approximately 0 and a standard deviation of 1. This ensures that no single feature dominates the clustering process due to its scale, allowing our distance-based algorithms to identify clusters based on the inherent relationships between features more accurately.
""")


