import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def main():
    st.title("Unsupervised Learning for Financial Asset Grouping")
    st.header("1. Notebook Overview")
    st.markdown("""
### Learning Goals
This Streamlit application aims to provide Financial Data Engineers with a hands-on, interactive experience in exploring and applying unsupervised clustering techniques—specifically k-Means and Hierarchical Clustering—to financial asset data. Upon completion, users will be able to:
*   Understand the principles and mechanics of k-Means and Hierarchical Clustering algorithms.
*   Generate and preprocess synthetic financial asset data suitable for clustering.
*   Implement k-Means clustering and interactively adjust the number of clusters ($k$).
*   Implement Hierarchical Clustering, select various linkage methods (e.g., single, complete, average, ward), and interactively define a cutoff distance for cluster formation.
*   Visualize clustering results effectively using scatter plots for k-Means (with centroids) and dynamic dendrograms for Hierarchical Clustering.
*   Calculate and interpret key evaluation metrics, including the Silhouette Score and Adjusted Rand Index (ARI), to assess clustering quality.
*   Discuss the practical financial applications of these techniques, such as portfolio construction and Hierarchical Risk Parity (HRP).

### Who the application is targeted to
This application is targeted towards **Financial Data Engineers** and data scientists with an interest in applying machine learning to financial markets. The content assumes a foundational understanding of data analysis and basic Python programming.
""")

    st.header("Section 1: Introduction to Financial Asset Grouping")
    st.markdown("""
Unsupervised learning techniques are powerful tools for uncovering hidden structures and patterns within data without relying on predefined labels. In the realm of finance, where "ground truth" labels for complex phenomena like market regimes or asset correlations are often elusive or expensive to obtain, unsupervised methods are invaluable.

Clustering, a prominent unsupervised technique, groups similar data points together based on their inherent characteristics. For Financial Data Engineers, applying clustering to assets (e.g., stocks, bonds, currencies) can reveal natural groupings that inform critical decisions in portfolio construction, risk management, and market analysis. By identifying assets that behave similarly or share common characteristics, we can build more diversified portfolios, understand systemic risk, and devise more robust trading strategies.

This application will focus on two fundamental clustering algorithms: **k-Means Clustering** and **Hierarchical Clustering**. We will explore their mechanisms, apply them to synthetic financial asset data, visualize their results, and evaluate their effectiveness using established metrics.
""")

    st.header("Section 2: Learning Objectives")
    st.markdown("""
By the end of this interactive application, you will be able to:
*   Articulate the core principles of k-Means and Hierarchical Clustering algorithms.
*   Generate and prepare a synthetic dataset of financial asset features.
*   Apply k-Means clustering and interactively adjust the number of clusters ($k$).
*   Implement Hierarchical Clustering, experiment with various linkage methods (e.g., 'single', 'complete', 'average', 'ward'), and dynamically set a cluster cutoff distance.
*   Generate and interpret interactive visualizations, including scatter plots for k-Means results and dynamic dendrograms for Hierarchical Clustering.
*   Calculate and understand the **Silhouette Score** and **Adjusted Rand Index (ARI)** for evaluating clustering quality.
*   Discuss the practical implications of these clustering techniques in financial contexts, such as optimizing portfolio diversification through strategies like Hierarchical Risk Parity (HRP) and enhancing portfolio construction.
""")

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

    def generate_financial_data(n_samples, n_features, n_clusters, cluster_std, random_state):
        """
        Generates a synthetic dataset of financial asset features.
        """
        X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, 
                               centers=n_clusters, cluster_std=cluster_std, random_state=random_state)
        
        df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Asset_ID'] = [f'Asset_{i}' for i in range(n_samples)]
        df['True_Cluster'] = y_true
        
        df = df[['Asset_ID'] + [f'Feature_{i+1}' for i in range(n_features)] + ['True_Cluster']]
        
        return df, y_true

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

    def scale_features(dataframe):
        """
        Scales numerical features using StandardScaler.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataframe)
        scaled_df = pd.DataFrame(scaled_data, columns=dataframe.columns)
        return scaled_df

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
