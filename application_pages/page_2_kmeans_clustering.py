import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

def main():
    st.header("Section 6: Introduction to k-Means Clustering")
    st.markdown("""
k-Means is one of the most widely used unsupervised clustering algorithms, known for its simplicity and efficiency. The core idea behind k-Means is to partition $n$ data points into $k$ distinct clusters, where each data point belongs to the cluster with the nearest mean (centroid).

The algorithm follows an iterative approach, as outlined in Figure 1 of the provided text:
1.  **Initialization**: Randomly select $k$ centroids.
2.  **Assignment Step**: Assign each data point to the cluster whose centroid is closest (e.g., using Euclidean distance).
3.  **Update Step**: Recalculate the centroids as the mean of all data points assigned to that cluster.
4.  **Convergence**: Repeat steps 2 and 3 until the centroids no longer change significantly or a maximum number of iterations is reached.

A key characteristic of k-Means is the requirement to pre-specify the number of clusters, $k$. In financial applications, k-Means can be used to group stocks based on continuous trend characteristics for portfolio construction, as highlighted by Wu, Wang, and Wu (2022) [1]. This can help in identifying groups of assets that behave similarly, aiding in diversification and risk management.
""")

    st.header("Section 7: Implementing k-Means Clustering")
    st.markdown("""
We will now apply the k-Means algorithm to our scaled financial data. We'll use `sklearn.cluster.KMeans` for this. A crucial aspect of k-Means is selecting the appropriate number of clusters, $k$. We will use an interactive slider to allow you to easily adjust $k$ and observe its impact on the clustering results.
""")

    def perform_kmeans_clustering(scaled_data, n_clusters_input, random_state):
        """
        Executes k-Means clustering and returns labels and centroids.
        """
        kmeans = KMeans(n_clusters=n_clusters_input, init='k-means++', n_init=10, random_state=random_state)
        kmeans.fit(scaled_data)
        return kmeans.labels_, kmeans.cluster_centers_

    n_clusters_k = st.slider(
        'Number of Clusters (k) for k-Means:',
        min_value=2,
        max_value=7,
        value=st.session_state.get('k_kmeans_value', 4), # Preserve state
        step=1,
        key='k_kmeans_slider'
    )
    st.session_state.k_kmeans_value = n_clusters_k # Update session state

    if st.button("Run k-Means Clustering"):
        if 'scaled_financial_df' in st.session_state:
            kmeans_labels, kmeans_centroids = perform_kmeans_clustering(st.session_state.scaled_financial_df, n_clusters_k, random_state=42)
            st.session_state.kmeans_labels = kmeans_labels
            st.session_state.kmeans_centroids = kmeans_centroids
            
            st.subheader("k-Means Clustering Results:")
            st.write(f"k-Means Labels (first 10): {st.session_state.kmeans_labels[:10]}")
            st.write(f"k-Means Centroids (shape): {st.session_state.kmeans_centroids.shape}")
            st.write(f"k-Means Centroids:\n{st.session_state.kmeans_centroids}")
        else:
            st.warning("Please generate and scale data on the 'Introduction and Data' page first.")
    else:
        if 'kmeans_labels' in st.session_state and 'kmeans_centroids' in st.session_state:
            st.subheader("k-Means Clustering Results (from last run):")
            st.write(f"k-Means Labels (first 10): {st.session_state.kmeans_labels[:10]}")
            st.write(f"k-Means Centroids (shape): {st.session_state.kmeans_centroids.shape}")
            st.write(f"k-Means Centroids:\n{st.session_state.kmeans_centroids}")
        else:
            st.info("Adjust the slider and click 'Run k-Means Clustering' to see results.")

    st.markdown("""
The k-Means algorithm has now been applied to our scaled financial data. By adjusting the `Number of Clusters (k)` slider, you can observe how assets are grouped into different clusters. The displayed labels show which cluster each asset belongs to, and the centroids represent the central point of each identified cluster in the feature space. These labels will be used for visualization and evaluation.
""")

    st.header("Section 8: Visualizing k-Means Clusters")
    st.markdown("""
Visualizing clustering results is crucial for understanding the separation and characteristics of the identified groups. For k-Means, a scatter plot is particularly effective, allowing us to see how assets are distributed in a 2D feature space and how the cluster centroids relate to these groupings (similar to Exhibit 1 in the provided text). We will use `plotly.express` for an interactive visualization.
""")

    def plot_kmeans_clusters(original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y):
        """
        Generates an interactive scatter plot for k-Means results.
        """
        plot_df = scaled_data.copy()
        plot_df['Asset_ID'] = original_data['Asset_ID']
        plot_df['Cluster'] = cluster_labels

        fig = px.scatter(
            plot_df,
            x=feature_x,
            y=feature_y,
            color=plot_df['Cluster'].astype(str), # Convert to string for discrete colors
            hover_name='Asset_ID',
            title='k-Means Clustering of Financial Assets',
            labels={feature_x: f'{feature_x} (Scaled)', feature_y: f'{feature_y} (Scaled)'}
        )

        centroids_df = pd.DataFrame(centroids, columns=scaled_data.columns)
        fig.add_scatter(
            x=centroids_df[feature_x],
            y=centroids_df[feature_y],
            mode='markers',
            marker=dict(size=15, symbol='x', color='black', line=dict(width=2, color='DarkSlateGrey')),
            name='Centroids',
            hoverinfo='text',
            hovertext=[f'Centroid {i}' for i in range(len(centroids_df))]
        )
        return fig

    if 'kmeans_labels' in st.session_state and 'kmeans_centroids' in st.session_state and 'financial_df' in st.session_state and 'scaled_financial_df' in st.session_state:
        st.subheader("k-Means Cluster Visualization:")
        kmeans_fig = plot_kmeans_clusters(
            original_data=st.session_state.financial_df,
            scaled_data=st.session_state.scaled_financial_df,
            cluster_labels=st.session_state.kmeans_labels,
            centroids=st.session_state.kmeans_centroids,
            feature_x='Feature_1',
            feature_y='Feature_2'
        )
        st.plotly_chart(kmeans_fig)
    else:
        st.info("Run k-Means clustering first to visualize results.")

    st.markdown("""
The interactive scatter plot above visually represents the k-Means clustering results. Each point corresponds to a financial asset, colored according to its assigned cluster. The prominent 'X' markers denote the cluster centroids. By observing the plot, we can assess the compactness and separation of the clusters, and how assets with similar `Daily_Return_Volatility` and `Average_Daily_Return` characteristics are grouped together.
""")
