import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def main():
    st.header("Section 9: Introduction to Hierarchical Clustering")
    st.markdown("""
Hierarchical Clustering, unlike k-Means, does not require a pre-specified number of clusters. Instead, it builds a tree-like structure of clusters called a dendrogram, illustrating the merging or splitting process. The most common approach is **Agglomerative Hierarchical Clustering**, which is a "bottom-up" method:
1.  **Initialization**: Each data point starts as its own individual cluster.
2.  **Merging**: Iteratively merge the two closest clusters until only one large cluster remains or a desired stopping criterion is met (Figure 3 in the provided text).
The "closeness" between clusters is determined by a **linkage method**, which defines how the distance between two clusters is calculated. Common linkage methods include [4, 5]:
*   **Single Linkage**: Distance between the closest points in the two clusters.
*   **Complete Linkage**: Distance between the farthest points in the two clusters.
*   **Average Linkage**: Average distance between all points in the two clusters.
*   **Ward Linkage**: Minimizes the variance within each merged cluster.

A key output is the **dendrogram** (Exhibit 2), which visually represents the hierarchy of clusters. In finance, Hierarchical Clustering is fundamental to concepts like Hierarchical Risk Parity (HRP) for portfolio diversification, where asset relationships are inferred from a hierarchical structure to optimize capital allocation [5].
""")

    st.header("Section 10: Implementing Hierarchical Clustering")
    st.markdown("""
We will implement Agglomerative Hierarchical Clustering using `sklearn.cluster.AgglomerativeClustering`. For visualizing the clustering hierarchy, we will generate a linkage matrix using `scipy.cluster.hierarchy.linkage` which is essential for plotting the dendrogram. We'll use interactive widgets to allow users to select different `linkage methods` and observe their impact.
""")

    def perform_hierarchical_clustering(scaled_data, n_clusters_hc, linkage_method):
        """
        Executes Agglomerative Hierarchical Clustering and returns labels and linkage matrix.
        """
        if scaled_data is None:
            return None, None
        # Perform AgglomerativeClustering to get labels
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_hc, linkage=linkage_method)
        hclust_labels = agg_clustering.fit_predict(scaled_data)
        
        # Compute the linkage matrix for the dendrogram
        linkage_matrix = linkage(scaled_data, method=linkage_method)
        
        return hclust_labels, linkage_matrix

    st.subheader("Hierarchical Clustering Parameters:")
    linkage_method_hc = st.selectbox(
        'Linkage Method:',
        options=('ward', 'complete', 'average', 'single'),
        value=st.session_state.get('linkage_method_hc_value', 'ward'),
        key='linkage_method_hc_select'
    )
    st.session_state.linkage_method_hc_value = linkage_method_hc

    n_clusters_hc = st.slider(
        'Number of Clusters (n) for Hierarchical Clustering:',
        min_value=2,
        max_value=7,
        value=st.session_state.get('n_clusters_hc_value', 4),
        step=1,
        key='n_clusters_hc_slider'
    )
    st.session_state.n_clusters_hc_value = n_clusters_hc

    if st.button("Run Hierarchical Clustering"):
        if 'scaled_financial_df' in st.session_state:
            hclust_labels, linkage_matrix_hc = perform_hierarchical_clustering(st.session_state.scaled_financial_df, n_clusters_hc, linkage_method_hc)
            st.session_state.hclust_labels = hclust_labels
            st.session_state.linkage_matrix_hc = linkage_matrix_hc

            if hclust_labels is not None and linkage_matrix_hc is not None:
                st.subheader("Hierarchical Clustering Results:")
                st.write(f"Selected Linkage Method: {linkage_method_hc}")
                st.write(f"Number of Clusters: {n_clusters_hc}")
                st.write(f"Hierarchical Clustering Labels (first 10): {st.session_state.hclust_labels[:10]}")
                st.write(f"Linkage Matrix (shape): {st.session_state.linkage_matrix_hc.shape}")
            else:
                st.warning("Could not perform Hierarchical Clustering. Ensure data is generated and scaled.")
        else:
            st.warning("Please generate and scale data on the 'Introduction and Data' page first.")
    else:
        if 'hclust_labels' in st.session_state and 'linkage_matrix_hc' in st.session_state:
            st.subheader("Hierarchical Clustering Results (from last run):")
            st.write(f"Selected Linkage Method: {linkage_method_hc}")
            st.write(f"Number of Clusters: {n_clusters_hc}")
            st.write(f"Hierarchical Clustering Labels (first 10): {st.session_state.hclust_labels[:10]}")
            st.write(f"Linkage Matrix (shape): {st.session_state.linkage_matrix_hc.shape}")
        else:
            st.info("Adjust parameters and click 'Run Hierarchical Clustering' to see results.")

    st.markdown("""
Hierarchical clustering has been performed using the selected linkage method and number of clusters. The `hclust_labels` indicate the cluster assignment for each asset. The `linkage_matrix` is a crucial output, as it encodes the full hierarchical structure, which we will use to generate a dendrogram for visual exploration of the merging process.
""")

    st.header("Section 11: Visualizing Hierarchical Clustering with a Dendrogram")
    st.markdown("""
The dendrogram is the primary visualization for Hierarchical Clustering, illustrating the sequence of merges or splits that occur during the clustering process. It's a powerful tool for discerning the natural groupings within the data and choosing an appropriate number of clusters by observing the 'height' (distance) at which merges occur (Exhibit 2).

A horizontal line across the dendrogram, representing a **cutoff distance**, can effectively define the clusters. Any vertical line (representing a cluster) that the cutoff line intersects corresponds to a distinct cluster at that distance level. We will use an interactive slider to adjust this `cutoff_distance` dynamically.
""")

    def plot_dendrogram(linkage_matrix, cutoff_distance_input):
        """
        Generates an interactive dendrogram with an adjustable cutoff.
        """
        fig, ax = plt.subplots(figsize=(12, 6)) 
        
        dendrogram(
            linkage_matrix,
            ax=ax,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            color_threshold=cutoff_distance_input, # Color clusters below this distance
            above_threshold_color='gray' # Optionally color merges above threshold grey
        )
        
        ax.axhline(y=cutoff_distance_input, color='r', linestyle='--', label=f'Cutoff at {cutoff_distance_input:.2f}')
        
        ax.set_title('Hierarchical Clustering Dendrogram with Dynamic Cutoff')
        ax.set_xlabel('Asset Index or Cluster')
        ax.set_ylabel('Distance')
        ax.legend()
        plt.tight_layout()
        return fig

    if 'linkage_matrix_hc' in st.session_state:
        st.subheader("Hierarchical Clustering Dendrogram:")
        cutoff_distance = st.slider(
            'Dendrogram Cutoff Distance:',
            min_value=0.0,
            max_value=15.0,
            value=st.session_state.get('cutoff_distance_value', 6.0),
            step=0.5,
            key='cutoff_distance_slider'
        )
        st.session_state.cutoff_distance_value = cutoff_distance
        
        dendrogram_fig = plot_dendrogram(st.session_state.linkage_matrix_hc, cutoff_distance)
        st.pyplot(dendrogram_fig)
        plt.close(dendrogram_fig) # Close the figure to prevent display issues on rerun
    else:
        st.info("Run Hierarchical Clustering first to generate the dendrogram.")

    st.markdown("""
The interactive dendrogram above provides a visual map of the hierarchical clustering process. Each merge is represented by a horizontal line, and the height of the line indicates the distance between the merged clusters. The dynamic red horizontal line represents the `Cutoff Distance`. By moving this slider, you can effectively "cut" the dendrogram at different height levels, observing how the number and composition of clusters change. This helps in deciding a suitable number of clusters based on the natural groupings suggested by the data's structure.
""")
