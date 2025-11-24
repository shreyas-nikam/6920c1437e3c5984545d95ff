
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Ensure required data is in session_state from previous pages
if 'scaled_financial_df' not in st.session_state or \
   'y_true_labels' not in st.session_state:
    st.warning("Please go back to 'Data Preparation' page to generate and scale data.")
    st.stop()

st.header("Section 12: Cluster Evaluation: Silhouette Score")
st.markdown("""
To quantitatively assess the quality of our clustering results, we use evaluation metrics. The **Silhouette Score** is an internal validation metric that measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation) [6, 7]. It ranges from -1 to 1:
*   Values close to +1 indicate that data points are well-matched to their own cluster and poorly matched to neighboring clusters (good clustering).
*   Values around 0 indicate overlapping clusters or points that are on or very close to the decision boundary.
*   Values close to -1 suggest that data points might have been assigned to the wrong cluster.

For each data point $i$, the silhouette coefficient $s(i)$ is calculated as:
$$ s(i) = \frac{b(i) - a(i)}{\max[a(i), b(i)]} $$
where:
*   $a(i)$ is the mean distance between $i$ and all other data points in the same cluster (mean intracluster distance).
*   $b(i)$ is the minimum mean distance between $i$ and all data points in any other cluster (mean intercluster distance to the nearest neighboring cluster).
The overall Silhouette Score for the clustering is the average $s(i)$ over all data points.
""")

# Function Definition
def calculate_silhouette_score(scaled_data, cluster_labels):
    """
    Computes the Silhouette Score.
    """
    # Ensure there are at least 2 clusters and more than 1 sample for silhouette_score to be valid
    if len(np.unique(cluster_labels)) < 2 or len(scaled_data) <= 1:
        return np.nan # Use NaN for invalid scores
    return silhouette_score(scaled_data, cluster_labels)

st.subheader("Silhouette Scores:")
kmeans_silhouette = np.nan
hclust_silhouette = np.nan

if 'kmeans_labels' in st.session_state:
    kmeans_silhouette = calculate_silhouette_score(st.session_state.scaled_financial_df, st.session_state.kmeans_labels)
    st.write(f"k-Means Silhouette Score: {kmeans_silhouette:.3f}")
else:
    st.write("k-Means Silhouette Score: N/A (Run k-Means first)")

if 'hclust_labels' in st.session_state:
    hclust_silhouette = calculate_silhouette_score(st.session_state.scaled_financial_df, st.session_state.hclust_labels)
    st.write(f"Hierarchical Clustering Silhouette Score: {hclust_silhouette:.3f}")
else:
    st.write("Hierarchical Clustering Silhouette Score: N/A (Run Hierarchical Clustering first)")

st.session_state.kmeans_silhouette = kmeans_silhouette
st.session_state.hclust_silhouette = hclust_silhouette

st.markdown("""
We have calculated the Silhouette Scores for both k-Means and Hierarchical Clustering. A higher silhouette score generally indicates better-defined and more separated clusters. These scores provide a quantitative measure of how well each algorithm grouped the financial assets based on their characteristics.
""")

st.header("Section 13: Cluster Evaluation: Adjusted Rand Index (ARI)")
st.markdown("""
The **Adjusted Rand Index (ARI)** is an external evaluation metric that measures the similarity between two clusterings, accounting for chance [7]. It is typically used when true labels (ground truth) are available, or to compare the similarity between the outputs of different clustering algorithms on the same dataset. The ARI ranges from -1 to 1:
*   A value of 1 indicates perfect agreement between the two clusterings.
*   A value of 0 indicates that the clusterings are independent (random labeling).
*   Negative values indicate worse-than-random agreement.

Since our synthetic dataset includes `True_Cluster` labels, we can use the ARI to compare how well our algorithms recover these underlying groups. The formula for ARI is:
$$ARI = \frac{RI - Expected_{RI}}{\max(RI) - Expected_{RI}}$$
where $RI$ is the Rand Index, and $Expected_{RI}$ is its expected value under a null hypothesis of random clustering [7].
""")

# Function Definition
def calculate_adjusted_rand_index(labels_true, labels_pred):
    """
    Computes the Adjusted Rand Index.
    """
    # Check if there's enough variety for meaningful ARI
    if len(np.unique(labels_true)) < 2 or len(np.unique(labels_pred)) < 2 or len(labels_true) < 2:
        return np.nan
    return adjusted_rand_score(labels_true, labels_pred)

st.subheader("Adjusted Rand Index (ARI) Scores:")
kmeans_ari = np.nan
hclust_ari = np.nan
inter_algo_ari = np.nan

if 'kmeans_labels' in st.session_state:
    kmeans_ari = calculate_adjusted_rand_index(st.session_state.y_true_labels, st.session_state.kmeans_labels)
    st.write(f"k-Means Adjusted Rand Index (vs True Labels): {kmeans_ari:.3f}")
else:
    st.write("k-Means Adjusted Rand Index (vs True Labels): N/A (Run k-Means first)")

if 'hclust_labels' in st.session_state:
    hclust_ari = calculate_adjusted_rand_index(st.session_state.y_true_labels, st.session_state.hclust_labels)
    st.write(f"Hierarchical Clustering Adjusted Rand Index (vs True Labels): {hclust_ari:.3f}")
else:
    st.write("Hierarchical Clustering Adjusted Rand Index (vs True Labels): N/A (Run Hierarchical Clustering first)")

if 'kmeans_labels' in st.session_state and 'hclust_labels' in st.session_state:
    inter_algo_ari = calculate_adjusted_rand_index(st.session_state.kmeans_labels, st.session_state.hclust_labels)
    st.write(f"Adjusted Rand Index (k-Means vs Hierarchical Clustering): {inter_algo_ari:.3f}")
else:
    st.write("Adjusted Rand Index (k-Means vs Hierarchical Clustering): N/A (Run both algorithms first)")

st.session_state.kmeans_ari = kmeans_ari
st.session_state.hclust_ari = hclust_ari
st.session_state.inter_algo_ari = inter_algo_ari

st.markdown("""
The Adjusted Rand Index scores provide a measure of similarity between our clustering results and the ground truth clusters, as well as between the two algorithms themselves. A higher ARI value, especially when compared to the `True_Cluster` labels, indicates that the algorithm successfully identified the underlying patterns in the data. Comparing the ARI between k-Means and Hierarchical Clustering also sheds light on how similarly these two distinct approaches group the assets.
""")

st.header("Section 14: Comparing Clustering Results")
st.markdown("""
Having evaluated both k-Means and Hierarchical Clustering using Silhouette Score and Adjusted Rand Index, we can now compare their performance. This comparison helps in understanding which algorithm might be more suitable for a given financial data analysis task.

In our case, the `True_Cluster` labels from the synthetic data generation provide an ideal benchmark for ARI. The Silhouette Score offers an intrinsic measure of cluster quality regardless of ground truth.
""")

st.subheader("Comparison Table:")
comparison_data = {
    "Clustering Algorithm": ["k-Means", "Hierarchical Clustering"],
    "Silhouette Score": [
        f"{st.session_state.kmeans_silhouette:.3f}" if not np.isnan(st.session_state.kmeans_silhouette) else "N/A",
        f"{st.session_state.hclust_silhouette:.3f}" if not np.isnan(st.session_state.hclust_silhouette) else "N/A"
    ],
    "Adjusted Rand Index (vs True Labels)": [
        f"{st.session_state.kmeans_ari:.3f}" if not np.isnan(st.session_state.kmeans_ari) else "N/A",
        f"{st.session_state.hclust_ari:.3f}" if not np.isnan(st.session_state.hclust_ari) else "N/A"
    ]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df.set_index("Clustering Algorithm"))

if not np.isnan(st.session_state.inter_algo_ari):
    st.write(f"\nAdjusted Rand Index (k-Means vs Hierarchical Clustering): {st.session_state.inter_algo_ari:.3f}")
else:
    st.write("\nAdjusted Rand Index (k-Means vs Hierarchical Clustering): N/A (Run both algorithms first)")

st.markdown("""
From the comparison, we can observe the strengths and weaknesses of each clustering algorithm on our synthetic financial dataset. For instance, one algorithm might yield better separation (higher Silhouette Score) while another might more accurately recover the predefined latent groups (higher ARI). This comprehensive evaluation guides us in selecting the most appropriate clustering approach for specific financial analysis needs.
""")

