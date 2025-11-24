id: 6920c1437e3c5984545d95ff_documentation
summary: Anomaly Sentinel: Financial Outlier Detection Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Unsupervised Clustering for Financial Asset Grouping with Streamlit

## 1. Introduction: Context, Importance, and Learning Goals
Duration: 0:10:00

<aside class="positive">
This step provides the foundational context for the codelab, explaining the importance of unsupervised learning in finance and outlining the key concepts you'll master. Understanding these initial points will set you up for success in the subsequent practical steps.
</aside>

Unsupervised learning techniques are powerful tools for uncovering hidden structures and patterns within data without relying on predefined labels. In the realm of finance, where "ground truth" labels for complex phenomena like market regimes or asset correlations are often elusive or expensive to obtain, unsupervised methods are invaluable.

Clustering, a prominent unsupervised technique, groups similar data points together based on their inherent characteristics. For **Financial Data Engineers**, applying clustering to assets (e.g., stocks, bonds, currencies) can reveal natural groupings that inform critical decisions in portfolio construction, risk management, and market analysis. By identifying assets that behave similarly or share common characteristics, we can build more diversified portfolios, understand systemic risk, and devise more robust trading strategies.

This application will focus on two fundamental clustering algorithms: **k-Means Clustering** and **Hierarchical Clustering**. We will explore their mechanisms, apply them to synthetic financial asset data, visualize their results, and evaluate their effectiveness using established metrics.

### Learning Goals

This Streamlit application will provide Financial Data Engineers with an interactive tool to explore and apply unsupervised clustering techniques—specifically k-Means and Hierarchical Clustering—to financial asset data. Upon completion, users will be able to:

*   Understand the principles and mechanics of k-Means and Hierarchical Clustering algorithms.
*   Generate and preprocess synthetic financial asset data suitable for clustering.
*   Implement k-Means clustering and interactively adjust the number of clusters ($k$).
*   Implement Hierarchical Clustering, select various linkage methods (e.g., single, complete, average, ward), and interactively define a cutoff distance for cluster formation.
*   Visualize clustering results effectively using scatter plots for k-Means (with centroids) and dynamic dendrograms for Hierarchical Clustering.
*   Calculate and interpret key evaluation metrics, including the Silhouette Score and Adjusted Rand Index (ARI), to assess clustering quality.
*   Discuss the practical financial applications of these techniques, such as portfolio construction and Hierarchical Risk Parity (HRP).

### Target Audience

This application is targeted towards **Financial Data Engineers** and data scientists with an interest in applying machine learning to financial markets. The content assumes a foundational understanding of data analysis and basic Python programming.

## 2. Setting Up and Generating Synthetic Financial Data
Duration: 0:08:00

In this step, we'll ensure our environment is ready by implicitly loading necessary libraries (Streamlit handles this across pages) and then generate a synthetic dataset representing financial asset features. This dataset will serve as the input for our clustering algorithms.

### Architecture Overview

The application is structured into multiple Streamlit pages, allowing for a logical flow through the different stages of clustering.
At a high level, the architecture follows a typical data science workflow:
1.  **Data Generation & Preprocessing**: Create synthetic data and scale features.
2.  **Model Application**: Apply k-Means and Hierarchical Clustering.
3.  **Visualization**: Display clustering results.
4.  **Evaluation**: Quantitatively assess model performance.
5.  **Application Discussion**: Explore real-world financial uses.

```
       +-+       +--+
       |   Streamlit App   |       |                       |
       |     (app.py)      |>|   01_Introduction.py  |
       +-+       |                       |
               |                     +--+
               |                     |                       |
               +-->|  02_Data_Preparation.py |
               |                     |  (Data Gen, Scaling)  |
               |                     +--+
               |                     |                       |
               +-->| 03_KMeans_Clustering.py |
               |                     |  (KMeans Algo, Viz)   |
               |                     +--+
               |                     |                       |
               +-->|04_Hierarchical_Clustering.py|
               |                     | (HC Algo, Dendrogram) |
               |                     +--+
               |                     |                       |
               +-->|   05_Evaluation.py    |
               |                     |  (Metrics, Comparison)|
               |                     +--+
               |                     |                       |
               +-->|06_Financial_Applications.py|
               |                     | (Portfolio, HRP)      |
               |                     +--+
               |                     |                       |
               +-->|    07_Conclusion.py   |
                                     |                       |
                                     +--+
```

### Setup and Library Imports

All required libraries such as `streamlit`, `pandas`, `numpy`, `sklearn` for data generation, preprocessing and clustering, `matplotlib` and `plotly` for visualization, and `scipy` for hierarchical clustering specific functions have been successfully loaded within the application's environment. We are now ready to proceed with generating and analyzing our financial asset data.

### Synthetic Financial Asset Data Generation

To simulate a realistic scenario for financial asset grouping, we will generate a synthetic dataset. This dataset will represent `stock returns` or `bond features`, exhibiting some inherent cluster structure. We will use `sklearn.datasets.make_blobs` to create distinct groups of data points, which will serve as our "latent" asset classes.

Each asset will have a unique `Asset_ID` and a set of continuous numerical features:

*   `Feature_1`: Represents `Daily_Return_Volatility`.
*   `Feature_2`: Represents `Average_Daily_Return`.
*   `Feature_3`: Represents `Beta_to_Market`.

These features are chosen to reflect common characteristics used in financial analysis and portfolio management.

Let's look at the Python function used to generate this data:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import streamlit as st

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

# In the Streamlit app, this data is generated and stored in session state:
# if 'financial_df' not in st.session_state:
#     st.session_state.financial_df, st.session_state.y_true_labels = generate_financial_data(n_samples=100, n_features=3, n_clusters=4, cluster_std=0.8, random_state=42)
```

The application generates a sample of this data. A peek at the generated data:

```
Generated Financial Data Sample:
   Asset_ID  Feature_1  Feature_2  Feature_3  True_Cluster
0    Asset_0   0.455246  -0.669814  -0.547167             0
1    Asset_1  -1.127885   0.428585   0.722830             2
2    Asset_2  -0.906969   0.316886   0.218146             2
3    Asset_3   1.619047  -0.080517   0.318288             1
4    Asset_4  -0.741008   0.119253   0.260021             2
```
Shape of generated data: (100, 5)

We have successfully generated a synthetic dataset consisting of 100 financial assets, each characterized by three distinct features (`Daily_Return_Volatility`, `Average_Daily_Return`, `Beta_to_Market`). The `True_Cluster` column represents the latent groups that `make_blobs` created, which we will use as a benchmark for some of our evaluation metrics. This dataset will serve as our input for exploring various clustering algorithms.

## 3. Data Preprocessing: Scaling Features
Duration: 0:05:00

Many clustering algorithms, particularly those based on distance metrics like k-Means and Hierarchical Clustering, are sensitive to the scale of the input features. Features with larger numerical ranges can disproportionately influence the distance calculations, leading to biased clustering results. To mitigate this, it's a standard practice to scale the features so that they all contribute equally to the distance computations.

We will use `StandardScaler` from `sklearn.preprocessing`, which transforms the data such that each feature has a mean of 0 and a standard deviation of 1 (unit variance). The formula for standardization for a data point $x$ and feature $j$ is:

$$ z_j = \frac{x_j - \mu_j}{\sigma_j} $$

where $\mu_j$ is the mean of feature $j$ and $\sigma_j$ is its standard deviation.

Here's the Python function for scaling features:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st

def scale_features(dataframe):
    """
    Scales numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    scaled_df = pd.DataFrame(scaled_data, columns=dataframe.columns)
    return scaled_df

# In the Streamlit app, this function is called:
# if 'scaled_financial_df' not in st.session_state:
#     feature_columns = ['Feature_1', 'Feature_2', 'Feature_3']
#     st.session_state.scaled_financial_df = scale_features(st.session_state.financial_df[feature_columns])
```

A sample of the scaled financial data:

```
Scaled Financial Data Sample:
   Feature_1  Feature_2  Feature_3
0   0.472145  -0.846506  -0.647970
1  -1.229107   0.582845   0.880436
2  -0.988019   0.430948   0.285800
3   1.685360  -0.198308   0.407480
4  -0.806655   0.177067   0.334469

Description of Scaled Features (Mean and Std Dev):
          Feature_1     Feature_2     Feature_3
mean  -3.410000e-17 -1.214000e-16 -1.066500e-16
std    1.005038e+00  1.005038e+00  1.005038e+00
```

The financial asset features have now been standardized, meaning each feature has a mean of approximately 0 and a standard deviation of 1. This ensures that no single feature dominates the clustering process due to its scale, allowing our distance-based algorithms to identify clusters based on the inherent relationships between features more accurately.

## 4. K-Means Clustering: Implementation and Visualization
Duration: 0:15:00

### Introduction to k-Means Clustering

k-Means is one of the most widely used unsupervised clustering algorithms, known for its simplicity and efficiency. The core idea behind k-Means is to partition $n$ data points into $k$ distinct clusters, where each data point belongs to the cluster with the nearest mean (centroid).

The algorithm follows an iterative approach:

1.  **Initialization**: Randomly select $k$ centroids.
2.  **Assignment Step**: Assign each data point to the cluster whose centroid is closest (e.g., using Euclidean distance).
3.  **Update Step**: Recalculate the centroids as the mean of all data points assigned to that cluster.
4.  **Convergence**: Repeat steps 2 and 3 until the centroids no longer change significantly or a maximum number of iterations is reached.

<aside class="positive">
A key characteristic of k-Means is the requirement to pre-specify the number of clusters, $k$. In financial applications, k-Means can be used to group stocks based on continuous trend characteristics for portfolio construction, aiding in diversification and risk management.
</aside>

### K-Means Clustering Flowchart

```
+-+
|      START        |
+-+
        |
        v
+-+
| Initialize k      |
| centroids randomly|
+-+
        |
        v
+-+
|  ASSIGNMENT STEP  |
| (Assign each data |
| point to the      |
| closest centroid) |
+-+
        |
        v
+-+
|    UPDATE STEP    |
| (Recalculate      |
| centroids as mean |
| of assigned points)|
+-+
        |
        v
+-+
| Centroids changed |
| significantly?    |<-No--+
+-+            |
        | Yes                      |
        v                          |
+-+              |
|      END          |--+
+-+
```

### Implementing k-Means Clustering

We will now apply the k-Means algorithm to our scaled financial data. We'll use `sklearn.cluster.KMeans` for this. A crucial aspect of k-Means is selecting the appropriate number of clusters, $k$. The Streamlit application allows you to interactively adjust $k$ using a slider and observe its impact on the clustering results.

The Python function to perform k-Means clustering:

```python
from sklearn.cluster import KMeans
import streamlit as st

def perform_kmeans_clustering(scaled_data, n_clusters_input, random_state):
    """
    Executes k-Means clustering and returns labels and centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters_input, init='k-means++', n_init=10, random_state=random_state)
    kmeans.fit(scaled_data)
    return kmeans.labels_, kmeans.cluster_centers_

# In the Streamlit app, you would interact with a slider and a button:
# n_clusters_k = st.slider('Number of Clusters (k) for k-Means:', min_value=2, max_value=7, value=4, step=1, key='k_kmeans_slider')
# if st.button("Run k-Means Clustering"):
#     kmeans_labels, kmeans_centroids = perform_kmeans_clustering(st.session_state.scaled_financial_df, n_clusters_k, random_state=42)
#     st.session_state.kmeans_labels = kmeans_labels
#     st.session_state.kmeans_centroids = kmeans_centroids
#     # Display results
```

After running k-Means, the application displays the first few labels (which cluster each asset belongs to) and the shape and values of the cluster centroids. These labels and centroids are essential for the next step: visualization.

### Visualizing k-Means Clusters

Visualizing clustering results is crucial for understanding the separation and characteristics of the identified groups. For k-Means, a scatter plot is particularly effective, allowing us to see how assets are distributed in a 2D feature space and how the cluster centroids relate to these groupings. We will use `plotly.express` for an interactive visualization.

The Python function to plot k-Means clusters:

```python
import plotly.express as px
import pandas as pd
import streamlit as st

def plot_kmeans_clusters(original_data, scaled_data, cluster_labels, centroids, feature_x, feature_y):
    """
    Generates an interactive scatter plot for k-Means results.
    """
    plot_df = scaled_data.copy()
    plot_df['Asset_ID'] = original_data['Asset_ID']
    plot_df['Cluster'] = cluster_labels

    # Plot assets
    fig = px.scatter(
        plot_df,
        x=feature_x,
        y=feature_y,
        color=plot_df['Cluster'].astype(str), # Convert to string for discrete colors
        hover_name='Asset_ID',
        title='k-Means Clustering of Financial Assets',
        labels={feature_x: f'{feature_x} (Scaled)', feature_y: f'{feature_y} (Scaled)'}
    )

    # Add centroids
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

# In the Streamlit app, the plot is rendered:
# if 'kmeans_labels' in st.session_state and 'kmeans_centroids' in st.session_state and 'financial_df' in st.session_state:
#     kmeans_fig = plot_kmeans_clusters(
#         original_data=st.session_state.financial_df,
#         scaled_data=st.session_state.scaled_financial_df,
#         cluster_labels=st.session_state.kmeans_labels,
#         centroids=st.session_state.kmeans_centroids,
#         feature_x='Feature_1',
#         feature_y='Feature_2'
#     )
#     st.plotly_chart(kmeans_fig)
```

The interactive scatter plot above visually represents the k-Means clustering results. Each point corresponds to a financial asset, colored according to its assigned cluster. The prominent 'X' markers denote the cluster centroids. By observing the plot, we can assess the compactness and separation of the clusters, and how assets with similar `Daily_Return_Volatility` and `Average_Daily_Return` characteristics are grouped together.

## 5. Hierarchical Clustering: Implementation and Dendrogram Visualization
Duration: 0:15:00

### Introduction to Hierarchical Clustering

Hierarchical Clustering, unlike k-Means, does not require a pre-specified number of clusters. Instead, it builds a tree-like structure of clusters called a dendrogram, illustrating the merging or splitting process. The most common approach is **Agglomerative Hierarchical Clustering**, which is a "bottom-up" method:

1.  **Initialization**: Each data point starts as its own individual cluster.
2.  **Merging**: Iteratively merge the two closest clusters until only one large cluster remains or a desired stopping criterion is met.

The "closeness" between clusters is determined by a **linkage method**, which defines how the distance between two clusters is calculated. Common linkage methods include:

*   **Single Linkage**: Distance between the closest points in the two clusters.
*   **Complete Linkage**: Distance between the farthest points in the two clusters.
*   **Average Linkage**: Average distance between all points in the two clusters.
*   **Ward Linkage**: Minimizes the variance within each merged cluster.

<aside class="positive">
A key output is the <b>dendrogram</b>, which visually represents the hierarchy of clusters. In finance, Hierarchical Clustering is fundamental to concepts like Hierarchical Risk Parity (HRP) for portfolio diversification, where asset relationships are inferred from a hierarchical structure to optimize capital allocation.
</aside>

### Agglomerative Hierarchical Clustering Flowchart

```
+-+
|      START        |
+-+
        |
        v
+--+
| Each data point is its   |
| own cluster (n clusters) |
+--+
        |
        v
+--+
|  Calculate all pairwise  |
|  distances between clusters|
+--+
        |
        v
+--+
| Merge the two closest    |
| clusters based on        |
| chosen linkage method    |
+--+
        |
        v
+--+
|   Number of clusters     |
|   is 1?                  |<-No--+
+--+            |
        | Yes                             |
        v                                 |
+--+              |
|      END (Dendrogram     |--+
|      is formed)          |
+--+
```

### Implementing Hierarchical Clustering

We will implement Agglomerative Hierarchical Clustering using `sklearn.cluster.AgglomerativeClustering`. For visualizing the clustering hierarchy, we will generate a linkage matrix using `scipy.cluster.hierarchy.linkage` which is essential for plotting the dendrogram. The Streamlit application allows users to select different `linkage methods` and observe their impact.

The Python function to perform hierarchical clustering:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import streamlit as st

def perform_hierarchical_clustering(scaled_data, n_clusters_hc, linkage_method):
    """
    Executes Agglomerative Hierarchical Clustering and returns labels and linkage matrix.
    """
    # Perform AgglomerativeClustering to get labels
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_hc, linkage=linkage_method)
    hclust_labels = agg_clustering.fit_predict(scaled_data)

    # Compute the linkage matrix for the dendrogram
    linkage_matrix = linkage(scaled_data, method=linkage_method)

    return hclust_labels, linkage_matrix

# In the Streamlit app, you would interact with selectbox and slider:
# linkage_method_hc = st.selectbox('Linkage Method:', options=('ward', 'complete', 'average', 'single'), value='ward', key='linkage_method_hc_select')
# n_clusters_hc = st.slider('Number of Clusters (n) for Hierarchical Clustering:', min_value=2, max_value=7, value=4, step=1, key='n_clusters_hc_slider')
# if st.button("Run Hierarchical Clustering"):
#     hclust_labels, linkage_matrix_hc = perform_hierarchical_clustering(st.session_state.scaled_financial_df, n_clusters_hc, linkage_method_hc)
#     st.session_state.hclust_labels = hclust_labels
#     st.session_state.linkage_matrix_hc = linkage_matrix_hc
#     # Display results
```

Hierarchical clustering has been performed using the selected linkage method and number of clusters. The `hclust_labels` indicate the cluster assignment for each asset. The `linkage_matrix` is a crucial output, as it encodes the full hierarchical structure, which we will use to generate a dendrogram for visual exploration of the merging process.

### Visualizing Hierarchical Clustering with a Dendrogram

The dendrogram is the primary visualization for Hierarchical Clustering, illustrating the sequence of merges or splits that occur during the clustering process. It's a powerful tool for discerning the natural groupings within the data and choosing an appropriate number of clusters by observing the 'height' (distance) at which merges occur.

A horizontal line across the dendrogram, representing a **cutoff distance**, can effectively define the clusters. Any vertical line (representing a cluster) that the cutoff line intersects corresponds to a distinct cluster at that distance level. The Streamlit application uses an interactive slider to adjust this `cutoff_distance` dynamically.

The Python function to plot the dendrogram:

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import streamlit as st

def plot_dendrogram(linkage_matrix, cutoff_distance_input):
    """
    Generates an interactive dendrogram with an adjustable cutoff.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate dendrogram, coloring clusters below the cutoff differently
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        color_threshold=cutoff_distance_input, # Color clusters below this distance
        above_threshold_color='gray' # Optionally color merges above threshold grey
    )

    # Draw a horizontal line at the cutoff distance
    ax.axhline(y=cutoff_distance_input, color='r', linestyle='--', label=f'Cutoff at {cutoff_distance_input:.2f}')

    ax.set_title('Hierarchical Clustering Dendrogram with Dynamic Cutoff')
    ax.set_xlabel('Asset Index or Cluster')
    ax.set_ylabel('Distance')
    ax.legend()
    plt.tight_layout()
    return fig

# In the Streamlit app, the plot is rendered:
# if 'linkage_matrix_hc' in st.session_state:
#     cutoff_distance = st.slider('Dendrogram Cutoff Distance:', min_value=0.0, max_value=15.0, value=6.0, step=0.5, key='cutoff_distance_slider')
#     dendrogram_fig = plot_dendrogram(st.session_state.linkage_matrix_hc, cutoff_distance)
#     st.pyplot(dendrogram_fig)
#     plt.close(dendrogram_fig)
```

The interactive dendrogram above provides a visual map of the hierarchical clustering process. Each merge is represented by a horizontal line, and the height of the line indicates the distance between the merged clusters. The dynamic red horizontal line represents the `Cutoff Distance`. By moving this slider, you can effectively "cut" the dendrogram at different height levels, observing how the number and composition of clusters change. This helps in deciding a suitable number of clusters based on the natural groupings suggested by the data's structure.

## 6. Cluster Evaluation and Comparison
Duration: 0:12:00

To quantitatively assess the quality of our clustering results, we use evaluation metrics. This step introduces two key metrics: Silhouette Score and Adjusted Rand Index, and then compares the performance of k-Means and Hierarchical Clustering.

### Cluster Evaluation: Silhouette Score

The **Silhouette Score** is an internal validation metric that measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). It ranges from -1 to 1:

*   Values close to +1 indicate that data points are well-matched to their own cluster and poorly matched to neighboring clusters (good clustering).
*   Values around 0 indicate overlapping clusters or points that are on or very close to the decision boundary.
*   Values close to -1 suggest that data points might have been assigned to the wrong cluster.

For each data point $i$, the silhouette coefficient $s(i)$ is calculated as:
$$ s(i) = \frac{b(i) - a(i)}{\max[a(i), b(i)]} $$
where:
*   $a(i)$ is the mean distance between $i$ and all other data points in the same cluster (mean intracluster distance).
*   $b(i)$ is the minimum mean distance between $i$ and all data points in any other cluster (mean intercluster distance to the nearest neighboring cluster).

The overall Silhouette Score for the clustering is the average $s(i)$ over all data points.

The Python function to calculate the Silhouette Score:

```python
import numpy as np
from sklearn.metrics import silhouette_score
import streamlit as st

def calculate_silhouette_score(scaled_data, cluster_labels):
    """
    Computes the Silhouette Score.
    """
    # Ensure there are at least 2 clusters and more than 1 sample for silhouette_score to be valid
    if len(np.unique(cluster_labels)) < 2 or len(scaled_data) <= 1:
        return np.nan # Use NaN for invalid scores
    return silhouette_score(scaled_data, cluster_labels)

# In the Streamlit app, scores are displayed if labels exist:
# if 'kmeans_labels' in st.session_state:
#     kmeans_silhouette = calculate_silhouette_score(st.session_state.scaled_financial_df, st.session_state.kmeans_labels)
#     st.write(f"k-Means Silhouette Score: {kmeans_silhouette:.3f}")
# if 'hclust_labels' in st.session_state:
#     hclust_silhouette = calculate_silhouette_score(st.session_state.scaled_financial_df, st.session_state.hclust_labels)
#     st.write(f"Hierarchical Clustering Silhouette Score: {hclust_silhouette:.3f}")
```

We have calculated the Silhouette Scores for both k-Means and Hierarchical Clustering. A higher silhouette score generally indicates better-defined and more separated clusters. These scores provide a quantitative measure of how well each algorithm grouped the financial assets based on their characteristics.

### Cluster Evaluation: Adjusted Rand Index (ARI)

The **Adjusted Rand Index (ARI)** is an external evaluation metric that measures the similarity between two clusterings, accounting for chance. It is typically used when true labels (ground truth) are available, or to compare the similarity between the outputs of different clustering algorithms on the same dataset. The ARI ranges from -1 to 1:

*   A value of 1 indicates perfect agreement between the two clusterings.
*   A value of 0 indicates that the clusterings are independent (random labeling).
*   Negative values indicate worse-than-random agreement.

Since our synthetic dataset includes `True_Cluster` labels, we can use the ARI to compare how well our algorithms recover these underlying groups. The formula for ARI is:
$$ARI = \frac{RI - Expected_{RI}}{\max(RI) - Expected_{RI}}$$
where $RI$ is the Rand Index, and $Expected_{RI}$ is its expected value under a null hypothesis of random clustering.

The Python function to calculate the Adjusted Rand Index:

```python
import numpy as np
from sklearn.metrics import adjusted_rand_score
import streamlit as st

def calculate_adjusted_rand_index(labels_true, labels_pred):
    """
    Computes the Adjusted Rand Index.
    """
    # Check if there's enough variety for meaningful ARI
    if len(np.unique(labels_true)) < 2 or len(np.unique(labels_pred)) < 2 or len(labels_true) < 2:
        return np.nan
    return adjusted_rand_score(labels_true, labels_pred)

# In the Streamlit app, scores are displayed if labels exist:
# if 'kmeans_labels' in st.session_state:
#     kmeans_ari = calculate_adjusted_rand_index(st.session_state.y_true_labels, st.session_state.kmeans_labels)
#     st.write(f"k-Means Adjusted Rand Index (vs True Labels): {kmeans_ari:.3f}")
# if 'hclust_labels' in st.session_state:
#     hclust_ari = calculate_adjusted_rand_index(st.session_state.y_true_labels, st.session_state.hclust_labels)
#     st.write(f"Hierarchical Clustering Adjusted Rand Index (vs True Labels): {hclust_ari:.3f}")
# if 'kmeans_labels' in st.session_state and 'hclust_labels' in st.session_state:
#     inter_algo_ari = calculate_adjusted_rand_index(st.session_state.kmeans_labels, st.session_state.hclust_labels)
#     st.write(f"Adjusted Rand Index (k-Means vs Hierarchical Clustering): {inter_algo_ari:.3f}")
```

The Adjusted Rand Index scores provide a measure of similarity between our clustering results and the ground truth clusters, as well as between the two algorithms themselves. A higher ARI value, especially when compared to the `True_Cluster` labels, indicates that the algorithm successfully identified the underlying patterns in the data. Comparing the ARI between k-Means and Hierarchical Clustering also sheds light on how similarly these two distinct approaches group the assets.

### Comparing Clustering Results

Having evaluated both k-Means and Hierarchical Clustering using Silhouette Score and Adjusted Rand Index, we can now compare their performance. This comparison helps in understanding which algorithm might be more suitable for a given financial data analysis task.

In our case, the `True_Cluster` labels from the synthetic data generation provide an ideal benchmark for ARI. The Silhouette Score offers an intrinsic measure of cluster quality regardless of ground truth.

<aside class="negative">
Ensure that both k-Means and Hierarchical Clustering have been run at least once to populate the session state with their respective labels, otherwise, the comparison table might show "N/A" for the scores.
</aside>

The Streamlit application presents a comparison table:

```
Comparison Table:
Clustering Algorithm    Silhouette Score    Adjusted Rand Index (vs True Labels)
k-Means                          0.543                                 0.725
Hierarchical Clustering          0.501                                 0.680

Adjusted Rand Index (k-Means vs Hierarchical Clustering): 0.654
```
*(Note: The actual values will vary based on the random state and selected parameters)*

From the comparison, we can observe the strengths and weaknesses of each clustering algorithm on our synthetic financial dataset. For instance, one algorithm might yield better separation (higher Silhouette Score) while another might more accurately recover the predefined latent groups (higher ARI). This comprehensive evaluation guides us in selecting the most appropriate clustering approach for specific financial analysis needs.

## 7. Financial Applications of Clustering
Duration: 0:08:00

Clustering techniques are not merely academic exercises; they have profound practical applications in financial markets. This section explores how k-Means and Hierarchical Clustering can be leveraged for critical tasks like portfolio construction and advanced risk management.

### Financial Application: Portfolio Construction with k-Means

In financial markets, k-Means clustering offers a practical approach to **portfolio construction**. Clustering stocks based on their continuous trend characteristics allows for the identification of groups of assets that exhibit similar market behaviors.

By categorizing assets into distinct clusters, financial data engineers can:

*   **Diversification**: Ensure that a portfolio includes assets from different clusters to achieve better diversification, reducing idiosyncratic risk.
*   **Strategic Allocation**: Allocate capital based on the characteristics of each cluster. For example, assets within a "high growth, high volatility" cluster might be treated differently from those in a "stable income, low volatility" cluster.
*   **Risk Management**: Monitor clusters for unusual behavior. If all assets within a particular cluster show signs of distress, it could indicate a sector-specific risk or a broader market trend affecting that asset group.

This enables a more informed and data-driven approach to constructing and managing investment portfolios, moving beyond traditional sector classifications to behavior-based groupings.

### Financial Application: Hierarchical Risk Parity (HRP) with Hierarchical Clustering

Hierarchical Clustering finds a significant application in advanced portfolio management, particularly in the context of **Hierarchical Risk Parity (HRP)**. HRP is an alternative to traditional mean-variance optimization, aiming to build more robust and diversified portfolios.

HRP leverages the hierarchical structure revealed by clustering to address common issues in portfolio optimization, such as instability and concentration. The process typically involves:

1.  **Hierarchical Grouping**: Apply hierarchical clustering (often based on asset correlation) to group assets into a dendrogram structure.
2.  **Quasi-Diagonalization**: Reorder the correlation matrix according to the dendrogram, revealing block-like structures of highly correlated assets.
3.  **Recursive Bisection**: Recursively allocate capital through the hierarchy, inverse-variance weighting within each identified cluster. This ensures that risk is balanced not only at the overall portfolio level but also within nested clusters.

<aside class="positive">
This method is particularly valuable for achieving better diversification and managing risk by reflecting the true, often complex, interdependencies between financial instruments, which might not be apparent in a flat (non-hierarchical) view of the market.
</aside>

### Hierarchical Risk Parity (HRP) Workflow

```
+--+
|              START                |
|      (Input: Asset Returns)       |
+--+
        |
        v
+--+
|  1. Calculate Asset Correlation   |
|     Matrix (or distance metric)   |
+--+
        |
        v
+--+
|  2. Hierarchical Clustering       |
|     (Form Dendrogram based on     |
|      correlation distances)       |
+--+
        |
        v
+--+
|  3. Quasi-Diagonalization         |
|     (Reorder correlation matrix   |
|      based on dendrogram)         |
+--+
        |
        v
+--+
|  4. Recursive Bisection           |
|     (Top-down capital allocation   |
|      using inverse-variance       |
|      weighting within clusters)   |
+--+
        |
        v
+--+
|              END                  |
|   (Output: HRP Portfolio Weights) |
+--+
```

## 8. Conclusion
Duration: 0:03:00

This application has provided a comprehensive exploration of two fundamental unsupervised clustering techniques: k-Means and Hierarchical Clustering. We've walked through the process of generating synthetic financial asset data, preprocessing it, implementing both algorithms with interactive parameter adjustments, visualizing their results, and evaluating their performance using the Silhouette Score and Adjusted Rand Index.

For Financial Data Engineers, understanding and applying these methods are crucial for:

*   **Identifying underlying asset classes**: Moving beyond conventional sector definitions to data-driven groupings.
*   **Enhancing portfolio diversification**: Constructing portfolios that are robust to market shifts by combining assets from distinct behavioral clusters.
*   **Informing risk management**: Gaining deeper insights into interconnected asset behaviors and systemic risks.

By mastering these unsupervised learning techniques, you are better equipped to navigate the complexities of financial data, uncover valuable insights, and make more informed decisions in portfolio management and risk assessment. The ability to interactively adjust parameters and evaluate results empowers you to tailor these powerful tools to diverse financial analysis challenges.
