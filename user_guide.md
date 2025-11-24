id: 6920c1437e3c5984545d95ff_user_guide
summary: Anomaly Sentinel: Financial Outlier Detection User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Unsupervised Learning for Financial Asset Grouping

## 1. Welcome to the Financial Asset Grouping Codelab
Duration: 0:05:00

Welcome to the "Unsupervised Learning for Financial Asset Grouping" Codelab! In this interactive guide, we will explore the powerful world of unsupervised learning, specifically focusing on **k-Means Clustering** and **Hierarchical Clustering**, to uncover hidden structures within financial asset data.

<aside class="positive">
<b>Why is this important for Financial Data Engineers?</b>
Financial markets are incredibly dynamic and complex. Traditional methods of categorizing assets, like sector classifications, might not always capture the full picture of how assets truly behave or interact. Unsupervised clustering offers a data-driven approach to:
<ul>
    <li>Identify underlying asset classes based on their inherent characteristics.</li>
    <li>Enhance portfolio diversification by grouping assets that behave similarly.</li>
    <li>Improve risk management strategies by understanding systemic risks within asset groups.</li>
</ul>
</aside>

This application is designed for **Financial Data Engineers** and data scientists interested in applying machine learning to financial markets. We'll provide a hands-on experience, allowing you to interact with key algorithms and visualize their impact.

**By the end of this codelab, you will be able to:**
*   Understand the core principles of k-Means and Hierarchical Clustering.
*   Generate and preprocess synthetic financial asset data.
*   Apply and interpret k-Means and Hierarchical Clustering results.
*   Visualize clustering outcomes using scatter plots and dendrograms.
*   Evaluate clustering quality using metrics like the Silhouette Score and Adjusted Rand Index.
*   Discuss practical financial applications, such as portfolio construction and Hierarchical Risk Parity (HRP).

The application starts on the **Introduction and Data** page (which is the current view). All necessary libraries have been loaded, and we are ready to dive into the exciting world of financial asset grouping!

## 2. Generating and Preprocessing Synthetic Financial Asset Data
Duration: 0:03:00

To begin our exploration, we need some data to work with. In real-world scenarios, you would use actual market data. For this codelab, we will generate a **synthetic dataset** that simulates financial asset features. This allows us to have a "ground truth" for evaluation later.

The generated dataset represents 100 financial assets, each with a unique `Asset_ID` and three key features:
*   **`Feature_1`**: Simulating `Daily_Return_Volatility`.
*   **`Feature_2`**: Representing `Average_Daily_Return`.
*   **`Feature_3`**: Modeling `Beta_to_Market`.

These features are chosen because they are commonly used in financial analysis to characterize asset behavior. The dataset is designed to have some inherent cluster structure, meaning there are natural groupings of assets that share similar characteristics.

**Action:**
1.  Navigate to the "Introduction and Data" page in the sidebar if you are not already there.
2.  Observe the "Generated Financial Data Sample" section. You will see a table displaying the first few rows of our synthetic dataset.
3.  Notice the `True_Cluster` column. This column represents the "latent" or actual underlying groups that were created during data generation. We will use this later to evaluate how well our clustering algorithms perform.

After data generation, the next crucial step in unsupervised learning is **data preprocessing**, specifically **feature scaling**. Many clustering algorithms are sensitive to the scale of features. If one feature has a much larger range of values than another, it can unfairly dominate the distance calculations, leading to biased clustering.

To address this, we apply **Standard Scaling**. This transformation adjusts the data so that each feature has a mean of 0 and a standard deviation of 1. The formula for standardization for a data point $x$ and feature $j$ is:
$$ z_j = \frac{x_j - \mu_j}{\sigma_j} $$
where $\mu_j$ is the mean of feature $j$ and $\sigma_j$ is its standard deviation. This ensures all features contribute equally to the clustering process.

**Action:**
1.  Scroll down to the "Scaled Financial Data Sample" section on the "Introduction and Data" page.
2.  Review the `head()` of the scaled data and the `describe()` table for mean and standard deviation. Notice how the means are close to 0 and standard deviations are close to 1 for all features.

## 3. Exploring k-Means Clustering
Duration: 0:07:00

Now that our data is ready, let's delve into our first clustering algorithm: **k-Means Clustering**. This algorithm is widely used due to its simplicity and efficiency. Its main goal is to partition data points into a predefined number of clusters, $k$, where each data point belongs to the cluster with the nearest mean (centroid).

The k-Means algorithm works iteratively:
1.  **Initialization**: Randomly select $k$ points from your data as initial cluster centroids.
2.  **Assignment Step**: Assign each data point to the cluster whose centroid is closest (e.g., using Euclidean distance).
3.  **Update Step**: Recalculate the centroids as the mean position of all data points assigned to that cluster.
4.  **Convergence**: Repeat steps 2 and 3 until the centroids no longer change significantly, indicating the clusters have stabilized.

A critical aspect of k-Means is that you need to specify the number of clusters, $k$, beforehand.

**Action:**
1.  Navigate to the **k-Means Clustering** page using the sidebar.
2.  In the "Implementing k-Means Clustering" section, you'll see a slider labeled **"Number of Clusters (k) for k-Means:"**. This allows you to interactively choose the value of $k$.
3.  Set the slider to a value of `4` (the true number of clusters in our synthetic data).
4.  Click the **"Run k-Means Clustering"** button.
5.  Observe the "k-Means Clustering Results" which display the cluster labels for the first few assets and the coordinates of the calculated centroids.

<aside class="positive">
<b>Tip: Experiment with `k`!</b>
Try adjusting the `Number of Clusters (k)` slider to different values (e.g., 2, 3, 5, 7) and re-running the clustering. Observe how the cluster assignments and centroid locations change. This helps build intuition about the algorithm's sensitivity to $k$.
</aside>

After running the clustering, visualization is key to understanding the results. A scatter plot helps us see how assets are distributed in a 2D feature space and how the cluster centroids relate to these groupings.

**Action:**
1.  Scroll down to the "Visualizing k-Means Clusters" section.
2.  Observe the interactive scatter plot. Each point represents a financial asset, colored according to its assigned cluster.
3.  The prominent **'X' markers** indicate the cluster centroids.
4.  Hover over individual data points to see their `Asset_ID` and their assigned cluster.
5.  Interact with the plot: zoom in, pan, and rotate to get a better view of the cluster separation.

By observing the plot, you can visually assess the compactness and separation of the clusters. Assets with similar `Daily_Return_Volatility` and `Average_Daily_Return` characteristics should be grouped together around their respective centroids.

## 4. Understanding Hierarchical Clustering
Duration: 0:07:00

Next, we move to **Hierarchical Clustering**, an algorithm that constructs a tree-like structure of clusters called a **dendrogram**. Unlike k-Means, it doesn't require you to pre-specify the number of clusters. The most common approach is **Agglomerative Hierarchical Clustering**, which is a "bottom-up" method:
1.  **Initialization**: Each data point starts as its own individual cluster.
2.  **Merging**: Iteratively merge the two closest clusters until only one large cluster remains.

The "closeness" between clusters is determined by a **linkage method**, which defines how the distance between two clusters is calculated. Common linkage methods include:
*   **Single Linkage**: Uses the distance between the closest points in the two clusters.
*   **Complete Linkage**: Uses the distance between the farthest points in the two clusters.
*   **Average Linkage**: Uses the average distance between all points in the two clusters.
*   **Ward Linkage**: Minimizes the variance within each merged cluster, often producing more spherical and compact clusters.

A key output is the **dendrogram**, which visually represents this hierarchy of merges.

**Action:**
1.  Navigate to the **Hierarchical Clustering** page using the sidebar.
2.  In the "Implementing Hierarchical Clustering" section, observe the "Hierarchical Clustering Parameters."
3.  You'll see a `selectbox` for **"Linkage Method:"** and a `slider` for **"Number of Clusters (n) for Hierarchical Clustering:"**.
4.  Select `'ward'` as the **Linkage Method** (a good starting point) and set the **Number of Clusters (n)** slider to `4`.
5.  Click the **"Run Hierarchical Clustering"** button.
6.  The "Hierarchical Clustering Results" will display the labels for the first few assets and the `Linkage Matrix` shape, which is the underlying data used to build the dendrogram.

<aside class="positive">
<b>Tip: Explore Linkage Methods!</b>
Try changing the `Linkage Method` (e.g., to 'complete', 'average', or 'single') and re-running the clustering. Notice how the cluster labels and the subsequent dendrogram structure can change significantly based on how "distance" between clusters is defined.
</aside>

The dendrogram is the primary visualization for Hierarchical Clustering. It illustrates the sequence of merges and the distance at which these merges occur. It's a powerful tool for discerning natural groupings and choosing an appropriate number of clusters. You can effectively "cut" the dendrogram at a certain height (distance) to define your clusters.

**Action:**
1.  Scroll down to the "Visualizing Hierarchical Clustering with a Dendrogram" section.
2.  Observe the interactive dendrogram. Each vertical line represents a data point or a merged cluster, and horizontal lines represent merges. The height of a horizontal line indicates the distance at which those clusters were merged.
3.  You'll see a slider labeled **"Dendrogram Cutoff Distance:"**. This is a dynamic red horizontal line on the dendrogram.
4.  Adjust the `Dendrogram Cutoff Distance` slider. As you move it, observe how the number of clusters (defined by the vertical lines below the cutoff) changes. For example, if you set the cutoff to approximately `6.0`, you might see 4 distinct clusters (colored differently below the line, with merges above in gray).

This interactive dendrogram helps you visually determine a suitable number of clusters based on the natural breaks or large distances between merges in the hierarchy.

## 5. Evaluating and Comparing Clustering Results
Duration: 0:08:00

After applying both k-Means and Hierarchical Clustering, it's essential to evaluate their performance to understand how well they grouped the financial assets. We will use two key metrics: the **Silhouette Score** and the **Adjusted Rand Index (ARI)**.

### Silhouette Score
The **Silhouette Score** is an internal validation metric. It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). It ranges from -1 to 1:
*   **+1**: Indicates a strong, clear separation of clusters.
*   **0**: Suggests overlapping clusters or points on cluster boundaries.
*   **-1**: Implies that data points might have been assigned to the wrong cluster.

For each data point $i$, the silhouette coefficient $s(i)$ is calculated as:
$$ s(i) = \frac{b(i) - a(i)}{\max[a(i), b(i)]} $$
where $a(i)$ is the mean distance to points in its own cluster, and $b(i)$ is the minimum mean distance to points in any other cluster. The overall Silhouette Score is the average of $s(i)$ for all data points.

**Action:**
1.  Navigate to the **Evaluation and Applications** page using the sidebar.
2.  In the "Cluster Evaluation: Silhouette Score" section, you will see the calculated Silhouette Scores for both k-Means and Hierarchical Clustering (assuming you have run both algorithms in previous steps).
3.  Observe and compare the scores. A higher score generally indicates better-defined and more separated clusters.

### Adjusted Rand Index (ARI)
The **Adjusted Rand Index (ARI)** is an external evaluation metric, meaning it requires knowledge of the "ground truth" labels (if available) or is used to compare two different clusterings. It measures the similarity between two clusterings, correcting for random chance. ARI ranges from -1 to 1:
*   **1**: Perfect agreement between the two clusterings.
*   **0**: Random agreement.
*   **Negative values**: Worse than random agreement.

Since our synthetic dataset includes `True_Cluster` labels, we can use ARI to see how well our algorithms recover these underlying groups. We can also use it to compare how similar the k-Means clustering results are to the Hierarchical Clustering results.

**Action:**
1.  Scroll down to the "Cluster Evaluation: Adjusted Rand Index (ARI)" section.
2.  You will see ARI scores comparing:
    *   k-Means labels against the `True_Cluster` labels.
    *   Hierarchical Clustering labels against the `True_Cluster` labels.
    *   k-Means labels against Hierarchical Clustering labels.
3.  Analyze these scores. An ARI closer to 1 (especially against `True_Cluster`) indicates that the algorithm successfully identified the original underlying patterns.

### Comparing Clustering Results
Finally, we can bring these evaluation metrics together to compare the performance of both algorithms on our synthetic financial dataset.

**Action:**
1.  Scroll down to the "Comparing Clustering Results" section.
2.  Review the comparison table, which summarizes the Silhouette Scores and ARI (vs. True Labels) for both algorithms.
3.  Also, note the ARI score comparing k-Means to Hierarchical Clustering.

This comparison helps you understand the strengths and weaknesses of each algorithm. For instance, one might excel at creating compact, separated clusters (high Silhouette Score), while another might be better at recovering the known underlying groups (high ARI). This quantitative analysis is crucial for selecting the most appropriate clustering approach for specific financial analysis needs.

## 6. Financial Applications and Conclusion
Duration: 0:05:00

Clustering is not just an academic exercise; it has profound practical implications in finance. Let's explore two significant applications:

### Financial Application: Portfolio Construction with k-Means
In portfolio construction, k-Means clustering can be used to group stocks based on their behavioral characteristics, such as `Daily_Return_Volatility`, `Average_Daily_Return`, and `Beta_to_Market`. By categorizing assets into distinct clusters, financial data engineers can:
*   **Enhance Diversification**: Ensure a portfolio includes assets from different clusters, reducing the risk that all assets move in the same direction.
*   **Strategic Capital Allocation**: Allocate capital based on the unique risk-return profiles of each cluster. For example, a "high growth, high volatility" cluster might warrant a different allocation strategy than a "stable income, low volatility" cluster.
*   **Improved Risk Management**: Monitor the health of individual clusters. Distress in a particular cluster could signal sector-specific risks or broader market trends affecting that asset group.

This approach allows for a more granular and data-driven strategy for building and managing investment portfolios, moving beyond traditional industry classifications to behavior-based groupings.

### Financial Application: Hierarchical Risk Parity (HRP) with Hierarchical Clustering
Hierarchical Clustering finds a powerful application in advanced portfolio management, particularly in the **Hierarchical Risk Parity (HRP)** strategy, introduced by López de Prado. HRP aims to build more robust and diversified portfolios compared to traditional mean-variance optimization, which can often lead to unstable and concentrated portfolios.

The HRP process typically involves:
1.  **Hierarchical Grouping**: Using hierarchical clustering, assets are grouped based on their correlation, forming a dendrogram structure that reveals their interdependencies.
2.  **Quasi-Diagonalization**: The correlation matrix is reordered according to the dendrogram, which helps to reveal "blocks" of highly correlated assets.
3.  **Recursive Bisection**: Capital is then allocated recursively through the hierarchy, with inverse-variance weighting applied within each identified cluster. This ensures that risk is balanced not only at the overall portfolio level but also within nested sub-clusters.

HRP is invaluable for managing risk and achieving better diversification by explicitly accounting for the complex, often non-linear, relationships between financial instruments.

### Conclusion

Congratulations! You have completed the "Unsupervised Learning for Financial Asset Grouping" codelab. You've journeyed through:
*   Generating and preparing synthetic financial asset data.
*   Implementing and visualizing both k-Means and Hierarchical Clustering.
*   Evaluating clustering performance using the Silhouette Score and Adjusted Rand Index.
*   Understanding the practical financial applications of these powerful techniques.

For Financial Data Engineers, mastering these unsupervised learning methods is crucial for:
*   Identifying subtle, data-driven asset classes beyond conventional definitions.
*   Constructing robust and diversified portfolios resilient to market shifts.
*   Gaining deeper insights into interconnected asset behaviors for enhanced risk management.

The ability to interactively adjust parameters, visualize, and evaluate results empowers you to effectively apply these tools to diverse and complex financial analysis challenges. Keep exploring and applying these concepts to real-world financial data!
