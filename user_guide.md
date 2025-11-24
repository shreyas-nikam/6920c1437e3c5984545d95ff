id: 6920c1437e3c5984545d95ff_user_guide
summary: Anomaly Sentinel: Financial Outlier Detection User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Exploring Financial Asset Grouping with Unsupervised Clustering

## Welcome and Learning Goals
Duration: 02:00

Welcome to this interactive codelab, designed to guide Financial Data Engineers through the fascinating world of unsupervised clustering for financial asset grouping. In today's dynamic financial markets, understanding the inherent relationships and structures within asset data is paramount for making informed decisions in portfolio management and risk assessment.

This application provides a hands-on experience with two fundamental unsupervised learning techniques: **k-Means Clustering** and **Hierarchical Clustering**. You will not only learn their core principles but also apply them to synthetic financial data, visualize the results, and evaluate their effectiveness. This will equip you with the skills to uncover hidden patterns that can significantly enhance portfolio diversification, risk management, and overall investment strategies.

This application is specifically tailored for **Financial Data Engineers** and data scientists with a foundational understanding of data analysis. Throughout this guide, we'll focus on the *concepts* behind these powerful algorithms and *how to use the application* to explore them, rather than diving deep into the underlying code.

Upon completing this codelab, you will be able to:
*   Understand the principles and mechanics of k-Means and Hierarchical Clustering algorithms.
*   Generate and preprocess synthetic financial asset data suitable for clustering.
*   Implement k-Means clustering and interactively adjust the number of clusters ($k$).
*   Implement Hierarchical Clustering, select various linkage methods (e.g., single, complete, average, ward), and interactively define a cutoff distance for cluster formation.
*   Visualize clustering results effectively using scatter plots for k-Means (with centroids) and dynamic dendrograms for Hierarchical Clustering.
*   Calculate and interpret key evaluation metrics, including the Silhouette Score and Adjusted Rand Index (ARI), to assess clustering quality.
*   Discuss the practical financial applications of these techniques, such as portfolio construction and Hierarchical Risk Parity (HRP).

## Introduction to Financial Asset Grouping
Duration: 01:30

In the world of finance, "ground truth" labels for complex phenomena like market regimes or asset correlations are often elusive or expensive to obtain. This is where **unsupervised learning** shines. It allows us to uncover hidden structures and patterns within data without relying on predefined categories.

**Clustering** is a core unsupervised technique that groups similar data points together based on their inherent characteristics. For Financial Data Engineers, applying clustering to assets (such as stocks, bonds, or currencies) can reveal natural groupings that are crucial for making informed decisions in portfolio construction, risk management, and market analysis. By identifying assets that behave similarly or share common characteristics, we can build more diversified portfolios, understand systemic risk, and devise more robust trading strategies.

This application will focus on two fundamental clustering algorithms: **k-Means Clustering** and **Hierarchical Clustering**. We will explore how they work, apply them to simulated financial asset data, visualize their results, and evaluate their effectiveness using established metrics.

## Preparing Synthetic Financial Asset Data
Duration: 02:30

To provide a hands-on experience, this application uses a **synthetic dataset** designed to simulate real-world financial assets. This approach allows us to demonstrate clustering techniques in a controlled environment where we know the "true" underlying groups, which is helpful for evaluating our results.

Navigate to the `Data Preparation` section in the sidebar.

The dataset represents `stock returns` or `bond features` and has an inherent cluster structure. Each asset is given a unique `Asset_ID` and three continuous numerical features:
*   `Feature_1`: Represents `Daily_Return_Volatility`.
*   `Feature_2`: Represents `Average_Daily_Return`.
*   `Feature_3`: Represents `Beta_to_Market`.

These features are chosen because they reflect common characteristics used in financial analysis and portfolio management to describe assets.

<aside class="positive">
You can observe a sample of the **Generated Financial Data** in the main panel. This initial data, including the `True_Cluster` column, serves as our ground truth for evaluation later.
</aside>

We've generated 100 financial assets, each with these three features. The `True_Cluster` column indicates the original, hidden groups that `make_blobs` created, which we will use as a benchmark. This dataset is now ready for preprocessing.

## Preprocessing Features: Scaling for Better Clustering
Duration: 01:30

Before applying clustering algorithms, **data preprocessing** is a critical step. Many clustering algorithms, especially those based on distance calculations like k-Means and Hierarchical Clustering, are sensitive to the scale of input features. If one feature has a much larger numerical range than others, it can unfairly dominate the distance calculations, leading to biased and inaccurate clustering results.

To address this, we use `StandardScaler` from `sklearn.preprocessing`. This technique transforms the data so that each feature has a mean of 0 and a standard deviation of 1 (unit variance). This ensures that all features contribute equally to the distance computations, allowing the algorithms to identify clusters based on the true relationships between features.

The formula for standardization for a data point $x$ and feature $j$ is:
$$ 
    z_j = \frac{x_j - \mu_j}{\sigma_j}
$$
where $\mu_j$ is the mean of feature $j$ and $\sigma_j$ is its standard deviation.

<aside class="positive">
Observe the **Scaled Financial Data Sample** and its **Description** in the `Data Preparation` section. Notice how the mean is now close to 0 and the standard deviation is close to 1 for each feature, signifying successful standardization.
</aside>

This standardization ensures that our clustering algorithms will identify groups based on the inherent patterns in asset characteristics, rather than being skewed by differences in their measurement scales.

## Understanding k-Means Clustering
Duration: 02:00

Now that our data is prepared, let's dive into our first clustering technique: **k-Means Clustering**. This is one of the most widely used unsupervised algorithms, valued for its simplicity and efficiency in partitioning data.

The core idea of k-Means is to divide $n$ data points into $k$ distinct clusters, where each data point belongs to the cluster whose central point (or **centroid**) is closest to it.

The algorithm works iteratively:
1.  <b>Initialization</b>: First, the algorithm randomly selects $k$ data points from the dataset to serve as initial cluster centroids.
2.  <b>Assignment Step</b>: Each data point in the dataset is then assigned to the cluster whose centroid it is closest to. Typically, Euclidean distance is used to measure this closeness.
3.  <b>Update Step</b>: After all points are assigned, the centroids are recalculated. Each new centroid is the mean position of all data points currently assigned to that cluster.
4.  <b>Convergence</b>: Steps 2 and 3 are repeated. The process continues until the centroids no longer change significantly between iterations, or a predefined maximum number of iterations is reached.

<aside class="positive">
A key characteristic of k-Means is that you must **pre-specify the number of clusters, $k$**. This is an important decision that can influence the clustering outcome.
</aside>

In finance, k-Means is powerful for grouping stocks based on their continuous trend characteristics. For example, it can help identify groups of assets that exhibit similar market behaviors, aiding in diversification and risk management for portfolio construction.

## Implementing and Visualizing k-Means Clusters
Duration: 03:00

Let's apply k-Means to our scaled financial data and observe its effects. Navigate to the `KMeans Clustering` section in the sidebar.

In this section, you will interactively explore the k-Means algorithm:
1.  **Adjust the `Number of Clusters (k)` slider**: Experiment with values between 2 and 7. This slider directly controls the $k$ in k-Means, allowing you to see how different numbers of clusters affect the grouping.
2.  Click the **`Run k-Means Clustering`** button.

Once executed, you will see the `k-Means Clustering Results`, including:
*   The `k-Means Labels`: These are numerical IDs indicating which cluster each of your 100 assets has been assigned to.
*   The `k-Means Centroids`: These are the coordinates in our feature space that represent the center of each cluster. Their shape will be `(k, 3)`, where `k` is your selected number of clusters and `3` corresponds to our three features.

<aside class="positive">
Pay close attention to how the centroids change as you adjust $k$. Each centroid is essentially the "average asset" within its cluster.
</aside>

### Visualizing k-Means Clusters

Visualizing the clustering results is essential for intuitive understanding. The application provides an interactive scatter plot:
*   Each point on the plot represents a financial asset.
*   The color of each point indicates its assigned cluster.
*   The prominent **'X' markers** denote the cluster centroids.
*   Hover over an asset point to see its `Asset_ID`.

The plot typically shows `Feature_1` (`Daily_Return_Volatility`) on the x-axis and `Feature_2` (`Average_Daily_Return`) on the y-axis, allowing you to visualize two key characteristics.

<aside class="positive">
Experiment with the `Number of Clusters (k)` slider, run the clustering, and observe how the asset groupings and centroid positions change on the scatter plot. Can you identify any assets that seem to be on the "edge" of a cluster or are surprisingly grouped?
</aside>

This visualization helps you assess the compactness and separation of the clusters and understand how assets with similar characteristics are grouped together.

## Understanding Hierarchical Clustering
Duration: 02:00

Now, let's explore **Hierarchical Clustering**, an alternative approach that doesn't require you to pre-specify the number of clusters. Instead, it builds a complete hierarchy of clusters, which can be visualized as a **dendrogram**.

The most common form is **Agglomerative Hierarchical Clustering**, a "bottom-up" method:
1.  <b>Initialization</b>: The process starts by treating each individual data point (each financial asset) as its own separate cluster. If you have 100 assets, you begin with 100 clusters.
2.  <b>Merging</b>: In each step, the algorithm identifies the two closest clusters and merges them into a single larger cluster. This merging process continues iteratively until only one large cluster remains, encompassing all data points, or until a specific stopping criterion is met.

The crucial concept here is how "closeness" between clusters is determined. This is defined by the **linkage method**:
*   **Single Linkage**: Measures the distance between the closest points in two clusters. It tends to form long, "chain-like" clusters.
*   **Complete Linkage**: Measures the distance between the farthest points in two clusters. It tends to form compact, spherical clusters.
*   **Average Linkage**: Calculates the average distance between all points in two clusters. It's a compromise between single and complete linkage.
*   **Ward Linkage**: Minimizes the variance within each merged cluster. This method often results in more balanced clusters and is frequently a good default choice.

The primary output of Hierarchical Clustering is the **dendrogram**, a tree-like diagram that visually represents the entire hierarchy of merges. This visualization is key to understanding the relationships between assets at different levels of granularity.

In finance, Hierarchical Clustering is fundamental to advanced concepts like **Hierarchical Risk Parity (HRP)** for portfolio diversification, where asset relationships are inferred from this hierarchical structure to optimize capital allocation.

## Implementing and Visualizing Hierarchical Clusters with a Dendrogram
Duration: 03:30

Let's implement Agglomerative Hierarchical Clustering. Navigate to the `Hierarchical Clustering` section in the sidebar.

You can interactively control the clustering process:
1.  **Select `Linkage Method`**: Choose from 'ward', 'complete', 'average', or 'single'. This directly influences how cluster distances are calculated.
2.  **Adjust `Number of Clusters (n)` slider**: While Hierarchical Clustering doesn't *require* a pre-specified $k$, you can still tell the algorithm to output a specific number of clusters by cutting the dendrogram at a certain height.
3.  Click the **`Run Hierarchical Clustering`** button.

The application will display:
*   The `Selected Linkage Method` and `Number of Clusters`.
*   `Hierarchical Clustering Labels`: The cluster assignments for each asset based on your chosen parameters.
*   `Linkage Matrix`: This array encodes the full hierarchical structure, which is then used to generate the dendrogram.

### Visualizing with a Dendrogram

The dendrogram is the key to understanding Hierarchical Clustering. It visually maps out the sequence of merges:
*   The leaves at the bottom of the dendrogram represent individual assets.
*   As you move up, horizontal lines indicate merges between clusters.
*   The **height** of each horizontal line represents the *distance* at which those clusters were merged.

<aside class="positive">
Use the **`Dendrogram Cutoff Distance` slider** to dynamically "cut" the dendrogram. As you move the red horizontal line, you'll see different branches (clusters) highlighted. Each distinct vertical line that intersects the cutoff corresponds to a separate cluster. This interactive tool helps you decide on a suitable number of clusters by visually inspecting natural groupings in the data.
</aside>

Experiment with different `Linkage Methods` and observe how the dendrogram's structure changes. Then, adjust the `Dendrogram Cutoff Distance` to see how the number and composition of clusters are affected.

## Evaluating Cluster Quality: Silhouette Score
Duration: 02:00

After implementing clustering, we need ways to quantitatively assess how good our clusters are. We'll start with the **Silhouette Score**, an internal validation metric. This means it measures the quality of clustering based *only* on the data and the assignments, without needing any "true" labels.

The Silhouette Score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). It ranges from -1 to 1:
*   Values close to **+1** indicate that data points are well-matched to their own cluster and distinct from neighboring clusters (this is good!).
*   Values around **0** suggest overlapping clusters, or points that are on or very close to the boundary between clusters.
*   Values close to **-1** imply that data points might have been assigned to the wrong cluster.

For each data point $i$, the silhouette coefficient $s(i)$ is calculated as:
$$ s(i) = \frac{b(i) - a(i)}{\max[a(i), b(i)]} $$
where:
*   $a(i)$ is the mean distance between data point $i$ and all other data points in the same cluster (mean intracluster distance).
*   $b(i)$ is the minimum mean distance between data point $i$ and all data points in any other cluster (mean intercluster distance to the nearest neighboring cluster).

The overall Silhouette Score for the entire clustering is simply the average $s(i)$ over all data points.

Navigate to the `Evaluation` section in the sidebar.

<aside class="positive">
Observe the **Silhouette Scores** for both k-Means and Hierarchical Clustering. A higher score indicates better-defined and more separated clusters.
</aside>

These scores provide a quantitative measure of how well each algorithm grouped the financial assets based on their characteristics, purely from an internal perspective.

## Evaluating Cluster Quality: Adjusted Rand Index (ARI)
Duration: 02:00

Next, we'll use the **Adjusted Rand Index (ARI)**, an external evaluation metric. Unlike the Silhouette Score, ARI measures the similarity between two clusterings and explicitly accounts for chance. This metric is incredibly useful when you have "true labels" (ground truth) for your data, or when you want to compare how similar the outputs of different clustering algorithms are on the same dataset.

The ARI ranges from -1 to 1:
*   A value of **1** indicates perfect agreement between the two clusterings (e.g., your algorithm perfectly reproduced the true labels).
*   A value of **0** suggests that the clusterings are independent (they are as similar as random labeling would be).
*   **Negative values** imply worse-than-random agreement.

Since our synthetic dataset includes `True_Cluster` labels (the original groups generated by `make_blobs`), we can use the ARI to see how well our algorithms recovered these underlying groups. We can also use it to compare the similarity between the k-Means and Hierarchical Clustering results.

The formula for ARI is:
$$ARI = \frac{RI - Expected_{RI}}{\max(RI) - Expected_{RI}}$$
where $RI$ is the Rand Index, and $Expected_{RI}$ is its expected value under a null hypothesis of random clustering.

Navigate to the `Evaluation` section in the sidebar.

<aside class="positive">
Examine the **Adjusted Rand Index (ARI) Scores**. You'll see scores for:
*   k-Means vs. True Labels
*   Hierarchical Clustering vs. True Labels
*   k-Means vs. Hierarchical Clustering
</aside>

A higher ARI, especially against the `True_Cluster` labels, means the algorithm successfully identified the underlying patterns. Comparing the ARI between the two algorithms gives insights into how consistently they group assets.

## Comparing Clustering Approaches
Duration: 01:00

Having evaluated both k-Means and Hierarchical Clustering using the Silhouette Score (internal quality) and the Adjusted Rand Index (external quality against ground truth and between algorithms), we can now make a direct comparison of their performance on our synthetic financial dataset.

Navigate to the `Evaluation` section in the sidebar.

<aside class="positive">
Review the **Comparison Table** provided. It summarizes the Silhouette Scores and Adjusted Rand Index (vs True Labels) for both clustering algorithms.
</aside>

From this comparison, you can observe the strengths and weaknesses of each approach. For example, one algorithm might yield better separation (a higher Silhouette Score) while another might more accurately recover the predefined latent groups (a higher ARI against true labels). This comprehensive evaluation is key to understanding which clustering approach might be more suitable for different financial analysis tasks.

Consider how the different parameters you chose (e.g., `k` for k-Means, `linkage method` for Hierarchical Clustering) affected these scores. This iterative process of experimenting and evaluating is central to applying machine learning effectively.

## Real-World Financial Applications
Duration: 02:00

Clustering techniques are not just theoretical tools; they have significant practical implications in finance. Let's explore two key applications:

### Portfolio Construction with k-Means

In financial markets, k-Means clustering provides a practical and data-driven approach to **portfolio construction**. By grouping stocks or other assets based on their continuous characteristics (like our `Daily_Return_Volatility`, `Average_Daily_Return`, and `Beta_to_Market` features), we can identify assets that exhibit similar market behaviors.

This allows Financial Data Engineers to:
*   <b>Diversification</b>: Move beyond traditional sector classifications to behavior-based groupings. By including assets from different clusters, portfolios can achieve better diversification, reducing idiosyncratic risk.
*   <b>Strategic Allocation</b>: Allocate capital based on the unique characteristics of each cluster. For example, assets within a "high growth, high volatility" cluster might be treated differently from those in a "stable income, low volatility" cluster.
*   <b>Risk Management</b>: Monitor clusters for unusual behavior. If all assets within a particular cluster show signs of distress, it could signal a sector-specific risk or a broader market trend affecting that asset group.

This approach leads to more informed and robust investment portfolios.

### Hierarchical Risk Parity (HRP) with Hierarchical Clustering

Hierarchical Clustering is particularly impactful in advanced portfolio management through the concept of **Hierarchical Risk Parity (HRP)**. HRP is an alternative to traditional mean-variance optimization, designed to build more robust and diversified portfolios by accounting for the complex dependencies between assets.

The HRP process, leveraging hierarchical clustering, typically involves:
1.  <b>Hierarchical Grouping</b>: Hierarchical clustering (often applied to asset correlation matrices) is used to group assets into a tree-like dendrogram structure, revealing their nested relationships.
2.  <b>Quasi-Diagonalization</b>: The correlation matrix is then reordered according to this dendrogram structure. This reordering tends to bring highly correlated assets closer together, revealing block-like structures.
3.  <b>Recursive Bisection</b>: Capital is recursively allocated through the hierarchy, inverse-variance weighting within each identified cluster. This ensures that risk is balanced not only at the overall portfolio level but also within each nested cluster, providing better diversification.

HRP is invaluable for managing risk by reflecting the true interdependencies between financial instruments, which are often complex and not visible in a simple, flat view of the market.

## Conclusion
Duration: 01:30

Congratulations! You have successfully navigated through this interactive exploration of unsupervised clustering for financial asset grouping. We've covered the entire journey, from generating and preprocessing synthetic financial asset data to implementing and evaluating both k-Means and Hierarchical Clustering algorithms. You've also gained insights into their practical applications in finance.

For Financial Data Engineers, mastering these unsupervised learning techniques is crucial for:
*   <b>Identifying underlying asset classes</b>: Moving beyond conventional sector definitions to uncover data-driven groupings.
*   <b>Enhancing portfolio diversification</b>: Constructing portfolios that are more robust to market shifts by combining assets from distinct behavioral clusters.
*   <b>Informing risk management</b>: Gaining deeper insights into interconnected asset behaviors and systemic risks.

The ability to interactively adjust parameters, visualize results, and quantitatively evaluate clustering performance empowers you to apply these powerful tools effectively to diverse financial analysis challenges. Keep exploring, experimenting, and refining your understanding of these techniques to unlock valuable insights in the complex world of financial data.
