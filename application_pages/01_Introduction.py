
import streamlit as st

st.title("Unsupervised Learning for Financial Asset Grouping")
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
