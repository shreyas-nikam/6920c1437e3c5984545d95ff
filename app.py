
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

# Overall Business Logic / Application Overview
st.markdown("""
### Learning Goals
This Streamlit application will provide Financial Data Engineers with an interactive tool to explore and apply unsupervised clustering techniques—specifically k-Means and Hierarchical Clustering—to financial asset data. Upon completion, users will be able to:
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

# Section 1: Introduction to Financial Asset Grouping
st.header("Section 1: Introduction to Financial Asset Grouping")
st.markdown("""
Unsupervised learning techniques are powerful tools for uncovering hidden structures and patterns within data without relying on predefined labels. In the realm of finance, where "ground truth" labels for complex phenomena like market regimes or asset correlations are often elusive or expensive to obtain, unsupervised methods are invaluable.

Clustering, a prominent unsupervised technique, groups similar data points together based on their inherent characteristics. For Financial Data Engineers, applying clustering to assets (e.g., stocks, bonds, currencies) can reveal natural groupings that inform critical decisions in portfolio construction, risk management, and market analysis. By identifying assets that behave similarly or share common characteristics, we can build more diversified portfolios, understand systemic risk, and devise more robust trading strategies.

This application will focus on two fundamental clustering algorithms: **k-Means Clustering** and **Hierarchical Clustering**. We will explore their mechanisms, apply them to synthetic financial asset data, visualize their results, and evaluate their effectiveness using established metrics.
""")

# Section 2: Learning Objectives
st.header("Section 2: Learning Objectives")
st.markdown("""
By the end of this interactive application, you will be able to:
*   Articulate the core principles of k-Means and Hierarchical Clustering algorithms.
*   Generate and prepare a synthetic dataset of financial asset features.
*   Apply k-Means clustering and interactively adjust the number of clusters ($k$).
*   Implement Hierarchical Clustering, experiment with various linkage methods (e.g., 