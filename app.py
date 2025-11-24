
import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, you will explore and apply unsupervised clustering techniques—specifically k-Means and Hierarchical Clustering—to financial asset data. Use the sidebar to navigate through the different sections of the application.

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

