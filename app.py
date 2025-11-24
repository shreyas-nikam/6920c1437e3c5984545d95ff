import streamlit as st

st.set_page_config(page_title="Anomaly Sentinel: Financial Outlier Detection", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("Anomaly Sentinel: Financial Outlier Detection")
st.divider()
st.markdown("""
In this lab, we will explore the fascinating world of unsupervised learning for financial asset grouping. Financial markets are dynamic and complex, often characterized by hidden patterns and interdependencies that are not immediately obvious. Unsupervised clustering techniques offer powerful tools to uncover these latent structures, enabling Financial Data Engineers to gain deeper insights into asset behavior, enhance portfolio diversification, and improve risk management strategies.

This application will guide you through the implementation and evaluation of two fundamental clustering algorithms: **k-Means Clustering** and **Hierarchical Clustering**. You will learn how to generate synthetic financial data, preprocess it, apply these algorithms with interactive controls, visualize their results, and assess their performance using key evaluation metrics. Finally, we will discuss their practical applications in critical financial domains like portfolio construction and Hierarchical Risk Parity (HRP).
""")

page = st.sidebar.selectbox(label="Navigation", options=[
    "Introduction and Data", 
    "k-Means Clustering", 
    "Hierarchical Clustering", 
    "Evaluation and Applications"
])

if page == "Introduction and Data":
    from application_pages.page_1_data_generation import main
    main()
elif page == "k-Means Clustering":
    from application_pages.page_2_kmeans_clustering import main
    main()
elif page == "Hierarchical Clustering":
    from application_pages.page_3_hierarchical_clustering import main
    main()
elif page == "Evaluation and Applications":
    from application_pages.page_4_evaluation_and_applications import main
    main()
