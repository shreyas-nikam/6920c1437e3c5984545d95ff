
import streamlit as st

st.header("Section 15: Financial Application: Portfolio Construction with k-Means")
st.markdown("""
In financial markets, k-Means clustering offers a practical approach to **portfolio construction**. As highlighted by Wu, Wang, and Wu (2022) [1], clustering stocks based on their continuous trend characteristics allows for the identification of groups of assets that exhibit similar market behaviors.

By categorizing assets into distinct clusters, financial data engineers can:
*   **Diversification**: Ensure that a portfolio includes assets from different clusters to achieve better diversification, reducing idiosyncratic risk.
*   **Strategic Allocation**: Allocate capital based on the characteristics of each cluster. For example, assets within a "high growth, high volatility" cluster might be treated differently from those in a "stable income, low volatility" cluster.
*   **Risk Management**: Monitor clusters for unusual behavior. If all assets within a particular cluster show signs of distress, it could indicate a sector-specific risk or a broader market trend affecting that asset group.

This enables a more informed and data-driven approach to constructing and managing investment portfolios, moving beyond traditional sector classifications to behavior-based groupings.
""")

st.header("Section 16: Financial Application: Hierarchical Risk Parity (HRP) with Hierarchical Clustering")
st.markdown("""
Hierarchical Clustering finds a significant application in advanced portfolio management, particularly in the context of **Hierarchical Risk Parity (HRP)**, as introduced by López de Prado (2016) [5]. HRP is an alternative to traditional mean-variance optimization, aiming to build more robust and diversified portfolios.

HRP leverages the hierarchical structure revealed by clustering to address common issues in portfolio optimization, such as instability and concentration. The process typically involves:
1.  **Hierarchical Grouping**: Apply hierarchical clustering (often based on asset correlation) to group assets into a dendrogram structure.
2.  **Quasi-Diagonalization**: Reorder the correlation matrix according to the dendrogram, revealing block-like structures of highly correlated assets.
3.  **Recursive Bisection**: Recursively allocate capital through the hierarchy, inverse-variance weighting within each identified cluster. This ensures that risk is balanced not only at the overall portfolio level but also within nested clusters.

This method is particularly valuable for achieving better diversification and managing risk by reflecting the true, often complex, interdependencies between financial instruments, which might not be apparent in a flat (non-hierarchical) view of the market.
""")

