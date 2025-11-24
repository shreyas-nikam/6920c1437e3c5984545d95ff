# Interactive Lab: Unsupervised Learning for Financial Asset Grouping

![Streamlit App](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## Project Title and Description

**Interactive Lab: Unsupervised Learning for Financial Asset Grouping**

This Streamlit application serves as an interactive laboratory for Financial Data Engineers and data scientists to explore and apply unsupervised clustering techniques to synthetic financial asset data. Financial markets are dynamic and complex, often characterized by hidden patterns and interdependencies. Unsupervised learning offers powerful tools to uncover these latent structures, providing deeper insights into asset behavior, enhancing portfolio diversification, and improving risk management strategies.

The application guides users through the entire workflow:
*   **Data Generation & Preprocessing**: Creating synthetic financial asset data and preparing it for analysis.
*   **Clustering Algorithms**: Implementing and interacting with k-Means and Hierarchical Clustering.
*   **Visualization**: Generating insightful plots such as scatter plots with centroids and dynamic dendrograms.
*   **Evaluation**: Quantitatively assessing clustering quality using metrics like Silhouette Score and Adjusted Rand Index.
*   **Financial Applications**: Discussing practical use cases in portfolio construction and Hierarchical Risk Parity (HRP).

The goal is to provide a hands-on experience, allowing users to interactively adjust parameters and immediately observe their impact on clustering results and evaluation metrics.

## Features

This interactive Streamlit application offers the following key features:

*   **Interactive Learning Goals**: Clearly defined learning objectives and target audience for the lab.
*   **Synthetic Financial Data Generation**:
    *   Generates a synthetic dataset of 100 financial assets, each with 3 features (`Daily_Return_Volatility`, `Average_Daily_Return`, `Beta_to_Market`).
    *   Includes a "True_Cluster" label for external evaluation.
*   **Data Preprocessing**: Applies `StandardScaler` to ensure features contribute equally to clustering distance calculations.
*   **k-Means Clustering Implementation**:
    *   Allows interactive adjustment of the number of clusters (`k`) via a slider.
    *   Outputs cluster labels and centroids.
    *   Visualizes k-Means results using an interactive Plotly scatter plot, displaying assets colored by cluster and marking centroids.
*   **Hierarchical Clustering Implementation**:
    *   Supports interactive selection of various `linkage methods` (e.g., 'ward', 'complete', 'average', 'single').
    *   Allows interactive adjustment of the number of clusters.
    *   Outputs cluster labels and the linkage matrix.
    *   Visualizes the hierarchical structure with a dynamic Matplotlib dendrogram, featuring an adjustable `cutoff distance` to define clusters.
*   **Cluster Evaluation Metrics**:
    *   Calculates and displays the **Silhouette Score** for both k-Means and Hierarchical Clustering, measuring cluster cohesion and separation.
    *   Calculates and displays the **Adjusted Rand Index (ARI)** against the true labels and between the two clustering algorithms, assessing similarity to ground truth and inter-algorithm agreement.
*   **Comparative Analysis**: Provides a summary table comparing the Silhouette Scores and ARIs of k-Means and Hierarchical Clustering.
*   **Financial Applications Discussion**: Explains the practical relevance of clustering in financial contexts, including:
    *   Portfolio Construction with k-Means.
    *   Hierarchical Risk Parity (HRP) with Hierarchical Clustering.
*   **Modular Page Navigation**: Uses Streamlit's multipage structure for clear organization of different sections of the lab.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8 or higher installed. You will also need `pip` for package management.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    (Replace `https://github.com/your-username/your-repo-name.git` with the actual repository URL)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.0
    pandas
    numpy
    scikit-learn
    plotly
    matplotlib
    scipy
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Navigate to the project directory** (if you haven't already):
    ```bash
    cd your-repo-name
    ```

2.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

This command will open the application in your default web browser (usually at `http://localhost:8501`).

### Basic Usage Instructions:

*   **Navigation**: Use the sidebar on the left to navigate between different sections of the lab: "Introduction and Data", "k-Means Clustering", "Hierarchical Clustering", and "Evaluation and Applications".
*   **Interactive Widgets**: On each page, adjust sliders, select boxes, and click buttons to interact with the models and visualizations.
*   **Data Flow**: Ensure you go through the "Introduction and Data" page first to generate and preprocess the data, as subsequent pages depend on this initial setup stored in Streamlit's session state.

## Project Structure

The project is organized into a main application file and a directory containing individual page scripts for better modularity.

```
.
├── app.py                            # Main Streamlit entry point, handles navigation.
├── application_pages/                # Directory containing scripts for each application page.
│   ├── __init__.py                   # Makes 'application_pages' a Python package.
│   ├── page_1_data_generation.py     # Introduction, synthetic data generation, and preprocessing.
│   ├── page_2_kmeans_clustering.py   # k-Means clustering implementation and visualization.
│   ├── page_3_hierarchical_clustering.py # Hierarchical Clustering implementation and visualization.
│   └── page_4_evaluation_and_applications.py # Cluster evaluation and financial application discussions.
└── requirements.txt                  # List of Python dependencies for the project.
```

## Technology Stack

*   **Python**: Programming language (3.8+)
*   **Streamlit**: Web framework for building interactive data apps
*   **Pandas**: Data manipulation and analysis
*   **NumPy**: Numerical computing
*   **Scikit-learn**: Machine learning algorithms (k-Means, AgglomerativeClustering, StandardScaler, Silhouette Score, Adjusted Rand Index)
*   **Plotly Express**: Interactive data visualization (for k-Means scatter plots)
*   **Matplotlib**: Static plotting library (for Hierarchical Clustering dendrograms)
*   **SciPy**: Scientific computing, specifically for hierarchical clustering linkage functions

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to:

*   **Your Name/Organization**: [Link to Website/GitHub/LinkedIn]
*   **Email**: `your.email@example.com`

---