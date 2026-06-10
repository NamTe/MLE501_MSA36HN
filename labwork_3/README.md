# Labwork 3

This lab applies K-Means and agglomerative hierarchical clustering (AHC) on two labeled datasets, evaluates cluster quality, then repeats clustering after dimensionality reduction.

The labels are not used for clustering. They are used only after clustering to calculate external validation metrics.

## Run

```powershell
python labwork3_clustering.py
```

The script creates:

- `labwork3_report.md`: report with tables, comments, and visualizations
- `outputs/clustering_results.csv`: all numeric results
- `outputs/*_clusters.png`: 2D cluster visualizations

## Methods

- Datasets: Wine and Breast Cancer Wisconsin from scikit-learn
- Clustering: K-Means and AHC
- Dimensionality reduction: PCA, SVD, and t-SNE
- Evaluation: ARI, NMI, homogeneity, completeness, and silhouette score
