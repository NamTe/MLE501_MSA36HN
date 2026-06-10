import os
import warnings
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, min(4, (os.cpu_count() or 2) - 1)))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def load_labeled_datasets():
    """Return labeled datasets. Labels are used only for evaluation."""
    return {
        "Wine": load_wine(as_frame=True),
        "Breast Cancer": load_breast_cancer(as_frame=True),
    }


def evaluate_clustering(x, y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "Silhouette": silhouette_score(x, y_pred),
    }


def run_clusterers(x, y_true, n_clusters):
    models = {
        "K-Means": KMeans(n_clusters=n_clusters, n_init=50, random_state=RANDOM_STATE),
        "AHC": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    }
    rows = []
    predictions = {}
    for model_name, model in models.items():
        y_pred = model.fit_predict(x)
        predictions[model_name] = y_pred
        metrics = evaluate_clustering(x, y_true, y_pred)
        rows.append({"Method": model_name, **metrics})
    return rows, predictions


def reduce_data(x):
    return {
        "PCA": PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x),
        "SVD": TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(x),
        "t-SNE": TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
            init="pca",
            learning_rate="auto",
            perplexity=30,
        ).fit_transform(x),
    }


def safe_name(name):
    return name.lower().replace(" ", "_").replace("-", "").replace("/", "_")


def plot_reduced_clusters(dataset_name, reduced_name, x_2d, y_true, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    plots = [("True labels", y_true), ("K-Means clusters", predictions["K-Means"]), ("AHC clusters", predictions["AHC"])]

    for ax, (title, labels) in zip(axes, plots):
        scatter = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap="tab10", s=24, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        legend = ax.legend(*scatter.legend_elements(), title="Group", loc="best", fontsize=8)
        ax.add_artist(legend)

    fig.suptitle(f"{dataset_name}: {reduced_name} 2D space", fontsize=13)
    path = OUTPUT_DIR / f"{safe_name(dataset_name)}_{safe_name(reduced_name)}_clusters.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(df):
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.4f}")
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def comment_on_dataset(df, dataset_name):
    original = df[df["Space"] == "Original"].sort_values("ARI", ascending=False).iloc[0]
    reduced = df[df["Space"] != "Original"].sort_values("ARI", ascending=False).iloc[0]
    worst = df.sort_values("ARI", ascending=True).iloc[0]

    lines = [
        f"- On {dataset_name}, the best original-space result is {original['Method']} "
        f"with ARI={original['ARI']:.3f} and NMI={original['NMI']:.3f}.",
        f"- The best reduced-space result is {reduced['Method']} on {reduced['Space']} "
        f"with ARI={reduced['ARI']:.3f} and NMI={reduced['NMI']:.3f}.",
    ]

    if reduced["ARI"] > original["ARI"] + 0.02:
        lines.append("- Dimensionality reduction improves clustering here, likely because noise or redundant features are compressed.")
    elif reduced["ARI"] < original["ARI"] - 0.02:
        lines.append("- Dimensionality reduction lowers clustering quality here, which suggests some discriminative structure was lost.")
    else:
        lines.append("- Reduced and original spaces perform similarly, so the 2D projection preserves much of the cluster structure.")

    lines.append(
        f"- The weakest result is {worst['Method']} on {worst['Space']} "
        f"with ARI={worst['ARI']:.3f}; this is visible in the plots when clusters overlap."
    )
    return "\n".join(lines)


def build_report(dataset_summaries, image_paths):
    report = [
        "# Labwork 3: Clustering and K-Means on Subspace",
        "",
        "## Datasets",
        "",
        "Two labeled datasets are used from scikit-learn:",
        "",
        "- Wine: 178 samples, 13 numeric features, 3 wine cultivar classes.",
        "- Breast Cancer Wisconsin: 569 samples, 30 numeric features, 2 diagnosis classes.",
        "",
        "The class labels are not used during clustering. They are used only after training to evaluate how well the unsupervised clusters match known classes.",
        "",
        "## Metrics",
        "",
        "- ARI and NMI compare predicted clusters with the true labels. Higher is better.",
        "- Homogeneity and completeness measure whether clusters contain one class and whether each class is assigned mostly to one cluster. Higher is better.",
        "- Silhouette uses only feature geometry and does not need labels. Higher is better, with values near 1 indicating better separated clusters.",
    ]

    for dataset_name, df in dataset_summaries.items():
        report.extend(
            [
                "",
                f"## {dataset_name}",
                "",
                markdown_table(df),
                "",
                "### Comments",
                "",
                comment_on_dataset(df, dataset_name),
                "",
                "### Visualizations",
                "",
            ]
        )
        for image_path in image_paths[dataset_name]:
            report.append(f"![{dataset_name} clusters]({image_path.as_posix()})")
            report.append("")

    report.extend(
        [
            "## Overall Conclusion",
            "",
            "K-Means and AHC can be trained directly on unlabeled feature data. Labels are useful only for external validation. "
            "Performance depends strongly on whether the actual classes form compact geometric groups. PCA and SVD often preserve global structure, "
            "while t-SNE can create clear 2D visual separation but may not always improve clustering metrics because its embedding is optimized for visualization rather than clustering.",
            "",
        ]
    )
    return "\n".join(report)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset_summaries = {}
    image_paths = {}

    for dataset_name, bunch in load_labeled_datasets().items():
        x_raw = bunch.data.to_numpy()
        y_true = bunch.target.to_numpy()
        n_clusters = len(np.unique(y_true))
        x_scaled = StandardScaler().fit_transform(x_raw)

        rows, _ = run_clusterers(x_scaled, y_true, n_clusters)
        for row in rows:
            row["Dataset"] = dataset_name
            row["Space"] = "Original"

        images = []
        for reduced_name, x_reduced in reduce_data(x_scaled).items():
            reduced_rows, predictions = run_clusterers(x_reduced, y_true, n_clusters)
            for row in reduced_rows:
                row["Dataset"] = dataset_name
                row["Space"] = reduced_name
            rows.extend(reduced_rows)
            images.append(plot_reduced_clusters(dataset_name, reduced_name, x_reduced, y_true, predictions))

        df = pd.DataFrame(rows)
        ordered_cols = ["Dataset", "Space", "Method", "ARI", "NMI", "Homogeneity", "Completeness", "Silhouette"]
        df = df[ordered_cols].sort_values(["Space", "Method"]).reset_index(drop=True)
        dataset_summaries[dataset_name] = df
        image_paths[dataset_name] = images

    report = build_report(dataset_summaries, image_paths)
    report_path = Path("labwork3_report.md")
    report_path.write_text(report, encoding="utf-8")

    csv_path = OUTPUT_DIR / "clustering_results.csv"
    pd.concat(dataset_summaries.values(), ignore_index=True).to_csv(csv_path, index=False)
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {sum(len(paths) for paths in image_paths.values())} plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
