# Week 3 Task: Unsupervised Learning and Clustering Analysis
# Dataset: Wine dataset from scikit-learn
# Algorithms: K-Means + Hierarchical (Ward)
# Includes preprocessing, model selection, evaluation and visualizations.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from scipy.cluster.hierarchy import dendrogram, linkage


# 1. LOAD DATASET
wine = load_wine(as_frame=True)
df = wine.frame

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# 2. SELECT FEATURES
# The target/class column is NOT used for clustering.
feature_names = wine.feature_names
X = df[feature_names]

print("\nFeatures used for clustering:")
print(feature_names)


# 3. EXPLORATORY VISUALIZATION
plt.figure(figsize=(12, 6))
plt.boxplot(
    [X[column].values for column in feature_names],
    labels=feature_names
)
plt.xticks(rotation=90)
plt.title("Distribution of Wine Chemistry Features")
plt.ylabel("Feature Value")
plt.tight_layout()
plt.show()


# 4. FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)


# 5. TEST DIFFERENT VALUES OF k
k_values = range(2, 9)

inertias = []
silhouette_scores = []
calinski_scores = []
davies_scores = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )
    calinski_scores.append(
        calinski_harabasz_score(X_scaled, labels)
    )
    davies_scores.append(
        davies_bouldin_score(X_scaled, labels)
    )


# 6. ELBOW METHOD
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# 7. SILHOUETTE ANALYSIS
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Values of k")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# Automatically select k with the highest silhouette score
best_k = list(k_values)[np.argmax(silhouette_scores)]

print("\nBest number of clusters:", best_k)
print("Best silhouette score:", max(silhouette_scores))


# 8. MODEL SELECTION METRICS
metrics_df = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertias,
    "Silhouette": silhouette_scores,
    "Calinski-Harabasz": calinski_scores,
    "Davies-Bouldin": davies_scores
})

print("\nModel Selection Metrics:")
print(metrics_df.round(3))


# 9. FINAL K-MEANS MODEL
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = kmeans.fit_predict(X_scaled)

df_clustered = X.copy()
df_clustered["Cluster"] = cluster_labels

print("\nClustered data:")
print(df_clustered.head())


# 10. CLUSTER SIZES
cluster_counts = (
    df_clustered["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nNumber of samples in each cluster:")
print(cluster_counts)

plt.figure(figsize=(7, 5))
plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)
plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.title("Number of Samples in Each Cluster")
plt.tight_layout()
plt.show()


# 11. PCA FOR 2D VISUALIZATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print("\nPCA explained variance ratio:")
print(explained_variance)

print(
    "Total variance explained by first two PCs:",
    explained_variance.sum()
)


# 12. K-MEANS CLUSTER VISUALIZATION
plt.figure(figsize=(9, 6))

for cluster in sorted(np.unique(cluster_labels)):
    mask = cluster_labels == cluster

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        alpha=0.75,
        label=f"Cluster {cluster}"
    )

# Project K-Means centroids into PCA space
centroids_pca = pca.transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.1f}% variance)"
)

plt.title("K-Means Clusters in PCA Space")
plt.legend()
plt.tight_layout()
plt.show()


# 13. CLUSTER PROFILE - ORIGINAL SCALE
cluster_means = (
    df_clustered
    .groupby("Cluster")
    .mean()
)

print("\nOriginal-scale cluster means:")
print(cluster_means.round(2))


# 14. STANDARDIZED CLUSTER CENTERS
cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=feature_names
)

cluster_centers.index.name = "Cluster"

print("\nStandardized cluster centers:")
print(cluster_centers.round(2))


# 15. CLUSTER FEATURE HEATMAP
plt.figure(figsize=(12, 6))

plt.imshow(
    cluster_centers,
    aspect="auto"
)

plt.colorbar(label="Standardized Mean")

plt.xticks(
    range(len(feature_names)),
    feature_names,
    rotation=90
)

plt.yticks(
    range(best_k),
    [f"Cluster {i}" for i in range(best_k)]
)

plt.title("Cluster Feature Signatures")
plt.tight_layout()
plt.show()


# 16. HIERARCHICAL CLUSTERING
hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

print("\nHierarchical cluster labels:")
print(hierarchical_labels)


# 17. HIERARCHICAL DENDROGRAM
Z = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Compressed Observation Groups")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()


# 18. COMPARE K-MEANS AND HIERARCHICAL CLUSTERING
kmeans_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print("\nK-Means Silhouette Score:",
      round(kmeans_silhouette, 3))

print("Hierarchical Silhouette Score:",
      round(hierarchical_silhouette, 3))


comparison_df = pd.DataFrame({
    "Method": [
        "K-Means",
        "Hierarchical (Ward)"
    ],
    "Silhouette": [
        silhouette_score(X_scaled, cluster_labels),
        silhouette_score(X_scaled, hierarchical_labels)
    ],
    "Calinski-Harabasz": [
        calinski_harabasz_score(X_scaled, cluster_labels),
        calinski_harabasz_score(X_scaled, hierarchical_labels)
    ],
    "Davies-Bouldin": [
        davies_bouldin_score(X_scaled, cluster_labels),
        davies_bouldin_score(X_scaled, hierarchical_labels)
    ]
})

print("\nClustering Method Comparison:")
print(comparison_df.round(3))


# 19. POST-HOC VALIDATION
# Original target labels are NOT used during training.
# They are used only after clustering for comparison.
ari = adjusted_rand_score(
    wine.target,
    cluster_labels
)

print(
    "\nAdjusted Rand Index (post-hoc validation):",
    round(ari, 3)
)


# 20. FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL CLUSTERING SUMMARY")
print("=" * 60)

print("Dataset: Wine dataset")
print("Number of observations:", X.shape[0])
print("Number of features:", X.shape[1])
print("Selected number of clusters:", best_k)

print(
    "K-Means Silhouette Score:",
    round(kmeans_silhouette, 3)
)

print(
    "K-Means Calinski-Harabasz Score:",
    round(
        calinski_harabasz_score(
            X_scaled,
            cluster_labels
        ), 3
    )
)

print(
    "K-Means Davies-Bouldin Score:",
    round(
        davies_bouldin_score(
            X_scaled,
            cluster_labels
        ), 3
    )
)

print(
    "Adjusted Rand Index:",
    round(ari, 3)
)

print("\nCluster sizes:")
print(cluster_counts)

print("\nCluster means:")
print(cluster_means.round(2))
