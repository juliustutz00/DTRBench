"""
Clustering-based subforest selection strategies for DTRBench.

This module provides various clustering-based strategies for selecting a subset of trees (subforest) from a random forest based on their pairwise distances.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

from dtrbench.selection_strategies.common import subforest_oob_mcc
from dtrbench.selection_strategies.registry import register_selection_strategy


@register_selection_strategy("k-medoid")
def select_subforest_via_kmedoid(distance_matrix, subforest_size, seed):
    """Select a subforest using k-medoid clustering based on the distance matrix.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""
    kmed_clustering = _cluster_labels(
        distance_matrix, subforest_size, method="k-medoid", seed=seed
    )

    subforest_indices = [int(x) for x in kmed_clustering.medoid_indices_]

    clustering_silhouette_score = silhouette_score(
        distance_matrix, kmed_clustering.labels_, metric="precomputed"
    )

    return subforest_indices, clustering_silhouette_score


@register_selection_strategy("agglomerative")
def select_subforest_via_agglomerative(distance_matrix, subforest_size, seed):
    """Select a subforest using agglomerative clustering based on the distance matrix.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""
    labels = _cluster_labels(
        distance_matrix, subforest_size, method="agglomerative", seed=seed
    )

    subforest_indices = []
    for cluster_id in range(subforest_size):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_index_within_cluster = cluster_indices[
            int(np.argmin(cluster_distances.sum(axis=1)))
        ]
        subforest_indices.append(medoid_index_within_cluster)

    clustering_silhouette_score = silhouette_score(
        distance_matrix, labels, metric="precomputed"
    )

    return subforest_indices, clustering_silhouette_score


@register_selection_strategy("k-medoid-performance")
def select_subforest_via_kmedoid_performance(
    distance_matrix,
    subforest_size,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    seed,
):
    """Select a subforest using k-medoid-performance clustering based on the distance matrix and out-of-bag predictions.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the training labels.
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""

    return _select_subforest_via_performance_clustering(
        distance_matrix=distance_matrix,
        subforest_size=subforest_size,
        method="k-medoid",
        all_oob_preds=all_oob_preds,
        oob_indices_list=oob_indices_list,
        y_train=y_train,
        n_classes=n_classes,
        seed=seed,
    )


@register_selection_strategy("agglomerative-performance")
def select_subforest_via_agglomerative_performance(
    distance_matrix,
    subforest_size,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    seed,
):
    """Select a subforest using agglomerative-performance clustering based on the distance matrix and out-of-bag predictions.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the training labels.
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""

    return _select_subforest_via_performance_clustering(
        distance_matrix=distance_matrix,
        subforest_size=subforest_size,
        method="agglomerative",
        all_oob_preds=all_oob_preds,
        oob_indices_list=oob_indices_list,
        y_train=y_train,
        n_classes=n_classes,
        seed=seed,
    )


def _select_subforest_via_performance_clustering(
    distance_matrix,
    subforest_size,
    method,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    seed,
):
    mcc_per_tree = np.array(
        [
            subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)
            for i in range(distance_matrix.shape[0])
        ]
    )

    if method == "k-medoid":
        kmed_clustering = _cluster_labels(
            distance_matrix, subforest_size, method="k-medoid", seed=seed
        )
        cluster_labels = kmed_clustering.labels_
        clustering_silhouette_score = silhouette_score(
            distance_matrix, cluster_labels, metric="precomputed"
        )
    elif method == "agglomerative":
        cluster_labels = _cluster_labels(
            distance_matrix, subforest_size, method="agglomerative", seed=seed
        )
        clustering_silhouette_score = silhouette_score(
            distance_matrix, cluster_labels, metric="precomputed"
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # select best performing tree from each cluster
    subforest_indices = []
    for cluster_id in range(subforest_size):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue

        # get OOB MCC for each tree in cluster
        cluster_performances = mcc_per_tree[cluster_indices]

        # select tree with highest performance
        valid_mask = ~np.isnan(cluster_performances)
        if np.any(valid_mask):
            best_local_idx = np.nanargmax(cluster_performances)
        else:
            best_local_idx = 0

        subforest_indices.append(int(cluster_indices[best_local_idx]))

    return subforest_indices, clustering_silhouette_score


def _cluster_labels(distance_matrix, subforest_size, method, seed):
    if method == "k-medoid":
        model = KMedoids(
            n_clusters=int(subforest_size), metric="precomputed", random_state=seed
        )
        model.fit(distance_matrix)
        return model
    elif method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=int(subforest_size), metric="precomputed", linkage="average"
        )
        return model.fit_predict(distance_matrix)
    raise ValueError(f"Unknown clustering method: {method}")
