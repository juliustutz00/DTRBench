"""
Similarity utilities for evaluating subforests in DTRBench.

This module provides functions for computing similarity measures between tree representations.
"""

import numpy as np


def compute_similarity_to_base_tree(
    similarity_fn,
    len_tree_representations,
    len_perturbations,
    len_intensities,
    len_perturbation_runs,
):
    """Compute the similarity of each tree representation to its corresponding base tree representation.
    
    Args:
        similarity_fn (callable): Function that computes the similarity between two tree indices.
        len_tree_representations (int): Total number of tree representations.
        len_perturbations (int): Number of perturbations.
        len_intensities (int): Number of intensity levels.
        len_perturbation_runs (int): Number of perturbation runs.

    Returns:
        list[dict]: A list of dictionaries containing the tree index and its similarity to the base tree.
    """
    similarity_values = []
    perturbation_offset = (
        len_perturbations * len_intensities * len_perturbation_runs + 1
    )
    for idx in range(len_tree_representations):
        if (idx % perturbation_offset) != 0:
            base_tree_idx = (idx // perturbation_offset) * perturbation_offset
            similarity = similarity_fn(idx, base_tree_idx)
            similarity_values.append(
                {"tree_idx": idx - 1, "similarity_to_base": similarity}
            )
    return similarity_values


def similarity_to_distance_matrix(similarity_fn, n):
    """Convert a similarity matrix to a distance matrix.
    
    Args:
        similarity_fn (callable): Function that computes the similarity between two tree indices.
        n (int): Number of trees.
        
    Returns:
        np.ndarray: A distance matrix derived from the similarity matrix.
    """
    similarity_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = similarity_fn(i, j)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    sim_min, sim_max = similarity_matrix.min(), similarity_matrix.max()
    similarity_matrix_normalized = (similarity_matrix - sim_min) / (
        sim_max - sim_min + 1e-10
    )
    distance_matrix = 1.0 - similarity_matrix_normalized
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix
