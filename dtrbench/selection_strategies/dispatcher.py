"""
Organization of selection strategies for subforest selection in DTRBench.

This module provides the select_subforest_via_selection_strategy function, which selects a subset of trees (subforest) from a random forest based on the provided distance matrix and selection strategy. The function validates the subforest size, retrieves the appropriate selection strategy function, and computes the subforest indices and clustering silhouette score if applicable.
"""

import inspect

import numpy as np

from dtrbench.selection_strategies.common import (
    precompute_all_oob_predictions,
    validate_subforest_size,
)
from dtrbench.selection_strategies.registry import get_selection_strategy


def select_subforest_via_selection_strategy(
    distance_matrix,
    subforest_size,
    selection_strategy,
    seed,
    random_forest_trees=None,
    X_train=None,
    y_train=None,
    oob_indices_list=None,
    density_sigma_grid=None,
    density_alpha_grid=None,
):
    """Select a subset of trees (subforest) from a random forest based on the provided distance matrix and selection strategy.
    
    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        selection_strategy (str): Name of the selection strategy to be used for subforest selection.
        seed (int): Random seed for reproducibility.
        random_forest_trees (list): List of decision tree estimators in the random forest. Required for some selection strategies that need access to the trees.
        X_train (np.ndarray): Train-set used to fit the random forest. Required for some selection strategies that need access to the training data.
        y_train (np.ndarray): Train-set labels. Required for some selection strategies that need access to the target labels.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest. Required for some selection strategies that need access to out-of-bag predictions.
        density_sigma_grid (list[float]): Grid of sigma values for density-based selection strategy. If not provided, a default grid will be used.
        density_alpha_grid (list[float]): Grid of alpha values for density-based selection strategy. If not provided, a default grid will be used.
        
    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float | None): Silhouette score of the clustering (if applicable).
    """
    validate_subforest_size(subforest_size, distance_matrix.shape[0])
    strategy_func = get_selection_strategy(selection_strategy)
    sig = inspect.signature(strategy_func)
    params = sig.parameters

    all_oob_preds = None
    n_classes = None
    if "all_oob_preds" in params or "n_classes" in params:
        if y_train is not None:
            n_classes = len(np.unique(y_train))
        if (
            random_forest_trees is not None
            and oob_indices_list is not None
            and X_train is not None
        ):
            all_oob_preds = precompute_all_oob_predictions(
                random_forest_trees, oob_indices_list, X_train
            )

    arg_pool = {
        "distance_matrix": distance_matrix,
        "subforest_size": subforest_size,
        "seed": seed,
        "all_oob_preds": all_oob_preds,
        "oob_indices_list": oob_indices_list,
        "y_train": y_train,
        "n_classes": n_classes,
        "sigma_grid": density_sigma_grid,
        "alpha_grid": density_alpha_grid,
        "mcc_computation": "per_tree",
    }
    kwargs = {k: v for k, v in arg_pool.items() if k in params}

    result = strategy_func(**kwargs)
    clustering_silhouette_score = None
    if isinstance(result, tuple):
        if len(result) == 2:
            subforest_indices, clustering_silhouette_score = result
        else:
            subforest_indices = result[0]
    else:
        subforest_indices = result

    return subforest_indices, clustering_silhouette_score
