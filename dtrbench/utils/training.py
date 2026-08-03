"""
Training utilities for evaluating subforests in DTRBench.

This module provides functions for training and managing random forests and their selected subforests.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def train_own_random_forest(X_train, y_train, n_trees, seed):
    """Train a random forest from scratch using the provided training data and parameters.

    Args:
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        n_trees (int): Number of trees in the random forest.
        seed (int): Random seed for reproducibility.

    Returns:
        random_forest_trees (list): List of trained decision tree estimators.
        bootstrap_indices_list (list): List of bootstrap sample indices for each tree.
        oob_indices_list (list): List of out-of-bag sample indices for each tree
    """

    random_forest_trees = []
    bootstrap_indices_list = []
    oob_indices_list = []
    for tree in range(n_trees):
        bootstrap_indices = _generate_sample_indices(
            seed + tree, X_train.shape[0], X_train.shape[0]
        )
        X_boot, y_boot = X_train[bootstrap_indices], y_train[bootstrap_indices]
        oob_indices = _generate_unsampled_indices(X_train.shape[0], bootstrap_indices)
        template_tree = DecisionTreeClassifier(
            max_depth=10, max_features="sqrt", random_state=seed
        )
        template_tree.fit(X_boot, y_boot)
        random_forest_trees.append(template_tree)
        bootstrap_indices_list.append(bootstrap_indices)
        oob_indices_list.append(oob_indices)
    return random_forest_trees, bootstrap_indices_list, oob_indices_list


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    random_instance = np.random.RandomState(random_state)
    sample_indices = random_instance.randint(
        low=0, high=n_samples, size=n_samples_bootstrap, dtype=np.int32
    )
    return sample_indices


def _generate_unsampled_indices(n_samples, sample_indices):
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices
