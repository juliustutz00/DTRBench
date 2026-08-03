"""
Common utility functions for subforest selection strategies in DTRBench.
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef


def subforest_oob_mcc(selected, all_oob_preds, oob_indices_list, y_train, n_classes):
    """Compute the Matthews correlation coefficient (MCC) for the out-of-bag predictions of the selected subforest.
    
    Args:
        selected (list[int]): List of indices of the selected trees in the subforest.
        all_oob_preds (list[np.ndarray]): List of out-of-bag predictions for each tree in the random forest.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the target labels.
        
    Returns:
        mcc (float): Matthews correlation coefficient for the out-of-bag predictions of the selected subforest. Returns NaN if there are not enough samples to compute the MCC."""
    n = y_train.shape[0]
    votes = np.zeros((n, n_classes), dtype=int)
    counts = np.zeros(n, dtype=int)

    for tidx in selected:
        preds = all_oob_preds[tidx]
        oob_idx = oob_indices_list[tidx]

        if preds is None:
            continue

        votes[oob_idx, preds] += 1
        counts[oob_idx] += 1

    mask = counts > 0
    if np.sum(mask) < 2:
        return np.nan

    y_hat = np.argmax(votes[mask], axis=1)
    return float(matthews_corrcoef(y_train[mask], y_hat))

def precompute_all_oob_predictions(trees, oob_indices_list, X_train):
    """Precompute the out-of-bag predictions for all trees in the random forest.
    
    Args:
        trees (list[DecisionTreeClassifier]): List of decision tree estimators in the random forest.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest.
        X_train (np.ndarray): Train-set used to fit the random forest.

    Returns:
        all_oob_preds (list[np.ndarray]): List of out-of-bag predictions for each tree in the random forest.
    """
    all_oob_preds = []
    for tidx, tree in enumerate(trees):
        oob_idx = oob_indices_list[tidx]
        if oob_idx is None or len(oob_idx) == 0:
            all_oob_preds.append(None)
        else:
            preds = tree.predict(X_train[oob_idx]).astype(int)
            all_oob_preds.append(preds)
    return all_oob_preds


def validate_subforest_size(subforest_size, n_trees):
    """Validate that the subforest size is within the valid range based on the number of trees in the random forest.
    
    Args:
        subforest_size (int): Desired size of the subforest to be selected.
        n_trees (int): Total number of trees in the random forest.
        
    Raises:
        ValueError: If subforest_size is not in the range [1, n_trees].
        """
    
    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")