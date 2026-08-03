"""

Density-based subforest selection strategy for DTRBench.

This module provides a density-based strategy for selecting a subset of trees (subforest) from a random forest based on their pairwise distances.
"""

import numpy as np

from dtrbench.selection_strategies.common import subforest_oob_mcc
from dtrbench.selection_strategies.registry import register_selection_strategy


@register_selection_strategy("density")
def select_subforest_via_density(
    distance_matrix,
    subforest_size,
    seed,
    all_oob_preds,
    y_train,
    n_classes,
    oob_indices_list=None,
    sigma_grid=None,
    alpha_grid=None,
):
    """Select a subforest using a density-based strategy based on the distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        seed (int): Random seed for reproducibility.
        all_oob_preds (list[np.ndarray]): List of out-of-bag predictions for each tree in the random forest.
        y_train (np.ndarray): Train-set labels. Required for some selection strategies that need access to the target labels.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest.
        sigma_grid (list[float]): Grid of sigma values. Sigma adjusts the influence radius for density calculations. If not provided, a default grid will be used.
        alpha_grid (list[float]): Grid of alpha values. Alpha controls the reduction of probabilities for trees with many close neighbors. If not provided, a default grid will be used.
        
    Returns:
        best_sel (list[int]): List of indices of the selected trees in the subforest.
        best_sigma (float): Sigma value that yielded the best MCC.
        best_alpha (float): Alpha value that yielded the best MCC.
    """
    sigma_candidates = (
        _default_sigma_grid(distance_matrix)
        if sigma_grid is None
        else np.asarray(sigma_grid, dtype=float)
    )
    sigma_candidates = sigma_candidates[
        np.isfinite(sigma_candidates) & (sigma_candidates > 0)
    ]
    if sigma_candidates.size == 0:
        sigma_candidates = np.array([1e-8], dtype=float)

    alpha_candidates = (
        _default_alpha_grid()
        if alpha_grid is None
        else np.asarray(alpha_grid, dtype=float)
    )
    alpha_candidates = alpha_candidates[
        np.isfinite(alpha_candidates) & (alpha_candidates > 0)
    ]
    if alpha_candidates.size == 0:
        alpha_candidates = np.array([1e-8], dtype=float)

    best_sigma = None
    best_alpha = None
    best_mcc = -np.inf
    best_sel = None

    trial = 0
    for s in sigma_candidates:
        for a in alpha_candidates:
            sel = _select_subforest_density_once(
                distance_matrix, subforest_size, seed + trial, float(s), float(a)
            )
            trial += 1
            mcc = subforest_oob_mcc(
                sel, all_oob_preds, oob_indices_list, y_train, n_classes
            )
            if np.isfinite(mcc) and mcc > best_mcc:
                best_mcc = mcc
                best_sigma = float(s)
                best_alpha = float(a)
                best_sel = sel


    return best_sel, best_sigma, best_alpha

def _select_subforest_density_once(
    distance_matrix, subforest_size, seed, sigma, alpha, return_trace=False
):
    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    densities = np.exp(-(distance_matrix**2) / max(float(sigma), 1e-12)).sum(axis=1)
    probs = densities / max(float(densities.sum()), 1e-12)

    remaining = set(range(n_trees))
    selected = []
    trace = []

    for step in range(subforest_size):
        rem = list(remaining)

        if return_trace:
            remaining_mask = np.zeros(n_trees, dtype=bool)
            remaining_mask[rem] = True

        p = np.array([probs[i] for i in rem], dtype=float)
        p_sum = p.sum()
        if p_sum <= 0 or not np.isfinite(p_sum):
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_sum
        chosen = int(rng.choice(rem, p=p))
        selected.append(chosen)
        remaining.remove(chosen)

        if return_trace:
            trace.append(
                {
                    "step": step + 1,
                    "probs": probs.copy(),
                    "selected_so_far": selected.copy(),
                    "chosen": chosen,
                    "remaining_mask": remaining_mask,
                }
            )

        for i in remaining:
            probs[i] *= distance_matrix[i, chosen] ** alpha

    if return_trace:
        return selected, trace
    return selected


def _default_sigma_grid(distance_matrix):
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    if upper.size == 0:
        return np.array([1.0], dtype=float)
    med = float(np.median(upper)) if np.median(upper) > 0 else 1e-12
    std = float(np.std(upper))
    k = std + 3
    return np.geomspace(med / k, med * k, 17)


def _default_alpha_grid():
    return np.geomspace(1 / 3, 3.0, 17, dtype=float)