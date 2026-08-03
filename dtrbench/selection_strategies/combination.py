"""
 Combination-based (performance & diversity) subforest selection strategies for DTRBench.

This module provides various combination-based (performance & diversity) strategies for selecting a subset of trees (subforest) from a random forest based on their pairwise distances.
"""

from dataclasses import dataclass

import numpy as np

from dtrbench.selection_strategies.common import subforest_oob_mcc
from dtrbench.selection_strategies.registry import register_selection_strategy

_ZSCALE_N_RANDOM_SOLUTIONS = 100


@register_selection_strategy("combination-greedy")
def select_subforest_via_combination_greedy(
    distance_matrix,
    subforest_size,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    mcc_computation,
    seed,
):
    """Select a subforest using greedy selection based on the distance matrix and out-of-bag predictions.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        all_oob_preds (list): List of out-of-bag predictions for each tree in the random forest.
        oob_indices_list (list): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the training labels.
        mcc_computation (str): Method for computing the performance score. Options are "per_tree" or "subforest". "per_tree" uses per-tree MCC, while "subforest" uses subforest MCC after adding a candidate tree (slower).
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""

    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    mcc_per_tree = np.array(
        [
            subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)
            for i in range(n_trees)
        ]
    )
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling
    mean_dist_to_all = distance_matrix.mean(axis=1)
    dist_scaler = _fit_zscaler(mean_dist_to_all)

    if mcc_computation == "subforest":
        _, perf_scaler = _fit_combination_zscalers_from_random_solutions(
            distance_matrix=distance_matrix,
            subforest_size=subforest_size,
            mcc_per_tree=mcc_per_tree,
            all_oob_preds=all_oob_preds,
            oob_indices_list=oob_indices_list,
            y_train=y_train,
            n_classes=n_classes,
            mcc_computation="subforest",
            seed=seed,
        )
    else:
        perf_scaler = _fit_zscaler(mcc_per_tree)

    selected = []
    remaining = set(range(n_trees))

    # first tree
    z_div0 = -dist_scaler.transform(mean_dist_to_all)
    z_perf0 = perf_scaler.transform(mcc_per_tree)
    score0 = (w_div * z_div0 + w_perf * z_perf0) / (w_div + w_perf)

    first = int(rng.choice(np.where(score0 == np.nanmax(score0))[0]))
    selected.append(first)
    remaining.remove(first)

    # subsequent trees
    while len(selected) < subforest_size:
        rem = np.array(list(remaining), dtype=int)

        # diversity
        avg_dist_to_selected = distance_matrix[np.ix_(rem, selected)].mean(axis=1)
        z_div = dist_scaler.transform(avg_dist_to_selected)

        # performance
        if mcc_computation == "subforest":
            perf_vals = []
            for cand in rem:
                mcc = subforest_oob_mcc(
                    selected + [int(cand)],
                    all_oob_preds,
                    oob_indices_list,
                    y_train,
                    n_classes,
                )
                perf_vals.append(mcc)
            perf_vals = np.array(perf_vals, dtype=float)
        else:
            k = len(selected)
            sel_sum = np.nansum(mcc_per_tree[selected])
            perf_vals = (sel_sum + mcc_per_tree[rem]) / (k + 1)

        z_perf = perf_scaler.transform(perf_vals)
        scores = (w_div * z_div + w_perf * z_perf) / (w_div + w_perf)

        best_idx = rem[np.where(scores == np.nanmax(scores))[0]]
        chosen = int(rng.choice(best_idx))
        selected.append(chosen)
        remaining.remove(chosen)

    return selected


@register_selection_strategy("combination-simulated_annealing")
def select_subforest_via_combination_simulated_annealing(
    distance_matrix,
    subforest_size,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    mcc_computation,
    seed,
):
    """Select a subforest using simulated annealing based on the distance matrix and out-of-bag predictions.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        all_oob_preds (list): List of out-of-bag predictions for each tree in the random forest.
        oob_indices_list (list): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the training labels.
        mcc_computation (str): Method for computing the performance score. Options are "per_tree" or "subforest". "per_tree" uses per-tree MCC, while "subforest" uses subforest MCC after adding a candidate tree (slower).
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""

    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    # per-tree MCC
    mcc_per_tree = np.array(
        [
            subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)
            for i in range(n_trees)
        ]
    )
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling
    dist_scaler, perf_scaler = _fit_combination_zscalers_from_random_solutions(
        distance_matrix=distance_matrix,
        subforest_size=subforest_size,
        mcc_per_tree=mcc_per_tree,
        all_oob_preds=all_oob_preds,
        oob_indices_list=oob_indices_list,
        y_train=y_train,
        n_classes=n_classes,
        mcc_computation="subforest",
        seed=seed,
    )

    best_solution_trees = current_solution_trees = rng.choice(
        n_trees, size=subforest_size, replace=False
    ).tolist()
    best_solution_score = current_solution_score = _score_solution(
        current_solution_trees,
        distance_matrix,
        mcc_per_tree,
        w_div,
        w_perf,
        dist_scaler,
        perf_scaler,
        all_oob_preds,
        oob_indices_list,
        y_train,
        n_classes,
        mcc_computation,
    )

    # hyperparameter testing
    average_delta_score = []
    for _ in range(25):
        candidate_solution_trees = current_solution_trees.copy()
        idx_to_replace = rng.choice(subforest_size)
        available_indices = list(set(range(n_trees)) - set(candidate_solution_trees))

        candidate_solution_trees[idx_to_replace] = rng.choice(available_indices)
        candidate_solution_score = _score_solution(
            candidate_solution_trees,
            distance_matrix,
            mcc_per_tree,
            w_div,
            w_perf,
            dist_scaler,
            perf_scaler,
            all_oob_preds,
            oob_indices_list,
            y_train,
            n_classes,
            mcc_computation,
        )

        average_delta_score.append(candidate_solution_score - current_solution_score)
    average_delta_score = float(np.nanmean(average_delta_score))

    # hyperparameters
    if not np.isfinite(average_delta_score) or abs(average_delta_score) < 1e-12:
        current_temp = 1.0
    else:
        current_temp = average_delta_score / np.log(0.8)
        if not np.isfinite(current_temp) or current_temp <= 0:
            current_temp = (
                abs(float(current_temp)) if np.isfinite(current_temp) else 1.0
            )
            current_temp = max(current_temp, 1e-3)
    min_temp = 0.1
    cooling_rate = 0.98
    steps_per_temp = n_trees

    while current_temp > min_temp:
        for _ in range(steps_per_temp):
            candidate_solution_trees = current_solution_trees.copy()
            idx_to_replace = rng.choice(subforest_size)
            available_indices = list(
                set(range(n_trees)) - set(candidate_solution_trees)
            )
            candidate_solution_trees[idx_to_replace] = rng.choice(available_indices)
            candidate_solution_score = _score_solution(
                candidate_solution_trees,
                distance_matrix,
                mcc_per_tree,
                w_div,
                w_perf,
                dist_scaler,
                perf_scaler,
                all_oob_preds,
                oob_indices_list,
                y_train,
                n_classes,
                mcc_computation,
            )

            if candidate_solution_score > best_solution_score:
                best_solution_trees = candidate_solution_trees.copy()
                best_solution_score = candidate_solution_score

            if candidate_solution_score > current_solution_score:
                current_solution_trees = candidate_solution_trees.copy()
                current_solution_score = candidate_solution_score
            else:
                score_diff = candidate_solution_score - current_solution_score
                acceptance_prob = (
                    np.exp(score_diff / current_temp) if score_diff < 0 else 1.0
                )
                if rng.rand() < acceptance_prob:
                    current_solution_trees = candidate_solution_trees.copy()
                    current_solution_score = candidate_solution_score

        current_temp *= cooling_rate

    return best_solution_trees


@register_selection_strategy("combination-genetic")
def select_subforest_via_combination_genetic(
    distance_matrix,
    subforest_size,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    mcc_computation,
    seed,
):
    """Select a subforest using a genetic algorithm based on the distance matrix and out-of-bag predictions.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        all_oob_preds (list): List of out-of-bag predictions for each tree in the random forest.
        oob_indices_list (list): List of out-of-bag indices for each tree in the random forest.
        y_train (np.ndarray): Train-set labels.
        n_classes (int): Number of unique classes in the training labels.
        mcc_computation (str): Method for computing the performance score. Options are "per_tree" or "subforest". "per_tree" uses per-tree MCC, while "subforest" uses subforest MCC after adding a candidate tree (slower).
        seed (int): Random seed for reproducibility.

    Returns:
        subforest_indices (list[int]): List of indices of the selected trees in the subforest.
        clustering_silhouette_score (float): Silhouette score of the clustering."""

    rng = np.random.RandomState(seed)
    n_trees = distance_matrix.shape[0]
    w_div = 0.5
    w_perf = 0.5

    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")

    # hyperparameters
    population_size = 100
    elite_percentage = 0.03
    crossover_probability = 0.9
    mutation_probability = 1 / (subforest_size * 2)
    max_generations = 50

    # per-tree MCC
    mcc_per_tree = np.array(
        [
            subforest_oob_mcc([i], all_oob_preds, oob_indices_list, y_train, n_classes)
            for i in range(n_trees)
        ]
    )
    if np.all(~np.isfinite(mcc_per_tree)):
        mcc_per_tree = np.zeros(n_trees, dtype=float)

    # scaling
    dist_scaler, perf_scaler = _fit_combination_zscalers_from_random_solutions(
        distance_matrix=distance_matrix,
        subforest_size=subforest_size,
        mcc_per_tree=mcc_per_tree,
        all_oob_preds=all_oob_preds,
        oob_indices_list=oob_indices_list,
        y_train=y_train,
        n_classes=n_classes,
        mcc_computation=mcc_computation,
        seed=seed,
        n_random_solutions=_ZSCALE_N_RANDOM_SOLUTIONS,
    )

    # initial population
    population = [
        rng.choice(n_trees, size=subforest_size, replace=False).tolist()
        for _ in range(population_size)
    ]

    best_solution = None
    best_score = -np.inf

    for _ in range(max_generations):
        scores = np.array(
            [
                _score_solution(
                    sol,
                    distance_matrix,
                    mcc_per_tree,
                    w_div,
                    w_perf,
                    dist_scaler,
                    perf_scaler,
                    all_oob_preds,
                    oob_indices_list,
                    y_train,
                    n_classes,
                    mcc_computation,
                )
                for sol in population
            ],
            dtype=float,
        )

        # track best
        gen_best_idx = int(np.nanargmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = float(scores[gen_best_idx])
            best_solution = population[gen_best_idx].copy()

        # elites
        elite_count = max(1, int(round(population_size * elite_percentage)))
        elite_indices = np.argsort(scores)[-elite_count:]
        next_generation = [population[i].copy() for i in elite_indices]

        # fill rest of generation
        while len(next_generation) < population_size:
            parent_indices = _sus_select(scores, 2, rng)
            p1 = population[parent_indices[0]]
            p2 = population[parent_indices[1]]

            if rng.rand() < crossover_probability and subforest_size > 1:
                cut = rng.randint(1, subforest_size)
                c1 = _crossover(p1, p2, cut)
                c2 = _crossover(p2, p1, cut)
                c1 = _mutate(c1, mutation_probability, n_trees, rng)
                c2 = _mutate(c2, mutation_probability, n_trees, rng)
                next_generation.append(c1)
                if len(next_generation) < population_size:
                    next_generation.append(c2)
            else:
                c1 = _mutate(p1.copy(), mutation_probability, n_trees, rng)
                c2 = _mutate(p2.copy(), mutation_probability, n_trees, rng)
                next_generation.append(c1)
                if len(next_generation) < population_size:
                    next_generation.append(c2)

        population = next_generation

    return best_solution


@dataclass(frozen=True)
class ZScaler:
    mean_: float
    std_: float
    eps: float = 1e-12

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        std = self.std_ if abs(self.std_) > self.eps else 1.0
        return (x - self.mean_) / std


def _score_solution(
    sol,
    distance_matrix,
    mcc_per_tree,
    w_dist,
    w_perf,
    dist_scaler,
    perf_scaler,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    mcc_computation,
):
    sol = list(sol)
    avg_dist = _avg_pairwise_distance(sol, distance_matrix)

    if mcc_computation == "subforest":
        perf = subforest_oob_mcc(
            sol, all_oob_preds, oob_indices_list, y_train, n_classes
        )
    else:
        perf = float(np.nanmean(mcc_per_tree[sol])) if len(sol) else np.nan

    z_dist = float(dist_scaler.transform([avg_dist])[0])
    z_perf = float(perf_scaler.transform([perf])[0]) if np.isfinite(perf) else 0.0

    return float((w_dist * z_dist + w_perf * z_perf) / (w_dist + w_perf))


def _crossover(p1, p2, cut):
    size = len(p1)

    child = p1[:cut]

    # right part of p2
    for gene in p2[cut:]:
        if gene not in child:
            child.append(gene)

        if len(child) == size:
            return child

    # left part of p2
    for gene in p2[:cut]:
        if gene not in child:
            child.append(gene)

        if len(child) == size:
            return child

    return child


def _mutate(child, mutation_probability, n_trees, rng):
    used = set(child)
    for i in range(len(child)):
        if rng.rand() < mutation_probability:
            available = list(set(range(n_trees)) - used)
            if not available:
                continue
            new_gene = int(rng.choice(available))
            used.remove(child[i])
            child[i] = new_gene
            used.add(new_gene)
    return child


def _sus_select(scores, n_select, rng):
    scores = np.asarray(scores, dtype=float)
    if np.all(~np.isfinite(scores)):
        return rng.choice(len(scores), size=n_select, replace=True).tolist()
    min_score = np.nanmin(scores)
    shift = -min_score if min_score < 0 else 0.0
    adj = scores + shift + 1e-12
    total = float(np.sum(adj))
    if total <= 0 or not np.isfinite(total):
        return rng.choice(len(scores), size=n_select, replace=True).tolist()
    step = total / n_select
    start = rng.uniform(0, step)
    points = start + step * np.arange(n_select)

    cum = np.cumsum(adj)
    selected = []
    i = 0
    for p in points:
        while cum[i] < p:
            i += 1
        selected.append(i)
    return selected


def _fit_zscaler(values, eps=1e-12):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ZScaler(mean_=0.0, std_=1.0, eps=eps)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=0))
    if not np.isfinite(std) or abs(std) <= eps:
        std = 1.0
    return ZScaler(mean_=mean, std_=std, eps=eps)


def _avg_pairwise_distance(sol, distance_matrix):
    sol = list(sol)
    if len(sol) <= 1:
        return 0.0
    sub = distance_matrix[np.ix_(sol, sol)]
    tri = sub[np.triu_indices_from(sub, k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.nanmean(tri))


def _sample_random_solutions(n_trees, subforest_size, n_samples, rng):
    if subforest_size <= 0 or subforest_size > n_trees:
        raise ValueError(f"subforest_size must be in the range [1, {n_trees}]")
    n_samples = int(max(1, n_samples))
    return [
        rng.choice(n_trees, size=subforest_size, replace=False).astype(int).tolist()
        for _ in range(n_samples)
    ]


def _fit_combination_zscalers_from_random_solutions(
    distance_matrix,
    subforest_size,
    mcc_per_tree,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
    mcc_computation,
    seed,
    n_random_solutions=_ZSCALE_N_RANDOM_SOLUTIONS,
):
    rng = np.random.RandomState(seed + 1337)
    n_trees = int(distance_matrix.shape[0])

    sols = _sample_random_solutions(n_trees, subforest_size, n_random_solutions, rng)

    dist_samples = np.array(
        [_avg_pairwise_distance(sol, distance_matrix) for sol in sols], dtype=float
    )
    dist_scaler = _fit_zscaler(dist_samples)

    if mcc_computation == "subforest":
        perf_samples = np.array(
            [
                subforest_oob_mcc(
                    sol, all_oob_preds, oob_indices_list, y_train, n_classes
                )
                for sol in sols
            ]
        )
        perf_scaler = _fit_zscaler(perf_samples)
    else:
        perf_scaler = _fit_zscaler(mcc_per_tree)

    return dist_scaler, perf_scaler
