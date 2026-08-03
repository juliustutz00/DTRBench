"""
Subforest benchmark module for evaluating the performance of different decision tree representations and selection strategies.

This module provides functions for running subforest benchmarks and comparing the effectiveness of various decision tree representations and selection strategies on the downstream task of subforest selection.

"""

import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from dtrbench.representations.indtree_representation import INDTreeRepresentation
from dtrbench.representations.topological_forest_representation import (
    TopologicalForestRepresentation,
)
from dtrbench.selection_strategies.dispatcher import (
    select_subforest_via_selection_strategy,
)
from dtrbench.utils.evaluation import (
    evaluate_forest,
    prediction_agreement,
    shared_metric_cols,
)
from dtrbench.utils.metrics import tree_metric_score
from dtrbench.utils.similarity import similarity_to_distance_matrix
from dtrbench.utils.visualization import plot_and_save_mds


def run_subforest_benchmark(
    X_train,
    X_test,
    y_train,
    y_test,
    representation_options,
    random_forest_trees,
    oob_indices_list=None,
    n_trees=100,
    subforest_size=[10],
    selection_strategies=["k-medoid"],
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    results_root=r"..\results",
    dataset_name=None,
    fold_idx=None,
    seed=0,
):
    """
    Runs the subforest benchmark on a given dataset and returns the results as a DataFrame. 

    Args:
        X_train (np.ndarray): Train-set.
        X_test (np.ndarray): Test-set.
        y_train (np.ndarray): Train-set labels.
        y_test (np.ndarray): Test-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        representation_options (list[dict]): Representation options to evaluate.
        random_forest_trees (list[DecisionTreeClassifier]): Decision trees from a random forest.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree.
        n_trees (int): Total number of trees in the random forest.
        subforest_size (list[int]): List of subforest sizes to evaluate.
        selection_strategies (list[str]): List of selection strategies to evaluate.
        run_topological_forest (bool): Whether to run the topological forest representation.
        run_indtree (bool): Whether to run the INDTree representation.
        print_progress (bool): Whether to print progress messages.
        results_root (str): Root directory to save the MDS results.
        dataset_name (str): Name of the dataset (for reporting purposes).
        fold_idx (int): Index of the fold (for reporting purposes).
        seed (int): Random seed for reproducibility.

    Returns:
        A DataFrame containing the results of the subforest benchmark, including multiple performance metrics.

    Raises:
        ValueError: If subforest_size is empty or contains invalid sizes.
    """
    if print_progress:
        print("Running subforest benchmark...")

    sizes = [int(s) for s in subforest_size]

    if selection_strategies is None:
        selection_strategies = []
    elif isinstance(selection_strategies, (list, tuple)):
        selection_strategies = list(selection_strategies)
    else:
        selection_strategies = [selection_strategies]
    if len(selection_strategies) == 0:
        selection_strategies = ["k-medoid"]

    sizes = sorted({s for s in sizes if s is not None})
    if len(sizes) == 0:
        raise ValueError("subforest_size must not be empty.")
    for s in sizes:
        if s <= 0 or s > n_trees:
            raise ValueError(
                f"Each subforest_size must be in the range [1, {n_trees}], got {s}."
            )

    results = []
    n_instances_test = X_test.shape[0]
    class_labels = np.unique(y_train)
    n_classes = len(class_labels)

    # compute metrics of full random forest
    full_forest_eval = evaluate_forest(
        X_test, y_test, random_forest_trees, n_instances_test, n_classes
    )

    results.append(
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "fold": int(fold_idx) if fold_idx is not None else None,
            "representation": "Full Forest",
            "selection_strategy": None,
            "full_forest_size": int(n_trees),
            "subforest_size": int(n_trees),
            **shared_metric_cols(full_forest_eval),
            "silhouette_score": np.nan,
            "agreement_with_full_forest": np.nan,
            "indices": np.nan,
        }
    )

    oob_mccs = []
    for idx, tree in enumerate(random_forest_trees):
        oob_idx = oob_indices_list[idx]
        if oob_idx is None or len(oob_idx) == 0:
            oob_mccs.append(np.nan)
        else:
            try:
                oob_mccs.append(
                    tree_metric_score(
                        tree, X_train[oob_idx], y_train[oob_idx], metric="mcc"
                    )
                )
            except:  # noqa: E722
                oob_mccs.append(np.nan)
    oob_mccs_np = np.array(oob_mccs)

    # Single Tree baseline
    single_DT = DecisionTreeClassifier(max_depth=10, random_state=seed)
    single_DT.fit(X_train, y_train)

    single_DT_eval = evaluate_forest(
        X_test, y_test, [single_DT], n_instances_test, n_classes
    )

    single_DT_agreement = prediction_agreement(
        single_DT_eval["hard_predictions"], full_forest_eval["hard_predictions"]
    )

    results.append(
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "fold": int(fold_idx) if fold_idx is not None else None,
            "representation": "Single DT",
            "selection_strategy": None,
            "full_forest_size": int(n_trees),
            "subforest_size": 1,
            **shared_metric_cols(single_DT_eval),
            "silhouette_score": np.nan,
            "agreement_with_full_forest": single_DT_agreement,
            "indices": -1,  # Placeholder, as single DT has only one tree that is not part of the random forest
        }
    )

    # Random Subforest baseline
    for s in sizes:
        random_subforest_evals = []
        random_subforest_agreements = []
        # 10 random subforests averaged for each size
        for idx in range(0, 10):
            rng = np.random.RandomState(seed + idx)
            random_subforest_indices = list(
                rng.choice(n_trees, size=int(s), replace=False)
            )

            random_subforest_eval = evaluate_forest(
                X_test,
                y_test,
                [random_forest_trees[idx] for idx in random_subforest_indices],
                n_instances_test,
                n_classes,
            )
            random_subforest_evals.append(random_subforest_eval)

            random_subforest_agreement = prediction_agreement(
                random_subforest_eval["hard_predictions"],
                full_forest_eval["hard_predictions"],
            )
            random_subforest_agreements.append(random_subforest_agreement)
        random_subforest_eval = dict(random_subforest_evals[0])
        random_subforest_eval["metrics"] = {
            key: float(
                np.nanmean(
                    [
                        eval_result["metrics"].get(key, np.nan)
                        for eval_result in random_subforest_evals
                    ]
                )
            )
            for key in random_subforest_evals[0]["metrics"].keys()
        }
        random_subforest_eval["feature_importances"] = np.nanmean(
            [
                eval_result["feature_importances"]
                for eval_result in random_subforest_evals
            ],
            axis=0,
        )
        random_subforest_eval["probabilities"] = np.nanmean(
            [eval_result["probabilities"] for eval_result in random_subforest_evals],
            axis=0,
        )
        random_subforest_agreement = float(np.nanmean(random_subforest_agreements))

        results.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "fold": int(fold_idx) if fold_idx is not None else None,
                "representation": "Random",
                "selection_strategy": None,
                "full_forest_size": int(n_trees),
                "subforest_size": int(s),
                **shared_metric_cols(random_subforest_eval),
                "silhouette_score": np.nan,
                "agreement_with_full_forest": random_subforest_agreement,
                "indices": sorted([int(i) for i in random_subforest_indices]),
            }
        )

    # Top OOB ACC Subforest baseline
    if oob_indices_list is None or len(oob_indices_list) < len(random_forest_trees):
        raise ValueError("OOB indices list is not available for all trees.")

    oob_accs = []
    for idx, tree in enumerate(random_forest_trees):
        oob_idx = oob_indices_list[idx]
        if oob_idx is None or len(oob_idx) == 0:
            oob_accs.append(np.nan)
        else:
            try:
                acc = tree_metric_score(
                    tree, X_train[oob_idx], y_train[oob_idx], metric="accuracy"
                )
            except Exception:
                acc = np.nan
            oob_accs.append(acc)

    accs_arr = np.array([a if not np.isnan(a) else -np.inf for a in oob_accs])
    oob_acc_ranked = list(np.argsort(-accs_arr))
    for s in sizes:
        top_oob_acc_indices = [int(i) for i in oob_acc_ranked[: int(s)]]

        top_oob_acc_eval = evaluate_forest(
            X_test,
            y_test,
            [random_forest_trees[idx] for idx in top_oob_acc_indices],
            n_instances_test,
            n_classes,
        )

        top_oob_acc_agreement = prediction_agreement(
            top_oob_acc_eval["hard_predictions"], full_forest_eval["hard_predictions"]
        )

        results.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "fold": int(fold_idx) if fold_idx is not None else None,
                "representation": "Top OOB ACC",
                "selection_strategy": None,
                "full_forest_size": int(n_trees),
                "subforest_size": int(s),
                **shared_metric_cols(top_oob_acc_eval),
                "silhouette_score": np.nan,
                "agreement_with_full_forest": top_oob_acc_agreement,
                "indices": sorted(top_oob_acc_indices),
            }
        )

    # Top OOB MCC Subforest baseline
    if oob_indices_list is None or len(oob_indices_list) < len(random_forest_trees):
        raise ValueError("OOB indices list is not available for all trees.")

    oob_mccs = []
    for idx, tree in enumerate(random_forest_trees):
        oob_idx = oob_indices_list[idx]
        if oob_idx is None or len(oob_idx) == 0:
            oob_mccs.append(np.nan)
        else:
            try:
                mcc = tree_metric_score(
                    tree, X_train[oob_idx], y_train[oob_idx], metric="mcc"
                )
            except Exception:
                mcc = np.nan
            oob_mccs.append(mcc)

    mccs_arr = np.array([m if not np.isnan(m) else -np.inf for m in oob_mccs])
    oob_mcc_ranked = list(np.argsort(-mccs_arr))
    for s in sizes:
        top_oob_mcc_indices = [int(i) for i in oob_mcc_ranked[: int(s)]]

        top_oob_mcc_eval = evaluate_forest(
            X_test,
            y_test,
            [random_forest_trees[idx] for idx in top_oob_mcc_indices],
            n_instances_test,
            n_classes,
        )

        top_oob_mcc_agreement = prediction_agreement(
            top_oob_mcc_eval["hard_predictions"], full_forest_eval["hard_predictions"]
        )

        results.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "fold": int(fold_idx) if fold_idx is not None else None,
                "representation": "Top OOB MCC",
                "selection_strategy": None,
                "full_forest_size": int(n_trees),
                "subforest_size": int(s),
                **shared_metric_cols(top_oob_mcc_eval),
                "silhouette_score": np.nan,
                "agreement_with_full_forest": top_oob_mcc_agreement,
                "indices": sorted(top_oob_mcc_indices),
            }
        )

    subforest_distance_matrices = []

    for name, R in representation_options.items():
        if print_progress:
            print("Starting subforest computation for representation:", name)
            start_time = time.time()

        collected_representations = [
            R.represent(tree, X_train) for tree in random_forest_trees
        ]

        num_trees = len(collected_representations)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: R.similarity(
                collected_representations[i], collected_representations[j]
            ),
            num_trees,
        )
        plot_and_save_mds(distance_matrix, oob_mccs_np, name, results_root, seed)
        subforest_distance_matrices.append(distance_matrix)

        for selection_strategy in selection_strategies:
            for s in sizes:
                subforest_indices, clustering_silhouette_score = (
                    select_subforest_via_selection_strategy(
                        distance_matrix,
                        int(s),
                        selection_strategy,
                        seed,
                        random_forest_trees,
                        X_train=X_train,
                        y_train=y_train,
                        oob_indices_list=oob_indices_list,
                    )
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes,
                )

                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"],
                )

                results.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "fold": int(fold_idx) if fold_idx is not None else None,
                        "representation": name,
                        "selection_strategy": selection_strategy,
                        "full_forest_size": int(n_trees),
                        "subforest_size": int(s),
                        **shared_metric_cols(subforest_eval),
                        "silhouette_score": clustering_silhouette_score,
                        "agreement_with_full_forest": subforest_agreement,
                        "indices": sorted([int(i) for i in subforest_indices]),
                    }
                )

        if print_progress:
            end_time = time.time()
            print(f"Finished in {end_time - start_time:.2f} seconds.")

    if run_topological_forest:
        if print_progress:
            print(
                "Starting subforest computation for representation: Topological Forest"
            )
            start_time = time.time()

        topological_forest_R = TopologicalForestRepresentation(tree_vectors=None)
        topological_forest_representations = [
            topological_forest_R.represent(t, X_train) for t in random_forest_trees
        ]
        topological_forest_R = TopologicalForestRepresentation(
            tree_vectors=topological_forest_representations
        )

        num_trees = len(topological_forest_representations)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: topological_forest_R.similarity(i, j), num_trees
        )
        plot_and_save_mds(
            distance_matrix, oob_mccs_np, "Topological Forest", results_root, seed
        )

        for selection_strategy in selection_strategies:
            for s in sizes:
                subforest_indices, clustering_silhouette_score = (
                    select_subforest_via_selection_strategy(
                        distance_matrix,
                        int(s),
                        selection_strategy,
                        seed,
                        random_forest_trees,
                        X_train=X_train,
                        y_train=y_train,
                        oob_indices_list=oob_indices_list,
                    )
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes,
                )

                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"],
                )

                results.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "fold": int(fold_idx) if fold_idx is not None else None,
                        "representation": "Topological Forest",
                        "selection_strategy": selection_strategy,
                        "full_forest_size": int(n_trees),
                        "subforest_size": int(s),
                        **shared_metric_cols(subforest_eval),
                        "silhouette_score": clustering_silhouette_score,
                        "agreement_with_full_forest": subforest_agreement,
                        "indices": sorted([int(i) for i in subforest_indices]),
                    }
                )

        if print_progress:
            end_time = time.time()
            print(
                f"Topological Forest finished in {end_time - start_time:.2f} seconds."
            )

    if run_indtree:
        if print_progress:
            print("Starting subforest computation for representation: INDTree")
            start_time = time.time()

        ind_R = INDTreeRepresentation(
            random_forest_trees, X_train, y_train, "direct", "output", seed
        )

        num_trees = len(random_forest_trees)
        distance_matrix = similarity_to_distance_matrix(
            lambda i, j: ind_R.similarity(i, j), num_trees
        )
        plot_and_save_mds(distance_matrix, oob_mccs_np, "INDTree", results_root, seed)

        for selection_strategy in selection_strategies:
            for s in sizes:
                subforest_indices, clustering_silhouette_score = (
                    select_subforest_via_selection_strategy(
                        distance_matrix,
                        int(s),
                        selection_strategy,
                        seed,
                        random_forest_trees,
                        X_train=X_train,
                        y_train=y_train,
                        oob_indices_list=oob_indices_list,
                    )
                )

                subforest_eval = evaluate_forest(
                    X_test,
                    y_test,
                    [random_forest_trees[idx] for idx in subforest_indices],
                    n_instances_test,
                    n_classes,
                )

                subforest_agreement = prediction_agreement(
                    subforest_eval["hard_predictions"],
                    full_forest_eval["hard_predictions"],
                )

                results.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "fold": int(fold_idx) if fold_idx is not None else None,
                        "representation": "INDTree",
                        "selection_strategy": selection_strategy,
                        "full_forest_size": int(n_trees),
                        "subforest_size": int(s),
                        **shared_metric_cols(subforest_eval),
                        "silhouette_score": clustering_silhouette_score,
                        "agreement_with_full_forest": subforest_agreement,
                        "indices": sorted([int(i) for i in subforest_indices]),
                    }
                )

        if print_progress:
            end_time = time.time()
            print(f"INDTree finished in {end_time - start_time:.2f} seconds.")

    results = pd.DataFrame(results)
    return results
