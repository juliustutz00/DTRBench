"""
Perturbation benchmark module for evaluating the robustness of decision tree representations under various perturbations.

This module provides functions for running perturbation benchmarks and measuring the sensitivity of different perturbations on model performance and feature importance.
"""

import pandas as pd

from dtrbench.perturbations.perturbation_ops import remove_nodes
from dtrbench.representations.feature_graph_representation import (
    FeatureGraphRepresentation,
)
from dtrbench.representations.indtree_representation import INDTreeRepresentation
from dtrbench.representations.topological_forest_representation import (
    TopologicalForestRepresentation,
)  # type: ignore
from dtrbench.representations.tree_descriptor_representation import (
    TreeDescriptorRepresentation,
)
from dtrbench.utils.metrics import (
    compute_feature_importance_difference,
    tree_metric_score,
)
from dtrbench.utils.similarity import compute_similarity_to_base_tree


def run_perturbation_benchmark(
    X_train,
    y_train,
    features_info,
    representation_options,
    perturbations,
    random_forest_trees,
    bootstrap_indices_list,
    oob_indices_list,
    intensities=[0.25, 0.5, 0.75, 1],
    perturbation_runs=1,
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    dataset_name=None,
    fold_idx=None,
    seed=0,
):
    """
    Runs the perturbation benchmark on a given dataset and returns the results as a DataFrame.

    Args:
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        representation_options (list[dict]): Representation options to evaluate.
        perturbations (list[callable]): Perturbation functions to apply.
        random_forest_trees (list[DecisionTreeClassifier]): Decision trees from a random forest.
        bootstrap_indices_list (list[np.ndarray]): List of bootstrap indices for each tree.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree.
        intensities (list[float]): List of perturbation intensities to evaluate.
        perturbation_runs (int): Number of runs for each perturbation and intensity.
        run_topological_forest (bool): Whether to run the topological forest representation.
        run_indtree (bool): Whether to run the INDTree representation.
        print_progress (bool): Whether to print progress messages.
        dataset_name (str): Name of the dataset (for reporting purposes).
        fold_idx (int): Index of the fold (for reporting purposes).
        seed (int): Random seed for reproducibility.

    Returns:
        A DataFrame containing the results of the perturbation benchmark, including performance metrics and similarity scores.
    """
    if print_progress:
        print("Running perturbation benchmark...")

    results = []
    topological_forest_representations = []
    topological_forest_R = None

    indtree_all_trees = []
    for idx, template_tree in enumerate(random_forest_trees):
        X_boot, y_boot = (
            X_train[bootstrap_indices_list[idx]],
            y_train[bootstrap_indices_list[idx]],
        )
        X_oob, y_oob = X_train[oob_indices_list[idx]], y_train[oob_indices_list[idx]]
        base_tree = remove_nodes(
            template_tree,
            None,
            X_boot,
            y_boot,
            features_info,
            intensity=0.5,
            seed=seed,
        )
        for name, R in representation_options.items():
            if name == "Feature Graph":
                original_feature_graph = R
                new_feature_graph = FeatureGraphRepresentation(
                    criterion=original_feature_graph.criterion, X=X_boot
                )
                representation_options[name] = new_feature_graph
            elif name == "Tree Descriptor":
                original_feature_graph = R
                new_tree_descriptor = TreeDescriptorRepresentation(
                    weights=R.weights, metric=R.metric, X=X_boot
                )
                representation_options[name] = new_tree_descriptor
        if run_topological_forest:
            topological_forest_R = TopologicalForestRepresentation(tree_vectors=None)
            representation_base_tree = topological_forest_R.represent(base_tree, X_boot)
            topological_forest_representations.append(representation_base_tree)
        if run_indtree:
            indtree_all_trees.append(base_tree)

        performance_base_tree = tree_metric_score(base_tree, X_oob, y_oob, metric="mcc")
        representations_base_tree = {
            name: R.represent(base_tree, X_boot)
            for name, R in representation_options.items()
        }

        for p_name, p_fn in perturbations.items():
            for i in intensities:
                for p_run in range(perturbation_runs):
                    # deepcopy is made in the perturbation function
                    perturbed_tree = p_fn(
                        base_tree,
                        template_tree,
                        X_boot,
                        y_boot,
                        features_info,
                        intensity=i,
                        seed=seed + idx + p_run,
                    )
                    if run_topological_forest:
                        representation_perturbed_tree = topological_forest_R.represent(
                            perturbed_tree, X_boot
                        )
                        topological_forest_representations.append(
                            representation_perturbed_tree
                        )
                    if run_indtree:
                        indtree_all_trees.append(perturbed_tree)
                    performance_perturbed_tree = tree_metric_score(
                        perturbed_tree, X_oob, y_oob, metric="mcc"
                    )
                    fi_difference = compute_feature_importance_difference(
                        base_tree, perturbed_tree, X_boot, correlation_adjustment=False
                    )
                    similarities = {}
                    for name, R in representation_options.items():
                        similarity = R.similarity(
                            representations_base_tree[name],
                            R.represent(perturbed_tree, X_boot),
                        )
                        similarities[name] = similarity
                    results.append(
                        {
                            "dataset": dataset_name,
                            "seed": int(seed + idx + p_run),
                            "fold": int(fold_idx) if fold_idx is not None else None,
                            "perturbation": p_name,
                            "intensity": i,
                            "performance_base": performance_base_tree,
                            "performance_perturbed": performance_perturbed_tree,
                            "feature_importance_difference": fi_difference,
                            **{f"sim_{k}": v for k, v in similarities.items()},
                        }
                    )

    for name, R in representation_options.items():
        if name == "Feature Graph":
            original_feature_graph = R
            new_feature_graph = FeatureGraphRepresentation(
                criterion=original_feature_graph.criterion, X=X_train
            )
            representation_options[name] = new_feature_graph
        elif name == "Tree Descriptor":
            original_feature_graph = R
            new_tree_descriptor = TreeDescriptorRepresentation(
                weights=R.weights, metric=R.metric, X=X_train
            )
            representation_options[name] = new_tree_descriptor
    results = pd.DataFrame(results)

    if run_topological_forest:
        topological_forest_R = TopologicalForestRepresentation(
            tree_vectors=topological_forest_representations
        )
        similarity_values = compute_similarity_to_base_tree(
            lambda i, j: topological_forest_R.similarity(i, j),
            len(topological_forest_representations),
            len(perturbations),
            len(intensities),
            perturbation_runs,
        )
        results["sim_Topological Forest"] = [
            sim_value["similarity_to_base"] for sim_value in similarity_values
        ]

    if run_indtree:
        # direct or repr3rows, encoding or output
        indtree_R = INDTreeRepresentation(
            indtree_all_trees, X_train, y_train, "direct", "output", seed
        )
        similarity_values = compute_similarity_to_base_tree(
            lambda i, j: indtree_R.similarity(i, j),
            len(indtree_all_trees),
            len(perturbations),
            len(intensities),
            perturbation_runs,
        )
        results["sim_INDTree"] = [
            sim_value["similarity_to_base"] for sim_value in similarity_values
        ]

    return results
