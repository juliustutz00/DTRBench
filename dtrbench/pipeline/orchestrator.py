"""
Pipeline orchestrator for running benchmarks in DTRBench.

This module provides the main entry point for running the benchmarks for decision tree representations in DTRBench.
"""

import os
from datetime import datetime, timezone

from dtrbench.benchmarks.io import save_benchmark
from dtrbench.benchmarks.perturbation_benchmark import run_perturbation_benchmark
from dtrbench.benchmarks.resource_benchmark import run_resource_benchmark
from dtrbench.benchmarks.subforest_benchmark import run_subforest_benchmark
from dtrbench.utils.training import train_own_random_forest


def run_benchmark(
    X_train,
    X_test,
    y_train,
    y_test,
    features_info,
    representation_options,
    perturbations,
    intensities=[0.25, 0.5, 0.75, 1],
    perturbation_runs=1,
    n_trees=100,
    subforest_size=[10],
    selection_strategies=["k-medoid"],
    resource_benchmark_sizes=[5, 10],
    run_topological_forest=False,
    run_indtree=False,
    print_progress=False,
    save_results=False,
    dataset_name=None,
    results_root=r"..\results",
    perturbation_benchmark=True,
    subforest_benchmark=True,
    resource_benchmark=True,
    run_id=None,
    fold_idx=None,
    existing_perturbation_results_path=None,
    existing_subforest_results_path=None,
    existing_resource_represent_results_path=None,
    existing_resource_similarity_results_path=None,
    seed=0,
):
    """Run the benchmark for decision tree representations.

    Args:
        X_train (np.ndarray): Train-set.
        X_test (np.ndarray): Test-set.
        y_train (np.ndarray): Train-set labels.
        y_test (np.ndarray): Test-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        representation_options (list[dict]): Representation options to evaluate.
        perturbations (list[callable]): Perturbation functions to apply.
        intensities (list[float]): List of perturbation intensities to evaluate.
        perturbation_runs (int): Number of runs for each perturbation and intensity.
        n_trees (int): Total number of trees in the random forest.
        subforest_size (list[int]): List of subforest sizes to evaluate.
        selection_strategies (list[str]): List of selection strategies to evaluate.
        resource_benchmark_sizes (list[int]): List of random forest sizes to evaluate for the resource benchmark.
        run_topological_forest (bool): Whether to run the topological forest representation.
        run_indtree (bool): Whether to run the INDTree representation.
        print_progress (bool): Whether to print progress messages.
        save_results (bool): Whether to save the results to disk.
        dataset_name (str): Name of the dataset (for reporting purposes).
        results_root (str): Root directory to save the MDS results.
        perturbation_benchmark (bool): Whether to run the perturbation benchmark.
        subforest_benchmark (bool): Whether to run the subforest benchmark.
        resource_benchmark (bool): Whether to run the resource benchmark.
        fold_idx (int): Index of the fold (for reporting purposes).
        existing_perturbation_results_path (str): Path to existing perturbation benchmark results (if any).
        existing_subforest_results_path (str): Path to existing subforest benchmark results (if any).
        existing_resource_represent_results_path (str): Path to existing resource benchmark representation results (if any).
        existing_resource_similarity_results_path (str): Path to existing resource benchmark similarity results (if any).
        seed (int): Random seed for reproducibility.

    Returns:
        df_perturbation (pd.DataFrame): DataFrame containing the perturbation benchmark results.
        df_subforest (pd.DataFrame): DataFrame containing the subforest benchmark results.
        df_resource_represent (pd.DataFrame): DataFrame containing the resource benchmark representation results.
        df_resource_similarity (pd.DataFrame): DataFrame containing the resource benchmark similarity results.

    Raises:
        ValueError: If none of the benchmark flags (perturbation_benchmark, subforest_benchmark, resource_benchmark) are set to True.
    """
    if (
        not perturbation_benchmark
        and not subforest_benchmark
        and not resource_benchmark
    ):
        raise ValueError(
            "At least one of perturbation_benchmark or subforest_benchmark or resource_benchmark has to be True."
        )

    random_forest_trees, bootstrap_indices_list, oob_indices_list = (
        train_own_random_forest(X_train, y_train, n_trees, seed)
    )

    df_perturbation = None
    df_subforest = None
    df_resource_represent = None
    df_resource_similarity = None

    if perturbation_benchmark:
        df_perturbation = run_perturbation_benchmark(
            X_train,
            y_train,
            features_info,
            representation_options,
            perturbations,
            random_forest_trees,
            bootstrap_indices_list,
            oob_indices_list,
            intensities,
            perturbation_runs,
            run_topological_forest,
            run_indtree,
            print_progress,
            dataset_name,
            int(fold_idx),
            seed,
        )

    if subforest_benchmark:
        df_subforest = run_subforest_benchmark(
            X_train,
            X_test,
            y_train,
            y_test,
            representation_options,
            random_forest_trees,
            oob_indices_list,
            n_trees,
            subforest_size,
            selection_strategies,
            run_topological_forest,
            run_indtree,
            print_progress,
            results_root,
            dataset_name,
            int(fold_idx),
            seed,
        )

    if resource_benchmark:
        df_resource_represent, df_resource_similarity = run_resource_benchmark(
            X_train,
            y_train,
            representation_options,
            run_topological_forest,
            run_indtree,
            random_forest_sizes=resource_benchmark_sizes,
            dataset_name=dataset_name,
            fold=int(fold_idx) if fold_idx is not None else None,
            seed=seed,
        )

    if perturbation_benchmark and df_perturbation is not None:
        if run_topological_forest:
            representation_options["Topological Forest"] = None
        if run_indtree:
            representation_options["INDTree"] = None

    if save_results:
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        os.makedirs(results_root, exist_ok=True)
        meta = {
            "run_id": run_id,
            "fold_idx": int(fold_idx) if fold_idx is not None else None,
            "dataset_name": dataset_name,
            "n_instances_train": int(X_train.shape[0]) if X_train is not None else None,
            "n_instances_test": int(X_test.shape[0]) if X_test is not None else None,
            "n_features": int(X_train.shape[1]) if X_train is not None else None,
            "representations": list(representation_options.keys()),
            "perturbations": list(perturbations.keys()),
            "intensities": intensities,
            "perturbation_runs": perturbation_runs,
            "random_forest_size": n_trees,
            "subforest_size": subforest_size,
            "selection_strategies": selection_strategies,
            "run_topological_forest": run_topological_forest,
            "run_indtree": run_indtree,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }

        if perturbation_benchmark and df_perturbation is not None:
            save_benchmark(
                results_root,
                "perturbation_benchmark_results",
                run_id,
                "perturbation_benchmark",
                df_perturbation,
                meta,
                existing_perturbation_results_path,
            )

        if subforest_benchmark and df_subforest is not None:
            save_benchmark(
                results_root,
                "subforest_benchmark_results",
                run_id,
                "subforest_benchmark",
                df_subforest,
                meta,
                existing_subforest_results_path,
            )

        if (
            resource_benchmark
            and df_resource_represent is not None
            and df_resource_similarity is not None
        ):
            save_benchmark(
                results_root,
                "resource_benchmark_results",
                run_id,
                "resource_benchmark_represent",
                df_resource_represent,
                meta,
                existing_resource_represent_results_path,
            )
            save_benchmark(
                results_root,
                "resource_benchmark_results",
                run_id,
                "resource_benchmark_similarity",
                df_resource_similarity,
                meta,
                existing_resource_similarity_results_path,
                False,
            )

    return df_perturbation, df_subforest, df_resource_represent, df_resource_similarity
