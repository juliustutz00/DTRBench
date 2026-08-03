"""
Plotting and reporting functions for the DTRBench analysis.

This module orchestrates functions to analyze the results of the benchmark runs and generate plots and statistics.
It works with results files containing one or multiple datasets.
If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.
"""

import warnings

from dtrbench.analysis.perturbation_results import (
    plot_rep_similarity_vs_performance_feature_importance,
    plot_similarity_vs_intensity_per_perturbation,
    read_perturbation_benchmark_results,
)
from dtrbench.analysis.resource_results import (
    plot_memory_analysis,
    plot_runtime_analysis,
    read_resource_benchmark_results,
)
from dtrbench.analysis.subforest_results import (
    plot_kendalls_w_vs_config,
    plot_mcc_boxplots,
    plot_mcc_representation_selection_strategy,
    plot_rf_compression,
    plot_spearman_vs_subforest_size,
    plot_std_representation_selection_strategy,
    print_config_vs_subforest_size,
    print_representation_vs_subforest_size,
    read_subforest_selection_result,
)
from dtrbench.config.loader import load_analysis_config

warnings.filterwarnings("ignore")


REP_NAMES = [
    "Tree Descriptor",
    "Leaf Profile",
    "Feature Graph",
    "Topological Forest",
    "INDTree",
]
SEL_STRATEGIES = [
    "k-medoid",
    "k-medoid-performance",
    "agglomerative",
    "agglomerative-performance",
    "density",
    "combination-greedy",
    "combination-genetic",
    "combination-simulated_annealing",
]
PERTURBATIONS = [
    "change_threshold",
    "change_feature",
    "swap_nodes",
    "remove_nodes",
    "add_nodes",
]
SUBFOREST_SIZES = [2, 3, 5, 10, 15, 20, 25, 30]


def report(config_path):
    """
    Analyze the results of the benchmark runs and generate plots and statistics.

    Works with results files containing one or multiple datasets. If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.

    Args:
        config_path (str): Path to the report configuration file.
    """

    config = load_analysis_config(config_path)

    output_dir = config["output_dir"]
    perturbation_benchmark_results_path = config.get("perturbation_benchmark_results_path")
    subforest_selection_results_path = config.get("subforest_selection_results_path")
    resource_benchmark_represent_results_path = config.get("resource_benchmark_represent_results_path")
    resource_benchmark_similarity_results_path = config.get("resource_benchmark_similarity_results_path")
    rep_similarity_vs_performance_feature_importance = config.get("rep_similarity_vs_performance_feature_importance", True)
    similarity_vs_intensity_per_perturbation = config.get("similarity_vs_intensity_per_perturbation", True)
    rf_compression = config.get("rf_compression", True)
    mcc_boxplots = config.get("mcc_boxplots", True)
    mcc_representation_selection_strategy = config.get("mcc_representation_selection_strategy", True)
    std_representation_selection_strategy = config.get("std_representation_selection_strategy", True)
    kendalls_w_vs_config = config.get("kendalls_w_vs_config", True)
    spearman_vs_subforest_size = config.get("spearman_vs_subforest_size", True)
    representation_vs_subforest_size = config.get("representation_vs_subforest_size", True)
    config_vs_subforest_size = config.get("config_vs_subforest_size", True)
    resource_benchmark_represent = config.get("resource_benchmark_represent", True)
    resource_benchmark_similarity = config.get("resource_benchmark_similarity", True)

    if perturbation_benchmark_results_path is not None:
        perturbation_data = read_perturbation_benchmark_results(
            perturbation_benchmark_results_path, REP_NAMES
        )
        if rep_similarity_vs_performance_feature_importance:
            plot_rep_similarity_vs_performance_feature_importance(
                perturbation_data, output_dir, REP_NAMES
            )
        if similarity_vs_intensity_per_perturbation:
            plot_similarity_vs_intensity_per_perturbation(
                perturbation_data, output_dir, REP_NAMES, PERTURBATIONS
            )

    if subforest_selection_results_path is not None:
        subforest_data = read_subforest_selection_result(
            subforest_selection_results_path, REP_NAMES, SUBFOREST_SIZES
        )
        if rf_compression:
            plot_rf_compression(subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES)
        if mcc_boxplots:
            plot_mcc_boxplots(subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES)
        if mcc_representation_selection_strategy:
            plot_mcc_representation_selection_strategy(
                subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES
            )
        if std_representation_selection_strategy:
            plot_std_representation_selection_strategy(
                subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES
            )
        if kendalls_w_vs_config:
            plot_kendalls_w_vs_config(
                subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES
            )
        if spearman_vs_subforest_size:
            plot_spearman_vs_subforest_size(
                subforest_data, output_dir, REP_NAMES, SEL_STRATEGIES
            )
        if representation_vs_subforest_size:
            print_representation_vs_subforest_size(
                subforest_data, output_dir, REP_NAMES
            )
        if config_vs_subforest_size:
            print_config_vs_subforest_size(subforest_data, output_dir)

    if (
        resource_benchmark_represent_results_path is not None
        or resource_benchmark_similarity_results_path is not None
    ):
        resource_benchmark_represent_data = read_resource_benchmark_results(
            resource_benchmark_represent_results_path
        )
        resource_benchmark_similarity_data = read_resource_benchmark_results(
            resource_benchmark_similarity_results_path
        )

        if resource_benchmark_represent_results_path is not None and resource_benchmark_represent:
            (
                plot_runtime_analysis(
                    resource_benchmark_represent_data, output_dir, REP_NAMES
                )
                if resource_benchmark_represent_data is not None
                else None
            )
            (
                plot_runtime_analysis(
                    resource_benchmark_similarity_data, output_dir, REP_NAMES
                )
                if resource_benchmark_similarity_data is not None
                else None
            )
        if resource_benchmark_similarity_results_path is not None and resource_benchmark_similarity:
            (
                plot_memory_analysis(
                    resource_benchmark_represent_data, output_dir, REP_NAMES
                )
                if resource_benchmark_represent_data is not None
                else None
            )
            (
                plot_memory_analysis(
                    resource_benchmark_similarity_data, output_dir, REP_NAMES
                )
                if resource_benchmark_similarity_data is not None
                else None
            )
