"""
Default configuration values for benchmark runs and analysis.
"""

import math

DEFAULT_REPRESENTATIONS = [
    "Tree Descriptor",
    "Leaf Profile",
    "Feature Graph",
]

DEFAULT_PERTURBATIONS = [
    "change_threshold",
    "change_feature",
    "swap_nodes",
    "remove_nodes",
    "add_nodes",
]

DEFAULT_SUBFOREST_SIZES = [5, 10, 15, 20, 25, 30]

DEFAULT_SELECTION_STRATEGIES = [
    "k-medoid",
    "k-medoid-performance",
    "agglomerative",
    "agglomerative-performance",
    "density",
    "combination-greedy",
    "combination-simulated_annealing",
    "combination-genetic",
]

DEFAULT_BENCHMARK_RUN = {
    "seed": 42,
    "print_progress": True,
    "save_results": True,
    "perturbation_benchmark": True,
    "subforest_benchmark": True,
    "resource_benchmark": True,
    "existing_perturbation_results_path": None,
    "existing_subforest_results_path": None,
    "existing_resource_represent_results_path": None,
    "existing_resource_similarity_results_path": None,
    "dataset": "iris",
    "n_splits": 3,
    "n_samples": math.inf,
    "representations": DEFAULT_REPRESENTATIONS,
    "perturbations": DEFAULT_PERTURBATIONS,
    "run_topological_forest": True,
    "run_indtree": True,
    "intensities": [0.2, 0.4, 0.6, 0.8, 1],
    "perturbation_runs": 1,
    "random_forest_size": 100,
    "subforest_size": [5, 10, 15, 20, 25, 30],
    "selection_strategies": DEFAULT_SELECTION_STRATEGIES,
    "resource_benchmark_sizes": [5, 10, 15, 20, 25, 30],
}

DEFAULT_BENCHMARK_RUNS = [
    {
        "dataset": "connectionist",
        "perturbation_benchmark": True,
        "subforest_benchmark": True,
        "resource_benchmark": True,
        "representations": [
            "Feature Graph",
            "Tree Descriptor",
            "Leaf Profile",
        ],
        "run_topological_forest": True,
        "run_indtree": True,
        "random_forest_size": 5,
        "subforest_size": [2, 3],
        "selection_strategies": [
            "k-medoid",
            "k-medoid-performance",
            "agglomerative",
            "agglomerative-performance",
            "density",
            "combination-greedy",
            "combination-simulated_annealing",
            "combination-genetic",
        ],
        "resource_benchmark_sizes": [2, 3, 4],
        "save_results": True,
        "perturbation_runs": 1,
    }
]

DEFAULT_ANALYSIS_RUN = {
    "output_dir": "analysis_results",
    "perturbation_benchmark_results_path": None,
    "subforest_selection_results_path": None,
    "resource_benchmark_represent_results_path": None,
    "resource_benchmark_similarity_results_path": None,
    "rep_similarity_vs_performance_feature_importance": True,
    "similarity_vs_intensity_per_perturbation": True,
    "rf_compression": True,
    "mcc_boxplots": True,
    "mcc_representation_selection_strategy": True,
    "std_representation_selection_strategy": True,
    "kendalls_w_vs_config": True,
    "spearman_vs_subforest_size": True,
    "representation_vs_subforest_size": True,
    "config_vs_subforest_size": True,
    "resource_benchmark_results_path": None,
    "resource_benchmark_similarity": True,
}

DEFAULT_ANALYSIS_RUNS = [
    {
        "output_dir": "analysis_results",
        "perturbation_benchmark_results_path": "results/perturbation_benchmark_results.csv",
        "subforest_selection_results_path": "results/subforest_selection_results.csv",
        "resource_benchmark_represent_results_path": "results/resource_benchmark_represent_results.csv",
        "resource_benchmark_similarity_results_path": "results/resource_benchmark_similarity_results.csv",
    }
]
