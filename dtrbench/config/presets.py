"""
Small helper presets for common benchmark configurations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from dtrbench.config.defaults import DEFAULT_BENCHMARK_RUN


def make_benchmark_run_preset(**overrides: Any) -> dict[str, Any]:
    """Create a benchmark run preset with the given overrides.

    Args:
        **overrides: Keyword arguments to override the default values in the preset.

    Returns:
        A dictionary representing the benchmark run preset with the specified overrides.
    """
    preset = deepcopy(DEFAULT_BENCHMARK_RUN)
    for key, value in overrides.items():
        preset[key] = value
    return preset


def perturbation_only_preset(dataset: str, **overrides: Any) -> dict[str, Any]:
    """Create a benchmark run preset for perturbation benchmarks only.

    Args:
        dataset (str): The dataset to use for the benchmark.
        **overrides: Keyword arguments to override the default values in the preset.

    Returns:
        A dictionary representing the benchmark run preset with the specified overrides.
    """
    return make_benchmark_run_preset(
        dataset=dataset,
        perturbation_benchmark=True,
        subforest_benchmark=False,
        resource_benchmark=False,
        random_forest_size=50,
        perturbation_runs=1,
        intensities=[0.2, 0.4, 0.6, 0.8, 1],
        n_splits=5,
        **overrides,
    )


def subforest_only_preset(dataset: str, **overrides: Any) -> dict[str, Any]:
    """Create a benchmark run preset for subforest benchmarks only.

    Args:
        dataset (str): The dataset to use for the benchmark.
        **overrides: Keyword arguments to override the default values in the preset.

    Returns:
        A dictionary representing the benchmark run preset with the specified overrides.
    """
    return make_benchmark_run_preset(
        dataset=dataset,
        perturbation_benchmark=False,
        subforest_benchmark=True,
        resource_benchmark=False,
        random_forest_size=1000,
        subforest_size=[
            2,
            5,
            10,
            15,
            20,
            25,
            30,
            50,
            100,
            250,
            500,
            750,
            900,
        ],
        selection_strategies=[
            "k-medoid",
            "k-medoid-performance",
            "agglomerative",
            "agglomerative-performance",
            "density",
            "combination-greedy",
            "combination-simulated_annealing",
            "combination-genetic",
        ],
        n_splits=5,
        **overrides,
    )


def resource_only_preset(dataset: str, **overrides: Any) -> dict[str, Any]:
    """Create a benchmark run preset for resource benchmarks only.

    Args:
        dataset (str): The dataset to use for the benchmark.
        **overrides: Keyword arguments to override the default values in the preset.

    Returns:
        A dictionary representing the benchmark run preset with the specified overrides.
    """
    return make_benchmark_run_preset(
        dataset=dataset,
        perturbation_benchmark=False,
        subforest_benchmark=False,
        resource_benchmark=True,
        resource_benchmark_sizes=[
            5,
            10,
            15,
            20,
            25,
            30,
            50,
            100,
            250,
            500,
            750,
            1000,
        ],
        n_splits=5,
        **overrides,
    )
