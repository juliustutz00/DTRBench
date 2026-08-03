"""
Typed configuration objects for benchmark execution and analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Typed configuration object for a single benchmark run."""

    seed: int = 42
    print_progress: bool = True
    save_results: bool = True
    perturbation_benchmark: bool = True
    subforest_benchmark: bool = True
    resource_benchmark: bool = True
    existing_perturbation_results_path: str | None = None
    existing_subforest_results_path: str | None = None
    existing_resource_represent_results_path: str | None = None
    existing_resource_similarity_results_path: str | None = None
    dataset: str = "iris"
    n_splits: int = 3
    n_samples: int | float = math.inf
    representations: list[str] = field(default_factory=list)
    perturbations: list[str] = field(default_factory=list)
    run_topological_forest: bool = True
    run_indtree: bool = True
    intensities: list[float] = field(default_factory=list)
    perturbation_runs: int = 1
    random_forest_size: int = 100
    subforest_size: list[int] = field(default_factory=list)
    selection_strategies: list[str] = field(default_factory=list)
    resource_benchmark_sizes: list[int] = field(default_factory=list)
    run_id: str | None = None
    fold_idx: int | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Typed configuration object for a benchmark execution."""

    defaults: BenchmarkRunConfig
    runs: list[BenchmarkRunConfig] = field(default_factory=list)
    source_path: str | None = None

