"""
Load and normalize benchmark configuration files.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from dtrbench.config.defaults import (
    DEFAULT_ANALYSIS_RUN,
    DEFAULT_ANALYSIS_RUNS,
    DEFAULT_BENCHMARK_RUN,
    DEFAULT_BENCHMARK_RUNS,
)
from dtrbench.config.schema import (
    BenchmarkConfig,
    BenchmarkRunConfig,
)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively merge two dictionaries, with values from the override dictionary taking precedence.

    Args:
        base (dict[str, Any]): The base dictionary to merge into.
        override (dict[str, Any] | None): The dictionary whose values will override those in the base.

    Returns:
        dict[str, Any]: A new dictionary containing the merged values from both input dictionaries."""
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_benchmark_run_config(payload: dict[str, Any]) -> BenchmarkRunConfig:
    known_fields = BenchmarkRunConfig.__dataclass_fields__.keys()
    known = {k: payload[k] for k in payload if k in known_fields}
    extra = {k: payload[k] for k in payload if k not in known_fields}
    return BenchmarkRunConfig(
        **known,
        extra=extra,
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and normalize a benchmark configuration file from the given path.

    Args:
        path (str | Path): The path to the benchmark configuration file.

    Returns:
        BenchmarkConfig: A normalized BenchmarkConfig object containing the defaults and runs."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults_payload = _normalize_representations(
        deep_merge(DEFAULT_BENCHMARK_RUN, raw.get("defaults", {}) or {})
    )
    defaults = _build_benchmark_run_config(defaults_payload)

    run_payloads = raw.get("runs") or DEFAULT_BENCHMARK_RUNS
    runs = [
        _build_benchmark_run_config(
            _normalize_representations(deep_merge(defaults_payload, run_payload or {}))
        )
        for run_payload in run_payloads
    ]

    return BenchmarkConfig(defaults=defaults, runs=runs, source_path=str(config_path))


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize an analysis configuration file from the given path.

    Args:
        path (str | Path): The path to the analysis configuration file.

    Returns:
        dict[str, Any]: A normalized dictionary containing the given parameters."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    defaults = deep_merge(
        DEFAULT_ANALYSIS_RUN,
        raw.get("defaults", {}) or {},
    )
    runs = raw.get("run") or [defaults]
    run_config = deep_merge(
        defaults,
        runs[0],
    )
    run_config = _normalize_representations(run_config)
    if run_config.get("output_dir") is None:
        raise ValueError("output_dir must be specified in the analysis configuration.")
    run_config["source_path"] = str(config_path)
    return run_config


def run_benchmark_config_to_dict(run_config: BenchmarkRunConfig) -> dict[str, Any]:
    """Convert a BenchmarkRunConfig dataclass instance to a dictionary, including extra fields.

    Args:
        run_config (BenchmarkRunConfig): The BenchmarkRunConfig instance to convert.

    Returns:
        dict[str, Any]: A dictionary representation of the BenchmarkRunConfig instance.
    """
    return {
        "seed": run_config.seed,
        "print_progress": run_config.print_progress,
        "save_results": True,
        "perturbation_benchmark": run_config.perturbation_benchmark,
        "subforest_benchmark": run_config.subforest_benchmark,
        "resource_benchmark": run_config.resource_benchmark,
        "existing_perturbation_results_path": run_config.existing_perturbation_results_path,
        "existing_subforest_results_path": run_config.existing_subforest_results_path,
        "existing_resource_represent_results_path": run_config.existing_resource_represent_results_path,
        "existing_resource_similarity_results_path": run_config.existing_resource_similarity_results_path,
        "dataset": run_config.dataset,
        "n_splits": run_config.n_splits,
        "n_samples": run_config.n_samples,
        "representations": deepcopy(run_config.representations),
        "perturbations": deepcopy(run_config.perturbations),
        "run_topological_forest": run_config.run_topological_forest,
        "run_indtree": run_config.run_indtree,
        "intensities": list(run_config.intensities),
        "perturbation_runs": 1,
        "random_forest_size": run_config.random_forest_size,
        "subforest_size": list(run_config.subforest_size),
        "selection_strategies": run_config.selection_strategies,
        "resource_benchmark_sizes": list(run_config.resource_benchmark_sizes),
        "run_id": run_config.run_id,
        "fold_idx": run_config.fold_idx,
        "analysis": deepcopy(run_config.analysis),
        "extra": deepcopy(run_config.extra),
    }


def _normalize_representations(payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(payload)
    representations = payload.get("representations", []) or []

    if "Topological Forest" in representations:
        payload["run_topological_forest"] = True
        representations.remove("Topological Forest")
    else:
        payload["run_topological_forest"] = False

    if "INDTree" in representations:
        payload["run_indtree"] = True
        representations.remove("INDTree")
    else:
        payload["run_indtree"] = False

    payload["representations"] = representations

    if payload.get("n_samples") is None:
        payload["n_samples"] = math.inf

    return payload
