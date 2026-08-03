"""
Runner module for executing benchmark tasks.

This module provides the main entry point for running benchmark configurations and managing the execution flow. It triggers the orchestration of benchmarks based on the provided configuration, handling dataset loading, perturbation application, representation selection, and result saving.
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import dtrbench.plugins  # noqa: F401
from dtrbench.config.loader import (
    load_benchmark_config,
    run_benchmark_config_to_dict,
)
from dtrbench.config.presets import (
    perturbation_only_preset,
    resource_only_preset,
    subforest_only_preset,
)
from dtrbench.datasets.registry import get_dataset
from dtrbench.perturbations.registry import get_perturbation
from dtrbench.pipeline.orchestrator import run_benchmark
from dtrbench.representations.registry import get_representation

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def run(
    config_path: str | None = None,
    dataset: str | None = None,
    mode: str | None = None,
):
    """Run the benchmark tasks based on the provided configuration.

    Args:
        config_path (str): Path to the benchmark configuration YAML file.
        dataset (str): Name of the dataset to use.
        mode (str): Which benchmark preset to run.
    """

    if dataset and mode:
        runs = [_build_mode_preset(mode=mode, dataset=dataset)]
    else:
        config = load_benchmark_config(config_path)
        runs = [run_benchmark_config_to_dict(run) for run in config.runs]

    for run_idx, cfg in enumerate(runs, start=1):
        dataset_name = cfg["dataset"]
        loader = get_dataset(dataset_name)
        dataset_dict = loader(
            n_splits=cfg.get("n_splits", 3),
            n_samples=cfg.get("n_samples", 100000),
            seed=cfg.get("seed", 0),
        )
        resolved_name = dataset_dict["name"]
        folds = dataset_dict["folds"]
        features_info = dataset_dict["features"]
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"_run{run_idx}_{dataset_name}"
        )

        print(
            f"--- Starting Task: {dataset_name} (PertBench: {cfg.get('perturbation_benchmark')}, SubfBench: {cfg.get('subforest_benchmark')}, ResoBench: {cfg.get('resource_benchmark')}) ---"
        )

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
            fold_seed = cfg.get("seed", 0) + fold_idx * cfg.get("n_splits", 10)
            X_train = X_train.astype("float32")
            X_test = X_test.astype("float32")

            representation_options = _build_representation_options(cfg, X_train)
            perturbations = _build_perturbations(cfg)

            (
                df_perturbations,
                df_subforest,
                df_resource_represent,
                df_resource_similarity,
            ) = run_benchmark(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                features_info=features_info,
                representation_options=representation_options,
                perturbations=perturbations,
                intensities=cfg.get("intensities", [0.2, 0.4, 0.6, 0.8, 1]),
                perturbation_runs=1,
                n_trees=cfg.get("random_forest_size", 100),
                subforest_size=cfg.get("subforest_size", [5, 10, 15, 20]),
                selection_strategies=cfg.get("selection_strategies", ["k-medoid"]),
                resource_benchmark_sizes=cfg.get("resource_benchmark_sizes", [5, 10]),
                run_topological_forest=cfg.get("run_topological_forest", False),
                run_indtree=cfg.get("run_indtree", False),
                print_progress=cfg.get("print_progress", False),
                save_results=True,
                dataset_name=f"{resolved_name}",
                results_root=str(RESULTS_ROOT),
                perturbation_benchmark=cfg.get("perturbation_benchmark", True),
                subforest_benchmark=cfg.get("subforest_benchmark", True),
                resource_benchmark=cfg.get("resource_benchmark", True),
                run_id=run_id,
                fold_idx=fold_idx,
                existing_perturbation_results_path=cfg.get(
                    "existing_perturbation_results_path", None
                ),
                existing_subforest_results_path=cfg.get(
                    "existing_subforest_results_path", None
                ),
                existing_resource_represent_results_path=cfg.get(
                    "existing_resource_represent_results_path", None
                ),
                existing_resource_similarity_results_path=cfg.get(
                    "existing_resource_similarity_results_path", None
                ),
                seed=fold_seed,
            )

            print(
                f"Finished run {run_idx}/{len(runs)} | "
                f"dataset={dataset_name} | fold={fold_idx + 1}/{len(folds)}"
            )


def _build_representation_options(cfg, X_train):
    names = cfg.get("representations", [])
    opts = {}
    for name in names:
        builder = get_representation(name)
        opts[name] = builder(X_train=X_train, seed=cfg.get("seed", 0))
    return opts


def _build_perturbations(cfg: dict) -> dict:
    names = cfg.get("perturbations", [])
    return {name: get_perturbation(name) for name in names}
    # here I removed the () after (name)


def _build_mode_preset(mode, dataset):
    if mode == "perturbation":
        return perturbation_only_preset(dataset)
    elif mode == "subforest":
        return subforest_only_preset(dataset)
    elif mode == "resource":
        return resource_only_preset(dataset)
    else:
        raise ValueError(f"Unknown mode: {mode}")
