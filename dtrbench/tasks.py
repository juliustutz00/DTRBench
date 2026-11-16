"""
Registry and runner for custom downstream tasks in DTRBench.

A task receives a `TaskContext` for one (dataset, fold): the data split, the trained random forest (trees,
bootstrap and out-of-bag indices) and the configured representations, and returns a pandas DataFrame of
results. The runner handles dataset loading, cross-validation folds, forest training, representation
construction, seeding and saving, exactly as the built-in benchmarks do, so a task only implements its own
evaluation logic (e.g. tree anomaly detection, drift detection, retrieval).

This module is self-contained: it reuses the existing registries and utilities and does not modify them.
Custom tasks can live anywhere; they are registered as soon as the defining module is imported
(`user_extensions.tasks` is imported automatically if it exists).

Example:
    from dtrbench.tasks import register_task, run_task

    @register_task("my_task")
    def my_task(ctx, **kwargs):
        D = ctx.distance_matrix("Tree Descriptor")
        ...
        return pd.DataFrame(rows)

    run_task("my_task", dataset="wine", representations=["Tree Descriptor", "Leaf Profile"])
"""

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import dtrbench.datasets  # noqa: F401  (registers datasets)
import dtrbench.perturbations  # noqa: F401  (registers perturbations)
import dtrbench.representations  # noqa: F401  (registers representations)
from dtrbench.datasets.registry import get_dataset
from dtrbench.representations.registry import get_representation
from dtrbench.utils.similarity import similarity_to_distance_matrix
from dtrbench.utils.training import train_own_random_forest

_TASKS = {}


def register_task(name):
    """Register a task function `fn(ctx: TaskContext, **kwargs) -> pd.DataFrame` under `name`."""

    def decorator(fn):
        if name in _TASKS:
            raise ValueError(f"Task '{name}' already registered")
        _TASKS[name] = fn
        return fn

    return decorator


def get_task(name):
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available_tasks()}")
    return _TASKS[name]


def available_tasks():
    return sorted(_TASKS.keys())


@dataclass
class TaskContext:
    """Everything a task needs for one (dataset, fold)."""

    dataset_name: str
    fold: int
    seed: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    features_info: pd.DataFrame
    trees: list
    bootstrap_indices: list
    oob_indices: list
    representations: dict  # name -> representation instance
    _cache: dict = field(default_factory=dict, repr=False)

    def distance_matrix(self, representation, trees=None, nan_similarity=None):
        """Pairwise distance matrix of `trees` (default: the forest) under a representation, computed as in
        the subforest benchmark (represent on X_train, similarity, min-max rescaled, 1 - similarity).
        Cached for the forest; recomputed when a custom list of trees is passed.

        An undefined (NaN) similarity would turn the whole rescaled matrix into NaN, so it raises an error
        unless `nan_similarity` gives a replacement value. Example: Prediction Profile's Cohen's kappa is
        0/0 for two identical constant prediction vectors (e.g. a perturbed tree predicting a single class),
        where nan_similarity=1.0 is the correct value."""
        use_cache = trees is None and nan_similarity is None
        if use_cache and representation in self._cache:
            return self._cache[representation]
        trees = self.trees if trees is None else trees
        R = self.representations[representation]
        reps = [R.represent(t, self.X_train) for t in trees]

        def sim(i, j):
            s = R.similarity(reps[i], reps[j])
            if s is None or np.isnan(s):
                if nan_similarity is None:
                    raise ValueError(f"{representation}: undefined similarity between trees {i} and {j}; "
                                     f"pass nan_similarity=<value> to replace it")
                return nan_similarity
            return s

        D = similarity_to_distance_matrix(sim, len(reps))
        if use_cache:
            self._cache[representation] = D
        return D


def run_task(
    name,
    dataset,
    representations,
    n_splits=5,
    n_trees=100,
    n_samples=100000,
    seed=42,
    task_kwargs=None,
    results_dir=None,
):
    """Run a registered task on every fold of a dataset (same folds, seeds and forest training as the
    built-in benchmarks) and return the concatenated results; saved as CSV if `results_dir` is given."""
    task = get_task(name)
    data = get_dataset(dataset)(n_splits=n_splits, n_samples=n_samples, seed=seed)
    out = []
    for fold, (X_train, X_test, y_train, y_test) in enumerate(data["folds"]):
        fold_seed = seed + fold * n_splits
        X_train, X_test = X_train.astype("float32"), X_test.astype("float32")
        trees, boots, oobs = train_own_random_forest(X_train, y_train, n_trees, fold_seed)
        reps = {r: get_representation(r)(X_train=X_train, seed=seed) for r in representations}
        ctx = TaskContext(data["name"], fold, fold_seed, X_train, X_test, y_train, y_test,
                          data["features"], trees, boots, oobs, reps)
        res = task(ctx, **(task_kwargs or {}))
        out.append(res.assign(dataset=data["name"], fold=fold, seed=fold_seed))
    results = pd.concat(out, ignore_index=True)
    if results_dir is not None:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        results.to_csv(Path(results_dir) / f"task_{name}_{data['name']}.csv", index=False)
    return results


try:  # user-defined tasks, mirroring the other user_extensions packages
    importlib.import_module("user_extensions.tasks")
except ModuleNotFoundError as e:  # absent is fine; errors inside an existing module are not swallowed
    if e.name not in ("user_extensions", "user_extensions.tasks"):
        raise
