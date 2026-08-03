"""
Resource benchmark module for evaluating the computational resources consumed by decision tree representations.

This module provides functions for running resource benchmarks and analyzing the memory and time usage of different representations.
"""

import gc
import os
import pickle
import random
import time
import tracemalloc
from collections import defaultdict
from itertools import combinations
from math import comb

import pandas as pd
import torch

from dtrbench.representations.indtree_representation import INDTreeRepresentation
from dtrbench.representations.topological_forest_representation import (
    TopologicalForestRepresentation,
)
from dtrbench.utils.training import train_own_random_forest


def run_resource_benchmark(
    X_train,
    y_train,
    representation_options,
    run_topological_forest,
    run_indtree,
    random_forest_sizes,
    dataset_name,
    fold,
    seed,
):
    """
    Runs the resource benchmark on a given dataset and returns the results as a DataFrame. 

    Args:
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        representation_options (list[dict]): Representation options to evaluate.
        run_topological_forest (bool): Whether to run the topological forest representation.
        run_indtree (bool): Whether to run the INDTree representation.
        random_forest_sizes (list[int]): List of random forest sizes to evaluate.
        dataset_name (str): Name of the dataset (for reporting purposes).
        fold_idx (int): Index of the fold (for reporting purposes).
        seed (int): Random seed for reproducibility.

    Returns:
        A DataFrame containing the results of the resource benchmark, including runtime and memory usage metrics.
    """
    print("Running resource benchmark...")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    runtimes_represent = []
    runtimes_similarity = []
    (initial_random_forest_trees, initial_bootstrap_indices_list, _) = (
        train_own_random_forest(
            X_train, y_train, max(random_forest_sizes) + 2, seed=seed
        )
    )
    for rf_size in random_forest_sizes:
        subforest = random.sample(initial_random_forest_trees, rf_size + 2)
        # warm up
        collected_warmup_representations = defaultdict(list)
        for idx in range(2):
            X_boot = X_train[initial_bootstrap_indices_list[idx]]
            for name, R in representation_options.items():
                warmup_tree_to_represent = subforest[idx]
                warmup_rep_tmp = R.represent(warmup_tree_to_represent, X_boot)
                collected_warmup_representations[name].append(warmup_rep_tmp)
        for name, reps in collected_warmup_representations.items():
            representation_options[name].similarity(reps[0], reps[1])
        print("Warmup done for random forest size:", rf_size)
        # warm up end
        
        collected_representations = defaultdict(list)
        topological_forest_R = TopologicalForestRepresentation(tree_vectors=None)
        topological_forest_runtime_init = None
        topological_forest_peak_ram_bytes = None
        topological_forest_peak_vram_bytes = None
        indtree_trees = []
        indtree_R = None
        indtree_runtime_init = None
        for idx in range(2, rf_size + 2):
            X_boot = X_train[initial_bootstrap_indices_list[idx]]

            rep_options = list(representation_options.items())
            if run_topological_forest:
                rep_options.append(("Topological Forest", topological_forest_R))
            for name, R in rep_options:
                tree_to_represent = subforest[idx]

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                tracemalloc.start()
                start_time = time.perf_counter()
                rep_tmp = R.represent(tree_to_represent, X_boot)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                _, peak_ram_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_vram_bytes = (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0
                )
                rep_size_bytes = len(pickle.dumps(rep_tmp))
                runtime_represent = end_time - start_time

                collected_representations[name].append(
                    {"idx": idx, "representation": rep_tmp}
                )
                runtimes_represent.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "fold": fold,
                        "representation": name,
                        "random_forest_size": rf_size,
                        "runtime": runtime_represent,
                        "peak_ram_kb": peak_ram_bytes / 1024,
                        "peak_vram_kb": peak_vram_bytes / 1024,
                        "rep_size_kb": rep_size_bytes / 1024,
                    }
                )

            if run_indtree:
                indtree_trees.append(subforest[idx])

        if run_topological_forest:
            topological_forest_representations = [
                rep["representation"]
                for rep in collected_representations["Topological Forest"]
            ]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            tracemalloc.start()
            start_time = time.perf_counter()
            topological_forest_R = TopologicalForestRepresentation(
                tree_vectors=topological_forest_representations
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            _, topological_forest_peak_ram_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            topological_forest_peak_vram_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            topological_forest_runtime_init = end_time - start_time

        if run_indtree:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            tracemalloc.start()
            start_time = time.perf_counter()
            indtree_R = INDTreeRepresentation(
                indtree_trees, X_train, y_train, "direct", "output", seed
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            _, peak_ram_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_vram_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            rep_size_bytes = sum(
                p.numel() * p.element_size()
                for p in indtree_R.lightning_model.metamodel_.module.parameters()
            )
            indtree_runtime_init = end_time - start_time
            runtime_represent = indtree_runtime_init / rf_size

            runtimes_represent.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "fold": fold,
                    "representation": "INDTree",
                    "random_forest_size": rf_size,
                    "runtime": runtime_represent,
                    "peak_ram_kb": peak_ram_bytes / 1024,
                    "peak_vram_kb": peak_vram_bytes / 1024,
                    "rep_size_kb": rep_size_bytes / 1024,
                }
            )

        print("Done collecting representations for random forest size:", rf_size)

        for name, reps in collected_representations.items():
            if run_topological_forest and name == "Topological Forest":
                continue

            R = representation_options[name]

            for a, b in combinations(reps, 2):
                representation_a = a["representation"]
                representation_b = b["representation"]

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                tracemalloc.start()
                start_time = time.perf_counter()
                R.similarity(representation_a, representation_b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                _, peak_ram_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_vram_bytes = (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0
                )
                runtime_similarity = end_time - start_time

                runtimes_similarity.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "fold": fold,
                        "representation": name,
                        "random_forest_size": rf_size,
                        "runtime": runtime_similarity,
                        "peak_ram_kb": peak_ram_bytes / 1024,
                        "peak_vram_kb": peak_vram_bytes / 1024,
                    }
                )

        if run_topological_forest or run_indtree:
            rep_options = []
            rep_options += ["Topological Forest"] if run_topological_forest else []
            rep_options += ["INDTree"] if run_indtree else []
            for name in rep_options:
                R = topological_forest_R if name == "Topological Forest" else indtree_R
                time_addition = (
                    topological_forest_runtime_init / comb(rf_size, 2)
                    if name == "Topological Forest"
                    else 0
                )
                for idx_a, idx_b in combinations(range(rf_size), 2):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    tracemalloc.start()
                    start_time = time.perf_counter()
                    R.similarity(idx_a, idx_b)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    _, peak_ram_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_vram_bytes = (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    )
                    runtime_similarity = (end_time - start_time) + time_addition

                    if name == "Topological Forest":
                        peak_ram_bytes = (
                            topological_forest_peak_ram_bytes
                            if peak_ram_bytes < topological_forest_peak_ram_bytes
                            else peak_ram_bytes
                        )
                        peak_vram_bytes = (
                            topological_forest_peak_vram_bytes
                            if peak_vram_bytes < topological_forest_peak_vram_bytes
                            else peak_vram_bytes
                        )

                    runtimes_similarity.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "fold": fold,
                            "representation": name,
                            "random_forest_size": rf_size,
                            "runtime": runtime_similarity,
                            "peak_ram_kb": peak_ram_bytes / 1024,
                            "peak_vram_kb": peak_vram_bytes / 1024,
                        }
                    )

        print("Done calculating similarities for random forest size:", rf_size)

    df_runtimes_represent = pd.DataFrame(runtimes_represent)
    df_runtimes_similarity = pd.DataFrame(runtimes_similarity)
    return df_runtimes_represent, df_runtimes_similarity
