"""
Plotting and reporting functions for the resource benchmark.

This module provides functions to analyze the results of the resource benchmark runs and generate plots and statistics.
It works with results files containing one or multiple datasets.
If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_resource_benchmark_results(results_path):
    """Read and process the resource benchmark results from a CSV file.
    
    Args:
        results_path (str): Path to the CSV file containing the resource benchmark results.

    Returns:
        pd.DataFrame: Processed resource benchmark results.
    """

    resource_df = pd.read_csv(results_path)
    resource_df["seed"] = pd.to_numeric(resource_df["seed"], errors="coerce")
    resource_df["fold"] = pd.to_numeric(resource_df["fold"], errors="coerce")
    resource_df["random_forest_size"] = pd.to_numeric(
        resource_df["random_forest_size"], errors="coerce"
    )
    resource_df["runtime"] = pd.to_numeric(resource_df["runtime"], errors="coerce")
    resource_df["peak_ram_kb"] = pd.to_numeric(
        resource_df["peak_ram_kb"], errors="coerce"
    )
    resource_df["peak_vram_kb"] = pd.to_numeric(
        resource_df["peak_vram_kb"], errors="coerce"
    )
    if "rep_size_kb" in resource_df.columns:
        resource_df["rep_size_kb"] = pd.to_numeric(
            resource_df["rep_size_kb"], errors="coerce"
        )

    value_cols = ["runtime", "peak_ram_kb", "peak_vram_kb"]
    if "rep_size_kb" in resource_df.columns:
        value_cols.append("rep_size_kb")
    resource_df = resource_df.groupby(
        ["dataset", "fold", "representation", "random_forest_size"],
        as_index=False,
    )[value_cols].mean()
    resource_df = resource_df.groupby(
        ["dataset", "representation", "random_forest_size"],
        as_index=False,
    )[value_cols].mean()
    return resource_df


def plot_runtime_analysis(data, output_dir, REP_NAMES):
    """Plot the runtime analysis results.
    
    Args:
        data (pd.DataFrame): DataFrame containing the runtime analysis results.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list): List of representation names to be used in the plots.
    """

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind")

    mode = "represent" if "rep_size_kb" in data.columns else "similarity"

    plt.figure()
    sns.lineplot(
        data=data,
        x="random_forest_size",
        y="runtime",
        hue="representation",
        hue_order=REP_NAMES,
        errorbar=None,
        marker="o",
        palette=palette,
    )

    plt.yscale("log")
    plt.xlabel("Random Forest Size")
    plt.ylabel("Average Runtime (s)")
    plt.legend(title="Representation")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/runtime_vs_random_forest_size_{mode}.png", dpi=600)

    print(f"Runtime/Memory Analysis: Runtime Analysis {mode.capitalize()} - done.")
    print()


def plot_memory_analysis(data, output_dir, REP_NAMES):
    """Plot the memory analysis results.
    
    Args:
        data (pd.DataFrame): DataFrame containing the memory analysis results.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list): List of representation names to be used in the plots.
    """
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind")

    mode = "represent" if "rep_size_kb" in data.columns else "similarity"

    def _plot_mem(data, y_col, y_label):
        plt.figure()
        sns.lineplot(
            data=data,
            x="random_forest_size",
            y=y_col,
            hue="representation",
            hue_order=REP_NAMES,
            estimator="mean",
            errorbar=None,
            marker="o",
            palette=palette,
        )

        plt.yscale("log")
        plt.xlabel("Random Forest Size")
        plt.ylabel(y_label)
        plt.legend(title="Representation")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{y_col}_vs_random_forest_size_{mode}.png", dpi=600)
        plt.close()

    _plot_mem(data, "peak_ram_kb", "Average Peak RAM (KB)")
    _plot_mem(data, "peak_vram_kb", "Average Peak VRAM (KB)")
    (
        _plot_mem(data, "rep_size_kb", "Average Representation Size (KB)")
        if "rep_size_kb" in data.columns
        else None
    )

    print(f"Runtime/Memory Analysis: Memory Analysis {mode.capitalize()} - done.")
    print()
