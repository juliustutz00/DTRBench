"""
Visualization utilities for evaluating subforests in DTRBench.

This module provides functions for creating visualizations of tree representations and their relationships.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


def plot_and_save_mds(distance_matrix, accuracies, name, savepath, seed):
    """Plot and save a 2D MDS projection of the decision tree representations based on the provided distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Precomputed distance matrix.
        accuracies (np.ndarray): Array of accuracy values for each tree.
        name (str): Name of the visualization.
        savepath (str): Path to save the visualization.
        seed (int): Random seed for reproducibility.
    """
    if np.all(np.isnan(accuracies)):
        color_data = np.zeros(len(accuracies))
    else:
        min_acc = np.nanmin(accuracies)
        color_data = np.nan_to_num(accuracies, nan=min_acc)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
        n_init=4,
    )
    coords = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_data,
        cmap="cividis",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
        s=60,
    )

    plt.colorbar(sc, label="Individual Tree OOB MCC")
    plt.title(f"MDS Tree Projection: {name}")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)

    plots_dir = os.path.join(savepath, "mds_plots")
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(
        plots_dir, f"mds_plot_{name.replace(' ', '_').lower()}_{seed}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
