"""
Feature Graph representation for decision trees in DTRBench.

This module provides the FeatureGraphRepresentation class, which represents a decision tree as a graph of features and their relationships and computes differences using a correlation-adjusted version of the Frobenius norm. The implementation is based on the work by Sirocchi et al. (2025) (https://github.com/ChristelSirocchi/urf-graphs/blob/main/utilities.r).
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dtrbench.representations.registry import register_representation

from .base import BaseRepresentation

register_representation(
    "Feature Graph",
    lambda X_train, seed: FeatureGraphRepresentation(criterion="sample", X=X_train),
)


class FeatureGraphRepresentation(BaseRepresentation):
    def __init__(self, criterion="sample", X=None):
        self.criterion = criterion
        self.X = X
        feature_correlation_matrix = np.corrcoef(X, rowvar=False)
        feature_correlation_matrix[np.isnan(feature_correlation_matrix)] = 0.0
        feature_correlation_matrix = np.abs(feature_correlation_matrix)
        tmp_weight = 1 - feature_correlation_matrix
        self.weight = np.pad(
            tmp_weight, ((0, 1), (0, 1)), constant_values=1.0
        )  # might make sense to choose another pad-value for the leaf nodes here

    def represent(self, tree, X_train):
        feature_graph, labels = self._compute_edge_matrix(tree, X_train)
        # self._visualize_feature_graph(feature_graph, labels)
        return feature_graph

    def similarity(self, representation_a, representation_b):

        return 1 / (
            1
            + self.compute_correlated_frobenius_norm(representation_a, representation_b)
        )

    def _fixation_traverse_tree(
        self,
        tree,
        node_id,
        X,
        edge_matrix,
        sample_size,
        feature_names,
        level=1,
        feature_counts={},
    ):
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        if children_left[node_id] == children_right[node_id]:
            return edge_matrix
        split_feature_idx = feature[node_id]
        split_value = threshold[node_id]
        left_child = children_left[node_id]
        right_child = children_right[node_id]

        left_samples = X[:, split_feature_idx] <= split_value
        right_samples = X[:, split_feature_idx] > split_value

        parent_name = feature_names[split_feature_idx]
        left_name = (
            "T" if feature[left_child] == -2 else feature_names[feature[left_child]]
        )
        right_name = (
            "T" if feature[right_child] == -2 else feature_names[feature[right_child]]
        )

        if self.criterion == "present":
            left_weight = 1.0
            right_weight = 1.0
        elif self.criterion == "fixation":
            feature_idx = feature[node_id]
            feature_counts[feature_idx] = feature_counts.get(feature_idx, 0) + 1
            left_weight = feature_counts[feature_idx]
            right_weight = feature_counts[feature_idx]
        elif self.criterion == "level":
            left_weight = 1.0 / level
            right_weight = 1.0 / level
        elif self.criterion == "sample":
            left_weight = np.sum(left_samples) / sample_size
            right_weight = np.sum(right_samples) / sample_size
        else:
            raise ValueError("Undefined criterion.")

        i_parent = feature_names.index(parent_name)
        i_left = (
            len(feature_names) if left_name == "T" else feature_names.index(left_name)
        )
        i_right = (
            len(feature_names) if right_name == "T" else feature_names.index(right_name)
        )

        edge_matrix[i_parent, i_left] += left_weight
        edge_matrix[i_parent, i_right] += right_weight

        if feature[left_child] != -2:
            edge_matrix = self._fixation_traverse_tree(
                tree,
                left_child,
                X[left_samples],
                edge_matrix,
                sample_size,
                feature_names,
                level=level + 1,
                feature_counts=feature_counts,
            )
        if feature[right_child] != -2:
            edge_matrix = self._fixation_traverse_tree(
                tree,
                right_child,
                X[right_samples],
                edge_matrix,
                sample_size,
                feature_names,
                level=level + 1,
                feature_counts=feature_counts,
            )

        return edge_matrix

    def _compute_edge_matrix(self, tree, X_train):
        n_features = X_train.shape[1]
        feature_names = [f"X{i}" for i in range(n_features)]
        edge_matrix = np.zeros((n_features + 1, n_features + 1), dtype=float)
        edge_matrix = self._fixation_traverse_tree(
            tree.tree_, 0, X_train, edge_matrix, X_train.shape[0], feature_names
        )
        return edge_matrix, feature_names + ["T"]

    def _visualize_feature_graph(self, edge_matrix, labels, weight_threshold=0.0):
        G = nx.DiGraph()
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                weight = edge_matrix[i, j]
                if weight > weight_threshold:
                    G.add_edge(src, dst, weight=weight)

        pos = nx.spring_layout(G, seed=42)
        edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1500,
            node_color="lightblue",
            arrowsize=20,
            width=edge_weights,
            font_size=10,
        )
        plt.title("Feature Graph (type = 'sample')")
        plt.show()

    def edge_matrix_to_edge_set(self, edge_matrix, threshold=0.0):
        edges = set()
        N = edge_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if edge_matrix[i, j] > threshold:
                    edges.add((i, j))
        return edges

    def compute_correlated_frobenius_norm(self, matrix_1, matrix_2):
        if matrix_1.shape != matrix_2.shape:
            raise ValueError(
                "2 Graphs have to have the same dimensions in order to be compared."
            )

        diff = (matrix_1 - matrix_2) ** 2 * self.weight

        return np.sqrt(np.sum(diff))
