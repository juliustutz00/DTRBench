"""
Leaf Profile representation for decision trees in DTRBench.

This module provides the LeafProfileRepresentation class, which represents a decision tree as a collection of leaf node profiles and computes distances using the Earth Mover's Distance (EMD).
"""

import numpy as np
import ot

from dtrbench.representations.registry import register_representation

from .base import BaseRepresentation

register_representation(
    "Leaf Profile", lambda X_train, seed: LeafProfileRepresentation(criterion="l2")
)


class LeafProfileRepresentation(BaseRepresentation):
    def __init__(self, criterion="l2"):
        self.criterion = criterion

    def represent(self, tree, X_train):
        clf = tree
        tree = tree.tree_
        node_count = tree.node_count
        is_leaf = (tree.children_left == -1) & (tree.children_right == -1)

        total_samples = tree.n_node_samples[0]
        ldp = []

        max_class_index = int(np.max(clf.classes_))
        target_dim = max_class_index + 1
        classes_map = [int(c) for c in clf.classes_]

        for i in range(node_count):
            if is_leaf[i]:
                mass = tree.n_node_samples[i] / total_samples
                raw_distribution = tree.value[i][0]
                if np.array_equal(classes_map, range(target_dim)):
                    aligned_distribution = raw_distribution
                else:
                    aligned_distribution = np.zeros(target_dim, dtype=float)
                    for raw_idx, target_idx in enumerate(classes_map):
                        if target_idx < target_dim:
                            aligned_distribution[target_idx] = raw_distribution[raw_idx]

                s = aligned_distribution.sum()
                if s > 0:
                    aligned_distribution = aligned_distribution / s

                ldp.append((mass, aligned_distribution))

        return ldp

    def similarity(self, representation_a, representation_b):
        return 1 / (
            1 + self.__compute_emd(representation_a, representation_b, self.criterion)
        )

    def __compute_emd(self, ldp1, ldp2, criterion="l2"):
        w1, p1 = zip(*ldp1)
        w2, p2 = zip(*ldp2)

        w1 = np.array(w1)
        w2 = np.array(w2)
        p1 = np.array(p1)
        p2 = np.array(p2)

        d1 = p1.shape[1]
        d2 = p2.shape[1]

        if d1 != d2:
            max_d = max(d1, d2)
            if d1 < max_d:
                pad = np.zeros((p1.shape[0], max_d - d1))
                p1 = np.hstack([p1, pad])
            if d2 < max_d:
                pad = np.zeros((p2.shape[0], max_d - d2))
                p2 = np.hstack([p2, pad])

        if criterion == "l1":
            M = np.sum(np.abs(p1[:, None, :] - p2[None, :, :]), axis=2)
        elif criterion == "l2":
            M = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
        else:
            raise ValueError("Unsupported metric.")

        w1 = w1 / w1.sum()
        w2 = w2 / w2.sum()

        return ot.emd2(w1, w2, M)
