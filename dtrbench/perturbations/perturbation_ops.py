"""
Perturbation operations for decision trees in DTRBench.

This module provides various perturbation functions that can be applied to decision trees for benchmarking and analysis purposes. Perturbations can only be applied to DecisionTreeClassifier from sklearn and work using the __setstate__ function.
"""

import copy

import numpy as np

from dtrbench.perturbations.registry import register_perturbation

# this file is a collection of perturbations that can be applied to a decision tree (sklearn implementation) using the __setstate__ function
# since its use is concerned with measuring structural differences between decision trees, these perturbations do not bother to adapt outcome-related attributes (such as samples passing the nodes)


@register_perturbation("change_threshold")
def change_threshold(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Change the threshold of [0, n_nodes] nodes of a given decision tree.

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree (not used in this perturbation).
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    internal_nodes = [i for i, n in enumerate(tree_state["nodes"]) if n["feature"] >= 0]
    n_change = int(len(internal_nodes) * intensity)
    change_nodes = rng.choice(internal_nodes, n_change, replace=False)

    for node_idx in change_nodes:
        feature = tree_state["nodes"][node_idx]["feature"]
        feature_column = X_train[:, feature]
        feature_type = str(features_info.loc[feature, "type"]).lower()

        if feature_type in ["real", "continuous", "integer"]:
            rng_value = np.max(feature_column) - np.min(feature_column)
            if rng_value == 0:
                continue
            perturbation = rng.normal(scale=rng_value)
            tree_state["nodes"][node_idx]["threshold"] += perturbation
            if feature_type == "integer":
                tree_state["nodes"][node_idx]["threshold"] = int(
                    round(tree_state["nodes"][node_idx]["threshold"])
                )
        elif feature_type in ["categorical", "binary"]:
            unique_values = np.unique(feature_column)
            if len(unique_values) <= 1:
                continue
            new_threshold = rng.choice(unique_values)
            tree_state["nodes"][node_idx]["threshold"] = new_threshold
        else:
            raise ValueError("Unsupported feature type for perturbation.")

    update_subtree(
        0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state
    )
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


@register_perturbation("change_feature")
def change_feature(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Change the feature of [0, n_nodes] nodes of a given decision tree. The threshold is set to the best value according to gini impurity.

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree (not used in this perturbation).
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    n_features = X_train.shape[1]
    internal_nodes = [
        i for i, node in enumerate(tree_state["nodes"]) if node["feature"] >= 0
    ]
    n_change = int(len(internal_nodes) * intensity)
    if n_change == 0:
        return tree_copy
    change_nodes = rng.choice(internal_nodes, n_change, replace=False)

    for node_idx in change_nodes:
        random_feature = rng.randint(0, n_features)
        random_feature_type = str(features_info.loc[random_feature, "type"]).lower()
        sample_indices = np.arange(len(X_train))
        sample_indices_at_node = _get_sample_indices_at_node(
            0, node_idx, sample_indices, tree_state, X_train, features_info
        ).astype(int)
        optimal_threshold = _find_optimal_threshold(
            X_train,
            y_train,
            random_feature,
            random_feature_type,
            sample_indices_at_node,
        )
        tree_state["nodes"][node_idx]["feature"] = random_feature
        tree_state["nodes"][node_idx]["threshold"] = optimal_threshold

    update_subtree(
        0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state
    )
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


@register_perturbation("swap_nodes")
def swap_nodes(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Swap the features and thresholds of [0, n_nodes] nodes of a given decision tree.

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree (not used in this perturbation).
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    internal_nodes = [
        i for i, node in enumerate(tree_state["nodes"]) if node["feature"] >= 0
    ]
    n_swaps = int((len(internal_nodes) * intensity) / 2)
    if n_swaps == 0:
        return tree_copy
    swap_pairs = rng.choice(internal_nodes, size=2 * n_swaps, replace=False)

    for i in range(n_swaps):
        node_a = swap_pairs[2 * i]
        node_b = swap_pairs[2 * i + 1]
        tmp_feature = tree_state["nodes"][node_a]["feature"]
        tmp_threshold = tree_state["nodes"][node_a]["threshold"]
        tree_state["nodes"][node_a]["feature"] = tree_state["nodes"][node_b]["feature"]
        tree_state["nodes"][node_a]["threshold"] = tree_state["nodes"][node_b][
            "threshold"
        ]
        tree_state["nodes"][node_b]["feature"] = tmp_feature
        tree_state["nodes"][node_b]["threshold"] = tmp_threshold

    update_subtree(
        0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state
    )
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


@register_perturbation("remove_nodes")
def remove_nodes(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Remove [0, n_nodes] nodes of a given decision tree. The parent node of the removed nodes will become a leaf node.

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree (not used in this perturbation).
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()
    removal_count = 0
    n_removals = int((tree_state["node_count"] - 1) * intensity)
    if n_removals == 0:
        return tree_copy

    while removal_count < n_removals:
        # parent_nodes refers to parent nodes with 2 children that can be removed (as they are both leafs) such that the parent node will become a leaf
        parent_nodes = _find_parent_with_two_leafs(tree_state)
        chosen_parent_node = rng.choice(parent_nodes)
        _remove_children(chosen_parent_node, tree_state)
        removal_count += 2

    update_subtree(
        0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state
    )
    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy


@register_perturbation("add_nodes")
def add_nodes(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Add [0, n_nodes] nodes to a given decision tree. The new nodes will be added to leaf nodes of the tree.

    Since this implementation might cause memory issues due to its internal cython representation, the following requirement has to be met: n_nodes(template_tree) > (n_nodes(tree)-1) * 2

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree that ensures enough memory is available.
        X_train (np.ndarray): Train-set.
        y_train (np.ndarray): Train-set labels.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""
    max_depth = 8
    rng = np.random.RandomState(seed)
    tree_copy = copy.deepcopy(tree)
    template_tree_copy = copy.deepcopy(template_tree)
    tree_state = tree_copy.tree_.__getstate__()
    template_tree_state = template_tree_copy.tree_.__getstate__()
    if not len(template_tree_state["nodes"]) > ((len(tree_state["nodes"]) - 1) * 2):
        raise ValueError(
            "add_node: Since this implementation might cause memory issues due to its internal cython representation the following requirement has to be met: n_nodes(template_tree) > (n_nodes(tree)-1) * 2"
        )
    addition_count = 0
    n_additions = int((tree_state["node_count"] - 1) * intensity)
    if n_additions == 0:
        return tree_copy

    while addition_count < n_additions:
        depths = _compute_node_depth(tree_state)
        leaf_nodes = _find_leaf_nodes(tree_state)
        expandable_leaf_nodes = [
            nid for nid in leaf_nodes if depths.get(nid, 0) < max_depth
        ]
        if len(expandable_leaf_nodes) == 0:
            break
        chosen_leaf = rng.choice(expandable_leaf_nodes)
        _add_children(chosen_leaf, tree_state, X_train, y_train, features_info, rng)
        addition_count += 2

    update_subtree(
        0, X_train, y_train, features_info, np.arange(len(X_train)), tree_state
    )
    template_tree_copy.tree_.__setstate__(tree_state)
    return template_tree_copy


def update_subtree(
    node_id,
    X,
    y,
    features_info,
    sample_indices,
    tree_state,
    parent_class_distribution=None,
):
    """Recursively update the subtree rooted at the given node_id with the provided sample indices and class distribution.

    Args:
        node_id (int): The index of the current node in the tree_state.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        features_info (pd.DataFrame): Information about the type of features.
        sample_indices (np.ndarray): Indices of samples that reach the current node.
        tree_state (dict): The state of the decision tree, including nodes and values.
        parent_class_distribution (np.ndarray): Class distribution of the parent node. Defaults to None.
    """
    nodes = tree_state["nodes"]
    node = nodes[node_id]
    feature = node["feature"]
    threshold = node["threshold"]

    # if the node is empty (n_node_samples == 0) the prediction value of its parent node is being used; if the root is empty a prediciton value np.zeroes(size=n_classes) is used
    unique_classes = np.unique(y)
    class_counts = np.array([np.sum(y[sample_indices] == c) for c in unique_classes])
    total = class_counts.sum()
    if total > 0:
        class_distribution = class_counts / total
    else:
        class_distribution = (
            parent_class_distribution.copy()
            if parent_class_distribution is not None
            else np.zeros_like(class_counts)
        )

    tree_state["values"][node_id][0] = class_distribution
    tree_state["nodes"][node_id]["n_node_samples"] = len(sample_indices)
    tree_state["nodes"][node_id]["weighted_n_node_samples"] = float(len(sample_indices))
    tree_state["nodes"][node_id]["impurity"] = _compute_gini(class_counts)

    if feature >= 0:
        X_node = X[sample_indices, feature]
        feature_type = str(features_info.loc[feature, "type"]).lower()
        if feature_type in ["real", "continuous", "integer"]:
            left_indices = sample_indices[X_node <= threshold]
            right_indices = sample_indices[X_node > threshold]
        elif feature_type in ["categorical", "binary"]:
            left_indices = sample_indices[X_node == threshold]
            right_indices = sample_indices[X_node != threshold]
        else:
            raise ValueError("Unsupported feature type for perturbation.")

        update_subtree(
            node["left_child"],
            X,
            y,
            features_info,
            left_indices,
            tree_state,
            class_distribution,
        )
        update_subtree(
            node["right_child"],
            X,
            y,
            features_info,
            right_indices,
            tree_state,
            class_distribution,
        )


def _compute_node_depth(tree_state):
    nodes = tree_state["nodes"]
    depths = {0: 0}
    stack = [0]
    while stack:
        nid = stack.pop()
        d = depths[nid]
        left = nodes[nid]["left_child"]
        right = nodes[nid]["right_child"]
        if left != -1:
            depths[left] = d + 1
            stack.append(left)
        if right != -1:
            depths[right] = d + 1
            stack.append(right)
    return depths


def _compute_gini(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [count / total for count in class_counts]
    return 1.0 - sum(p**2 for p in probs)


def _find_optimal_threshold(X, y, feature_idx, feature_type, sample_indices):
    if len(sample_indices) == 0:
        return np.mean(X[:, feature_idx])
    feature_values = X[sample_indices, feature_idx]

    unique_values = np.unique(feature_values)
    if len(unique_values) == 1:
        return unique_values[0]
    best_gini = float("inf")
    best_threshold = None

    if feature_type in ["real", "continuous", "integer"]:
        for i in range(1, len(unique_values)):
            threshold = (unique_values[i - 1] + unique_values[i]) / 2
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[right_mask]

            left_gini = _compute_gini(np.bincount(y[left_indices]))
            right_gini = _compute_gini(np.bincount(y[right_indices]))
            total_samples = len(sample_indices)
            weighted_gini = (len(left_indices) / total_samples) * left_gini + (
                len(right_indices) / total_samples
            ) * right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_threshold = threshold

        best_threshold = (
            int(round(best_threshold)) if feature_type == "integer" else best_threshold
        )

    elif feature_type in ["categorical", "binary"]:
        for value in unique_values:
            left_mask = feature_values == value
            right_mask = ~left_mask
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[right_mask]

            left_gini = _compute_gini(np.bincount(y[left_indices]))
            right_gini = _compute_gini(np.bincount(y[right_indices]))
            total_samples = len(sample_indices)
            weighted_gini = (len(left_indices) / total_samples) * left_gini + (
                len(right_indices) / total_samples
            ) * right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_threshold = value

    else:
        raise ValueError("Unsupported feature type for perturbation.")

    return best_threshold


def _get_sample_indices_at_node(
    current_node_idx,
    searched_node_idx,
    sample_indices,
    tree_state,
    X_train,
    features_info,
):
    if current_node_idx == searched_node_idx:
        if sample_indices is None or len(sample_indices) == 0:
            sample_indices = np.array([])
        return sample_indices
    node = tree_state["nodes"][current_node_idx]
    feature = node["feature"]
    threshold = node["threshold"]

    if feature >= 0:
        X_node = X_train[sample_indices, feature]
        feature_type = str(features_info.loc[feature, "type"]).lower()
        if feature_type in ["real", "continuous", "integer"]:
            left_mask = X_node <= threshold
            right_mask = ~left_mask
        elif feature_type in ["categorical", "binary"]:
            left_mask = X_node == threshold
            right_mask = ~left_mask
        else:
            raise ValueError("Unsupported feature type for perturbation.")

        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[right_mask]

        # recursively traverse to left and right child nodes
        if node["left_child"] >= 0:
            left_indices = _get_sample_indices_at_node(
                node["left_child"],
                searched_node_idx,
                left_indices,
                tree_state,
                X_train,
                features_info,
            )
        if node["right_child"] >= 0:
            right_indices = _get_sample_indices_at_node(
                node["right_child"],
                searched_node_idx,
                right_indices,
                tree_state,
                X_train,
                features_info,
            )
        return np.concatenate((left_indices, right_indices))
    else:
        return np.array([])


def _find_parent_with_two_leafs(tree_state):
    parents_with_two_leafs = []
    for node_idx, node in enumerate(tree_state["nodes"]):
        left_node = node["left_child"]
        right_node = node["right_child"]
        if left_node == -1 or right_node == -1:
            continue
        if (
            tree_state["nodes"][left_node]["left_child"] == -1
            and tree_state["nodes"][left_node]["right_child"] == -1
            and tree_state["nodes"][right_node]["left_child"] == -1
            and tree_state["nodes"][right_node]["right_child"] == -1
        ):
            parents_with_two_leafs.append(node_idx)
    return parents_with_two_leafs


def _find_leaf_nodes(tree_state):
    leaf_nodes = []
    for node_idx, node in enumerate(tree_state["nodes"]):
        if node["left_child"] == -1 and node["right_child"] == -1:
            leaf_nodes.append(node_idx)
    return leaf_nodes


def _remove_children(parent_node, tree_state):
    left_child_idx = tree_state["nodes"][parent_node][0]
    right_child_idx = tree_state["nodes"][parent_node][1]
    to_delete = sorted([left_child_idx, right_child_idx], reverse=True)
    tree_state["nodes"] = np.delete(tree_state["nodes"], to_delete, axis=0)
    tree_state["values"] = np.delete(tree_state["values"], to_delete, axis=0)

    for node in tree_state["nodes"]:
        for child in ["left_child", "right_child"]:
            if node[child] in to_delete:
                node[child] = -1
            else:
                for deleted_idx in to_delete:
                    if node[child] > deleted_idx:
                        node[child] -= 1

    new_parent = parent_node
    for d in to_delete:
        if parent_node > d:
            new_parent -= 1

    tree_state["nodes"][new_parent]["left_child"] = -1
    tree_state["nodes"][new_parent]["right_child"] = -1
    tree_state["nodes"][new_parent]["feature"] = -2
    tree_state["nodes"][new_parent]["threshold"] = -2.0

    tree_state["node_count"] += -2
    tree_state["max_depth"] = _calculate_depth(0, tree_state)


def _add_children(leaf_node, tree_state, X_train, y_train, features_info, rng):
    new_left_node_idx = len(tree_state["nodes"])
    new_right_node_idx = new_left_node_idx + 1
    node_dtype = tree_state["nodes"].dtype
    new_nodes = np.array(
        [(-1, -1, -2, -2.0, 0, 0, 0.0, 0), (-1, -1, -2, -2.0, 0, 0, 0.0, 0)],
        dtype=node_dtype,
    )
    # it's not possible to use np.concatenate/append as the dtype behaves weird
    combined_nodes = np.empty(
        tree_state["nodes"].shape[0] + new_nodes.shape[0], dtype=node_dtype
    )
    combined_nodes[: tree_state["nodes"].shape[0]] = tree_state["nodes"]
    combined_nodes[tree_state["nodes"].shape[0] :] = new_nodes
    tree_state["nodes"] = combined_nodes
    leaf_node_value = tree_state["values"][leaf_node].copy()
    tree_state["values"] = np.append(
        tree_state["values"], [leaf_node_value, leaf_node_value], axis=0
    )
    tree_state["nodes"][leaf_node]["left_child"] = new_left_node_idx
    tree_state["nodes"][leaf_node]["right_child"] = new_right_node_idx
    n_features = X_train.shape[1]
    random_feature_new_parent = rng.randint(0, n_features)
    random_feature_new_parent_type = str(
        features_info.loc[random_feature_new_parent, "type"]
    ).lower()
    sample_indices = np.arange(len(X_train))
    sample_indices_at_node = _get_sample_indices_at_node(
        0, leaf_node, sample_indices, tree_state, X_train, features_info
    ).astype(int)
    optimal_threshold_new_parent = _find_optimal_threshold(
        X_train,
        y_train,
        random_feature_new_parent,
        random_feature_new_parent_type,
        sample_indices_at_node,
    )
    tree_state["nodes"][leaf_node]["feature"] = random_feature_new_parent
    tree_state["nodes"][leaf_node]["threshold"] = optimal_threshold_new_parent

    tree_state["node_count"] += 2
    tree_state["max_depth"] = _calculate_depth(0, tree_state)


def _calculate_depth(idx, tree_state):
    # depth is calculated according to sklearn's implementation
    if (
        tree_state["nodes"][idx]["left_child"] == -1
        and tree_state["nodes"][idx]["right_child"] == -1
    ):
        return 1
    return 1 + max(
        _calculate_depth(tree_state["nodes"][idx]["left_child"], tree_state),
        _calculate_depth(tree_state["nodes"][idx]["right_child"], tree_state),
    )
