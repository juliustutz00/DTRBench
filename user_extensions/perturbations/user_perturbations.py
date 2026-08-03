"""
Use this module to add your custom DecisionTreeClassifier perturbations.

You can use the `register_perturbation` decorator to register your perturbation functions. Each function should return a perturbed version of the input DecisionTreeClassifier model.
"""

import copy

from dtrbench.perturbations.perturbation_ops import update_subtree
from dtrbench.perturbations.registry import register_perturbation


# Uncomment the decorator and replace `sample_perturbation` with a unique name for your perturbation
# @register_perturbation("sample_perturbation")
def sample_perturbation(
    tree, template_tree, X_train, y_train, features_info, intensity, seed=None
):
    """Template for a custom perturbation function. Replace the contents of this function with your own perturbation logic.

    The function needs to return a perturbed copy of the input DecisionTreeClassifier model. The input tree must not be modified in place. Always return a modified copy.
    You can use the __getstate__() and __setstate__(state) methods of the DecisionTreeClassifier to access and modify the internal state of the tree.
    You can also use the `update_subtree` function to update the subtree of a node after perturbing it.

    Args:
        tree (DecisionTreeClassifier): The decision tree to perturb.
        template_tree (DecisionTreeClassifier): A template decision tree with more memory available. Can be used if nodes are added to the tree.
        X_train (np.ndarray): Train-set that tree was fitted on.
        y_train (np.ndarray): Train-set labels that tree was fitted on.
        features_info (pd.DataFrame): Information about the type of features.
        intensity (float): The proportion of nodes to perturb (between 0 and 1).
        seed (int): Random seed for reproducibility."""

    tree_copy = copy.deepcopy(tree)
    tree_state = tree_copy.tree_.__getstate__()

    # TODO: add your perturbation logic here
    #
    # Example:
    # tree_state["nodes"][node_id]["feature"] = new_feature
    # update_subtree(tree_copy, node_id, X_train, y_train)
    #
    # See the perturbations implemented in `dtrbench/perturbations/perturbation_ops.py` for complete examples.

    tree_copy.tree_.__setstate__(tree_state)
    return tree_copy
