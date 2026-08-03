"""
Use this module to add custom selection strategies for subforest selection.

You can use the `register_selection_strategy` decorator to register your selection strategies. Each strategy should define how to select subtrees from a forest using a distance matrix and oob-performance values.
"""

from dtrbench.selection_strategies.registry import register_selection_strategy


# Uncomment the decorator below and replace `sample_selection_strategy` with a unique name for your selection strategy
# @register_selection_strategy("sample_selection_strategy")
def sample_selection_strategy(
    distance_matrix,
    subforest_size,
    seed,
    all_oob_preds,
    oob_indices_list,
    y_train,
    n_classes,
):
    """Template for a custom selection strategy. Replace the contents of this function with your own selection logic.

    A selection strategy should select a subset of trees (subforest) from a random forest based on the provided distance matrix and oob-performance values.

    Not all arguments need to be used. Ignore those that are not required by your selection strategy.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix of the trees in the random forest.
        subforest_size (int): Desired size of the subforest to be selected.
        seed (int): Random seed for reproducibility.
        all_oob_preds (list): List of out-of-bag predictions for each tree in the random forest.
        oob_indices_list (list[np.ndarray]): List of out-of-bag indices for each tree in the random forest. Required for some selection strategies that need access to out-of-bag predictions.
        y_train (np.ndarray): Train-set labels. Required for some selection strategies that need access to the target labels.
        n_classes (int): Number of unique classes in the training labels.

    Returns:
        list: Indices of the selected trees in the subforest. The list should contain `subforest_size` unique indices referring to the input forest.
    """

    # TODO: Implement your selection strategy here. You can use the provided distance_matrix, all_oob_preds, oob_indices_list, y_train, and n_classes to inform your selection logic.
    subforest_indices = []
    return subforest_indices
