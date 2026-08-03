"""
Use this module to add custom decision tree representations.

You can use the `register_representation` function to register a factory that creates an instance of your representation class. Each function should return a representation of the decision tree given.
"""

from dtrbench.representations.registry import register_representation
from dtrbench.representations.base import BaseRepresentation


# Uncomment the line below and replace 'Sample Representation Name' and 'SampleRepresentation' with a unique name for your representation
# register_representation("Sample Representation Name", lambda X_train, seed: SampleRepresentation(X_train=X_train, seed=seed))
class SampleRepresentation(BaseRepresentation):
    """Template for a custom representation class. A representation converts a decision tree into an arbitrary object that can later be compared using `similarity()`.

    Replace the contents of this class with your own representation logic."""

    def __init__(self, X_train, seed):
        """Initialize your representation with any parameters you need. You may use X_train and seed as they are passed to each representation when instantiated (you may also ignore them)."""
        self.X_train = X_train
        self.seed = seed
        # TODO: add your initialization logic here

    def represent(self, tree, X_train):
        """Compute the representation of the given decision tree.

        Args:
            tree: Decision tree to represent.
            X_train: Train-set used to train the decision tree.

        Returns:
            object: A representation of the decision tree.
        """
        # TODO: implement your representation logic here
        pass

    def similarity(self, representation_a, representation_b):
        """Compute the similarity between two representations.

        Args:
            representation_a: The first representation.
            representation_b: The second representation.

        Returns:
            float: A similarity score between the two representations. Larger values indicate more similar trees. Normalize to range [0, 1] whenever possible.
        """
        # TODO: implement your similarity computation logic here
        pass
