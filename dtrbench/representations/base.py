"""
Base class for decision tree representations in DTRBench.

This module provides the base class for implementing various decision tree representations used in DTRBench. Each representation must implement the `represent` and `similarity` methods to define how a decision tree is represented and how similarity between representations is computed.

"""

from abc import ABC, abstractmethod


class BaseRepresentation(ABC):
    @abstractmethod
    def represent(self, tree, X_train):
        """Convert a decision tree into its representation.
        
        Args:
            tree: The decision tree to be represented.
            X_train: The training data used to fit the decision tree.
            
        Returns:
            A representation of the decision tree."""
        pass

    @abstractmethod
    def similarity(self, representation_a, representation_b):
        """Compute the similarity score between two representations from the same class.
        
        Args:
            representation_a: The first representation to compare.
            representation_b: The second representation to compare.
            
        Returns:
            A similarity score (float) indicating how similar the two representations are.
        """
        pass
