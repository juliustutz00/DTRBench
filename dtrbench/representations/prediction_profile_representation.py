"""
Prediction Profile representation for decision trees in DTRBench.

This module provides the PredictionProfileRepresentation class, which represents a decision tree as its predictions on the whole training data.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score

from dtrbench.representations.registry import register_representation

from .base import BaseRepresentation

register_representation(
    "Prediction Profile", lambda X_train, seed: PredictionProfileRepresentation(X_train, seed)
)


class PredictionProfileRepresentation(BaseRepresentation):
    def __init__(self, X_train, seed):
        self.X_train = X_train
        self.seed = seed

    def represent(self, tree, X_boot):
        predictions = tree.predict(self.X_train)

        return {
            "predictions": predictions,
            "classes": tree.classes_.copy(),
        }

    def similarity(self, representation_a, representation_b):
        if (representation_a["predictions"] is None or representation_b["predictions"] is None):
            return 0.0

        classes_a = representation_a["classes"]
        classes_b = representation_b["classes"]
        if not np.array_equal(classes_a, classes_b):
            raise ValueError("The two trees do not have the same class labels.")

        predictions_a = representation_a["predictions"]
        predictions_b = representation_b["predictions"]

        cohen_kappa = cohen_kappa_score(predictions_a, predictions_b)

        return float((cohen_kappa + 1.0) / 2.0)
