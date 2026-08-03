"""
Metrics for evaluating subforests in DTRBench.

This module provides functions for computing various metrics to assess the performance and characteristics of selected subforests.
"""

import numpy as np
import ot
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
)


def compute_feature_importance_difference(
    original, variation, X_train, correlation_adjustment=False
):
    """Compute the difference in feature importance between two trees, optionally adjusting for feature correlation.
    
    Args:
        original (DecisionTreeClassifier): Original decision tree.
        variation (DecisionTreeClassifier): Perturbed decision tree.
        X_train (np.ndarray): Train-set.
        correlation_adjustment (bool): Whether to adjust for feature correlation using optimal transport."""

    def feature_importance_vector(tree, n_features: int) -> np.ndarray:
        """Return feature_importances_ padded/truncated to n_features, normalized to sum=1 if possible."""
        if tree is None or not hasattr(tree, "feature_importances_"):
            return np.zeros(n_features, dtype=float)

        fi = np.asarray(tree.feature_importances_, dtype=float).ravel()
        if fi.size < n_features:
            fi = np.pad(fi, (0, n_features - fi.size), mode="constant")
        else:
            fi = fi[:n_features]

        fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
        fi[fi < 0] = 0.0

        s = float(fi.sum())
        if s > 0:
            fi = fi / s
        return fi

    n_features = X_train.shape[1]
    p = feature_importance_vector(original, n_features)
    q = feature_importance_vector(variation, n_features)

    if not correlation_adjustment:
        return np.linalg.norm(p - q, ord=1)
    else:
        corr = np.corrcoef(X_train, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        D = 1 - np.abs(corr)

        p = np.maximum(p, 0)
        q = np.maximum(q, 0)
        if p.sum() > 0:
            p = p / p.sum()
        else:
            p = np.ones_like(p) / len(p)

        if q.sum() > 0:
            q = q / q.sum()
        else:
            q = np.ones_like(q) / len(q)

        return ot.emd2(p, q, D)


def tree_metric_score(tree, X, y, metric="mcc", average="macro"):
    """Compute a specified metric score for a decision tree on a given dataset.
    
    Args:
        tree (DecisionTreeClassifier): Decision tree to evaluate.
        X (np.ndarray): Feature matrix for evaluation.
        y (np.ndarray): True labels for evaluation.
        metric (str): Metric to compute ('accuracy', 'f1', or 'mcc').
        average (str): Averaging method for multi-class F1 score ('macro', 'micro', etc.)."""
    if X is None or y is None or len(y) == 0:
        return np.nan

    y_pred = tree.predict(X)

    if metric == "accuracy":
        return float(np.mean(y_pred == y))
    if metric == "f1":
        return float(f1_score(y, y_pred, average=average, zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y, y_pred))

    raise ValueError(f"Unsupported tree metric: {metric}")
