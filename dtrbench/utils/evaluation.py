"""
Evaluation utilities for DTRBench.

This module provides functions for evaluating the performance of random forests and their selected subforests.
"""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_forest(X_test, y_test, trees, n_instances_test, n_classes):
    """Evaluate the performance of a random forest or subforest on a test set.
    
    Args:
        X_test (np.ndarray): Test-set.
        y_test (np.ndarray): Test-set labels.
        trees (list[DecisionTreeClassifier]): List of decision trees in the random forest or subforest.
        n_instances_test (int): Number of instances in the test-set.
        n_classes (int): Number of unique classes in the test-set.
        
    Returns:
        dict: A dictionary containing hard predictions, probabilities, evaluation metrics, and feature importances.
    """
    all_preds = np.vstack([t.predict(X_test) for t in trees])
    hard_preds = np.zeros(n_instances_test, dtype=int)

    for i in range(n_instances_test):
        counts = np.bincount(all_preds[:, i].astype(int), minlength=n_classes)
        hard_preds[i] = counts.argmax()

    probas_list = [_safe_predict_proba(t, X_test, n_classes) for t in trees]
    probas = np.mean(probas_list, axis=0)

    metrics = {
        "accuracy": accuracy_score(y_test, hard_preds),
        "macro_f1": f1_score(y_test, hard_preds, average="macro"),
        "mcc": matthews_corrcoef(y_test, hard_preds),
    }

    if len(np.unique(y_test)) <= 1:
        print(
            f"Warning: Only one class {np.unique(y_test)} in y_test. AUC not defined."
        )

    # ROC AUC
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probas[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, probas, multi_class="ovr")
    except ValueError:
        if n_classes == 2:
            metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc_ovr"] = np.nan

    # PR AUC
    try:
        if n_classes == 2:
            metrics["pr_auc"] = average_precision_score(y_test, probas[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            metrics["pr_auc_ovr"] = average_precision_score(
                y_test_bin, probas, average="macro"
            )
    except ValueError:
        if n_classes == 2:
            metrics["pr_auc"] = np.nan
        else:
            metrics["pr_auc_ovr"] = np.nan

    # minority class metrics (one-vs-rest)
    y_test_int = np.asarray(y_test).astype(int)
    class_counts = np.bincount(y_test_int, minlength=n_classes)
    present_classes = np.where(class_counts > 0)[0]

    if present_classes.size > 0:
        minority_class = int(present_classes[np.argmin(class_counts[present_classes])])
        metrics["minority_class"] = minority_class
        metrics["minority_support"] = int(class_counts[minority_class])

        metrics["minority_precision"] = float(
            precision_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0,
            )[0]
        )
        metrics["minority_recall"] = float(
            recall_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0,
            )[0]
        )
        metrics["minority_f1"] = float(
            f1_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0,
            )[0]
        )
    else:
        metrics["minority_class"] = np.nan
        metrics["minority_support"] = np.nan
        metrics["minority_precision"] = np.nan
        metrics["minority_recall"] = np.nan
        metrics["minority_f1"] = np.nan

    # feature importances
    n_features = X_test.shape[1]
    importances = []
    for t in trees:
        if hasattr(t, "feature_importances_"):
            fi = np.asarray(t.feature_importances_, dtype=float)
            if fi.shape[0] != n_features:
                fi = np.pad(fi, (0, max(0, n_features - fi.shape[0])), mode="constant")
            importances.append(fi[:n_features])
        else:
            importances.append(np.zeros(n_features, dtype=float))

    if len(importances) > 0:
        feature_importances = np.mean(importances, axis=0)
    else:
        feature_importances = np.zeros(n_features, dtype=float)

    return {
        "hard_predictions": hard_preds,
        "probabilities": probas,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


def prediction_agreement(preds_a, preds_b):
    """Compute the prediction agreement between two sets of predictions.

    Args:
        preds_a (np.ndarray): Predictions from the first model.
        preds_b (np.ndarray): Predictions from the second model.

    Returns:
        float: The proportion of instances where the predictions agree.
    """

    return np.mean(preds_a == preds_b)


def shared_metric_cols(eval_result):
    """Extract shared metric columns from the evaluation result dictionary.
    
    Args:
        eval_result (dict): Evaluation result dictionary containing metrics and feature importances.
        
    Returns:
        dict: A dictionary containing shared metric columns and feature importances as a JSON string.
    """
    
    m = eval_result["metrics"]
    return {
        "acc": m.get("accuracy", np.nan),
        "macro_f1": m.get("macro_f1", np.nan),
        "mcc": m.get("mcc", np.nan),
        "roc_auc": m.get("roc_auc", m.get("roc_auc_ovr", np.nan)),
        "pr_auc": m.get("pr_auc", m.get("pr_auc_ovr", np.nan)),
        "minority_class": m.get("minority_class", np.nan),
        "minority_support": m.get("minority_support", np.nan),
        "minority_precision": m.get("minority_precision", np.nan),
        "minority_recall": m.get("minority_recall", np.nan),
        "minority_f1": m.get("minority_f1", np.nan),
        "feature_importances": json.dumps(
            [
                float(f"{x:.10f}")
                for x in np.asarray(
                    eval_result.get("feature_importances", np.array([]))
                )
            ]
        ),
    }


def _safe_predict_proba(tree, X, n_classes):
    probas = tree.predict_proba(X)

    if probas.shape[1] == n_classes:
        return probas

    full_probas = np.zeros((X.shape[0], n_classes), dtype=probas.dtype)

    for k, class_label in enumerate(tree.classes_):
        if class_label < n_classes:
            full_probas[:, int(class_label)] = probas[:, k]

    return full_probas
