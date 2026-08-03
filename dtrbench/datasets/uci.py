"""
UCI repository dataset loading and preprocessing module.

This module provides functions for loading and preprocessing datasets from the UCI Machine Learning Repository. It includes functionality for downloading datasets, handling missing values, encoding categorical features, and splitting the data into training and testing folds for benchmarking purposes.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from dtrbench.datasets.registry import register_dataset

for name in [
    "iris",
    "balance_scale",
    "cervical_cancer",
    "cirrhosis",
    "connectionist",
    "credit_approval",
    "cylinder_bands",
    "DARWIN",
    "diabetic_retinopathy",
    "eeg_eye",
    "fertility",
    "heart_disease",
    "heart_failure",
    "isolet",
    "japanese_credit",
    "letter_recognition",
    "monk_problem",
    "musk_1",
    "statlog_australian",
    "statlog_german",
    "statlog_heart",
    "statlog_vehicle",
    "support2",
    "vertebral_column",
    "waveform",
    "wine",
]:

    register_dataset(name)(
        lambda n_splits=3,
               n_samples=math.inf,
               seed=0,
               name=name:

            _load_ucirepo_dataset(
                name,
                n_splits,
                n_samples,
                seed
            )
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets"
if not DATA_ROOT.exists():
    raise RuntimeError(f"DATA_ROOT does not exist: {DATA_ROOT}")

def _load_ucirepo_dataset(uci_name, n_splits=3, n_samples=math.inf, seed=None):
    rng = np.random.RandomState(seed)

    base_path = DATA_ROOT / "UCI" / uci_name

    X = np.load(base_path / "X.npy", allow_pickle=True)
    y = np.load(base_path / "y.npy", allow_pickle=True)
    vars_df = pd.read_csv(base_path / "features.csv")

    features_df = vars_df[vars_df["role"] == "Feature"][["type"]].reset_index(drop=True)

    X = np.asarray(X)

    # drop categorical and binary features
    allowed_types = {"continuous", "integer", "int", "numeric", "real"}
    feature_types = features_df["type"].astype(str).str.lower()
    if uci_name == "balance_scale":
        feature_types = feature_types.replace({"categorical": "integer"})
    keep_mask = feature_types.isin(allowed_types)
    keep_indices = np.where(keep_mask)[0]

    X = X[:, keep_indices]
    features_df = features_df.iloc[keep_indices].reset_index(drop=True)

    y = np.asarray(y)

    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        elif y.shape[0] == X.shape[0]:
            y = y[:, 0]
        elif y.shape[1] == X.shape[0]:
            y = y.T[:, 0]
        else:
            y = y.ravel()

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if uci_name == "statlog_vehicle":
        statlog_vehicle_mask = y != "204"
        X = X[statlog_vehicle_mask]
        y = y[statlog_vehicle_mask]

    y = LabelEncoder().fit_transform(y.ravel())
    y = y.astype(np.int64)

    if X.shape[0] != y.shape[0]:
        min_n = min(X.shape[0], y.shape[0])
        X = X[:min_n]
        y = y[:min_n]

    if n_samples is not None and n_samples != math.inf and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X_obj = np.array(X, dtype=object)
    n_features_in = X_obj.shape[1]

    for col_idx in range(n_features_in):
        try:
            ftype = str(features_df.loc[col_idx, "type"]).lower()
        except Exception:
            ftype = "continuous"

        col = X_obj[:, col_idx]

        if ftype in ["categorical", "binary"]:
            le = LabelEncoder()
            X_obj[:, col_idx] = le.fit_transform(col.astype(str))
        else:
            try:
                X_obj[:, col_idx] = col.astype(float)
            except Exception:
                coerced = pd.to_numeric(col, errors="coerce")
                mean = np.nanmean(coerced)
                if np.isnan(mean):
                    mean = 0.0
                X_obj[:, col_idx] = np.where(np.isnan(coerced), mean, coerced)

    X_numeric = X_obj.astype(float)
    if np.isnan(X_numeric).any():
        imputer = KNNImputer(n_neighbors=5)
        X_numeric = imputer.fit_transform(X_numeric)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, test_idx in skf.split(X_numeric, y):
        X_train, X_test = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    return {"name": uci_name, "folds": folds, "features": features_df}

def download_UCI_datasets(ids, names, save_dir):
    """Download one or more datasets from the UCI repository and save them to the specified directory.
    
    Args:
        ids (list[int]): A list of UCI dataset IDs to download.
        names (list[str]): A list of names corresponding to the datasets.
        save_dir (str): The directory where the datasets will be saved.
    """
    for id, name in zip(ids, names):
        ds = fetch_ucirepo(id=id)
        features_df = ds.variables
        X = ds.data.features
        y = ds.data.targets
        os.makedirs(save_dir + "/" + str(name), exist_ok=True)
        np.save(save_dir + "/" + str(name) + "/X.npy", X.to_numpy())
        np.save(save_dir + "/" + str(name) + "/y.npy", y.to_numpy())
        features_df.to_csv(save_dir + "/" + str(name) + "/features.csv", index=False)
