"""
Use this module to add loading functions for your custom datasets.

You can use the `register_dataset` decorator to register your dataset loading functions. Each function should return a dictionary with the following keys:
- `name`: The name of the dataset.
- `folds`: A list of tuples, each containing the training and test data for a fold.
- `features`: A DataFrame containing information about the type of the features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from dtrbench.datasets.registry import register_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets"


# Uncomment the decorator below and replace `sample_dataset_name` with a unique name for your dataset
# @register_dataset("sample_dataset_name")
def load_sample_dataset(n_splits, n_samples, seed):
    """Template to load a custom dataset. Replace the contents of this function with your own dataset loading logic.

    Your dataset should be stored in a folder with the following structure:
    - sample_dataset_name/X.npy (numpy array)
    - sample_dataset_name/y.npy (numpy array)
    - sample_dataset_name/features.csv (CSV file with a column "type" indicating the type of each feature: "categorical", "binary", "integer", or "continuous")

    Args:
        n_splits (int): Number of folds for cross-validation.
        n_samples (int): Number of samples to use from the dataset.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing exactly the dataset name, folds, and feature information."""

    # ============================================================
    # User configuration
    # ============================================================
    # TODO: set the name of your dataset
    name = "sample_dataset_name"

    # TODO: set the path to the folder where the dataset is stored
    base_path = Path("custom/path/to/your/dataset")

    # you can also store your dataset in the `datasets` folder of the project and use the following line instead:
    # base_path = DATA_ROOT / "sample_dataset_name"

    # ============================================================
    # Default preprocessing (can usually remain unchanged)
    # ============================================================
    # the remaining code performs the default preprocessing used by DTRBench
    # you can leave it unchanged or customize it if needed

    # load your dataset and labels here; X and y should be numpy arrays, and features_df should be a pandas DataFrame with a column "type" indicating the type of each feature (either "categorical", "binary", "integer", or "continuous")
    X = np.load(base_path / "X.npy", allow_pickle=True)
    y = np.load(base_path / "y.npy", allow_pickle=True)
    features_df = pd.read_csv(base_path / "features.csv")

    # sklearns DecisionTreeClassifier does not support categorical features, so we will drop them here. You can also choose to encode them instead.
    allowed_types = {"continuous", "integer", "int", "numeric", "real"}
    feature_types = features_df["type"].astype(str).str.lower()
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

    # encode target labels with value between 0 and n_classes-1
    y = LabelEncoder().fit_transform(y.ravel())
    y = y.astype(np.int64)

    if X.shape[0] != y.shape[0]:
        min_n = min(X.shape[0], y.shape[0])
        X = X[:min_n]
        y = y[:min_n]

    if n_samples is not None and n_samples < len(X):
        rng = np.random.RandomState(seed)
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

    # impute missing values
    X_numeric = X_obj.astype(float)
    if np.isnan(X_numeric).any():
        imputer = KNNImputer(n_neighbors=5)
        X_numeric = imputer.fit_transform(X_numeric)

    # create stratified folds for cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(X_numeric, y):
        X_train, X_test = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    return {"name": name, "folds": folds, "features": features_df}
