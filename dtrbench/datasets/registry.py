"""
Registry for datasets in DTRBench.

This module provides a registry for datasets used in DTRBench. It allows for the registration and retrieval of dataset loaders, enabling users to easily add custom datasets and access them by name.
"""

_DATASETS = {}


def register_dataset(name):

    def decorator(loader):

        if name in _DATASETS:
            raise ValueError(
                f"Dataset '{name}' already registered"
            )

        _DATASETS[name] = loader
    return decorator


def get_dataset(name):

    if name not in _DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {available_datasets()}"
        )

    return _DATASETS[name]


def available_datasets():

    return sorted(_DATASETS.keys())
