"""
Registry for perturbations in DTRBench.

This module provides a registry for perturbations used in DTRBench. It allows for the registration and retrieval of perturbation functions, enabling users to easily add custom perturbations and access them by name.
"""

_PERTURBATIONS = {}


def register_perturbation(name):
    def decorator(loader):
        if name in _PERTURBATIONS:
            raise ValueError(f"Perturbation '{name}' already registered")

        _PERTURBATIONS[name] = loader
        return loader

    return decorator


def get_perturbation(name):

    if name not in _PERTURBATIONS:
        raise KeyError(
            f"Unknown perturbation '{name}'. Available perturbations: {available_perturbations()}"
        )

    return _PERTURBATIONS[name]


def available_perturbations():

    return sorted(_PERTURBATIONS.keys())
