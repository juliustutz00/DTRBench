"""
Registry for decision tree representations in DTRBench.

This module provides a registry for decision tree representations in DTRBench. It allows for the registration and retrieval of representation classes, enabling users to easily add custom representations and access them by name.
"""

_REPRESENTATIONS = {}


def register_representation(name, builder):

    if name in _REPRESENTATIONS:
        raise ValueError(f"Representation '{name}' already registered")

    _REPRESENTATIONS[name] = builder


def get_representation(name):

    if name not in _REPRESENTATIONS:
        raise KeyError(
            f"Unknown representation '{name}'. Available representations: {available_representations()}"
        )

    return _REPRESENTATIONS[name]


def available_representations():

    return sorted(_REPRESENTATIONS.keys())
