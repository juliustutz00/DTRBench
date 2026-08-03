"""
Registry for selection strategies in DTRBench.

This module provides a registry for selection strategies used in DTRBench. It allows for the registration and retrieval of selection functions, enabling users to easily add custom selection strategies and access them by name.
"""

_SELECTION_STRATEGIES = {}


def register_selection_strategy(name):

    def decorator(func):
        if name in _SELECTION_STRATEGIES:
            raise ValueError(
                f"Selection strategy '{name}' already registered"
            )

        _SELECTION_STRATEGIES[name] = func
        return func

    return decorator


def get_selection_strategy(name):

    if name not in _SELECTION_STRATEGIES:
        raise KeyError(
            f"Unknown selection strategy '{name}'. Available strategies: {available_selection_strategies()}"
        )

    return _SELECTION_STRATEGIES[name]


def available_selection_strategies():

    return sorted(_SELECTION_STRATEGIES.keys())
