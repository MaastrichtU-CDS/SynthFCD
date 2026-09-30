"""
Helper function for marking functions as deprecated.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    message: str,
    *,
    version: str | None = None,
    replacement: str | None = None,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Mark a callable as deprecated and emit a warning on use.
    """
    parts = [message]
    if version is not None:
        parts.append(f"Deprecated since version {version}.")
    if replacement is not None:
        parts.append(f"Use `{replacement}` instead.")
    warning_message = " ".join(parts)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(warning_message, category, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
