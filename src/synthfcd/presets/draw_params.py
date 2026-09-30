"""
Support for drawing parameter values from presets.
"""

from __future__ import annotations

__all__ = [
    "Draw",
    "draw_params",
]

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synthfcd.presets.utils import _normalize_draw_output
from synthfcd.utils._aliases import _RNGType
from synthfcd.utils._validators import validate_obj_type
from synthfcd.utils.misc import get_rng


@dataclass(frozen=True)
class Draw:
    """
    A callable dataclass that draws a value from a sampler.

    Note:
        If callable sampler is provided, it must accept a ``random_state``
        keyword argument.

    Args:
        sampler (str | Callable[..., Any]):
            The sampler to use. Can be a string representing a method name on the RNG
            instance, or a callable that takes the RNG instance as an argument.
        args (tuple[Any, ...]):
            The positional arguments to pass to the sampler.
        kwargs (dict[str, Any]):
            The keyword arguments to pass to the sampler.

    Returns:
        Any:
            The drawn value from the sampler.
    """

    sampler: str | Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the sampler and arguments.
        """
        if not isinstance(self.sampler, str) and not callable(self.sampler):
            raise TypeError("`sampler` must be a string or a callable.")

        validate_obj_type(self.args, "args", tuple)
        validate_obj_type(self.kwargs, "kwargs", dict)

    def __call__(self, rng: _RNGType) -> Any:
        """
        Draw a value from the sampler.

        Args:
            rng (np.random.Generator | np.random.RandomState):
                The random number generator instance to use.

        Returns:
            Any:
                The drawn value from the sampler.
        """
        validate_obj_type(rng, "rng", (np.random.Generator, np.random.RandomState))

        if callable(self.sampler):
            return self.sampler(*self.args, random_state=rng, **self.kwargs)

        # Manual routing to RNG methods that are known to differ between
        # `np.random.Generator` and `np.random.RandomState`.
        if self.sampler == "integers":
            fn = rng.integers if isinstance(rng, np.random.Generator) else rng.randint
        else:
            try:
                fn = (
                    getattr(rng, self.sampler)
                    if isinstance(self.sampler, str)
                    else self.sampler
                )
            except AttributeError as e:
                raise AttributeError(
                    f"Random number generator does not have a method named {self.sampler}."
                ) from e

        return _normalize_draw_output(fn(*self.args, **self.kwargs))


def draw_params(
    params: Any,
    *,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
    path: str = "params",
) -> Any:
    """
    Draw concrete values from a parameter specification, which can be a ``Draw``
    instance, or a container (dictionary, list, tuple) of parameter specifications.

    Notes:
        - `Draw` instances are sampled with a shared RNG.
        - Containers are traversed recursively.

    Args:
        params (Any):
            The container of parameter specifications to draw values from.
        random_seed (int | None, optional):
            The random seed for the random number generator. If ``None``, the global
            NumPy RNG is used.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            Random number generator for reproducibility. If ``None``, the global
            NumPy RNG is used.
        path (str, optional):
            The path to the parameter in the nested specification.
            Used for error reporting. Defaults to ``"params"``.

    Returns:
        Any:
            The materialized parameter value(s).
    """
    rng = get_rng(seed=random_seed, random_state=random_state)

    if isinstance(params, Draw):
        try:
            return params(rng)
        except Exception as e:
            raise RuntimeError(f"Failed to draw parameter `{path}`.") from e

    if isinstance(params, dict):
        return {
            key: draw_params(item, random_state=rng, path=f"{path}.{key}")
            for key, item in params.items()
        }

    if isinstance(params, list):
        return [
            draw_params(item, random_state=rng, path=f"{path}[{i}]")
            for i, item in enumerate(params)
        ]

    if isinstance(params, tuple):
        return tuple(
            draw_params(item, random_state=rng, path=f"{path}[{i}]")
            for i, item in enumerate(params)
        )

    return deepcopy(params)
