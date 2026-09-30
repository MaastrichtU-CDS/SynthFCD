"""
Helper functions for initializing parameter dictionaries.
"""

from __future__ import annotations

__all__ = [
    "sample_neg_trunc_lognormal",
    "sample_trunc_lognormal",
    "seed_params",
]

import warnings
from collections.abc import Iterator
from typing import Any, Literal, cast

import numpy as np
from scipy.stats import truncnorm

from synthfcd.utils._aliases import _RNGType
from synthfcd.utils._const import SEED_MAX
from synthfcd.utils._validators import RealNoBool, validate_obj_type
from synthfcd.utils.misc import get_rng, read_path_dict


def sample_trunc_lognormal(
    range: tuple[float | int, float | int],
    mean: float | None = None,
    cv: float = 0.5,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
) -> float:
    """
    Sample a value from a truncated log-normal distribution.

    The distribution is bounded over the interval ``[low, high]`` and parametrized
    by a target mean and coefficient of variation (CV). If ``mean`` is ``None``,
    the geometric mean of the interval is used as the central tendency.

    Args:
        range (tuple[float | int, float | int]):
            Lower and upper bounds of the support interval. Must satisfy
            ``range[0] < range[1]``.
        mean (float | int | None, optional):
            Desired arithmetic mean of the distribution. If ``None``, the geometric
            mean of ``range`` is used. Defaults to ``None``.
        cv (float | int, optional):
            Coefficient of variation (standard deviation divided by mean) in
            log-normal space. Defaults to ``0.5``.
        random_seed (int | None, optional):
            The random seed for the random number generator. If ``None``, the global
            NumPy RNG is used.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            Random number generator for reproducibility. If ``None``, the global
            NumPy RNG is used.

    Returns:
        float:
            A sampled value drawn from the truncated log-normal distribution.

    Raises:
        ValueError:
            If ``range`` is invalid, unordered, or zero-width.
        TypeError:
            If argument types are incompatible (e.g., non-numeric parameters or
            unsupported RNG types).
    """
    validate_obj_type(range, "range", tuple)
    if len(range) != 2:
        raise ValueError("`range` must be a tuple of length 2")
    if not isinstance(range[0], RealNoBool) or not isinstance(range[1], RealNoBool):
        raise TypeError("`range` must be a tuple of real numbers")
    if range[0] >= range[1]:
        raise ValueError(
            "`range` must be a tuple of length 2 with `range[0] < range[1]`"
        )

    validate_obj_type(mean, "mean", (RealNoBool, type(None)))
    validate_obj_type(cv, "cv", RealNoBool)

    rng = get_rng(seed=random_seed, random_state=random_state)

    low, high = float(range[0]), float(range[1])

    # Compute log‑normal parameters
    if mean is None:
        mean = np.sqrt(low * high)  # geometric midpoint

    sigma = np.sqrt(np.log(float(cv) ** 2 + 1))
    mu = np.log(cast(float, mean)) - 0.5 * sigma**2

    # Compute truncation in log‑space
    a = (np.log(low) - mu) / sigma
    b = (np.log(high) - mu) / sigma

    # Sample from the truncated normal in log‑space
    tn = truncnorm(a, b, loc=mu, scale=sigma)
    sample_log = tn.rvs(random_state=rng)

    # Transform back to the original scale
    return float(np.exp(sample_log))


def sample_neg_trunc_lognormal(
    range: tuple[float | int, float | int],
    **kwargs,
) -> float:
    """
    Sample a value from a truncated log-normal distribution with negative values.

    Args:
        range (tuple[float | int, float | int]):
            Lower and upper bounds of the support interval. Must satisfy
            ``range[0] < range[1]``.
        **kwargs:
            Keyword arguments to pass to ``sample_trunc_lognormal``.

    Returns:
        float:
            A sampled value drawn from the truncated log-normal distribution with negative values.
    """
    low, high = range
    return -sample_trunc_lognormal((-high, -low), **kwargs)


def seed_params(
    rng: _RNGType,
    params: dict[str, Any],
    *,
    paths_to_seed: list[str] | Literal["all"] = "all",
) -> None:
    """
    Inject a unique random seed as a ``random_seed`` key into provided paths of nested
    parameter dictionaries using a random number generator.

    Notes:
        - Modifies ``params`` in place.
        - If ``paths_to_seed="all"``, seeds terminal dictionaries only.
        - If paths are provided, each path must point to a dictionary.

    Args:
        rng (np.random.Generator | np.random.RandomState):
            The random number generator to use.
        params (dict[str, Any]):
            The nested parameter dictionary to seed.
        paths_to_seed (list[str] | Literal["all"], optional):
            The paths to the parameter dictionaries to seed. If ``"all"``, all terminal
            dictionaries are seeded. If a list of paths is provided, only the dictionaries
            at the given paths are seeded.

    Raises:
        TypeError: If argument types are invalid.
        ValueError: If ``params`` is not a dictionary, or ``paths_to_seed``
            is empty or does not contain valid dictionary paths.
    """
    validate_obj_type(rng, "rng", (np.random.Generator, np.random.RandomState))
    validate_obj_type(params, "params", dict)

    if paths_to_seed == "all":
        targets = list(_iter_terminal_dicts(params))
    else:
        validate_obj_type(paths_to_seed, "paths_to_seed", list)
        if len(paths_to_seed) == 0:
            raise ValueError("`paths_to_seed` must be a non-empty list")

        targets = []
        for path in paths_to_seed:
            validate_obj_type(path, "paths_to_seed", str)
            target = read_path_dict(params, path)
            if not isinstance(target, dict):
                raise ValueError(f"Path `{path}` must point to a dictionary.")
            targets.append((path, target))

    for path, target in targets:
        if "random_seed" in target:
            warnings.warn(
                f"Skipping random seed generation for `{path}` as "
                "`random_seed` is already provided.",
                UserWarning,
            )
            continue

        if "random_state" in target:
            warnings.warn(
                f"Skipping random seed generation for `{path}` as "
                "`random_state` is already provided.",
                UserWarning,
            )
            continue

        if isinstance(rng, np.random.Generator):
            target["random_seed"] = int(rng.integers(0, SEED_MAX))
        else:
            target["random_seed"] = int(rng.randint(0, SEED_MAX))


def _iter_terminal_dicts(
    params: dict[str, Any],
    *,
    path: str = "",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yield terminal dictionaries in a nested parameter dictionary.

    Args:
        params (dict[str, Any]):
            The nested parameter dictionary to traverse.
        path (str, optional):
            The dotted path to ``params`` within the original root dictionary.
            Used internally during recursion. Defaults to ``""``.

    Yields:
        tuple[str, dict[str, Any]]:
            The dotted path to each terminal dictionary and the dictionary itself.
    """
    child_dicts = {
        key: value for key, value in params.items() if isinstance(value, dict)
    }

    if not child_dicts:
        yield path, params
        return

    for key, value in child_dicts.items():
        child_path = key if path == "" else f"{path}.{key}"
        yield from _iter_terminal_dicts(value, path=child_path)


def _normalize_draw_output(value: Any) -> Any:
    """
    Normalize NumPy scalar/container outputs from RNG samplers to plain Python values.

    This keeps preset materialization friendlier for downstream validators that expect
    bool/int/float rather than np.bool_/np.integer/np.floating.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {k: _normalize_draw_output(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_normalize_draw_output(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_normalize_draw_output(v) for v in value)

    return value
