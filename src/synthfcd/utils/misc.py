"""
General-purpose helper functions.

All modules except `_validators` can import from this module.
"""

from __future__ import annotations

__all__ = [
    "callable_name",
    "dcopy_arrays",
    "get_leaf_paths_dict",
    "get_rng",
    "make_stable_id",
    "perf_step",
    "perf_total",
    "pretty_print_dict",
    "read_path_dict",
    "replace_paths_dict",
    "start_perf_tracking",
    "stop_perf_tracking",
]

import hashlib
import math
import time
import tracemalloc
import warnings
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, cast

import numpy as np

from synthfcd.utils._aliases import _RNGType
from synthfcd.utils._const import SEED_MAX
from synthfcd.utils._validators import validate_obj_type


def get_rng(
    seed: int | None = None, random_state: _RNGType | None = None, *, offset: int = 0
) -> _RNGType:
    """
    Get a random number generator based on provided seed or random state.

    Args:
        seed (int | None, optional):
            The seed to use for the random number generator. If ``None``, a default
            random number generator is used.
        random_state (np.random.RandomState | np.random.Generator | None, optional):
            The random number generator to use. If ``None``, a default random number
            generator is used.

    Returns:
        np.random.RandomState | np.random.Generator: The random number generator.

    Warnings:
        UserWarning:
            If both `seed` and `random_state` are provided, returning `random_state`.

    Raises:
        TypeError:
            If input parameters are not of the correct type.
    """
    validate_obj_type(seed, name="seed", target_type=(int, type(None)))
    validate_obj_type(
        random_state,
        name="random_state",
        target_type=(np.random.RandomState, np.random.Generator, type(None)),
    )
    validate_obj_type(offset, name="offset", target_type=int)

    if seed is None and random_state is None:
        return np.random.default_rng()

    if seed is not None and random_state is None:
        seed = (seed + offset) % SEED_MAX
        return np.random.default_rng(seed=seed)

    if seed is not None and random_state is not None:
        warnings.warn(
            "Both `seed` and `random_state` are provided; returning `random_state`.",
            UserWarning,
        )
        return random_state

    return cast(_RNGType, random_state)


def callable_name(obj: Any | None) -> str:
    """
    Get the name of a callable or a descriptive string if None/non-callable.

    Args:
        obj (Any | None): The callable or None.

    Returns:
        str: The name of the callable or a descriptive string if None/non-callable.
    """
    if obj is None:
        return "None"
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)


def _normalize_for_hash(value: Any) -> Any:
    """
    Recursively convert arbitrary objects into a deterministic, hashable structure.
    """
    # Primitive scalar types
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value

    # Float normalization (handle NaN/Inf deterministically)
    if isinstance(value, float):
        if math.isnan(value):
            return ("__float__", "nan")
        if math.isinf(value):
            return ("__float__", "inf" if value > 0 else "-inf")
        return ("__float__", repr(value))

    # NumPy scalar
    if isinstance(value, np.generic):
        return ("__npscalar__", str(value.dtype), _normalize_for_hash(value.item()))

    # NumPy array
    if isinstance(value, np.ndarray):
        h = hashlib.blake2b(digest_size=16)
        h.update(str(value.shape).encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(value.tobytes(order="C"))
        return ("__ndarray__", value.shape, str(value.dtype), h.hexdigest())

    # Mappings: sort by key repr for deterministic ordering
    if isinstance(value, dict):
        items = []
        for k in sorted(value.keys(), key=lambda x: repr(x)):
            items.append((_normalize_for_hash(k), _normalize_for_hash(value[k])))
        return ("__dict__", tuple(items))

    # Ordered iterables preserve order
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_normalize_for_hash(v) for v in value))

    # Unordered iterables sort normalized repr
    if isinstance(value, set):
        norm = [_normalize_for_hash(v) for v in value]
        norm_sorted = tuple(sorted(norm, key=lambda x: repr(x)))
        return ("set", norm_sorted)

    # Fallback for other objects
    return ("__repr__", type(value).__qualname__, repr(value))


def make_stable_id(payload: dict[str, Any], *, digest_size: int = 16) -> str:
    """
    Create a deterministic hash ID from a dictionary payload.

    - Stable across key order changes
    - Handles nested dict/list/tuple/set
    - Handles NumPy arrays by hashing shape/dtype/bytes
    """
    validate_obj_type(payload, name="payload", target_type=dict)

    canonical = _normalize_for_hash(payload)
    data = repr(canonical).encode("utf-8")

    h = hashlib.blake2b(digest_size=digest_size)
    h.update(data)
    return h.hexdigest()


def _format_bytes(num_bytes: int) -> str:
    """
    Format bytes into a human-readable unit string.
    """
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def start_perf_tracking(enabled: bool) -> tuple[float, tuple[int, int], bool]:
    """
    Initialize wall-time and traced-memory tracking.

    Returns:
        tuple[float, tuple[int, int], bool]:
            (start_time, (current_mem, peak_mem), started_tracing_here)
    """
    if not enabled:
        return 0.0, (0, 0), False

    started_tracing_here = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracing_here = True

    start_time = time.perf_counter()
    mem = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
    return start_time, mem, started_tracing_here


def perf_step(
    label: str,
    start_time: float,
    start_mem: tuple[int, int],
) -> tuple[str, tuple[int, int], float]:
    """
    Build a formatted performance message for one step.

    Args:
        label (str):
            The label of the step.
        start_time (float):
            The start time of the step.
        start_mem (tuple[int, int]):
            The start traced memory (current and peak) of the step.

    Returns:
        tuple[str, tuple[int, int], float]:
            (message, (current_mem, peak_mem), elapsed_seconds)
    """
    elapsed = time.perf_counter() - start_time
    if not tracemalloc.is_tracing():
        return f"[Perf] {label}: {elapsed:.3f}s", start_mem, elapsed

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    delta_current = current_mem - start_mem[0]
    delta_peak = peak_mem - start_mem[1]
    message = (
        f"[Perf] {label}: {elapsed:.3f}s | "
        f"traced current={_format_bytes(current_mem)} ({delta_current:+,d} B) | "
        f"traced peak={_format_bytes(peak_mem)} ({delta_peak:+,d} B)"
    )
    return message, (current_mem, peak_mem), elapsed


def perf_total(total_start: float, mem: tuple[int, int]) -> str:
    """
    Build a formatted total pipeline performance message.

    Args:
        total_start (float):
            The start wall-time of the total pipeline.
        mem (tuple[int, int]):
            The current and peak traced memory of the total pipeline.

    Returns:
        str:
            The formatted total pipeline performance message.
    """
    elapsed = time.perf_counter() - total_start
    return (
        f"[Perf] Pipeline total: {elapsed:.3f}s | "
        f"traced current={_format_bytes(mem[0])} | "
        f"traced peak={_format_bytes(mem[1])}"
    )


def stop_perf_tracking(started_tracing_here: bool) -> None:
    """
    Stop traced-memory tracking only if this call started it.

    Args:
        started_tracing_here (bool):
            Whether this call started the traced-memory tracking.
    """
    if started_tracing_here and tracemalloc.is_tracing():
        tracemalloc.stop()


def dcopy_arrays(
    masks_dict: dict[str, np.ndarray],
    /,
) -> dict[str, np.ndarray]:
    """
    Copy only the arrays in a dictionary.

    Args:
        masks_dict: The dictionary whose arrays to copy.

    Returns:
        dict[str, np.ndarray]: The output dictionary.

    Raises:
        TypeError: If the `masks_dict` is not a dictionary.
    """
    validate_obj_type(masks_dict, "masks_dict", dict)
    return {
        k: v.copy() if isinstance(v, np.ndarray) else v for k, v in masks_dict.items()
    }


def pretty_print_dict(obj: Any, indent: int = 2, width: int = 80) -> str:
    """
    Nicely format a dictionary or dataclass in a YAML-like style.

    - No outer braces
    - Lists are printed with '-'
    - NumPy scalars are converted to Python scalars
    """
    # Convert dataclass to dict recursively
    if is_dataclass(obj):
        obj = asdict(cast(Any, obj))

    if not isinstance(obj, dict):
        raise TypeError(
            f"`pretty_print_dict` expected a dict or dataclass, got {type(obj).__name__}"
        )

    # Convert NumPy scalars to Python scalars
    def to_py(x: Any) -> Any:
        return x.item() if isinstance(x, np.generic) else x

    # Format the dictionary
    def fmt(value: Any, level: int) -> list[str]:
        value = to_py(value)
        pad = " " * level
        if isinstance(value, dict):
            lines: list[str] = []
            for k, v in value.items():
                v = to_py(v)
                if isinstance(v, (dict, list, tuple)):
                    lines.append(f"{pad}{k}:")
                    lines.extend(fmt(v, level + indent))
                else:
                    lines.append(f"{pad}{k}: {v}")
            return lines
        if isinstance(value, (list, tuple)):
            lines = []
            for item in value:
                item = to_py(item)
                if isinstance(item, (dict, list, tuple)):
                    lines.append(f"{pad}-")
                    lines.extend(fmt(item, level + indent))
                else:
                    lines.append(f"{pad}- {item}")
            return lines
        return [f"{pad}{value}"]

    return "\n".join(fmt(obj, 0))


def get_leaf_paths_dict(params: dict[str, Any], *, prefix: str = "") -> set[str]:
    """
    Return dotted paths to all non-dictionary values in a nested dictionary.

    Args:
        params (dict[str, Any]):
            The nested dictionary to inspect.
        prefix (str, optional):
            Prefix prepended to returned paths. Used internally during recursion.
            Defaults to ``""``.

    Returns:
        set[str]:
            Dotted paths to all terminal non-dictionary values.
    """
    validate_obj_type(params, "params", dict)
    validate_obj_type(prefix, "prefix", str)

    paths: set[str] = set()

    for key, value in params.items():
        path = key if prefix == "" else f"{prefix}.{key}"

        if isinstance(value, dict):
            paths |= get_leaf_paths_dict(value, prefix=path)
        else:
            paths.add(path)

    return paths


def read_path_dict(params: dict[str, Any], path: str) -> Any:
    """
    Read a value from a dotted path in a dictionary.

    Args:
        params (dict[str, Any]):
            The parameter dictionary to query.
        path (str):
            Dotted path to the target value, for example
            ``"intensity_params.hyperintensity"``.

    Returns:
        Any:
            The value found at the requested path.

    Raises:
        ValueError:
            If ``path`` does not exist in ``params``
    """
    validate_obj_type(params, "params", dict)
    validate_obj_type(path, "path", str)

    value: Any = params

    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Path `{path}` does not exist in `params`.")
        value = value[part]

    return value


def replace_paths_dict(params: dict[str, Any], /) -> dict[str, Any]:
    """
    Replace any values of a parameters dict that are written as dotted paths with the
    values from the path.
    """
    params = deepcopy(params)

    def _walk(value: Any) -> Any:
        """
        Recursively find dotted paths and replace with values from `params`.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and "." in item:
                    value[key] = read_path_dict(params, item)
                else:
                    value[key] = _walk(item)
            return value
        return value

    return _walk(params)
