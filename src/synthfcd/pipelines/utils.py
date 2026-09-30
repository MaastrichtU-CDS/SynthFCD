"""
Helper functions for creating the pipelines.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from synthfcd.core.utils import generate_application_field
from synthfcd.utils._aliases import _DistanceType
from synthfcd.utils._const import (
    TEXTURE_SUPPORT_ROLLOFF_FROM_BLUR,
    TEXTURE_SUPPORT_THRES_FROM_BLUR,
)
from synthfcd.utils._validators import validate_obj_type
from synthfcd.utils.misc import make_stable_id


def texture_app_field_from_blur_effect(
    blur_effect: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    threshold: float = TEXTURE_SUPPORT_THRES_FROM_BLUR,
    rolloff_mm: _DistanceType = TEXTURE_SUPPORT_ROLLOFF_FROM_BLUR,
) -> np.ndarray:
    """
    Build a soft ``[0, 1]`` texture support field from boundary-blurring deltas.

    Min-max normalizes ``|blur_effect|``, uses the non-zero support as a mask, and
    generates an inward application field within that mask.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("`threshold` must be between 0.0 and 1.0")

    blur_mag = np.abs(blur_effect, dtype=np.float64)
    peak = float(blur_mag.max())
    if peak <= 0.0:
        return np.zeros(blur_effect.shape, dtype=np.float32)

    normalized = blur_mag / peak
    mask = normalized > threshold
    return generate_application_field(
        mask=mask,
        spacing=spacing,
        field_type="inward",
        rolloff_mm=rolloff_mm,
    )


def resolve_cache_settings(
    use_cache: bool | Literal["auto"],
    cache_key: str | Literal["auto"] | None,
    cache: dict[str, Any],
    **kwargs,
) -> tuple[str | None, bool, bool]:
    """
    Resolve the cache read/write settings.

    Args:
        use_cache (bool | Literal["auto"]):
            The mode to use for the cache.
            - ``"auto"``:read the cache only if requested key is available;
                always write the result to the cache.
            - ``True``: always use the cache for reading and writing.
            - ``False``: don't use the cache for any reading or writing.
        cache_key (str | Literal["auto"] | None):
            The key to use for accessing the cache.
            - ``"auto"``: generate a key from the provided kwargs.
            - ``str``: use the provided key.
            - ``None``: empty key to denote no cache usage.
        cache (dict[str, Any]):
            The cache dictionary to use.
        **kwargs (Any):
            Any keyword arguments to use for generating the cache key.

    Returns:
        tuple[str | None, bool, bool]:
            The cache key, whether to read the cache, and whether to write
            the result to the cache.

    Raises:
        TypeError:
            If the ``use_cache`` is not a boolean or ``"auto"``, or if
            the ``cache_key`` is not a string.
    """
    if not any([use_cache == "auto", isinstance(use_cache, bool)]):
        raise TypeError("`use_cache` must be a boolean or 'auto'")

    # Early exit if cache not used
    if not use_cache:
        return None, False, False

    # Resolve cache key
    if cache_key == "auto":
        cache_key = make_stable_id(kwargs)
    else:
        validate_obj_type(cache_key, "cache_key", str)

    # Resolve cache read/write
    if use_cache == "auto":
        return cache_key, cache_key in cache, True

    return cache_key, True, True


def readable_cache_key(cache_key: str | None, fn_names: list[str]) -> str:
    """
    Generate a readable cache key for message printing.

    Args:
        cache_key (str):
            The cache key.
        fn_names (list[str]):
            The human-readable names of the functions within an effects/
            target growth function group.

    Returns:
        str:
            The readable cache key.
    """
    return (
        f"spacing, group ({', '.join(fn_names)}), and keyword arguments"
        if cache_key == "auto"
        else f"key {cache_key!r}"
    )
