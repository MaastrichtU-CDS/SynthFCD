"""
Functions and mixins for growing a target region (e.g., lesion mask).
"""

from __future__ import annotations

__all__ = [
    "ToTarget",
    "grow_target",
]

import warnings
from copy import deepcopy
from inspect import isabstract, signature
from typing import Any, ClassVar, Literal, cast

import numpy as np

from synthfcd.core.utils import (
    bbox_from_mask,
    generate_application_field,
)
from synthfcd.pipelines.utils import (
    readable_cache_key,
    resolve_cache_settings,
)
from synthfcd.utils._aliases import (
    _CacheAccessType,
    _CacheKeyType,
    _DistanceType,
    _GrowTargetFnType,
    _GrowTargetResultType,
)
from synthfcd.utils._validators import (
    validate_3d_numpy_array,
    validate_dict_and_keys,
    validate_obj_type,
)
from synthfcd.utils.misc import callable_name


def grow_target(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    growth_fns: list[_GrowTargetFnType] | tuple[_GrowTargetFnType, ...],
    growth_params: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    allow_disabled: bool = False,
    **kwargs,
) -> _GrowTargetResultType:
    """
    Grow the target region by sequentially applying the target growth functions.

    Note:
        Callers can pass ``enable=False`` in each growth function parameters to
        disable the particular function.

    Args:
        seg_mask (np.ndarray):
            The input segmentation mask to use for the target growth.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image for computing the application field.
        growth_fns (list[Callable] | tuple[Callable, ...]):
            The list of target growth functions to apply.
        growth_params (list[dict[str, Any]] | tuple[dict[str, Any], ...]):
            The list of parameters to pass to each target growth function as keyword arguments.
        allow_disabled (bool, optional):
            Whether to allow any target growth function to be disabled. Defaults to ``False``.
        **kwargs (Any):
            Any keyword arguments to pass to all target growth functions.

    Returns:
        dict[str, Any]:
            A dictionary containing the following key-value pairs:
            - ``target``: The target region after applying the growth functions.
            - ``growth_stats``: A dictionary of growth statistics, keyed by the
                callable name of each growth function.

    Raises:
        TypeError:
            If the ``growth_fns`` or ``growth_params`` are not of the correct type.
        ValueError:
            If the ``growth_fns`` or ``growth_params`` are not of the correct length,
            or if the ``growth_fns`` are not callable objects, or if any growth
            function does not return a valid result.
    """
    validate_obj_type(growth_fns, "growth_fns", (list, tuple))
    if len(growth_fns) == 0:
        raise ValueError("`growth_fns` must be a non-empty list or tuple")
    if not all(callable(e) for e in growth_fns):
        raise ValueError("`growth_fns` must be a list or tuple of callable objects")

    validate_obj_type(growth_params, "growth_params", (list, tuple))
    if len(growth_params) != len(growth_fns):
        raise ValueError(
            "`growth_params` must be a list or tuple of the same length as `growth_fns`"
        )
    if not all(isinstance(p, dict) for p in growth_params):
        raise ValueError("`growth_params` must be a list or tuple of dictionaries")

    validate_obj_type(allow_disabled, "allow_disabled", bool)

    target_mask = np.zeros_like(seg_mask, dtype=np.bool_)
    growth_fn_names = [callable_name(e) for e in growth_fns]
    growth_stats: dict[str, Any] = {}

    for i, (fn, params) in enumerate(zip(growth_fns, growth_params, strict=True)):
        if not params.get("enable", True):
            if not allow_disabled:
                raise ValueError(
                    f"Growth function `{growth_fn_names[i]}` is disabled while "
                    "`allow_disabled=False`; re-run with `allow_disabled=True`."
                )
            continue

        # Call growth fn
        call_kwargs = {
            "seg_mask": seg_mask,
            "spacing": spacing,
            **params,
            **kwargs,
        }
        try:
            signature(fn).bind(**call_kwargs)
        except TypeError as e:
            raise TypeError(
                f"`{growth_fn_names[i]}` failed to bind arguments. "
                f"Provided params keys={list(params.keys())}, kwargs keys={list(kwargs.keys())}."
            ) from e
        result = fn(**call_kwargs)

        # Ensure valid return type, target and growth stats
        validate_dict_and_keys(
            result, f"result of `{growth_fn_names[i]}`", ("target", "growth_stats")
        )
        validate_3d_numpy_array(
            result["target"],
            f"`target` of `{growth_fn_names[i]}`",
            shape=seg_mask.shape,
        )
        validate_obj_type(
            result["growth_stats"], f"`growth_stats` of `{growth_fn_names[i]}`", dict
        )

        target_mask += result["target"]
        growth_stats[growth_fn_names[i]] = result["growth_stats"]

    return {
        "target": target_mask,
        "growth_stats": growth_stats,
    }


class ToTarget:
    """
    Mixin class for any pipeline that operates on a target region.

    Attributes that must be defined by subclasses:
    - TARGET_GROWTH (tuple[Callable, ...]):
        A tuple of callable objects that sequentially grow the target region.
    """

    TARGET_GROWTH: ClassVar[tuple[_GrowTargetFnType, ...]]

    def __init_subclass__(cls, **kwargs):
        """
        Ensures subclasses define a sequence of target growth functions; this is only
        enforced on concrete classes.
        """
        super().__init_subclass__(**kwargs)

        if isabstract(cls):
            return

        if not hasattr(cls, "TARGET_GROWTH"):
            raise TypeError(f"{cls.__name__} must define `TARGET_GROWTH`")
        if not isinstance(cls.TARGET_GROWTH, tuple):
            raise TypeError(f"{cls.__name__} must provide `TARGET_GROWTH` as a tuple")
        if len(cls.TARGET_GROWTH) == 0:
            raise TypeError(
                f"{cls.__name__} must provide at least one target growth function."
            )
        if not all(callable(e) for e in cls.TARGET_GROWTH):
            raise TypeError(
                f"{cls.__name__} must provide `TARGET_GROWTH` as callable objects"
            )

    def __init__(
        self,
        *args,
        growth_params: list[dict[str, Any]],
        **kwargs,
    ) -> None:
        """
        Initialize the target growth pipeline.

        Args:
            growth_params (list[dict[str, Any]]):
                The list of parameters to pass to each target growth function as keyword arguments.

        Raises:
            TypeError:
                If the ``growth_params`` are not of the correct type.
            ValueError:
                If the ``growth_params`` are not of the correct length,
                or if the ``growth_params`` are not dictionaries.
        """
        super().__init__(*args, **kwargs)

        self.growth_params = growth_params

        # Initialize cache
        self._cached_target: dict[str, np.ndarray] = {}
        self._cached_growth_stats: dict[str, dict[str, Any]] = {}
        self._cached_app_fields: dict[
            str, tuple[tuple[slice, slice, slice], np.ndarray]
        ] = {}

    @property
    def growth_params(self) -> tuple[dict[str, Any], ...]:
        """
        Retrieve the parameters of the target growth functions.
        """
        return deepcopy(self._growth_params)

    @growth_params.setter
    def growth_params(self, value: list[dict[str, Any]]) -> None:
        """
        Safely store the parameters of the target growth functions.
        """
        validate_obj_type(value, "growth_params", list)
        if len(value) != len(self.TARGET_GROWTH):
            raise ValueError(
                f"`growth_params` must match the length of `TARGET_GROWTH`: "
                f"{len(self.TARGET_GROWTH)}; got {len(value)}."
            )
        if not all(isinstance(p, dict) for p in value):
            raise ValueError("`growth_params` must be a list of dictionaries")
        self._growth_params = tuple(deepcopy(value))

        if hasattr(self, "_cached_target"):
            self.clear_cached_target()

    def _resolve_group_to_growth_fns(
        self,
        all_growth_fns: tuple[_GrowTargetFnType, ...],
        all_growth_params: tuple[dict[str, Any], ...],
        group: tuple[int, ...] | None,
    ) -> tuple[tuple[_GrowTargetFnType, ...], tuple[dict[str, Any], ...]]:
        """
        Safely route the requested group of target growth functions to the corresponding
        functions and parameters.

        Args:
            all_growth_fns (tuple[_GrowTargetFnType, ...]):
                The tuple of all target growth functions.
            all_growth_params (tuple[dict[str, Any], ...]):
                The tuple of all target growth parameters.
            group (tuple[int, ...] | None):
                The tuple of indices of the target growth functions to run, or ``None``
                to run functions sequentially.

        Returns:
            tuple[tuple[_GrowTargetFnType, ...], tuple[dict[str, Any], ...]]:
                A tuple containing the tuple of target growth functions and their parameters.

        Raises:
            ValueError:
                If the ``group`` is not a tuple of integers or if the indices
                are not within the range of the ``all_growth_fns``.
        """
        validate_obj_type(all_growth_fns, "all_growth_fns", tuple)
        validate_obj_type(all_growth_params, "all_growth_params", tuple)
        if len(all_growth_fns) != len(all_growth_params):
            raise ValueError(
                "`all_growth_fns` and `all_growth_params` must have the same length"
            )

        if group is None:
            return all_growth_fns, all_growth_params

        validate_obj_type(group, "group", tuple)
        if not all(isinstance(i, int) for i in group):
            raise ValueError("`group` must be a tuple of integers")
        if not all(0 <= i < len(all_growth_fns) for i in group):
            raise ValueError(
                f"indices in `group` must be within the range [0, {len(all_growth_fns)})"
            )
        return (
            tuple(all_growth_fns[i] for i in group),
            tuple(all_growth_params[i] for i in group),
        )

    def grow_target(
        self,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        group: tuple[int, ...] | None = None,
        allow_disabled: bool = False,
        use_cache: _CacheAccessType = "auto",
        cache_key: _CacheKeyType | None = "auto",
        **kwargs,
    ) -> _GrowTargetResultType:
        """
        Grow the target region by sequentially applying the target growth functions.

        Args:
            seg_mask (np.ndarray):
                The input segmentation mask to use for the target growth.
            spacing (tuple[float | int, float | int, float | int]):
                The voxel spacing of the image for computing the application field.
            group (tuple[int, ...] | None, optional):
                The tuple of indices of the target growth functions to apply, or ``None``
                to apply all ``TARGET_GROWTH`` sequentially. Defaults to ``None``.
            allow_disabled (bool, optional):
                Whether to allow any target growth function to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the target region.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            cache_key (str | Literal["auto"] | None, optional):
                The key to use for writing/retrieving the cached target region. Defaults to
                ``"auto"``; i.e., generate key from provided spacing, group, and kwargs.
            **kwargs (Any):
                Any keyword arguments to pass to all target growth functions.

        Returns:
            dict[str, Any]:
                A dictionary containing the target region and growth statistics.

        Raises:
            ValueError:
                If the ``use_cache`` is ``True`` and no cached target is found.
            ValueError:
                If the grown target region is empty.
        """
        validate_3d_numpy_array(seg_mask, "seg_mask", dtype=np.integer)
        validate_obj_type(allow_disabled, "allow_disabled", bool)

        # Resolve target growth functions to run
        growth_fns, growth_params = self._resolve_group_to_growth_fns(
            all_growth_fns=self.TARGET_GROWTH,
            all_growth_params=self.growth_params,
            group=group,
        )

        # Configure cache read/write
        cache_key, read_cache, write_cache = resolve_cache_settings(
            use_cache=use_cache,
            cache_key=cache_key,
            cache=self._cached_target,
            spacing=spacing,
            group=group,
            **kwargs,
        )

        # Early return if target exists in cache
        if read_cache:
            if cache_key not in self._cached_target:
                raise ValueError(
                    f"No cached target region available for provided "
                    f"{readable_cache_key(cache_key, [callable_name(e) for e in growth_fns])}; "
                    "re-run with `use_cache='auto'` or `False`."
                )
            return {
                "target": self._cached_target[cache_key].copy(),
                "growth_stats": deepcopy(self._cached_growth_stats.get(cache_key, {})),
            }

        # Grow target and guard against all-empty
        result = grow_target(
            seg_mask=seg_mask,
            spacing=spacing,
            growth_fns=growth_fns,
            growth_params=growth_params,
            allow_disabled=allow_disabled,
            **kwargs,
        )

        # Exit without cache update no growth functions were run
        # (catches all-disabled; empty targets from enabled growth functions still pass)
        if len(result["growth_stats"]) == 0:
            return {
                "target": np.zeros_like(seg_mask, dtype=np.bool_),
                "growth_stats": {},
            }

        if not result["target"].any():
            warnings.warn(
                "Grown target region is empty and will be cached as such; make "
                "sure this is acceptable.",
                UserWarning,
            )

        # Update cache with new target region and growth statistics
        if write_cache:
            self._cached_target[cast(str, cache_key)] = result["target"]
            self._cached_growth_stats[cast(str, cache_key)] = deepcopy(
                result["growth_stats"]
            )

        return {
            "target": result["target"].copy(),
            "growth_stats": result["growth_stats"],
        }

    def compute_application_field(
        self,
        target: np.ndarray,
        field_type: Literal["inward", "outward"],
        rolloff: _DistanceType,
        bbox_thres: float,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        use_cache: _CacheAccessType = "auto",
        cache_key: _CacheKeyType | None = "auto",
    ) -> tuple[tuple[slice, slice, slice], np.ndarray]:
        """
        Compute a bounding box and an application field for the target region.

        Note:
            Floating-point targets are interpreted as application fields, so make
            sure that the ``TARGET_GROWTH`` functions follow this convention.

        Args:
            target (np.ndarray):
                The target region to generate the application field for.
            field_type (Literal["inward", "outward"]):
                The type of application field to generate.
            rolloff (float):
                The rolloff distance for the application field type.
            bbox_thres (float):
                The threshold for the bounding box.
            spacing (tuple[float | int, float | int, float | int]):
                The voxel spacing of the image.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the application field.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            cache_key (str | Literal["auto"] | None, optional):
                The key to use for writing/retrieving the cached application field. Defaults to
                ``"auto"``; i.e., generate key from provided target, field type, rolloff,
                bbox threshold, and spacing.

        Returns:
            tuple[tuple[slice, slice, slice], np.ndarray]:
                A tuple containing the bounding box and application field.

        Raises:
            ValueError:
                If the ``use_cache`` is ``False`` and no target region is available.
            ValueError:
                If the ``use_cache`` is ``True`` and the cached application field
                is not available.
            ValueError:
                If the ``bbox_thres`` is not between 0 and 1.
        """
        # Check if the target is available
        validate_3d_numpy_array(target, "target")
        if not target.any():
            raise ValueError("Target region is empty")

        validate_obj_type(bbox_thres, "bbox_thres", float)
        if bbox_thres < 0.0 or bbox_thres > 1.0:
            raise ValueError("`bbox_thres` must be between 0 and 1")

        # Configure cache read/write
        cache_key, read_cache, write_cache = resolve_cache_settings(
            use_cache=use_cache,
            cache_key=cache_key,
            cache=self._cached_app_fields,
            target=target,
            field_type=field_type,
            rolloff=rolloff,
            bbox_thres=bbox_thres,
            spacing=spacing,
        )

        # Early return if app field exists in cache
        if read_cache:
            if cache_key not in self._cached_app_fields:
                cache_key_msg = (
                    "target, field type, rolloff, bbox threshold, and spacing"
                    if cache_key == "auto"
                    else f"key {cache_key!r}"
                )
                raise ValueError(
                    f"No cached application field available for provided {cache_key_msg}; "
                    "re-run with `use_cache='auto'` or `False`."
                )
            bbox, app_field = self._cached_app_fields[cache_key]
            return bbox, app_field.copy()

        # Generate new application field
        if not np.issubdtype(target.dtype, np.floating):
            # Generate bbox around target +/- rolloff to speed up the computation
            # of Euclidean distances inside `generate_application_field`.
            pad = (
                int(rolloff / spacing[0]) + 1,
                int(rolloff / spacing[1]) + 1,
                int(rolloff / spacing[2]) + 1,
            )
            bbox = bbox_from_mask(target, pad=pad)

            # Generate smooth application field around target
            app_field = np.zeros_like(target, dtype=np.float32)
            app_field[bbox] = generate_application_field(
                mask=target[bbox],
                field_type=field_type,
                rolloff_mm=rolloff,
                spacing=spacing,
            )
        else:
            # Interpret floating-point targets as application fields
            app_field = target

        # Threshold the application field to get the final bbox
        bbox = bbox_from_mask(app_field > bbox_thres)

        # Update cache with new application field
        if write_cache:
            if cache_key in self._cached_app_fields:
                warnings.warn(
                    f"Cached application field already exists for key {cache_key!r}; "
                    "overwriting.",
                    UserWarning,
                )
            self._cached_app_fields[cast(str, cache_key)] = (bbox, app_field[bbox])

        return bbox, app_field[bbox].copy()

    def clear_cached_target(self) -> None:
        """
        Clear the cached target region, growth statistics, and application fields.
        """
        self._cached_target = {}
        self._cached_growth_stats = {}
        self._cached_app_fields = {}
