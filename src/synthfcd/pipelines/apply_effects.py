"""
Functions and mixins for applying deformation and intensity effects to an image.
"""

from __future__ import annotations

__all__ = [
    "AppliesEffects",
    "apply_effects",
]

import warnings
from copy import deepcopy
from inspect import isabstract, signature
from typing import Any, ClassVar, cast

import numpy as np

from synthfcd.core.utils import warp
from synthfcd.pipelines.utils import (
    readable_cache_key,
    resolve_cache_settings,
)
from synthfcd.utils._aliases import (
    _CacheAccessType,
    _CacheKeyType,
    _DeformationResultType,
    _DistanceType,
    _EffectFnType,
    _IntensityResultType,
)
from synthfcd.utils._validators import (
    validate_3d_numpy_array,
    validate_dict_and_keys,
    validate_obj_type,
)
from synthfcd.utils.misc import callable_name


def apply_effects(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    effects: list[_EffectFnType] | tuple[_EffectFnType, ...],
    effect_params: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    allow_disabled: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Apply a list of effects to an image; delegates image and parameter checking to
    individual effects. Also performs return value validation.

    Notes:
        - The function aggregates all effects required to reproduce the pipeline in a single
        ``effect_fields`` key, comprising both deformation and intensity fields.
        - The caller is responsible for distinguishing warping and additive intensity fields.
        - Callers can pass ``enable=False`` in each effect parameters to disable the particular
          effect.

    Args:
        image (np.ndarray):
            The input image to apply the effects to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for the effects.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image for computing the application field.
        effects (list[Callable] | tuple[Callable, ...]):
            The list of effects to apply, each of which is a callable object.
        effect_params (list[dict[str, Any]] | tuple[dict[str, Any], ...]):
            The list of parameters to pass to each effect as keyword arguments;
            must match the length of the ``effects`` list.
        allow_disabled (bool, optional):
            Whether to allow any effect to be disabled. Defaults to ``False``.
        **kwargs (Any): Any keyword arguments to pass to all effects.

    Raises:
        TypeError:
            If the ``effects`` or ``effect_params`` are not of the correct type.
        ValueError:
            If the ``effects`` or ``effect_params`` are not of the correct length,
            or if the ``effects`` are not callable objects, or if any effect
            does not return a valid result.
    """
    validate_obj_type(effects, "effects", (list, tuple))
    if len(effects) == 0:
        raise ValueError("`effects` must be a non-empty list or tuple")
    if not all(callable(e) for e in effects):
        raise ValueError("`effects` must be a list or tuple of callable objects")

    validate_obj_type(effect_params, "effect_params", (list, tuple))
    if len(effect_params) != len(effects):
        raise ValueError(
            "`effect_params` must be a list or tuple of the same length as `effects`"
        )
    if not all(isinstance(p, dict) for p in effect_params):
        raise ValueError("`effect_params` must be a list or tuple of dictionaries")

    validate_obj_type(allow_disabled, "allow_disabled", bool)

    image = image.copy()
    seg_mask = seg_mask.copy()
    effect_names = [callable_name(e) for e in effects]
    effect_fields: list[
        np.ndarray | list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = []

    for i, (fn, params) in enumerate(zip(effects, effect_params, strict=True)):
        if not params.get("enable", True):
            if not allow_disabled:
                raise ValueError(
                    f"Effect `{effect_names[i]}` is disabled while `allow_disabled=False`; "
                    "re-run with `allow_disabled=True`."
                )
            continue

        # Call effect fn
        call_kwargs = {
            "image": image,
            "seg_mask": seg_mask,
            "spacing": spacing,
            **params,
            **kwargs,
        }
        try:
            signature(fn).bind(**call_kwargs)
        except TypeError as e:  # explicitly catch binding errors
            raise TypeError(
                f"`{effect_names[i]}` failed to bind arguments. "
                f"Provided params keys={list(params.keys())}, kwargs keys={list(kwargs.keys())}."
            ) from e
        result = fn(**call_kwargs)

        # Ensure valid return type, output image (+ segmentation mask) and effect field
        validate_dict_and_keys(
            result, f"result of `{effect_names[i]}`", ("out_image", "effect_field")
        )
        validate_3d_numpy_array(result["out_image"], "out_image", shape=image.shape)
        if "out_seg_mask" in result:
            validate_3d_numpy_array(
                result["out_seg_mask"],
                f"`out_seg_mask` of `{effect_names[i]}`",
                shape=seg_mask.shape,
            )

        # Ensure appropriate type and shape of effect field
        if isinstance(result["effect_field"], list):  # deformation field
            for field in result["effect_field"]:
                if not isinstance(field, tuple) or len(field) != 3:
                    raise ValueError(
                        f"Invalid deformation field from `{effect_names[i]}`; "
                        f"expected a list of tuples of 3 numpy arrays."
                    )
                for dim in field:
                    validate_3d_numpy_array(
                        dim,
                        f"deformation field of `{effect_names[i]}`",
                        shape=image.shape,
                    )

        else:  # intensity field
            validate_3d_numpy_array(
                result["effect_field"],
                f"intensity field of `{effect_names[i]}`",
                shape=image.shape,
            )

        image = result["out_image"]
        if "out_seg_mask" in result:
            seg_mask = result["out_seg_mask"]
        effect_fields.append(result["effect_field"])

    return {
        "out_image": image,
        "out_seg_mask": seg_mask,
        "effect_fields": effect_fields,
    }


class AppliesEffects:
    """
    Mixin class for any pipeline that applies deformation and intensity effects to an
    image.

    Supports caching deformation and intensity fields for faster re-use across
    multimodal data.

    Notes:
        - It is recommended that provided effect parameters are not input-dependent,
          since they will be called automatically for each effect. Pass input-dependent
          arguments directly when calling either `apply_deformations` or `apply_intensities`.
        - If such input-dependent arguments become too general for the effects to be applied,
          then this is a sign that a more specific group of effects should be applied, using
          the ``group`` argument to `apply_deformations` or `apply_intensities`.
        - The effect groups should be considered as the finest granularity for externally controlling
          effect-specific behavior; e.g., restricting effects to a local region.
    """

    DEFORMATION_EFFECTS: ClassVar[tuple[_EffectFnType, ...]]
    INTENSITY_EFFECTS: ClassVar[tuple[_EffectFnType, ...]]

    def __init_subclass__(cls, **kwargs):
        """
        Ensures subclasses define sequences of deformation and intensity effects; this
        is only enforced on concrete classes.
        """
        super().__init_subclass__(**kwargs)

        if isabstract(cls):
            return

        # Check effects
        if not hasattr(cls, "DEFORMATION_EFFECTS"):
            raise TypeError(
                f"{cls.__name__} must define `DEFORMATION_EFFECTS`; set to `tuple()` "
                "to disable deformation effects."
            )
        if not isinstance(cls.DEFORMATION_EFFECTS, tuple):
            raise TypeError(
                f"{cls.__name__} must provide `DEFORMATION_EFFECTS` as a tuple"
            )
        if not all(callable(e) for e in cls.DEFORMATION_EFFECTS):
            raise TypeError(
                f"{cls.__name__} must provide `DEFORMATION_EFFECTS` as callable objects"
            )

        if not hasattr(cls, "INTENSITY_EFFECTS"):
            raise TypeError(
                f"{cls.__name__} must define `INTENSITY_EFFECTS`; set to `tuple()` "
                "to disable intensity effects."
            )
        if not isinstance(cls.INTENSITY_EFFECTS, tuple):
            raise TypeError(
                f"{cls.__name__} must provide `INTENSITY_EFFECTS` as a tuple"
            )
        if not all(callable(e) for e in cls.INTENSITY_EFFECTS):
            raise TypeError(
                f"{cls.__name__} must provide `INTENSITY_EFFECTS` as callable objects"
            )

        if len(cls.DEFORMATION_EFFECTS) == 0 and len(cls.INTENSITY_EFFECTS) == 0:
            raise TypeError(
                f"{cls.__name__} must provide at least one deformation or intensity effect"
            )

    def __init__(
        self,
        *args,
        deformation_params: list[dict[str, Any]],
        intensity_params: list[dict[str, Any]],
        **kwargs,
    ) -> None:
        """
        Initialize the deformation and intensity effects pipeline.

        Args:
            deformation_params (list[dict[str, Any]]):
                The list of parameters to pass to each deformation effect as keyword arguments.
            intensity_params (list[dict[str, Any]]):
                The list of parameters to pass to each intensity effect as keyword arguments.

        Raises:
            TypeError:
                If the ``deformation_params`` or ``intensity_params`` are not of the correct type.
            ValueError:
                If the ``deformation_params`` or ``intensity_params`` are not of the correct length,
                or if the ``deformation_params`` or ``intensity_params`` are not dictionaries.
        """
        super().__init__(*args, **kwargs)

        self.deformation_params = deformation_params
        self.intensity_params = intensity_params

        # Initialize cached fields
        self._cached_deformation_fields: dict[
            str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        self._cached_intensity_fields: dict[str, np.ndarray] = {}

    @property
    def deformation_params(self) -> tuple[dict[str, Any], ...]:
        """
        Retrieve the parameters of deformation effects.
        """
        return deepcopy(self._deformation_params)

    @property
    def intensity_params(self) -> tuple[dict[str, Any], ...]:
        """
        Retrieve the parameters of intensity effects.
        """
        return deepcopy(self._intensity_params)

    @deformation_params.setter
    def deformation_params(self, value: list[dict[str, Any]]) -> None:
        """
        Safely store the parameters of deformation effects.
        """
        validate_obj_type(value, "deformation_params", list)
        if len(value) != len(self.DEFORMATION_EFFECTS):
            raise ValueError(
                f"`deformation_params` must match the length of `DEFORMATION_EFFECTS`: "
                f"{len(self.DEFORMATION_EFFECTS)}; got {len(value)}."
            )
        if not all(isinstance(p, dict) for p in value):
            raise ValueError("`deformation_params` must be a list of dictionaries")
        self._deformation_params = tuple(deepcopy(value))

        # Sync cached fields
        if hasattr(self, "_cached_deformation_fields"):
            self.clear_cached_fields(deformation_only=True)

    @intensity_params.setter
    def intensity_params(self, value: list[dict[str, Any]]) -> None:
        """
        Safely store the parameters of intensity effects.
        """
        validate_obj_type(value, "intensity_params", list)
        if len(value) != len(self.INTENSITY_EFFECTS):
            raise ValueError(
                f"`intensity_params` must match the length of `INTENSITY_EFFECTS`: "
                f"{len(self.INTENSITY_EFFECTS)}; got {len(value)}."
            )
        if not all(isinstance(p, dict) for p in value):
            raise ValueError("`intensity_params` must be a list of dictionaries")
        self._intensity_params = tuple(deepcopy(value))

        # Sync cached fields
        if hasattr(self, "_cached_intensity_fields"):
            self.clear_cached_fields(intensity_only=True)

    def _route_group_to_effects(
        self,
        all_effects: tuple[_EffectFnType, ...],
        all_params: tuple[dict[str, Any], ...],
        group: tuple[int, ...] | None,
    ) -> tuple[tuple[_EffectFnType, ...], tuple[dict[str, Any], ...]]:
        """
        Safely route the requested group of effects to the corresponding functions and
        parameters.

        Args:
            all_effects (tuple[_EffectFnType, ...]):
                The tuple of all effect functions.
            all_params (tuple[dict[str, Any], ...]):
                The tuple of all effect parameters.
            group (tuple[int, ...] | None):
                The tuple of indices of the effects to apply, or ``None`` to apply all
                effects sequentially.

        Returns:
            tuple[tuple[_EffectFnType, ...], tuple[dict[str, Any], ...]]:
                A tuple containing the tuple of effect functions and their parameters.

        Raises:
            TypeError:
                If the ``all_effects`` or ``all_params`` are not of the correct type.
            ValueError:
                If the ``all_effects`` and ``all_params`` are not of the same length,
                or if the ``group`` indices are not within allowed bounds.
        """
        validate_obj_type(all_effects, "all_effects", tuple)
        validate_obj_type(all_params, "all_params", tuple)
        if len(all_effects) != len(all_params):
            raise ValueError("`all_effects` and `all_params` must have the same length")

        if group is None:
            return all_effects, all_params

        validate_obj_type(group, "group", tuple)
        if not all(isinstance(i, int) for i in group):
            raise ValueError("`group` must be a tuple of integers")
        if not all(0 <= i < len(all_effects) for i in group):
            raise ValueError(
                f"indices in `group` must be within the range [0, {len(all_effects)})"
            )
        return (
            tuple(all_effects[i] for i in group),
            tuple(all_params[i] for i in group),
        )

    def apply_deformations(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        group: tuple[int, ...] | None = None,
        allow_disabled: bool = False,
        use_cache: _CacheAccessType = "auto",
        cache_key: _CacheKeyType | None = "auto",
        **kwargs,
    ) -> _DeformationResultType:
        """
        Apply the deformation effects to the image and segmentation mask.

        Note:
            If an effect returns an empty deformation field, it is treated as a valid
            no-op. The result will still be cached even if all effects return empty fields
            to avoid re-computation.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image for computing the application field.
            group (tuple[int, ...] | None, optional):
                The tuple of indices of the effects to apply, or ``None`` to apply all
                ``DEFORMATION_EFFECTS`` sequentially. Defaults to ``None``.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving deformation fields.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            cache_key (str | Literal["auto"] | None, optional):
                The key to use for writing/retrieving the cached fields. Defaults to
                ``"auto"``; i.e., generate key from provided spacing, group, and kwargs.
            **kwargs (Any):
                Any keyword arguments to pass to all effects.

        Returns:
            dict[str, Any]:
                A dictionary containing the output image, segmentation mask, and effect fields.

        Warnings:
            UserWarning:
                If no deformation effects are defined.

        Raises:
            ValueError:
                If the ``use_cache`` is ``True`` and the cached deformation fields are not available.
        """
        validate_3d_numpy_array(image, "image", dtype=np.floating)
        validate_3d_numpy_array(
            seg_mask,
            "seg_mask",
            dtype=np.integer,
            shape=image.shape,
        )
        validate_obj_type(allow_disabled, "allow_disabled", bool)

        # Resolve effects and parameters for the requested group
        effects, params = self._route_group_to_effects(
            all_effects=self.DEFORMATION_EFFECTS,
            all_params=self.deformation_params,
            group=group,
        )

        # Configure cache read/write
        cache_key, read_cache, write_cache = resolve_cache_settings(
            use_cache=use_cache,
            cache_key=cache_key,
            cache=self._cached_deformation_fields,
            spacing=spacing,
            group=group,
            **kwargs,
        )

        # Early return if deformation fields exist in cache
        if read_cache:
            if cache_key not in self._cached_deformation_fields:
                raise ValueError(
                    f"No cached deformation fields available for provided "
                    f"{readable_cache_key(cache_key, [callable_name(e) for e in effects])}; "
                    "re-run with `use_cache='auto'` or `False`."
                )

            deformation_fields = self._cached_deformation_fields[cache_key]
            if len(deformation_fields) == 0:  # no-op fields
                return {
                    "out_image": image,
                    "out_seg_mask": seg_mask,
                    "effect_field": [],
                }

            outputs = warp(
                images=[image, seg_mask],
                deformation_fields=deformation_fields,
                interp_order=[3, 0],
            )
            return {
                "out_image": outputs[0],
                "out_seg_mask": outputs[1],
                "effect_field": deepcopy(deformation_fields),
            }

        # Apply effects with new deformation fields
        outputs = apply_effects(
            image=image,
            seg_mask=seg_mask,
            spacing=spacing,
            effects=effects,
            effect_params=params,
            allow_disabled=allow_disabled,
            **kwargs,
        )

        # Exit without cache update if no deformation fields present
        # (catches all-disabled; no-op fields from enabled effects still pass)
        if not outputs["effect_fields"]:
            return {
                "out_image": image,
                "out_seg_mask": seg_mask,
                "effect_field": [],
            }

        # Validate and aggregate output fields
        out_fields = []
        for i, field in enumerate(outputs["effect_fields"]):
            if not isinstance(field, list):  # rest guaranteed from `apply_effects`
                raise ValueError(
                    f"Invalid result from deformation function "
                    f"`{callable_name(effects[i])}`; expected a list of tuples of 3 numpy arrays."
                )
            if len(field) == 0:
                warnings.warn(
                    f"Effect `{callable_name(effects[i])}` returned an empty deformation field; "
                    "accepting it as a no-op."
                )
            out_fields.extend(field)

        # Update the cached deformation fields
        if write_cache:
            self._cached_deformation_fields[cast(str, cache_key)] = out_fields

        return {
            "out_image": outputs["out_image"],
            "out_seg_mask": outputs["out_seg_mask"],
            "effect_field": deepcopy(out_fields),
        }

    def apply_intensity(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        group: tuple[int, ...] | None = None,
        allow_disabled: bool = False,
        use_cache: _CacheAccessType = "auto",
        cache_key: _CacheKeyType | None = "auto",
        **kwargs,
    ) -> _IntensityResultType:
        """
        Apply the intensity effects to the image.

        Note:
            Caching intensity fields only works when the effects are purely additive.
            Non-additive intensity effects should be grouped separately.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image for computing the application field.
            group (tuple[int, ...] | None, optional):
                The tuple of indices of the effects to apply, or ``None`` to apply all
                ``INTENSITY_EFFECTS`` sequentially. Defaults to ``None``.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving intensity fields.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            cache_key (str | Literal["auto"] | None, optional):
                The key to use for writing/retrieving the cached fields. Defaults to
                ``"auto"``; i.e., generate key from provided spacing, group, and kwargs.
            **kwargs (Any):
                Any keyword arguments to pass to all effects.

        Returns:
            dict[str, Any]:
                A dictionary containing the output image, segmentation mask, and effect fields.

        Warnings:
            UserWarning:
                If no intensity effects are defined.

        Raises:
            ValueError:
                If the ``use_cache`` is ``True`` and the cached intensity fields are not available.
        """
        validate_3d_numpy_array(image, "image", dtype=np.floating)
        validate_3d_numpy_array(
            seg_mask,
            "seg_mask",
            dtype=np.integer,
            shape=image.shape,
        )
        validate_obj_type(allow_disabled, "allow_disabled", bool)

        # Resolve effects and parameters for the requested group
        effects, params = self._route_group_to_effects(
            all_effects=self.INTENSITY_EFFECTS,
            all_params=self.intensity_params,
            group=group,
        )

        # Configure cache read/write
        cache_key, read_cache, write_cache = resolve_cache_settings(
            use_cache=use_cache,
            cache_key=cache_key,
            cache=self._cached_intensity_fields,
            spacing=spacing,
            group=group,
            **kwargs,
        )

        # Early return if intensity field exists in cache
        if read_cache:
            if cache_key not in self._cached_intensity_fields:
                raise ValueError(
                    f"No cached intensity fields available for provided "
                    f"{readable_cache_key(cache_key, [callable_name(e) for e in effects])}; "
                    "re-run with `use_cache='auto'` or `False`."
                )

            intensity_field = self._cached_intensity_fields[cache_key]
            if intensity_field.shape != image.shape:
                raise ValueError(
                    f"Cached intensity field for provided "
                    f"{readable_cache_key(cache_key, [callable_name(e) for e in effects])} "
                    f"does not match image shape ({intensity_field.shape} vs. {image.shape}); "
                    f"re-run with `use_cache='auto'` or `False`."
                )

            image = image + intensity_field

            return {
                "out_image": image,
                "effect_field": intensity_field.copy(),
            }

        # Apply effects with new intensity fields
        outputs = apply_effects(
            image=image,
            seg_mask=seg_mask,
            spacing=spacing,
            effects=effects,
            effect_params=params,
            allow_disabled=allow_disabled,
            **kwargs,
        )

        # Exit without cache update if no intensity fields present
        # (catches all-disabled; no-op fields from enabled effects still pass)
        if not outputs["effect_fields"]:
            return {
                "out_image": image,
                "effect_field": np.zeros_like(image),
            }

        # Validate and aggregate output fields
        out_field = np.zeros_like(image)
        for i, field in enumerate(outputs["effect_fields"]):
            # Ensure intensity effect; rest guaranteed from `apply_effects`.
            if not isinstance(field, np.ndarray):
                raise ValueError(
                    f"Invalid result from intensity function `{callable_name(effects[i])}`; "
                    f"expected a numpy array; got a list."
                )
            if not np.any(field):
                warnings.warn(
                    f"Effect `{callable_name(effects[i])}` returned an empty intensity field; "
                    "accepting it as a no-op."
                )
            out_field += field

        # Update the cached intensity field
        if write_cache:
            self._cached_intensity_fields[cast(str, cache_key)] = out_field

        return {
            "out_image": outputs["out_image"],
            "effect_field": out_field.copy(),
        }

    def clear_cached_fields(
        self,
        *,
        deformation_only: bool = False,
        intensity_only: bool = False,
    ) -> None:
        """
        Clear the cached deformation and intensity fields.
        """
        if not deformation_only:
            self._cached_intensity_fields = {}
        if not intensity_only:
            self._cached_deformation_fields = {}
