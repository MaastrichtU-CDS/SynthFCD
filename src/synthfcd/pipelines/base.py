"""
Base classes for all simulation pipelines.
"""

from __future__ import annotations

__all__ = [
    "LesionSimulationPipeline",
]

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np

from synthfcd.core.utils import postprocess_mask, warp
from synthfcd.pipelines.apply_effects import AppliesEffects
from synthfcd.pipelines.grow_target import ToTarget
from synthfcd.utils._aliases import (
    _CacheAccessType,
    _CacheKeyType,
    _DistanceType,
)
from synthfcd.utils._const import (
    APP_FIELD_BBOX_THRES,
    APP_FIELD_ROLLOFF,
    APP_FIELD_TYPE,
)
from synthfcd.utils._validators import (
    validate_3d_numpy_array,
    validate_literal_str,
)
from synthfcd.utils.misc import callable_name


class SimulationPipeline(ABC):
    """
    Abstract interface for any pipeline that applies some processing to an image.

    Each image must be coupled with a segmentation mask.

    Note:
        We acknowledge the possible desire to include certain anatomy-invariant effects.
        In such cases, we recommend that the segmentation mask is still provided,
        but internally ignored. This design choice is motivated by the philosophy
        of creating an anatomy-aware simulation engine rather than a general-purpose
        image processing pipeline.
    """

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the pipeline on the input image, coupled with a segmentation mask, and voxel
        spacing as a bridge between image and physical units.

        Accepts additional keyword arguments and is flexible to return any kind of
        result dictionary, including stats.
        """
        ...


class LesionSimulationPipeline(SimulationPipeline, AppliesEffects, ToTarget):
    """Base class for any lesion simulation pipeline:
    - Generates a target lesion mask
    - Generates a bounding box and a smooth application (blending) field for
      efficient local application of effects.
    - Provides helper functions to apply deformation and intensity effect pipelines.
    - Provides a helper function to efficiently apply the effects only around the
      generated target and smoothly blend with the original data.
    - Maintains caches for the target, application, and effects, for efficient
      re-use across multimodal data.
    """

    def _get_effects_group_and_params(
        self,
        effects_type: Literal["deformations", "intensity"],
        group: tuple[int, ...] | None = None,
        allow_disabled: bool = False,
    ) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
        """
        Centralized logic for group-to-effect (functions and parameters) routing and
        filtering out disabled effects to avoid any unnecessary computation of local
        application fields.

        Args:
            effects_type (Literal["deformations", "intensity"]):
                The type of effects to extract parameters for.
            group (tuple[int, ...] | None):
                The tuple of indices of the effects to extract parameters for.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.

        Returns:
            tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
                A tuple containing the filtered group indices and their parameters.

        Raises:
            ValueError:
                If any effect is disabled while ``allow_disabled=False``.
        """
        # Resolve effects and parameters for the requested effects type and group
        effects, params = self._route_group_to_effects(
            all_effects=(
                self.DEFORMATION_EFFECTS
                if effects_type == "deformations"
                else self.INTENSITY_EFFECTS
            ),
            all_params=(
                self.deformation_params
                if effects_type == "deformations"
                else self.intensity_params
            ),
            group=group,
        )
        if group is None:
            group = tuple(range(len(effects)))

        # Filter any disabled effects while preserving the order
        filtered_group = []
        filtered_params = []
        for idx, effect, params in zip(group, effects, params, strict=True):
            if not params.get("enable", True):
                if not allow_disabled:
                    raise ValueError(
                        f"Effect `{callable_name(effect)}` is disabled while "
                        f"`allow_disabled=False`; re-run with `allow_disabled=True`."
                    )
                continue
            filtered_group.append(idx)
            filtered_params.append(params)

        return tuple(filtered_group), tuple(filtered_params)

    def _get_app_field_params(
        self, params: tuple[dict[str, Any], ...], **kwargs
    ) -> dict[str, Any]:
        """
        Find the application field parameters from provided settings.

        The following logic is used:
        - If `app_field_type` is provided in ANY of the effect-specific parameters
          or in the keyword arguments, then it will be used. If provided in multiple
          places, the consistency of the values is checked.
        - Same for `app_rolloff` and `bbox_thres`, each of them independently.
        - If no parameters are provided, default values are used.

        Args:
            params (tuple[dict[str, Any], ...]):
                The list of effect-specific parameters.
            **kwargs (Any):
                Additional keyword arguments to pass to all effects.

        Returns:
            dict[str, Any]:
                A dictionary containing the application field parameters
                as keys `field_type`, `rolloff`, and `bbox_thres`.
        """
        # Scan kwargs first
        scan_for = ("app_field_type", "app_rolloff", "bbox_thres")
        found = {k: kwargs.get(k, None) for k in scan_for}

        # Scan effect-specific parameters and check consistency
        for param in params:
            for key in scan_for:
                found_value = param.get(key, None)
                # Not found; skip
                if found_value is None:
                    continue
                # Found; first encounter
                if found[key] is None:
                    found[key] = found_value
                    continue
                # Found; already encountered, should match
                if found[key] != found_value:
                    raise ValueError(
                        f"Inconsistent `{key}` in effect-specific parameters/"
                        f"keyword arguments: {found[key]} vs {found_value}."
                    )

        # Set defaults if nothing is found
        default_values = {
            "app_field_type": APP_FIELD_TYPE,
            "app_rolloff": APP_FIELD_ROLLOFF,
            "bbox_thres": APP_FIELD_BBOX_THRES,
        }
        for key, value in found.items():
            if value is None:
                warnings.warn(
                    f"No `{key}` found in effect-specific parameters ({params}) or keyword "
                    f"arguments ({kwargs}); using default `{default_values[key]}`.",
                    UserWarning,
                )
                found[key] = default_values[key]

        return {
            "field_type": found["app_field_type"],
            "rolloff": found["app_rolloff"],
            "bbox_thres": found["bbox_thres"],
        }

    def apply_effects_locally(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        target: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        effects_type: Literal["deformations", "intensity"],
        group: tuple[int, ...] | None = None,
        allow_disabled: bool = False,
        *,
        use_cache: _CacheAccessType = "auto",
        cache_key: _CacheKeyType | None = "auto",
        crop_kwargs: tuple[str, ...] | Literal["auto"] = "auto",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Wrapper function for applying effects (deformation or intensity) to the image
        and segmentation mask only within the bounding box of the target region using a
        smooth application field. Results are stitched back to the original inputs.

        Notes:
            - The effect-specific parameters are called internally in the ``effects_fn``
              so it's recommended that all modifiable arrays are provided as ``**kwargs``.
            - If not "auto" or "None", a single cache key is used for accessing the cached
              application and effect fields. It is the caller's responsibility to ensure that
              the cache key is compatible with the effects to be applied.
            - Input arrays are *NOT* modified in place.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            target (np.ndarray):
                The input target to use for locally applying the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image for computing the application field.
            effects_type (Literal["deformations", "intensity"]):
                The type of effects to apply; ``"deformations"`` for ``DEFORMATION_EFFECTS``
                and ``"intensity"`` for ``INTENSITY_EFFECTS``.
            group (tuple[int, ...] | None):
                The indices to apply a subset of effects of the given type; set to
                ``None`` to apply all.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the application field.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            cache_key (str | Literal["auto"] | None, optional):
                The key to use for writing/retrieving the cached application field. Defaults to
                ``"auto"``; i.e., generate key from provided target, field type, rolloff,
                bbox threshold, and spacing.
            crop_kwargs (tuple[str, ...] | Literal["auto"], optional):
                The tuple of keys to also crop to the bounding box for applying the effects.
                If "auto", all numpy arrays in ``**kwargs`` are cropped.
            **kwargs (Any):
                Additional keyword arguments to pass to all effects. Also used to search for
                application field parameters and externally provided application fields.

        Returns:
            dict[str, Any]:
                A dictionary containing the output image, segmentation mask, target,
                and effect fields.
        """
        validate_3d_numpy_array(image, "image", dtype=np.floating)
        validate_3d_numpy_array(
            seg_mask, "seg_mask", dtype=np.integer, shape=image.shape
        )
        validate_3d_numpy_array(target, "target", shape=image.shape)
        validate_literal_str(
            effects_type, "effects_type", ("deformations", "intensity")
        )

        # Extract effect-specific parameters while filtering disabled effects
        group, params = self._get_effects_group_and_params(
            effects_type=effects_type,
            group=group,
            allow_disabled=allow_disabled,
        )
        # Early exit if all disabled
        if len(group) == 0:
            return {
                "out_image": image.copy(),
                "out_seg_mask": seg_mask.copy(),
                "out_target": target.copy(),
                "effect_field": (
                    [] if effects_type == "deformations" else np.zeros_like(image)
                ),
            }

        # Search for application field parameters; set defaults if not provided
        app_field_params = self._get_app_field_params(params, **kwargs)

        # Compute application field for the requested target
        bbox, app_field = self.compute_application_field(
            target=target,
            field_type=app_field_params["field_type"],
            rolloff=app_field_params["rolloff"],
            bbox_thres=app_field_params["bbox_thres"],
            spacing=spacing,
            use_cache=use_cache,
            cache_key=cache_key,
        )

        # Extract bbox for image and seg mask
        tmp_image = image[bbox].copy()
        tmp_seg_mask = seg_mask[bbox].copy()
        tmp_target = target[bbox]  # not passed to effects

        # Crop any additional arrays in kwargs to bbox
        tmp_kwargs = deepcopy(kwargs)
        for k, v in tmp_kwargs.items():
            # Skip non-numpy arrays
            if not isinstance(v, np.ndarray):
                if crop_kwargs != "auto" and k in crop_kwargs:
                    raise ValueError(
                        f"Key {k} in `crop_kwargs` for group {group} is not a numpy array; "
                        "cannot crop to bbox."
                    )
                continue

            # Skip while guarding against accidental omissions
            if crop_kwargs != "auto" and k not in crop_kwargs:
                if v.shape != tmp_image.shape:
                    raise ValueError(
                        f"Key '{k}' was excluded from cropping but does not match the bbox shape "
                        f"({v.shape} vs. {tmp_image.shape}); adjust the settings (`app_field_type`, "
                        f"`app_rolloff`, `bbox_thres`) to match {k} or include `{k}` in `crop_kwargs`."
                    )
                continue

            # Crop while guarding against mismatches that will silently propagate to effects
            if v.shape != image.shape:
                raise ValueError(
                    f"Cannot crop {k} to bbox, as it has different shape than image "
                    f"for which the bbox was generated ({v.shape} vs. {image.shape})."
                )
            tmp_kwargs[k] = v[bbox]

        # Pass application field to each effect as anatomical constraint
        # if not externally provided
        if not ("app_field" in tmp_kwargs or any("app_field" in p for p in params)):
            tmp_kwargs["app_field"] = app_field

        # Apply effects
        effects_fn = (
            super().apply_deformations
            if effects_type == "deformations"
            else super().apply_intensity
        )
        result = effects_fn(
            image=tmp_image,
            seg_mask=tmp_seg_mask,
            spacing=spacing,
            group=group,
            allow_disabled=False,  # disabled effects already filtered
            use_cache=use_cache,
            cache_key=cache_key,
            **tmp_kwargs,
        )

        # Warp target if apply_fn returns deformation fields
        if (
            isinstance(result["effect_field"], list)
            and len(result["effect_field"]) != 0  # skip no-ops
        ):
            tmp_target = warp(
                [tmp_target],
                result["effect_field"],
                interp_order=0,
                mode="nearest",
                clip=True,
            )[0]
            # Minimally postprocess target
            tmp_target = postprocess_mask(
                tmp_target,
                smooth_sigma=None,
                closing=None,
                fill_holes=True,
                keep_largest_component=True,
                spacing=spacing,
            )

        # Stitch back to original inputs
        out_image = image.copy()
        out_image[bbox] = result["out_image"]
        out_seg_mask = seg_mask.copy()
        if "out_seg_mask" in result:
            out_seg_mask[bbox] = result["out_seg_mask"]

        out_target = target.copy()
        out_target[bbox] = tmp_target

        return {
            "out_image": out_image,
            "out_seg_mask": out_seg_mask,
            "out_target": out_target,
            "effect_field": result["effect_field"],
        }
