"""
Concrete lesion simulation pipelines.
"""

from __future__ import annotations

__all__ = [
    "SimpleFCD",
    "simple_fcd_simulator",
]

import time
from typing import Any

import numpy as np

from synthfcd.core import (
    abnormal_gyration,
    boundary_blurring,
    cortical_thickening,
    grow_random_lesion,
    hyperintensity,
    sulcal_widening,
    texture_restoration,
)
from synthfcd.pipelines.base import LesionSimulationPipeline
from synthfcd.pipelines.utils import texture_app_field_from_blur_effect
from synthfcd.utils import (
    SynthSegLabel,
    callable_name,
    perf_step,
    perf_total,
    start_perf_tracking,
    stop_perf_tracking,
)
from synthfcd.utils._aliases import (
    _CacheAccessType,
    _DistanceType,
    _HemisphereType,
    _LabelEnumType,
)
from synthfcd.utils._validators import (
    validate_3d_numpy_array,
    validate_dict_and_keys,
    validate_obj_type,
)


class SimpleFCD(LesionSimulationPipeline):
    """
    Concrete FCD simulation pipeline that first generates a lesion mask and then locally
    applies the effects.

    "Simple" refers to the effects not being applied in a neurodevelopmentally-
    plausible manner, as the GM-WM body is generated in advance. A more complex
    implementation would involve generating a smooth "migration" field from the
    ventricles to the cortical lesion and applying the effects along the trajectory. A
    binary lesion mask could be generated afterwards.
    """

    TARGET_GROWTH = (grow_random_lesion,)
    DEFORMATION_EFFECTS = (abnormal_gyration, cortical_thickening, sulcal_widening)
    INTENSITY_EFFECTS = (boundary_blurring, texture_restoration, hyperintensity)

    def apply_deformations(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        target: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        hemisphere: _HemisphereType | None = None,
        allow_disabled: bool = False,
        use_cache: _CacheAccessType = "auto",
        label_enum: _LabelEnumType = SynthSegLabel,
    ) -> dict[str, Any]:
        """
        Apply the deformation effects locally to the image, segmentation mask, and
        target.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            target (np.ndarray):
                The input target to use for locally applying the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image for computing the application field.
            hemisphere (Literal["left", "right"] | None, optional):
                The hemisphere to restrict the deformation effects to. Defaults to ``None``.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the deformation fields.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            label_enum (type[LabelEnum], optional):
                The label enumeration to use for reading ``seg_mask``.
                Defaults to ``synthfcd.utils.SynthSegLabel``.

        Returns:
            dict[str, Any]:
                A dictionary containing the warped image, segmentation mask, and target.

        Raises:
            ValueError:
                If the ``target`` does not match the ``image`` shape.
        """
        # Ensure target matches images; detailed checks between image and seg_mask
        # are performed in the effects_fn.
        if isinstance(image, np.ndarray):
            validate_3d_numpy_array(target, "target", shape=image.shape)

        # Abnormal gyration; put first so that anatomically-constrained
        # effects are not affected by this free-form deformation
        result = self.apply_effects_locally(
            image=image,
            seg_mask=seg_mask,
            target=target,
            spacing=spacing,
            effects_type="deformations",
            group=(0,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="gyration",
            hemisphere=hemisphere,
            label_enum=label_enum,
        )

        # Cortical thickening
        result = self.apply_effects_locally(
            image=result["out_image"],
            seg_mask=result["out_seg_mask"],
            target=result["out_target"],
            spacing=spacing,
            effects_type="deformations",
            group=(1,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="thickening",
            hemisphere=hemisphere,
            label_enum=label_enum,
        )

        # Fill-in "pushed" cortical voxels from cortical expansion
        result["out_target"] |= target

        # Cortical thinning (sulcal widening)
        result = self.apply_effects_locally(
            image=result["out_image"],
            seg_mask=result["out_seg_mask"],
            target=result["out_target"],
            spacing=spacing,
            effects_type="deformations",
            group=(2,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="thinning",
            hemisphere=hemisphere,
            label_enum=label_enum,
        )

        return {
            "out_image": result["out_image"],
            "out_seg_mask": result["out_seg_mask"],
            "out_target": result["out_target"],
        }

    def apply_intensity(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        target: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        hemisphere: _HemisphereType | None = None,
        allow_disabled: bool = False,
        use_cache: _CacheAccessType = "auto",
        label_enum: _LabelEnumType = SynthSegLabel,
    ) -> dict[str, Any]:
        """
        Apply the intensity effects locally to the image.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            target (np.ndarray):
                The input target to use for locally applying the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image for computing the application field.
            hemisphere (Literal["left", "right"] | None, optional):
                The hemisphere to restrict the intensity effects to. Defaults to ``None``.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the intensity fields.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.
            label_enum (type[LabelEnum], optional):
                The label enumeration to use for reading ``seg_mask``.
                Defaults to ``synthfcd.utils.SynthSegLabel``.

        Returns:
            dict[str, Any]:
                A dictionary containing only the output image.

        Raises:
            ValueError:
                If the ``target`` does not match the ``image`` shape.
        """
        # Ensure target matches images; detailed tests between image and seg_mask
        # are performed in the effects_fn.
        if isinstance(image, np.ndarray):
            validate_3d_numpy_array(target, "target", shape=image.shape)

        orig_image = image.copy()

        # GM-WM boundary blurring
        result = self.apply_effects_locally(
            image=image,
            seg_mask=seg_mask,
            target=target,
            spacing=spacing,
            effects_type="intensity",
            group=(0,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="blurring",
            hemisphere=hemisphere,
            label_enum=label_enum,
        )

        # Texture restoration: inward support from the blurred region.
        texture_app_field = texture_app_field_from_blur_effect(
            result["effect_field"], spacing
        )
        result = self.apply_effects_locally(
            image=result["out_image"],
            seg_mask=seg_mask,  # should not change in intensity effects
            target=target,  # same
            spacing=spacing,
            effects_type="intensity",
            group=(1,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="texture",
            orig=orig_image,  # for ``texture_restoration``
            app_field=texture_app_field,
            hemisphere=hemisphere,
            crop_kwargs=("orig",),  # ``app_field`` already cropped
            label_enum=label_enum,
        )

        # Hyperintensity
        result = self.apply_effects_locally(
            image=result["out_image"],
            seg_mask=seg_mask,
            target=target,
            spacing=spacing,
            effects_type="intensity",
            group=(2,),
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            cache_key="hyperintensity",
            hemisphere=hemisphere,
            label_enum=label_enum,
        )

        return {
            "out_image": result["out_image"],
        }

    def __call__(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        label_enum: _LabelEnumType = SynthSegLabel,
        allow_disabled: bool = False,
        verbose: bool = False,
        use_cache: _CacheAccessType = "auto",
    ) -> dict[str, Any]:
        """
        Forward pass of the FCD simulation pipeline for a given image, segmentation
        mask, and spacing.

        The following steps are performed in order:
        - Grow FCD lesion
        - Apply local deformation effects to lesion; separately for each effect,
          as each effect modifies the lesion mask.
        - Apply local intensity effects to lesion; separately for each effect,
          as each effect spreads differently outward/inward from the lesion boundary.
        - Cache the lesion, local deformation, and local intensity fields for faster
          re-use across future calls with the same parameters.

        Args:
            image (np.ndarray):
                The input image to apply the effects to.
            seg_mask (np.ndarray):
                The input segmentation mask to use for the effects.
            spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
                The voxel spacing of the image (and segmentation mask).
            label_enum (type[LabelEnum], optional):
                The label enumeration to use for reading ``seg_mask``.
                Defaults to ``synthfcd.utils.SynthSegLabel``.
            allow_disabled (bool, optional):
                Whether to allow any effect to be disabled. Defaults to ``False``.
            verbose (bool, optional):
                Whether to print pipeline starts and performance measures.
                Defaults to ``False``.
            use_cache (bool | Literal["auto"], optional):
                Whether to use the cache for writing/retrieving the deformation and intensity fields.
                Defaults to ``"auto"``; i.e., use when available, otherwise only update.

        Returns:
            dict[str, Any]:
                A dictionary containing the output image, segmentation mask, target,
                original target before any deformations, and growth statistics.
        """
        total_start, step_mem, started_tracing_here = start_perf_tracking(verbose)
        step_start = total_start

        # Grow FCD lesion
        target = self.grow_target(
            seg_mask=seg_mask,
            spacing=spacing,
            use_cache=use_cache,
            cache_key="target",
            allow_disabled=False,  # always grow a lesion
            label_enum=label_enum,
        )
        orig_target = target["target"].copy()

        growth_key = callable_name(self.TARGET_GROWTH[0])
        hemisphere = target["growth_stats"].get(growth_key, {}).get("hemisphere", None)
        if hemisphere not in ("left", "right"):
            raise ValueError(
                f"Internal contract violated: growth statistics in `SimpleFCD` should "
                f"contain a valid hemisphere (`left` or `right`) under `{growth_key}` "
                "key."
            )

        if verbose:
            print("=" * 80)
            message, step_mem, _ = perf_step(
                label="Grow lesion",
                start_time=step_start,
                start_mem=step_mem,
            )
            print(message)
            step_start = time.perf_counter()

        # Apply local deformation effects
        result = self.apply_deformations(
            image=image,
            seg_mask=seg_mask,
            target=target["target"],
            spacing=spacing,
            hemisphere=hemisphere,
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            label_enum=label_enum,
        )
        out_seg_mask = result["out_seg_mask"]
        out_target = result["out_target"]

        if verbose:
            message, step_mem, _ = perf_step(
                label="Apply local deformations",
                start_time=step_start,
                start_mem=step_mem,
            )
            print(message)
            step_start = time.perf_counter()

        # Apply local intensity effects
        result = self.apply_intensity(
            image=result["out_image"],
            seg_mask=result["out_seg_mask"],
            target=result["out_target"],
            spacing=spacing,
            hemisphere=hemisphere,
            allow_disabled=allow_disabled,
            use_cache=use_cache,
            label_enum=label_enum,
        )

        if verbose:
            message, step_mem, _ = perf_step(
                label="Apply local intensities",
                start_time=step_start,
                start_mem=step_mem,
            )
            print(message)
            print(perf_total(total_start=total_start, mem=step_mem))
            stop_perf_tracking(started_tracing_here)

        growth_stats = target["growth_stats"].get("grow_random_lesion", {})
        if verbose and growth_stats:
            print(f"Growth statistics: {growth_stats}")

        return {
            "out_image": result["out_image"],
            "out_seg_mask": out_seg_mask,
            "orig_target": orig_target,
            "out_target": out_target,
            "stats": growth_stats,
        }

    def clear(self) -> None:
        """
        Clear all cached fields.
        """
        self.clear_cached_target()
        self.clear_cached_fields()


def simple_fcd_simulator(
    images: list[np.ndarray],
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    growth_params: dict[str, Any],
    deformation_params: dict[str, dict[str, Any]],
    intensity_params: list[dict[str, dict[str, Any]]],
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
    allow_disabled: bool = True,
    verbose: bool = True,
    use_cache: _CacheAccessType = "auto",
) -> dict[str, Any]:
    """
    Simulate a simple FCD lesion on a given list of images and segmentation mask.

    Args:
        images (list[np.ndarray]):
            The list of input images to simulate the lesions on. The images are assumed
            to belong to the same subject and should be co-registered to the same space.
        seg_mask (np.ndarray):
            The input segmentation mask to use for the effects.
        spacing (tuple[_DistanceType, _DistanceType, _DistanceType]):
            The voxel spacing of the images. It should be the same for all images.
        growth_params (dict[str, Any]):
            The parameters to pass to the ``grow_random_lesion`` function for generating
            the synthetic FCD lesion shape.
        deformation_params (dict[str, dict[str, Any]]):
            The parameters to pass to each deformation effect. Expected keys:
            - ``"cortical_thickening"``
        intensity_params (list[dict[str, dict[str, Any]]]):
            The parameters to pass to each intensity effect. Should be provided as a list,
            matching the length of ``images``, so that unique intensity effects are applied
            per sequence (e.g., different noise, or hypo/hyper-intensity in T1w vs FLAIR).
            Expected keys per list item:
            - ``"boundary_blurring"``
            - ``"texture_restoration"``
            - ``"hyperintensity"``
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.
        allow_disabled (bool, optional):
            Whether to allow any effect to be disabled. Defaults to ``True``.
        verbose (bool, optional):
            Whether to print pipeline logs and performance measures.
            Defaults to ``True``.
        use_cache (bool | Literal["auto"], optional):
            Whether to use the cache for writing/retrieving the deformation and intensity fields.
            Defaults to ``"auto"``; i.e., use when available, otherwise only update.

    Returns:
        dict[str, Any]:
            A dictionary containing the following keys:
            - ``"out_images"``: the list of output images after the simulation.
            - ``"out_seg_mask"``: the output segmentation mask after the simulation.
            - ``"orig_target"``: the original generated lesion mask before any effects.
            - ``"out_target"``: the resulting lesion mask after all effects.

    Raises:
        TypeError:
            If input parameters are not of the correct type.
        ValueError:
            If images or seg_mask are not matching, or if any input parameters
            do not match the expected keys or/and length.
        ValueError:
            If the output target and anatomy (e.g., ``seg_mask``) do not match across
            images
    """
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=np.integer)

    # All images should match in shape the segmentation mask
    validate_obj_type(images, "images", list)
    if len(images) == 0:
        raise ValueError("`images` must be a non-empty list")
    for i, img in enumerate(images):
        validate_3d_numpy_array(
            img, f"images[{i}]", dtype=np.floating, shape=seg_mask.shape
        )

    validate_obj_type(growth_params, "growth_params", dict)

    # Ensure all deformation parameters are present
    deformation_names = [callable_name(k) for k in SimpleFCD.DEFORMATION_EFFECTS]
    validate_dict_and_keys(
        deformation_params,
        "deformation_params",
        tuple(deformation_names),
        strict_match=True,
    )

    # Ensure all intensity parameters are present
    intensity_names = [callable_name(k) for k in SimpleFCD.INTENSITY_EFFECTS]
    validate_obj_type(intensity_params, "intensity_params", list)
    if len(intensity_params) != len(images):
        raise ValueError(
            "`intensity_params` must be a list of the same length as `images`"
        )
    for i, p in enumerate(intensity_params):
        validate_dict_and_keys(
            p,
            f"intensity_params[{i}]",
            tuple(intensity_names),
            strict_match=True,
        )

    # Instantiate the pipeline, use first image's intensity params
    pipeline = SimpleFCD(
        growth_params=[growth_params],
        deformation_params=[deformation_params[n] for n in deformation_names],
        intensity_params=[intensity_params[0][n] for n in intensity_names],
    )

    # Simulate the lesion on first image
    result = pipeline(
        image=images[0],
        seg_mask=seg_mask,
        spacing=spacing,
        allow_disabled=allow_disabled,
        verbose=verbose,
        use_cache=use_cache,
        label_enum=label_enum,
    )
    out_images = [result["out_image"]]
    out_seg_mask = result["out_seg_mask"]
    orig_target = result["orig_target"]
    out_target = result["out_target"]
    stats = result["stats"]

    # Simulate remaining images
    for i in range(1, len(images)):
        # Update intensity params for each image
        pipeline.intensity_params = [intensity_params[i][n] for n in intensity_names]

        result = pipeline(
            image=images[i],
            seg_mask=seg_mask,
            spacing=spacing,
            allow_disabled=allow_disabled,
            verbose=verbose,
            use_cache=use_cache,
            label_enum=label_enum,
        )

        # Sanity checks to ensure target anatomy remains fixed across multimodal data
        if not all(
            [
                np.array_equal(result["orig_target"], orig_target),
                np.array_equal(result["out_target"], out_target),
                np.array_equal(result["out_seg_mask"], out_seg_mask),
            ]
        ):
            raise ValueError(
                f"Mismatch in output anatomy across `images[0]` and `images[{i}]`; "
                "this can be avoided either by explicitly passing a random seed/state "
                "to each effect, or by setting ``use_cache='auto'``."
            )

        out_images.append(result["out_image"])

    return {
        "out_images": out_images,
        "out_seg_mask": out_seg_mask,
        "stats": stats,
        "extras": {
            "orig_target": orig_target,
            "out_target": out_target,
        },
    }
