"""
MONAI transforms for SimpleFCD.
"""

from __future__ import annotations

__all__ = [
    "RandSimpleFCDd",
]

import warnings
from collections.abc import Hashable, Mapping
from copy import deepcopy
from typing import Any, cast

from monai.transforms.transform import Randomizable

from synthfcd.pipelines import simple_fcd_simulator
from synthfcd.presets import draw_simple_fcd_params, get_simple_fcd_presets
from synthfcd.transforms.base import MONAIBase
from synthfcd.utils._aliases import (
    _DistanceType,
    _LabelEnumType,
    _sMRIType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._validators import (
    validate_class_type,
    validate_dict_and_keys,
    validate_literal_str,
    validate_obj_type,
)
from synthfcd.utils.misc import get_leaf_paths_dict
from synthfcd.utils.seg_labels import SynthSegLabel


class RandSimpleFCDd(MONAIBase, Randomizable):
    """
    Randomizable MONAI-compatible dictionary transform for the ``SimpleFCD`` pipeline.

    Calls the ``simple_fcd_simulator`` function that:
    - Generates a synthetic FCD lesion mask based on the provided brain anatomy
    - Applies local deformation effects to the lesion area (same for all
      input images)
    - Applies local intensity effects to the lesion area (different for each
      input image and sequence)

    The transform replaces the original images and segmentation mask with the simulated
    equivalents and optionally adds:
        - The growth statistics of the lesion (e.g, DKT atlas location)
        - The output synthetic lesion mask
        - The initial lesion mask prior to applying any effects
    """

    def __init__(
        self,
        keys: str | list[str] | tuple[str, ...],
        sequences: list[_sMRIType],
        seg_mask_key: str = "seg_mask",
        custom_presets: dict[str, Any] | None = None,
        *,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType] | None = None,
        spacing_key: str | None = None,
        out_stats_key: str | None = "synthfcd_stats",
        out_lesion_mask_key: str | None = "synthfcd_lesion",
        orig_lesion_mask_key: str | None = None,
        label_enum: _LabelEnumType = SynthSegLabel,
        allow_missing_keys: bool = False,
    ):
        """Args:
        keys (str | list[str] | tuple[str, ...]):
            The input dictionary keys to apply the transform to. The transformed images
            will be written to the same keys.
        sequences (list[_sMRIType]):
            The MRI sequence names of the input images. Should match the length
            and order of the ``keys`` and be one of ``("T1like", "T2like")``.
        seg_mask_key (str, optional):
            The dictionary key to retrieve the segmentation mask from.
            Defaults to ``"seg_mask"``.
        custom_presets (dict[str, Any] | None, optional):
            Custom presets to use for drawing the simulation parameters.
            Defaults to ``None``; i.e., extracts presets from ``synthfcd.get_simple_fcd_presets``.
        spacing (tuple[_DistanceType, _DistanceType, _DistanceType] | None, optional):
            The voxel spacing of the input images. Defaults to ``None``; i.e.,
            the spacing will be retrieved from ``spacing_key`` (see below) or inferred
            from the input metadata (if ``MetaTensor`` inputs are used).
        spacing_key (str | None, optional):
            The dictionary key to retrieve the voxel spacing from. Defaults to ``None``;
            i.e., the spacing will be retrieved from ``spacing`` or inferred from
            the input metadata (if ``MetaTensor`` inputs are used).
        out_stats_key (str | None, optional):
            The dictionary key to write the lesion growth statistics to.
            Defaults to ``"synthfcd_stats"``.
        out_lesion_mask_key (str | None, optional):
            The dictionary key to write the synthetic lesion mask to.
            Defaults to ``"synthfcd_lesion"``.
        orig_lesion_mask_key (str | None, optional):
            The dictionary key to write the initially grown lesion mask to.
            Defaults to ``None``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading the segmentation mask.
            Defaults to ``synthfcd.utils.SynthSegLabel``.
        allow_missing_keys (bool, optional):
            Whether to allow missing keys in the input dictionary. Defaults to ``False``.

        Raises:
            TypeError/ValueError: If provided arguments do not match the expected types
                and/or values.

        """
        super().__init__(
            keys=keys,
            seg_mask_key=seg_mask_key,
            spacing=spacing,
            spacing_key=spacing_key,
            allow_missing_keys=allow_missing_keys,
        )

        validate_obj_type(sequences, "sequences", list)
        if len(sequences) != len(self.keys):
            raise ValueError(
                "`sequences` must be the same length as the number of keys"
            )
        for i, s in enumerate(sequences):
            validate_literal_str(
                s,
                name=f"sequences[{i}]",
                target_literals=("T1like", "T2like"),
            )
        self._sequences = sequences

        if custom_presets is not None:
            validate_obj_type(
                custom_presets,
                name="custom_presets",
                target_type=dict,
            )
            if get_leaf_paths_dict(custom_presets) != get_leaf_paths_dict(
                get_simple_fcd_presets()
            ):
                raise ValueError(
                    "`custom_presets` must contain the same keys and structure as the "
                    "default presets (see `synthfcd.get_simple_fcd_presets`)."
                )
        self._custom_presets = custom_presets

        if out_stats_key is not None:
            validate_obj_type(out_stats_key, "out_stats_key", str)
        self._out_stats_key = out_stats_key

        if out_lesion_mask_key is not None:
            validate_obj_type(out_lesion_mask_key, "out_lesion_mask_key", str)
        self._out_lesion_mask_key = out_lesion_mask_key

        if orig_lesion_mask_key is not None:
            validate_obj_type(orig_lesion_mask_key, "orig_lesion_mask_key", str)
        self._orig_lesion_mask_key = orig_lesion_mask_key

        validate_class_type(label_enum, "label_enum", _LabelEnum)
        self._label_enum = label_enum

        # Initialize simulation parameters; will be populated in `self.randomize()`
        self._params: dict[str, Any] | None = None

    def randomize(self) -> None:
        """
        Randomize the simulation parameters using MONAI's internal RNG.
        """
        self._params = draw_simple_fcd_params(
            sequences=self._sequences,
            custom_presets=self._custom_presets,
            random_seed=None,
            random_state=self.R,
            label_enum=self._label_enum,
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
        """
        Apply the transform to the input data.
        """
        d = dict(data)

        # Extract input-dependent arguments
        images, seg_mask, spacing = self.resolve_inputs(d)
        if len(images) == 0:
            warnings.warn("No images to process; returning input dictionary unchanged.")
            return d

        # Use first (present) image as reference for aligning extra outputs
        ref_key = next(self.key_iterator(d))

        # Randomize simulation parameters
        self.randomize()
        try:
            validate_dict_and_keys(
                self._params,
                "`self._params`",
                ("growth_params", "deformation_params", "intensity_params"),
            )
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "Internal contract violated: `self.randomize()`of `RandSimpleFCDd` "
                "did not provide `self._params` with the expected keys."
            ) from e
        params = cast(dict[str, Any], deepcopy(self._params))

        # Run simulation pipeline
        result = simple_fcd_simulator(
            images=images,
            seg_mask=seg_mask,
            spacing=spacing,
            growth_params=params["growth_params"],
            deformation_params=params["deformation_params"],
            intensity_params=params["intensity_params"],
            verbose=False,
            label_enum=self._label_enum,
        )
        try:
            validate_dict_and_keys(
                result,
                "`result`",
                ("out_images", "out_seg_mask", "stats", "extras"),
            )
            validate_dict_and_keys(
                result["extras"],
                "`result['extras']`",
                ("orig_target", "out_target"),
            )
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "Internal contract violated: `simple_fcd_simulator` did not return "
                "the expected keys."
            ) from e

        # Convert to original format and write back to `keys` and `seg_mask_key`
        for i, key in enumerate(self.key_iterator(d)):  # same order as inputs
            d[key] = self.numpy_to_image(result["out_images"][i], key)

        d[self.seg_mask_key] = self.numpy_to_image(
            result["out_seg_mask"], self.seg_mask_key
        )

        # Write extra outputs
        if self._out_lesion_mask_key is not None:
            d[self._out_lesion_mask_key] = self.numpy_to_image(
                result["extras"]["out_target"], ref_key
            )
        if self._orig_lesion_mask_key is not None:
            d[self._orig_lesion_mask_key] = self.numpy_to_image(
                result["extras"]["orig_target"], ref_key
            )
        if self._out_stats_key is not None:
            d[self._out_stats_key] = result["stats"]

        return d
