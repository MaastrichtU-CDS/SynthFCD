"""
Generic MONAI transform for lesion simulation.
"""

from __future__ import annotations

__all__ = [
    "MONAIBase",
]

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import torch
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends

from synthfcd.transforms.utils import (
    _arr_from_numpy,
    _arr_to_numpy,
    _matches_spacing,
    _maybe_spacing_from_metatensors,
)
from synthfcd.utils._aliases import (
    _DistanceType,
    _MONAIArrayType,
)
from synthfcd.utils._validators import (
    validate_obj_type,
    validate_spacing,
)


class MONAIBase(MapTransform):
    """
    Generic dictionary-based transform to simulate some effects on input image(s).

    It extends MONAI's dictionary transform base class (``MapTransform``) to provide
    support for resolving the inputs required by all simulation pipelines (``images``,
    ``seg_mask``, ``spacing``) and converting all input arrays to NumPy (and vice versa)
    while preserving the original type, device, and any metadata.

    Notes:
        - There are two optional ways users can specify the voxel spacing:

            (i) by providing a predefined spacing during initialization
                (best option if spacing is known and constant)

            (ii) by providing the key of the input dictionary that contains the
                spacing (best option if spacing is known but different for each image).

            The spacing will also be automatically inferred from the image metadata
            (applicable only for ``MetaTensor`` inputs) and checked against that from
            any of the above methods.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        keys: str | list[str] | tuple[str, ...],
        seg_mask_key: str,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType] | None = None,
        spacing_key: str | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """Args:
            keys (str | list[str] | tuple[str, ...]):
                The keys that contain the images to apply the transform to.
            seg_mask_key (str):
                The key of the segmentation mask in the input dictionary.
            spacing (tuple[float | int, float | int, float | int] | None, optional):
                The voxel spacing of the input images. Defaults to ``None``.
            spacing_key (str | None, optional):
                The key of the voxel spacing in the input dictionary. Defaults to ``None``.
            allow_missing_keys (bool):
                Whether to allow missing keys in the input dictionary. Defaults to ``False``.

        Raises:
            TypeError: If provided arguments are not of the correct type.
            ValueError: If provided spacing is not a tuple of 3 elements.
            ValueError: If provided label enumeration is not a subclass of ``LabelEnum``.

        """
        super().__init__(keys, allow_missing_keys)

        validate_obj_type(seg_mask_key, "seg_mask_key", str)
        if seg_mask_key in self.keys:
            raise ValueError(
                "Segmentation mask key cannot be one of the input image keys "
            )
        self.seg_mask_key = seg_mask_key

        if spacing is not None:
            validate_spacing(spacing, "spacing")
        self._spacing = spacing

        if spacing_key is not None:
            validate_obj_type(spacing_key, "spacing_key", str)
        self._spacing_key = spacing_key

        self._meta: dict[str, dict[str, Any]] = {}

    def clear_metadata(self) -> None:
        """
        Clear the input array metadata for all keys.

        Callers cannot selectively clear or update metadata keys to ensure complete
        reset in each call and no accidental mutations throughout the transform.
        """
        self._meta.clear()

    def image_to_numpy(self, arr: _MONAIArrayType, /, key: str) -> np.ndarray:
        """
        Convert a MONAI array to a NumPy array and store the metadata internally.

        Args:
            arr (MetaTensor | torch.Tensor | np.ndarray):
                The array to convert.
            key (str):
                The key of the array in the input dictionary.

        Returns:
            The NumPy array.

        Raises:
            TypeError: If the array is not a PyTorch tensor, a NumPy array,
            or a MONAI MetaTensor.
            ValueError: If the metadata already exists for the given key.
        """
        if key in self._meta:
            raise ValueError(
                f"Metadata already exists for key `{key}`; run `clear_metadata` first."
            )
        orig_type = type(arr)
        device = arr.device if isinstance(arr, torch.Tensor) else None
        spacing = _maybe_spacing_from_metatensors(arr)
        arr, meta = _arr_to_numpy(arr)
        self._meta[key] = {
            "MONAI_meta": meta,
            "spacing": spacing,
            "orig_type": orig_type,
            "device": device,
        }
        return arr

    def numpy_to_image(
        self,
        arr: np.ndarray,
        /,
        key: str,
    ) -> _MONAIArrayType:
        """
        Convert a NumPy array to the original type and device.

        Args:
            arr (np.ndarray):
                The NumPy array to convert.
            key (str):
                The key of the array in the input dictionary.

        Returns:
            The original array.

        Raises:
            ValueError: If the metadata is not found for the given key.
        """
        if key not in self._meta:
            raise ValueError(
                f"No metadata found for key `{key}`; run `image_to_numpy` first "
                f"with `key={key}`."
            )
        return _arr_from_numpy(
            arr,
            meta=self._meta[key]["MONAI_meta"],
            orig_type=self._meta[key]["orig_type"],
            device=self._meta[key]["device"],
            key=key,
        )

    def resolve_inputs(
        self,
        d: Mapping[Hashable, Any],
        /,
    ) -> tuple[
        list[np.ndarray], np.ndarray, tuple[_DistanceType, _DistanceType, _DistanceType]
    ]:
        """
        Resolve the provided ``keys``, ``seg_mask_key``, and ``spacing*`` arguments.

        Args:
            d (Mapping[Hashable, Any]):
                The input dictionary.

        Returns:
            tuple[list[np.ndarray], np.ndarray, tuple[int | float, int | float, int | float]]:
                A tuple containing the input images, segmentation mask, and voxel spacing.
        """
        # Fresh start at each call
        self.clear_metadata()

        # Extract input images and segmentation mask
        images = []
        for key in self.key_iterator(d):
            images.append(self.image_to_numpy(d[key], key))
        if self.seg_mask_key not in d:
            raise ValueError(
                f"Segmentation mask key `{self.seg_mask_key}` not found in input dictionary."
            )
        seg_mask = self.image_to_numpy(d[self.seg_mask_key], self.seg_mask_key)

        # Extract voxel spacing from:
        # 1) predefined spacing
        spacing_from_init = None
        if self._spacing is not None:
            spacing_from_init = self._spacing

        # 2) input dictionary
        spacing_from_dict = None
        if self._spacing_key is not None:
            if self._spacing_key not in d:
                raise ValueError(
                    f"Spacing key `{self._spacing_key}` not found in input dictionary."
                )
            spacing_from_dict = d[self._spacing_key]
            validate_spacing(spacing_from_dict, f"key `{self._spacing_key}`")
            # Ensure no mismatch with any predefined spacing
            if spacing_from_init is not None and not _matches_spacing(
                spacing_from_dict, spacing_from_init
            ):
                raise ValueError(
                    f"Spacing from key `{self._spacing_key}` does not match predefined "
                    f"spacing ({spacing_from_dict} vs {spacing_from_init})."
                )

        # 3) metadata of MONAI MetaTensor inputs
        spacing_from_meta = None
        for key, meta in self._meta.items():
            curr = meta["spacing"]
            if curr is None:
                continue

            validate_spacing(curr, f"key `{key}`")

            # Ensure no mismatch with any previously inferred spacing
            if spacing_from_meta is not None and not _matches_spacing(
                spacing_from_meta, curr
            ):
                raise ValueError("Spacing mismatch between metatensors of input data.")
            spacing_from_meta = curr

            # Ensure no mismatch with any predefined spacing
            if spacing_from_init is not None and not _matches_spacing(
                spacing_from_meta, spacing_from_init
            ):
                raise ValueError(
                    f"Spacing from metadata of key `{key}` does not match predefined "
                    f"spacing ({spacing_from_meta} vs {spacing_from_init})."
                )

            # Ensure no mismatch with any spacing provided in input dictionary
            if spacing_from_dict is not None and not _matches_spacing(
                spacing_from_meta, spacing_from_dict
            ):
                raise ValueError(
                    f"Spacing from metadata of key `{key}` ({spacing_from_meta}) does not "
                    f"match spacing from key `{self._spacing_key}` ({spacing_from_dict})."
                )

        # Pick the first valid spacing
        spacing = spacing_from_init or spacing_from_dict or spacing_from_meta
        if spacing is None:
            raise ValueError(
                "No spacing could be inferred from the input metadata, "
                "nor provided during initialization or as a dictionary key."
            )

        return images, seg_mask, spacing
