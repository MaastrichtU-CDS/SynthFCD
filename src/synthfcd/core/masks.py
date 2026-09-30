"""
Operations on masks preceding lesion growth.

Cannot import from other synthfcd.core modules to avoid circular imports.
"""

from __future__ import annotations

__all__ = [
    "extract_masks",
    "get_binary_mask",
]

import warnings
from typing import Any, cast

import numpy as np

from synthfcd.utils._aliases import (
    _HemisphereType,
    _LabelEnumType,
    _LabelType,
    _MaskType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._deprecated import deprecated
from synthfcd.utils._validators import (
    IntNoBool,
    validate_3d_numpy_array,
    validate_class_type,
    validate_literal_str,
    validate_obj_type,
)
from synthfcd.utils.seg_labels import SynthSegLabel


def resolve_label(label_enum: _LabelEnumType, label: _LabelType) -> set[int]:
    """Normalize a label specifier into an integer enum value that is compatible with the
    ``LabelEnum`` class.
    """
    validate_class_type(label_enum, "label_enum", _LabelEnum)
    if isinstance(label, (list, tuple, set)):
        if not label:
            raise ValueError("`label` must not be empty.")

        set_labels = set()
        for l in label:
            set_labels.update(resolve_label(label_enum, l))
        return set_labels

    validate_obj_type(label, "label", (_LabelEnum, IntNoBool, str))

    if isinstance(label, _LabelEnum):
        if not isinstance(label, label_enum):
            raise TypeError(
                f"`label` enum members must belong to {label_enum.__name__}. "
                f"Got {type(label).__name__}."
            )
        return {int(label)}

    if isinstance(label, int):
        try:
            return {int(label_enum(label))}
        except ValueError as exc:
            raise ValueError(
                f"`label` contains {label}, which is not a valid {label_enum.__name__} value."
            ) from exc

    if isinstance(label, str):
        stripped_label = label.strip()
        enum_label = label_enum.from_readable_name(stripped_label)
        if enum_label is not None:
            return {int(enum_label)}

        # Attempt to retrieve requested label
        enum_name = stripped_label.upper().replace(" ", "_")
        try:
            return {int(label_enum[enum_name])}
        except KeyError:
            pass

        if stripped_label.lstrip("-").isdigit():
            return resolve_label(label_enum, int(stripped_label))

        raise ValueError(
            f"`label` contains {label!r}, which does not match a readable name, "
            f"member name, or integer value in {label_enum.__name__}."
        )


def get_binary_mask(
    seg_mask: np.ndarray,
    mask_type: _MaskType | None = None,
    labels: _LabelType | None = None,
    hemisphere: _HemisphereType | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> np.ndarray:
    """
    Get a boolean mask from a segmentation mask using either a built-in mask type or
    explicitly specified labels.

    Note:
        The ``seg_mask`` input is expected to be generated using ``SynthSeg``
        (https://github.com/BBillot/SynthSeg) with the ``--parc`` flag enabled. We recommend
        users to use the following version of ``SynthSeg`` with more modern TensorFlow and
        CUDA wheels: https://github.com/MGH-LEMoN/Photo-SynthSeg/tree/synthseg_tf2.15.

        If wishing to use a different segmentation mask with custom labels, create a custom
        label enumeration by subclassing ``synthfcd.utils.LabelEnum`` and implementing the
        required methods. This can then be passed via the ``label_enum`` argument.

    Args:
        seg_mask (np.ndarray):
            The segmentation mask.
        mask_type (Literal["all", "gm", "wm", "ventricles", "subcortical", "background-csf"] | None, optional):
            Built-in mask types using the label enumeration helper functions.
        labels (LabelType | None, optional):
            Custom labels to use.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the mask to.
        extra_kwargs (dict[str, Any] | None, optional):
            Extra keyword arguments to pass to the label enumeration helper functions;
            e.g., ``lateral_only`` when ``mask_type`` is set to ``"ventricles"``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use. Defaults to ``SynthSegLabel``.

    Returns:
        ndarray[bool]:
            The boolean mask.

    Raises:
        TypeError: If input arguments have invalid types.
        ValueError: If both/neither `mask_type` and `labels` are provided.
    """
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=(np.integer))
    validate_class_type(label_enum, "label_enum", _LabelEnum)

    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    if mask_type is not None:
        validate_literal_str(
            mask_type,
            "mask_type",
            ("all", "gm", "wm", "ventricles", "subcortical", "background-csf"),
        )

    validate_obj_type(extra_kwargs, "extra_kwargs", (dict, type(None)))
    extra_kwargs = extra_kwargs or {}

    if (mask_type is None) == (labels is None):
        raise ValueError("Provide exactly one of `mask_type` or `labels`.")

    if mask_type is None:
        return np.isin(
            seg_mask,
            list(resolve_label(label_enum, cast(_LabelType, labels))),
        )

    if mask_type == "all":
        target_labels = label_enum.get_all_labels(hemisphere=hemisphere, **extra_kwargs)
    elif mask_type == "gm":
        target_labels = label_enum.get_cortical_labels(
            hemisphere=hemisphere, **extra_kwargs
        )
    elif mask_type == "wm":
        target_labels = label_enum.get_white_matter_labels(
            hemisphere=hemisphere, **extra_kwargs
        )
    elif mask_type == "subcortical":
        target_labels = label_enum.get_subcortical_labels(
            hemisphere=hemisphere, **extra_kwargs
        )
    elif mask_type == "background-csf":
        target_labels = label_enum.get_background_csf_labels(**extra_kwargs)
    elif mask_type == "ventricles":
        target_labels = label_enum.get_ventricle_labels(
            hemisphere=hemisphere, **extra_kwargs
        )
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    return np.isin(seg_mask, list(target_labels))


@deprecated(
    "This function is deprecated.",
    version="0.1.0",
    replacement="get_binary_mask",
)
def extract_masks(
    seg_mask: np.ndarray,
    hemisphere: _HemisphereType | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract grey matter (GM) and white matter (WM) masks from a whole-brain segmentation
    mask. Optionally, the masks can be limited to a specified hemisphere.

    Note:
        The ``seg_mask`` input is expected to be generated using ``SynthSeg``
        (https://github.com/BBillot/SynthSeg) with the ``--parc`` flag enabled. To assist
        users, we provide a function ``generate_seg_mask`` that runs a containerized ``SynthSeg``
        with GPU acceleration support and modern CUDA wheels, together with file exploration
        using ``nifti-finder`` (https://github.com/pkoutsouvelis/nifti-finder) for configurable
        batch processing of large datasets.

    Args:
        seg_mask (np.ndarray):
            The whole-brain segmentation mask.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to extract the masks for. Defaults to None.

    Returns:
        dict[str, np.ndarray]:
            Dictionary of extracted masks, including:
            - ``'seg_mask'``: original segmentation mask.
            - ``'gm_mask'``: gray matter mask.
            - ``'wm_mask'``: white matter mask.

    Warnings:
        UserWarning:
            Emitted if extracted GM or WM masks are empty.

    Raises:
        TypeError:
            If `seg_mask` or `hemisphere` have invalid types.
        ValueError:
            If `seg_mask` is empty, `hemisphere` is not "left" or "right",
            or input segmentation mask is not a 3D integer array.
    """
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=(np.integer))
    if not seg_mask.any():
        raise ValueError("`seg_mask` is empty; cannot extract masks")

    validate_obj_type(hemisphere, "hemisphere", (str, type(None)))
    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    gm_labels = SynthSegLabel.get_cortical_labels(cast(str, hemisphere))
    wm_labels = SynthSegLabel.get_white_matter_labels(cast(str, hemisphere))
    all_labels = SynthSegLabel.get_all_labels(
        cast(str, hemisphere), include_neutral=True
    )

    gm_mask = np.isin(seg_mask, list(gm_labels))
    wm_mask = np.isin(seg_mask, list(wm_labels))
    seg_mask = np.where(np.isin(seg_mask, list(all_labels)), seg_mask, 0)

    if not gm_mask.any() or not wm_mask.any():
        warnings.warn(
            "Extracted GM or WM masks are empty; visually check `seg_mask` for errors",
            UserWarning,
        )

    return {
        "seg_mask": seg_mask,
        "gm_mask": gm_mask,
        "wm_mask": wm_mask,
    }
