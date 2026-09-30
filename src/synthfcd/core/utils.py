"""
Utility functions for core `synthFCD` functionality.
"""

from __future__ import annotations

__all__ = [
    "bbox_from_mask",
    "compute_distance_fields",
    "create_meshgrid",
    "generate_application_field",
    "get_hemisphere_at_point",
    "get_largest_component",
    "postprocess_mask",
    "warp",
    "warp_image",
]


import warnings
from typing import Literal, cast, overload

import numpy as np
from scipy import ndimage

from synthfcd.core.masks import get_binary_mask
from synthfcd.utils import get_rng
from synthfcd.utils._aliases import (
    _DistanceType,
    _HemisphereType,
    _LabelEnumType,
    _RNGType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._const import (
    GAUSSIAN_ROLLOFF_TARGET,
    TANH_ROLLOFF_TARGET,
)
from synthfcd.utils._validators import (
    IntNoBool,
    RealNoBool,
    validate_3d_numpy_array,
    validate_class_type,
    validate_literal_str,
    validate_obj_type,
    validate_spacing,
)
from synthfcd.utils.seg_labels import SynthSegLabel


@overload
def compute_distance_fields(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    surface: Literal["both"] = "both",
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def compute_distance_fields(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    surface: Literal["gm_wm"] = "gm_wm",
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> np.ndarray: ...


@overload
def compute_distance_fields(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    surface: Literal["pial"] = "pial",
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> np.ndarray: ...


def compute_distance_fields(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    surface: Literal["gm_wm", "pial", "both"] = "both",
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Get two Euclidean signed distance fields (SDFs) from (i) the GM-WM boundary and (ii)
    the pial surface.

    Args:
        seg_mask (np.ndarray):
            A 3D boolean array representing the segmentation mask.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the segmentation mask in mm.
        surface (Literal["gm_wm", "pial", "both"], optional):
            The surface to compute the distance field from. Defaults to ``"both"``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use. Defaults to ``SynthSegLabel``.

    Returns:
        tuple[np.ndarray, np.ndarray] | np.ndarray:
            A tuple of the two SDFs in the order ``(gm_wm SDF, pial SDF)``.
            If ``surface`` is ``"gm_wm"`` or ``"both"``, the first element is the GM-WM SDF.
            If ``surface`` is ``"pial"``, the second element is the pial SDF.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If input arrays are malformed or inconsistent in shape.
            If ``spacing`` is invalid.
    """
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=(np.integer))
    validate_spacing(spacing, "spacing")
    validate_literal_str(surface, "surface", ("gm_wm", "pial", "both"))
    validate_class_type(label_enum, "label_enum", _LabelEnum)

    inner_labels = (
        label_enum.get_white_matter_labels()
        | label_enum.get_subcortical_labels()
        | label_enum.get_ventricle_labels()
    )

    # Signed distance field from the GM-WM boundary toward the WM
    if surface in ("gm_wm", "both"):
        inner_mask = get_binary_mask(
            seg_mask,
            labels=inner_labels,
            label_enum=label_enum,
        )
        gm_wm_sdf = cast(
            np.ndarray,
            ndimage.distance_transform_edt(inner_mask, sampling=spacing),
        )
        gm_wm_sdf -= cast(
            np.ndarray,
            ndimage.distance_transform_edt(~inner_mask, sampling=spacing),
        )

    if surface == "gm_wm":
        return gm_wm_sdf

    # GM or both
    # Signed distance field from the outer GM surface
    outer_mask = get_binary_mask(
        seg_mask,
        labels=inner_labels | label_enum.get_cortical_labels(),
        label_enum=label_enum,
    )
    pial_sdf = cast(
        np.ndarray,
        ndimage.distance_transform_edt(outer_mask, sampling=spacing),
    )
    pial_sdf -= cast(
        np.ndarray,
        ndimage.distance_transform_edt(~outer_mask, sampling=spacing),
    )

    if surface == "pial":
        return pial_sdf

    return gm_wm_sdf, pial_sdf


def warp_image(
    image: np.ndarray,
    deformation_field: tuple[np.ndarray, np.ndarray, np.ndarray],
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    interp_order: int = 3,
    mode: str = "nearest",
    clip: bool = True,
    preserve_zero_displacement: bool = True,
) -> np.ndarray:
    """
    Warp an image using a voxel-space deformation field.

    The deformation field defines a spatial transform where each voxel stores a
    displacement vector. The function performs backward (pull-based) interpolation,
    meaning each output voxel samples from a corresponding location in the input
    image. Displacements are assumed to be expressed in voxel units.

    Args:
        image (np.ndarray):
            Input 3D image to warp.
        deformation_field (tuple[np.ndarray, np.ndarray, np.ndarray]):
            Three arrays representing the displacement along the ``x``, ``y``,
            and ``z`` axes. Must match the shape of ``image``.
        meshgrid (tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional):
            Precomputed spatial grid for sampling. If ``None``, a grid is generated
            internally. Providing one is useful when warping multiple aligned images.
        interp_order (int, optional):
            Interpolation order passed to ``scipy.ndimage.map_coordinates``.
            ``0`` for nearest neighbor, ``1`` (linear), ``3`` (cubic), etc.
            Defaults to ``3``.
        mode (str, optional):
            Boundary handling mode for values sampled outside the image.
            See ``map_coordinates`` documentation. Defaults to ``'nearest'``.
        clip (bool, optional):
            If ``True``, sample coordinates are clipped to remain inside
            valid image bounds. Recommended for large warps. Defaults to ``True``.
        preserve_zero_displacement (bool, optional):
            If ``True``, voxels where the displacement is exactly zero are left
            unchanged, even when higher-order interpolation would otherwise alter
            their values. Defaults to ``True``.

    Returns:
        np.ndarray:
            A warped version of ``image`` with the same shape.

    Raises:
        TypeError:
            If argument types are invalid, including elements of ``meshgrid``
            and ``deformation_field``.
        ValueError:
            If input arrays are malformed or inconsistent in shape.
    """
    validate_3d_numpy_array(image, "image")

    validate_obj_type(deformation_field, "deformation_field", tuple)
    if len(deformation_field) != 3:
        raise ValueError("`deformation_field` must be a tuple of length 3")
    for i, obj in enumerate(deformation_field):
        validate_3d_numpy_array(
            obj, f"deformation_field_{i}", dtype=np.floating, shape=image.shape
        )

    validate_obj_type(meshgrid, "meshgrid", (tuple, type(None)))
    if meshgrid is not None:
        if len(meshgrid) != 3:
            raise ValueError("`meshgrid` must be a tuple of length 3")
        for i, obj in enumerate(meshgrid):
            validate_3d_numpy_array(
                obj, f"meshgrid_{i}", dtype=(np.number), shape=image.shape
            )

    validate_obj_type(interp_order, "interp_order", IntNoBool)
    validate_obj_type(mode, "mode", str)
    validate_obj_type(clip, "clip", bool)
    validate_obj_type(preserve_zero_displacement, "preserve_zero_displacement", bool)

    ux, uy, uz = deformation_field
    Dx, Dy, Dz = image.shape

    # Compute sampling grid
    if meshgrid is None:
        X, Y, Z = np.meshgrid(
            np.arange(Dx), np.arange(Dy), np.arange(Dz), indexing="ij"
        )
    else:
        X, Y, Z = meshgrid

    Xs = (X + ux).astype(np.float32)
    Ys = (Y + uy).astype(np.float32)
    Zs = (Z + uz).astype(np.float32)

    # Optional clipping
    if clip:
        Xs = np.clip(Xs, 0, Dx - 1)
        Ys = np.clip(Ys, 0, Dy - 1)
        Zs = np.clip(Zs, 0, Dz - 1)

    # Warp image
    warped_image = ndimage.map_coordinates(
        image, [Xs, Ys, Zs], order=interp_order, mode=mode
    )

    if preserve_zero_displacement:
        zero_displacement = (ux == 0) & (uy == 0) & (uz == 0)
        return np.where(zero_displacement, image, warped_image)

    return warped_image


def warp(
    images: list[np.ndarray],
    deformation_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    interp_order: int | list[int] = 3,
    mode: str | list[str] = "nearest",
    clip: bool | list[bool] = True,
    preserve_zero_displacement: bool = True,
) -> list[np.ndarray]:
    """
    Warp a list of images using a list of deformation fields.

    Args:
        images (list[np.ndarray]):
            A list of input images to warp.
        deformation_fields (list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
            A list of deformation fields to apply to each image.
        meshgrid (tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional):
            A meshgrid to use for sampling. If ``None``, a meshgrid is generated internally.
        interp_order (int | list[int], optional):
            The interpolation order to use for each image. If provided as a single value
            it will be used for all images; must match the length of ``images``.
        mode (str | list[str], optional):
            The boundary handling mode to use for each image; must match the length of ``images``.
        clip (bool | list[bool], optional):
            Whether to clip each image after warping; must match the length of ``images``.
        preserve_zero_displacement (bool, optional):
            Passed to ``warp_image``. Defaults to ``True``.

    Returns:
        list[np.ndarray]:
            A list of warped images.

    Raises:
        TypeError: If argument types are invalid.
        ValueError: If input lists are inconsistent in length or if any warping parameters are invalid.
    """
    validate_obj_type(images, "images", list)
    if len(images) == 0:
        raise ValueError("`images` must be a non-empty list")

    validate_obj_type(deformation_fields, "deformation_fields", list)
    if len(deformation_fields) == 0:
        raise ValueError("`deformation_fields` must be a non-empty list")

    validate_obj_type(interp_order, "interp_order", (int, list))
    if isinstance(interp_order, list):
        if len(interp_order) != len(images):
            raise ValueError(
                "`interp_order` must be a list of the same length as `images`"
            )
    else:
        interp_order = [interp_order] * len(images)

    validate_obj_type(mode, "mode", (str, list))
    if isinstance(mode, list):
        if len(mode) != len(images):
            raise ValueError("`mode` must be a list of the same length as `images`")
    else:
        mode = [mode] * len(images)

    validate_obj_type(clip, "clip", (bool, list))
    if isinstance(clip, list):
        if len(clip) != len(images):
            raise ValueError("`clip` must be a list of the same length as `images`")
    else:
        clip = [clip] * len(images)

    validate_obj_type(preserve_zero_displacement, "preserve_zero_displacement", bool)

    outs = []
    for i, o, m, c in zip(images, interp_order, mode, clip):
        for df in deformation_fields:
            i = warp_image(
                image=i,
                deformation_field=df,
                meshgrid=meshgrid,
                interp_order=o,
                mode=m,
                clip=c,
                preserve_zero_displacement=preserve_zero_displacement,
            )
        outs.append(i)

    return outs


def create_meshgrid(
    grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a voxel-space meshgrid.

    The returned arrays represent the coordinate indices along each axis of a
    regular grid. This is typically used as a base sampling grid for deformation
    fields, interpolation, and spatial transforms.

    Args:
        grid_shape (tuple[int, int, int]):
            The shape of the target grid as ``(Dx, Dy, Dz)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Three 3D arrays representing the ``x``, ``y``, and ``z`` coordinates of
            the grid, each matching ``grid_shape``.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If ``grid_shape`` does not contain exactly three positive integers.
    """
    validate_obj_type(grid_shape, "grid_shape", tuple)
    if len(grid_shape) != 3:
        raise ValueError("`grid_shape` must be a tuple of length 3")
    if not all(isinstance(s, IntNoBool) for s in grid_shape):
        raise ValueError("All elements of `grid_shape` must be integers")

    Dx, Dy, Dz = grid_shape
    return np.meshgrid(np.arange(Dx), np.arange(Dy), np.arange(Dz), indexing="ij")


def generate_application_field(
    mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    field_type: Literal["inward", "outward"] = "outward",
    rolloff_mm: _DistanceType = 3.0,
    *,
    tanh_target: float = TANH_ROLLOFF_TARGET,
    gaussian_target: float = GAUSSIAN_ROLLOFF_TARGET,
) -> np.ndarray:
    """
    Generate a weighting field for blending an effect with an image.

    If the field type is ``"inward"``, a Euclidean distance transform is computed
    from the mask boundaries to the center of the mask. The `tanh` function is then
    applied to create an application field that ranges between 0 and 1 and peaks at
    the center of the mask. Suitable for more localized effects, such as white matter
    hyperintensities or deformations.

    If the field type is ``"outward"``,  a Euclidean distance transform is computed
    from the mask boundaries outward the mask. A Gaussian function is then applied to
    create an application field that ranges between 0 and 1 and peaks at the mask boundaries.
    Suitable for deformation effects that require a smoother transition beyond the lesion
    boundary, such as cortical thickening.

    Args:
        mask (np.ndarray):
            A 3D boolean array representing the region where the effect is applied.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the mask in mm.
        field_type (Literal['inward', 'outward'], optional):
            The type of field to generate. Choose between ``'inward'`` and ``'outward'``.
            Defaults to ``'outward'``.
        rolloff_mm (float, optional):
            The rolloff distance for the field type. Defaults to ``3.0``.
            - For ``'inward'``, this is the distance where the ``tanh`` function
              reaches the target value (set by ``tanh_target``).
            - For ``'outward'``, this is the distance where the Gaussian function
              reaches the target value (set by ``gaussian_target``).
            Lower values result in a more abrupt application field.
        tanh_target (float, optional):
            The target value of the ``tanh`` function at the rolloff distance.
            Defaults to ``0.99``.
        gaussian_target (float, optional):
            The target value of the Gaussian function at the rolloff distance.
            Defaults to ``0.1``.

    Returns:
        np.ndarray:
            A weighting field for blending an effect with an image.

    Warnings:
        UserWarning:
            Emitted if the mask contains no ``True`` voxels.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If the input arrays are incompatible in shape or if ``mask`` is not
            boolean or if other arguments are invalid.
    """
    validate_3d_numpy_array(mask, "mask", dtype=(np.bool_, np.integer))
    if not mask.any():
        warnings.warn(
            "Mask is empty, returning an array of zeros.",
            UserWarning,
        )
        return np.zeros(mask.shape, dtype=np.float32)

    validate_spacing(spacing, "spacing")

    validate_literal_str(field_type, "field_type", ("inward", "outward"))

    validate_obj_type(rolloff_mm, "rolloff_mm", RealNoBool)
    if float(rolloff_mm) <= 0.0:
        raise ValueError("`rolloff_mm` must be greater than 0")

    validate_obj_type(tanh_target, "tanh_target", float)
    if tanh_target <= 0 or tanh_target >= 1:
        raise ValueError("`tanh_target` must be between 0 and 1")

    validate_obj_type(gaussian_target, "gaussian_target", float)
    if gaussian_target <= 0 or gaussian_target >= 1:
        raise ValueError("`gaussian_target` must be between 0 and 1")

    if field_type == "inward":
        dist = cast(
            np.ndarray, ndimage.distance_transform_edt(mask, sampling=spacing)
        ).astype(np.float32)

        slope = float(np.arctanh(tanh_target) / rolloff_mm)

        r = np.tanh(dist * slope).astype(np.float32)

        return np.clip(r / tanh_target, 0.0, 1.0)  # map to [0,1]; 1 at rolloff

    dist = cast(
        np.ndarray, ndimage.distance_transform_edt(~mask, sampling=spacing)
    ).astype(np.float32)

    sigma = float(rolloff_mm / np.sqrt(-2.0 * np.log(gaussian_target)))

    return np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Get the largest connected component of a binary mask.

    Args:
        mask (np.ndarray):
            A binary mask.

    Returns:
        np.ndarray:
            The largest connected component of the mask.

    Warnings:
        UserWarning:
            Emitted if the mask contains no ``True`` voxels.

    Raises:
        TypeError:
            If ``mask`` is not a numpy array.
        ValueError:
            If ``mask`` is not 3D or boolean.
    """
    validate_3d_numpy_array(mask, "mask", dtype=np.bool_)
    if not mask.any():
        warnings.warn(
            "`mask` is empty, returning an array of zeros.",
            UserWarning,
        )
        return np.zeros(mask.shape, dtype=bool)

    labeled_mask, _ = cast(tuple[np.ndarray, int], ndimage.label(mask))
    component_sizes = np.bincount(labeled_mask.ravel())
    largest_label = cast(int, component_sizes[1:].argmax() + 1)  # skip background
    return labeled_mask == largest_label


def bbox_from_mask(
    mask: np.ndarray,
    pad: int | tuple[int, int, int] = 5,
    *,
    ensure_square: bool = False,
) -> tuple[slice, slice, slice]:
    """
    Compute the bounding box of a binary mask with optional padding.

    The bounding box encloses all ``True`` voxels in the mask. Optional symmetric
    padding is applied to each dimension, with bounds clipped to remain within the
    original image space.

    Args:
        mask (np.ndarray):
            A 3D boolean or integer array representing the region of interest.
        pad (int | tuple[int, int, int], optional):
            Padding (in voxels) to extend the bounding box. If an integer, the same
            padding is applied across all axes. If a tuple, it must contain three
            integers corresponding to padding along each axis. Defaults to ``5``.
        ensure_square (bool, optional):
            If ``True``, expand the padded bounding box so that all three axes share
            the same length (the maximum axis length after padding). The square is
            centered on the padded bounding box when possible; otherwise it is shifted
            along each axis to fit within the image bounds. If the square is larger
            than the image along any axis, a warning is emitted and the non-square
            padded result is returned instead. Defaults to ``False``.

    Returns:
        tuple[slice, slice, slice]:
            Three slice objects that index into the mask's spatial extent.

    Warnings:
        UserWarning:
            Emitted if the mask contains no non-zero voxels, or if
            ``ensure_square`` is ``True`` but the square box is larger than the image
            along at least one axis.

    Raises:
        TypeError:
            If input types are invalid (e.g., mask is not a numpy array or
            ``pad`` is neither an integer nor a tuple of integers).
        ValueError:
            If ``mask`` is not a 3D boolean/integer array, or if ``pad`` is a tuple that
            does not contain exactly three integer elements.
    """
    validate_3d_numpy_array(mask, "mask", dtype=(np.bool_, np.integer))
    validate_obj_type(ensure_square, "ensure_square", bool)
    validate_obj_type(pad, "pad", (IntNoBool, tuple))
    if isinstance(pad, tuple) and len(pad) != 3:
        raise ValueError("`pad` must be a tuple of 3 integers")
    if isinstance(pad, tuple) and any(not isinstance(p, IntNoBool) for p in pad):
        raise ValueError("`pad` must be a tuple of integers")

    if isinstance(pad, int):
        pad = (pad, pad, pad)

    non_zero = np.argwhere(mask)

    # Fallback to full image if no non-zero voxels
    if non_zero.size == 0:
        warnings.warn(
            "`mask` is empty, returning the full image.",
            UserWarning,
        )
        return (slice(None), slice(None), slice(None))

    mins = non_zero.min(axis=0)
    maxs = non_zero.max(axis=0) + 1

    # add padding, clip to image bounds
    center = (mins + maxs) // 2
    mins = np.minimum(np.maximum(mins - pad, 0), center)
    maxs = np.maximum(
        np.minimum(np.maximum(maxs + pad, center + 1), mask.shape), mins + 1
    )
    result = (
        slice(mins[0], maxs[0]),
        slice(mins[1], maxs[1]),
        slice(mins[2], maxs[2]),
    )

    if not ensure_square:
        return result

    sizes = maxs - mins
    target_size = int(np.max(sizes))
    shape = np.array(mask.shape)

    if np.any(target_size > shape):
        warnings.warn(
            "Square bounding box is larger than the image, "
            "returning the non-square result.",
            UserWarning,
        )
        return result

    half = target_size // 2
    preferred_mins = center - half
    contain_mins = maxs - target_size
    contain_maxs = mins
    bounds_maxs = shape - target_size

    sq_mins = np.empty(3, dtype=int)
    for axis in range(3):
        valid_lo = max(contain_mins[axis], 0)
        valid_hi = min(contain_maxs[axis], bounds_maxs[axis])
        sq_mins[axis] = int(np.clip(preferred_mins[axis], valid_lo, valid_hi))

    sq_maxs = sq_mins + target_size
    return (
        slice(sq_mins[0], sq_maxs[0]),
        slice(sq_mins[1], sq_maxs[1]),
        slice(sq_mins[2], sq_maxs[2]),
    )


def get_hemisphere_at_point(
    seg_mask: np.ndarray,
    point: tuple[int, int, int],
    *,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> _HemisphereType | None:
    """
    Return the hemisphere of a voxel in a segmentation mask.

    The hemisphere is inferred from the segmentation label at ``point`` by checking
    membership in ``label_enum.get_all_labels(hemisphere=...)``. Labels that belong
    to neither lateralized set (e.g. background or midline structures) yield
    ``None``.

    Args:
        seg_mask (np.ndarray):
            A 3D integer segmentation mask.
        point (tuple[int, int, int]):
            Voxel indices ``(i, j, k)`` into ``seg_mask``.
        label_enum (type[LabelEnum], optional):
            The label enumeration used to interpret ``seg_mask``. Defaults to
            ``SynthSegLabel``.

    Returns:
        Literal["left", "right"] | None:
            The hemisphere of the label at ``point``, or ``None`` when the label
            is not lateralized.

    Raises:
        TypeError:
            If input types are invalid.
        ValueError:
            If ``point`` is out of bounds for ``seg_mask``.
    """
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=(np.integer))

    validate_obj_type(point, "point", tuple)
    if len(point) != 3 or not all(isinstance(c, IntNoBool) for c in point):
        raise TypeError("`point` must be a tuple of 3 integers")

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    for axis, (coord, size) in enumerate(zip(point, seg_mask.shape)):
        if coord < 0 or coord >= size:
            raise ValueError(
                f"`point[{axis}]`={coord} is out of bounds for shape {seg_mask.shape}"
            )

    label = int(seg_mask[point])
    if label in label_enum.get_all_labels(hemisphere="left"):
        return "left"
    if label in label_enum.get_all_labels(hemisphere="right"):
        return "right"
    return None


def postprocess_mask(
    mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    smooth_sigma: _DistanceType | Literal["auto"] | None = "auto",
    threshold: float = 0.5,
    closing: _DistanceType | None = None,
    fill_holes: bool = True,
    limit_to: np.ndarray | None = None,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """
    Postprocess a binary mask.

    The postprocessing steps are:
    - Gaussian smoothing
    - Morphological closing
    - Filling holes
    - Limiting to a mask
    - Getting the largest component
    - Thresholding

    Args:
        mask (np.ndarray):
            The input binary mask to postprocess.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the mask in mm.
        smooth_sigma (float | Literal["auto"] | None, optional):
            The sigma in mm for the Gaussian smoothing. Will be converted to voxels using the spacing.
            Defaults to ``"auto"``, where the minimum spacing is used to get at least 1-voxel smoothing
            along the finest axis. Set to ``None`` to disable smoothing.
        threshold (float | int, optional):
            Binarization threshold for the smoothed mask. Defaults to ``0.5``. Will have no
            effect if smoothing is disabled.
        closing (float | int | None, optional):
            The size of the structuring element in mm for morphological closing. This will be converted
            to voxels using the minimum spacing. Defaults to ``None``; i.e., no closing is performed.
        fill_holes (bool, optional):
            Whether to fill holes in the mask. Defaults to ``True``.
        limit_to (np.ndarray | None, optional):
            A mask to limit the postprocessed mask to; e.g., a combined GM & WM mask.
            Defaults to ``None``.
        keep_largest_component (bool, optional):
            Whether to keep only the largest component of the mask. Defaults to ``True``.

    Returns:
        np.ndarray:
            The postprocessed mask.

    Warnings:
        UserWarning:
            Emitted if the mask contains no ``True`` voxels.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If argument values are invalid; e.g., non-boolean mask,
            negative ``smooth_sigma``, invalid ``spacing``, etc.
    """
    validate_3d_numpy_array(mask, "mask", dtype=np.bool_)
    if not mask.any():
        warnings.warn(
            "`mask` is empty, returning an array of zeros.",
            UserWarning,
        )
        return np.zeros(mask.shape, dtype=np.bool_)

    validate_spacing(spacing, "spacing")

    if smooth_sigma is not None and smooth_sigma != "auto":
        validate_obj_type(smooth_sigma, "smooth_sigma", RealNoBool)
        if float(smooth_sigma) <= 0.0:
            raise ValueError("`smooth_sigma` must be positive")

    validate_obj_type(threshold, "threshold", RealNoBool)
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("`threshold` must be between 0 and 1")

    if closing is not None:
        validate_obj_type(closing, "closing", RealNoBool)
        if float(closing) <= 0.0:
            raise ValueError("`closing` must be positive")
        closing = max(1, int(np.ceil(float(closing) / min(spacing))))

    validate_obj_type(fill_holes, "fill_holes", bool)

    if limit_to is not None:
        validate_3d_numpy_array(limit_to, "limit_to", dtype=np.bool_, shape=mask.shape)
        if not limit_to.any():
            warnings.warn(
                "`limit_to` is empty, returning an array of zeros.",
                UserWarning,
            )
            return np.zeros(mask.shape, dtype=np.bool_)

    validate_obj_type(keep_largest_component, "keep_largest_component", bool)

    if smooth_sigma == "auto":
        smooth_sigma = min(spacing)

    if smooth_sigma is not None:
        s = (
            smooth_sigma / spacing[0],
            smooth_sigma / spacing[1],
            smooth_sigma / spacing[2],
        )
        mask = cast(
            np.ndarray, ndimage.gaussian_filter(mask.astype(np.float32), sigma=s)
        )

        mask = mask > threshold
        if not mask.any():
            warnings.warn(
                "`mask` is empty after thresholding, returning an array of zeros; "
                "consider checking the `threshold`.",
                UserWarning,
            )
            return np.zeros(mask.shape, dtype=np.bool_)

    if closing is not None:
        mask = cast(np.ndarray, ndimage.binary_closing(mask, iterations=closing))
        if not mask.any():
            warnings.warn(
                "`mask` is empty after closing, returning an array of zeros; "
                "consider checking the `closing`.",
                UserWarning,
            )
            return np.zeros(mask.shape, dtype=np.bool_)

    if fill_holes:
        mask = cast(np.ndarray, ndimage.binary_fill_holes(mask))
        if not mask.any():
            warnings.warn(
                "`mask` is empty after filling holes, returning an array of zeros; "
                "consider checking the `fill_holes`.",
                UserWarning,
            )
            return np.zeros(mask.shape, dtype=np.bool_)

    if limit_to is not None:
        mask &= limit_to
        if not mask.any():
            warnings.warn(
                "`mask` is empty after limiting to `limit_to`, returning an array of zeros. If this "
                "is not expected, consider checking the `limit_to` mask or lowering the `threshold`.",
                UserWarning,
            )
            return np.zeros(mask.shape, dtype=np.bool_)

    if keep_largest_component:
        mask = cast(np.ndarray, get_largest_component(mask))

    return mask


def generate_noise_field(
    shape: tuple[int, int, int],
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    corr_sigma: _DistanceType | None = None,
    hpf_sigma: _DistanceType | None = None,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
) -> np.ndarray:
    """
    Generate a white noise field that is optionally correlated and high-pass filtered to
    control clustering.

    Args:
        shape (tuple[int, int, int]):
            The shape of the noise field in voxels.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the noise field in mm.
        corr_sigma (float | None, optional):
            The sigma in mm for correlating the noise field via Gaussian smoothing.
            Defaults to ``None``.
        hpf_sigma (float | None, optional):
            The sigma in mm for high-pass filtering the noise field. Defaults to ``None``.
        random_seed (int | None, optional):
            The random seed for the noise field. Defaults to ``None``.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            The random state for the noise field. Defaults to ``None``.

    Returns:
        np.ndarray:
            The noise field.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If argument values are not appropriate; e.g., a lower ``hpf_sigma`` than ``corr_sigma``,
            or malformed ``shape`` and ``spacing``.
    """
    validate_obj_type(shape, "shape", tuple)
    if len(shape) != 3:
        raise ValueError("`shape` must contain 3 elements")
    if not all(isinstance(s, IntNoBool) and int(s) > 0 for s in shape):
        raise ValueError("`shape` must be a tuple of positive integers")

    validate_spacing(spacing, "spacing")

    validate_obj_type(corr_sigma, "corr_sigma", (RealNoBool, type(None)))
    if corr_sigma is not None and float(corr_sigma) < 0.0:
        raise ValueError("`corr_sigma` must be non-negative")

    validate_obj_type(hpf_sigma, "hpf_sigma", (RealNoBool, type(None)))
    if hpf_sigma is not None and float(hpf_sigma) <= 0.0:
        raise ValueError("`hpf_sigma` must be positive")
    if (
        hpf_sigma is not None
        and corr_sigma is not None
        and float(hpf_sigma) <= float(corr_sigma)
    ):
        raise ValueError("`hpf_sigma` must be no less than `corr_sigma`")

    rng = get_rng(seed=random_seed, random_state=random_state)

    noise = rng.normal(0.0, 1.0, size=shape)

    if corr_sigma is not None:
        sigma_vox = tuple([float(corr_sigma) / s for s in spacing])
        noise = ndimage.gaussian_filter(noise, sigma=sigma_vox)

    if hpf_sigma is not None:
        sigma_vox = tuple([float(hpf_sigma) / s for s in spacing])
        noise -= ndimage.gaussian_filter(noise, sigma=sigma_vox)

    return (noise - noise.min()) / (noise.max() - noise.min())
