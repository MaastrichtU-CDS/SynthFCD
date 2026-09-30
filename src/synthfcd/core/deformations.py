"""
Apply cortical deformations to the image and masks.
"""

from __future__ import annotations

__all__ = [
    "abnormal_gyration",
    "compose_deformation_fields",
    "cortical_expansion",
    "cortical_thickening",
    "exponentiate_velocity_field",
    "get_deformation_fields_from_sdf",
    "get_random_deformation_field",
    "sulcal_widening",
]


import warnings
from typing import Literal, cast

import numpy as np
from scipy import ndimage

from synthfcd.core.utils import (
    compute_distance_fields,
    create_meshgrid,
    generate_application_field,
    generate_noise_field,
    get_rng,
    warp_image,
)
from synthfcd.utils._aliases import (
    _DeformationResultType,
    _DistanceType,
    _HemisphereType,
    _LabelEnumType,
    _RNGType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._const import (
    SEED_OFFSET_GYRATION,
    SMOOTH_SIGMA_DEFORM,
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


def get_deformation_fields_from_sdf(
    sdf: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    scaling_field: np.ndarray | RealNoBool | None = None,
    smooth_sigma_grad: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a vector deformation field opposite to the direction of a signed distance
    field for pull-based warping.

    Args:
        sdf (np.ndarray):
            The signed distance field.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the signed distance field in mm.
        scaling_field (np.ndarray | float | int | None, optional):
            Per-voxel scaling array or a uniform scalar multiplier.
            Defaults to ``None``.
        smooth_sigma_grad (float | None, optional):
            The sigma used for the smoothing of the gradient of the SDF.
            Defaults to ``None``; i.e., no smoothing is applied.
        smooth_sigma_field (float | None, optional):
            The sigma used for the smoothing the resulting deformation field.
            Defaults to ``None``; i.e., no smoothing is applied.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The x, y, and z components of the deformation field.

    Raises:
        TypeError:
            If parameter types are incorrect.
        ValueError:
            If parameter values are invalid.
    """
    validate_3d_numpy_array(sdf, "sdf", dtype=np.floating)
    if sdf.ndim != 3:
        raise ValueError("`sdf` must be a 3D array")

    validate_spacing(spacing, "spacing")
    sx, sy, sz = spacing

    if scaling_field is not None:
        validate_obj_type(scaling_field, "scaling_field", (np.ndarray, RealNoBool))
        if isinstance(scaling_field, np.ndarray):
            validate_3d_numpy_array(
                scaling_field, "scaling_field", dtype=np.floating, shape=sdf.shape
            )

    validate_obj_type(smooth_sigma_grad, "smooth_sigma_grad", (RealNoBool, type(None)))
    if smooth_sigma_grad is not None and float(smooth_sigma_grad) <= 0.0:
        raise ValueError("`smooth_sigma_grad` must be greater than 0")

    validate_obj_type(
        smooth_sigma_field, "smooth_sigma_field", (RealNoBool, type(None))
    )
    if smooth_sigma_field is not None and float(smooth_sigma_field) <= 0.0:
        raise ValueError("`smooth_sigma_field` must be greater than 0")

    # Initialize deformation field with gradient of SDF
    ux, uy, uz = np.gradient(sdf, sx, sy, sz)
    ux, uy, uz = -ux, -uy, -uz  # inverse warping

    if smooth_sigma_grad is not None:
        s = (smooth_sigma_grad / sx, smooth_sigma_grad / sy, smooth_sigma_grad / sz)
        ux = cast(np.ndarray, ndimage.gaussian_filter(ux, s))
        uy = cast(np.ndarray, ndimage.gaussian_filter(uy, s))
        uz = cast(np.ndarray, ndimage.gaussian_filter(uz, s))

    # Normalize gradient magnitude to 1
    gradient_magnitude = np.sqrt(ux**2 + uy**2 + uz**2) + 1e-8
    ux /= gradient_magnitude
    uy /= gradient_magnitude
    uz /= gradient_magnitude

    # Optionally scale by external field
    if scaling_field is not None:
        ux *= scaling_field
        uy *= scaling_field
        uz *= scaling_field

    # Optionally smooth deformation field
    if smooth_sigma_field is not None:
        s = (smooth_sigma_field / sx, smooth_sigma_field / sy, smooth_sigma_field / sz)
        ux = cast(np.ndarray, ndimage.gaussian_filter(ux, s))
        uy = cast(np.ndarray, ndimage.gaussian_filter(uy, s))
        uz = cast(np.ndarray, ndimage.gaussian_filter(uz, s))

    # Convert mm to voxel indices
    ux /= sx
    uy /= sy
    uz /= sz

    return ux, uy, uz


def get_random_deformation_field(
    shape: tuple[int, int, int],
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    scaling_field: np.ndarray | RealNoBool | None = None,
    corr_sigma: _DistanceType | None = None,
    hpf_sigma: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = None,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random 3D deformation field that is optionally correlated and high-pass
    filtered to control clustering.

    Args:
        shape (tuple[int, int, int]):
            The shape of the deformation field in voxels.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the deformation field in mm.
        corr_sigma (float | None, optional):
            The sigma used for the correlating the deformation field via Gaussian smoothing.
            Defaults to ``None``; i.e., no correlation is applied.
        hpf_sigma (float | None, optional):
            The sigma used for the high-pass filtering the deformation field.
            Defaults to ``None``; i.e., no high-pass filtering is applied.
        scaling_field (np.ndarray | float | int | None, optional):
            Per-voxel scaling array or a uniform scalar multiplier.
            Defaults to ``None``; i.e., no scaling is applied.
        smooth_sigma_field (float | None, optional):
            The sigma used for smoothing the resulting deformation field. Defaults to ``None``;
            i.e., no smoothing is applied.
        random_seed (int | None, optional):
            The random seed for the deformation field. Defaults to ``None``.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            The random state for the deformation field. Defaults to ``None``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The x, y, and z components of the deformation field.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If ``spacing`` is invalid or if ``corr_sigma`` or ``hpf_sigma`` are invalid.
    """
    validate_obj_type(shape, "shape", tuple)
    if len(shape) != 3:
        raise ValueError("`shape` must contain 3 elements")
    if not all(isinstance(s, int) and s > 0 for s in shape):
        raise ValueError("`shape` must be a tuple of positive integers")

    validate_spacing(spacing, "spacing")
    sx, sy, sz = spacing

    if scaling_field is not None:
        validate_obj_type(scaling_field, "scaling_field", (np.ndarray, RealNoBool))
        if isinstance(scaling_field, np.ndarray):
            validate_3d_numpy_array(
                scaling_field, "scaling_field", dtype=np.floating, shape=shape
            )

    validate_obj_type(
        smooth_sigma_field, "smooth_sigma_field", (RealNoBool, type(None))
    )
    if smooth_sigma_field is not None and float(smooth_sigma_field) <= 0.0:
        raise ValueError("`smooth_sigma_field` must be greater than 0")

    rng = get_rng(
        seed=random_seed, random_state=random_state, offset=SEED_OFFSET_GYRATION
    )

    ux = generate_noise_field(
        shape=shape,
        spacing=spacing,
        corr_sigma=corr_sigma,
        hpf_sigma=hpf_sigma,
        random_state=rng,
    )
    uy = generate_noise_field(
        shape=shape,
        spacing=spacing,
        corr_sigma=corr_sigma,
        hpf_sigma=hpf_sigma,
        random_state=rng,
    )
    uz = generate_noise_field(
        shape=shape,
        spacing=spacing,
        corr_sigma=corr_sigma,
        hpf_sigma=hpf_sigma,
        random_state=rng,
    )

    # Center to zero mean
    ux -= ux.mean()
    uy -= uy.mean()
    uz -= uz.mean()

    # Normalize to 1mm length
    norm = np.sqrt(ux**2 + uy**2 + uz**2) + 1e-8
    ux /= norm
    uy /= norm
    uz /= norm

    if scaling_field is not None:
        ux *= scaling_field
        uy *= scaling_field
        uz *= scaling_field

    # Final smoothing of deformation field
    if smooth_sigma_field is not None:
        s = (smooth_sigma_field / sx, smooth_sigma_field / sy, smooth_sigma_field / sz)
        ux = ndimage.gaussian_filter(ux, s)
        uy = ndimage.gaussian_filter(uy, s)
        uz = ndimage.gaussian_filter(uz, s)

    # Convert mm to voxel indices
    ux /= sx
    uy /= sy
    uz /= sz

    return ux, uy, uz


def compose_deformation_fields(
    field_first: tuple[np.ndarray, np.ndarray, np.ndarray],
    field_second: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compose two pull-based deformation fields.

    The returned field is equivalent to applying ``field_first`` first and
    ``field_second`` second using ``warp_image``:

    ``warp_image(warp_image(image, field_first), field_second)``

    Args:
        field_first:
            First pull-based deformation field.
        field_second:
            Second pull-based deformation field.
        meshgrid:
            Optional precomputed meshgrid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The composed pull-based deformation field.
    """
    validate_obj_type(field_first, "field_first", tuple)
    if len(field_first) != 3:
        raise ValueError("`deformation_field` must be a tuple of length 3")
    for i, obj in enumerate(field_first):
        if i == 0:
            validate_3d_numpy_array(obj, f"deformation_field_{i}", dtype=np.floating)
            continue
        validate_3d_numpy_array(
            obj, f"deformation_field_{i}", dtype=np.floating, shape=field_first[0].shape
        )

    validate_obj_type(field_second, "field_second", tuple)
    if len(field_second) != 3:
        raise ValueError("`field_second` must be a tuple of length 3")
    for i, obj in enumerate(field_second):
        validate_3d_numpy_array(
            obj, f"field_second_{i}", dtype=np.floating, shape=field_first[0].shape
        )

    ax, ay, az = field_first
    bx, by, bz = field_second

    if meshgrid is None:
        meshgrid = create_meshgrid(ax.shape)

    ax_warped = warp_image(ax, field_second, meshgrid=meshgrid, interp_order=1)
    ay_warped = warp_image(ay, field_second, meshgrid=meshgrid, interp_order=1)
    az_warped = warp_image(az, field_second, meshgrid=meshgrid, interp_order=1)

    return (
        bx + ax_warped,
        by + ay_warped,
        bz + az_warped,
    )


def exponentiate_velocity_field(
    velocity_field: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    scaling_steps: int = 5,
    smooth_sigma: _DistanceType | None = None,
    meshgrid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exponentiate a stationary velocity field using scaling and squaring.

    The input field is interpreted as a stationary velocity field in voxel
    units, using the same pull-based convention as ``warp_image``.

    Conceptually, this computes:

    ``phi = exp(v)``

    where ``phi`` is represented as a pull displacement field.

    Args:
        velocity_field (tuple[np.ndarray, np.ndarray, np.ndarray]):
            Tuple containing the x, y, and z components of the velocity field.
            Components must be in voxel units, not mm.
        spacing (tuple[float | int, float | int, float | int]):
            Voxel spacing in mm.
        scaling_steps (int, optional):
            Number of scaling-and-squaring steps. Larger values make the
            initial deformation closer to identity. Defaults to ``5``.
        smooth_sigma (float | None, optional):
            Optional Gaussian smoothing sigma in mm applied to the velocity
            field before exponentiation.
        meshgrid (tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional):
            Optional precomputed meshgrid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The exponentiated pull-based displacement field.

    Raises:
        TypeError:
            If argument types are invalid.
        ValueError:
            If values are invalid.

    """
    validate_obj_type(velocity_field, "deformation_field", tuple)
    if len(velocity_field) != 3:
        raise ValueError("`deformation_field` must be a tuple of length 3")
    for i, obj in enumerate(velocity_field):
        if i == 0:
            validate_3d_numpy_array(obj, f"deformation_field_{i}", dtype=np.floating)
            continue
        validate_3d_numpy_array(
            obj,
            f"deformation_field_{i}",
            dtype=np.floating,
            shape=velocity_field[0].shape,
        )

    validate_obj_type(meshgrid, "meshgrid", (tuple, type(None)))
    if meshgrid is not None:
        if len(meshgrid) != 3:
            raise ValueError("`meshgrid` must be a tuple of length 3")
        for i, obj in enumerate(meshgrid):
            validate_3d_numpy_array(
                obj, f"meshgrid_{i}", dtype=(np.number), shape=velocity_field[0].shape
            )

    validate_spacing(spacing, "spacing")
    sx, sy, sz = spacing

    validate_obj_type(scaling_steps, "scaling_steps", IntNoBool)
    if scaling_steps < 0:
        raise ValueError("`scaling_steps` must be non-negative")

    validate_obj_type(smooth_sigma, "smooth_sigma", (RealNoBool, type(None)))
    if smooth_sigma is not None and float(smooth_sigma) <= 0.0:
        raise ValueError("`smooth_sigma` must be positive")

    vx, vy, vz = velocity_field

    # Copy to avoid mutating caller-owned arrays.
    vx = vx.astype(np.float64, copy=True)
    vy = vy.astype(np.float64, copy=True)
    vz = vz.astype(np.float64, copy=True)

    if smooth_sigma is not None:
        s = (
            float(smooth_sigma) / float(sx),
            float(smooth_sigma) / float(sy),
            float(smooth_sigma) / float(sz),
        )
        vx = ndimage.gaussian_filter(vx, s)
        vy = ndimage.gaussian_filter(vy, s)
        vz = ndimage.gaussian_filter(vz, s)

    if meshgrid is None:
        meshgrid = create_meshgrid(cast(tuple[int, int, int], vx.shape))

    scale = float(2**scaling_steps)

    ux = vx / scale
    uy = vy / scale
    uz = vz / scale

    field = (ux, uy, uz)

    for _ in range(scaling_steps):
        field = compose_deformation_fields(
            field,
            field,
            meshgrid=meshgrid,
        )

    return field


def cortical_expansion(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    n_iters: int = 1,
    mm_per_iter: _DistanceType = 1.0,
    normal_to: Literal["gm_wm", "pial"] = "gm_wm",
    pial_lower_bound: _DistanceType = -1.0,
    pial_upper_bound: _DistanceType | None = None,
    gwb_lower_bound: _DistanceType | None = None,
    gwb_upper_bound: _DistanceType = 2.0,
    edge_rolloff: _DistanceType = 1.0,
    smooth_sigma_grad: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = SMOOTH_SIGMA_DEFORM,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _DeformationResultType:
    """
    Apply cortical expansion to the image and segmentation mask.

    Args:
        image (np.ndarray):
            The input image to apply the cortical expansion to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing the expansion.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the expansion to. Defaults to ``None``.
        n_iters (int, optional):
            The number of iterations to apply the expansion. Together with ``mm_per_iter``
            it determines the total cortical expansion in mm. Defaults to ``1``.
        mm_per_iter (float | int, optional):
            The amount of cortical expansion in mm per iteration. Defaults to ``1.0`` mm.
        normal_to (Literal["gm_wm", "pial"], optional):
            The surface to use for computing the normal vectors that correspond
            to the expansion direction. Defaults to ``"gm_wm"``.
        pial_lower_bound (float | int, optional):
            The lower bound to constrain the cortical expansion, defined in terms of
            the signed distance to the pial surface. Defaults to ``-1.0`` mm.
        pial_upper_bound (float | int, optional):
            The upper bound to constrain the cortical expansion, defined in terms of
            the signed distance to the pial surface. Defaults to ``None``. If provided,
            the stricter upper bound between this and ``gwb_upper_bound`` (see below)
            prevails.
        gwb_lower_bound (float | int, optional):
            The lower bound to constrain the cortical expansion, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``None``. If provided,
            stricter lower bound between this and ``pial_lower_bound`` prevails.
        gwb_upper_bound (float | int, optional):
            The upper bound to constrain the cortical expansion, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``2.0`` mm.
        edge_rolloff (float | int, optional):
            The rolloff distance in mm from the boundaries defined by the above settings
            for the effect to reach 90% of its maximum value.
        smooth_sigma_grad (float | int, optional):
            The sigma used for the smoothing the vector field normal to the GM-WM boundary,
            prior to computing the deformation field. Defaults to ``1.0`` mm.
        smooth_sigma_field (float | int, optional):
            The sigma used for the smoothing the resulting deformation field.
            Defaults to ``0.5`` mm.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image, segmentation mask, and deformation field.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"out_seg_mask"``: the resulting segmentation mask.
            - ``"effect_field"``: a list of tuples of 3 numpy arrays, each representing the
              x, y, and z components of the deformation field for each iteration.

    Warnings:
        UserWarning:
            If no region to expand within the selected bounds is found at a given iteration,
            the expansion is stopped and a warning is issued.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If invalid values are provided for numerical arguments, as well as any
            shape mismatches in provided arrays.
    """
    validate_3d_numpy_array(image, "image", dtype=np.floating)
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=np.integer, shape=image.shape)
    validate_spacing(spacing, "spacing")

    if app_field is not None:
        validate_3d_numpy_array(
            app_field, "app_field", dtype=np.floating, shape=image.shape
        )
        if float(app_field.sum()) <= 1e-8:
            warnings.warn(
                "Application field has no support; returning image unchanged.",
                UserWarning,
            )
            return {
                "out_image": image,
                "out_seg_mask": seg_mask,
                "effect_field": [],
            }

    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    validate_obj_type(n_iters, "n_iters", IntNoBool)
    if n_iters <= 0:
        raise ValueError("`n_iters` must be positive")

    validate_obj_type(mm_per_iter, "mm_per_iter", RealNoBool)
    if float(mm_per_iter) == 0.0:
        raise ValueError("`mm_per_iter` must be non-zero")

    validate_literal_str(normal_to, "normal_to", ("gm_wm", "pial"))

    validate_obj_type(pial_lower_bound, "pial_lower_bound", RealNoBool)

    validate_obj_type(pial_upper_bound, "pial_upper_bound", (RealNoBool, type(None)))
    if pial_upper_bound is not None and float(pial_upper_bound) <= float(
        pial_lower_bound
    ):
        raise ValueError("`pial_upper_bound` must be greater than `pial_lower_bound`")

    validate_obj_type(gwb_lower_bound, "gwb_lower_bound", (RealNoBool, type(None)))

    validate_obj_type(gwb_upper_bound, "gwb_upper_bound", RealNoBool)
    if gwb_lower_bound is not None and float(gwb_lower_bound) >= float(gwb_upper_bound):
        raise ValueError("`gwb_lower_bound` must be less than `gwb_upper_bound`")

    validate_obj_type(edge_rolloff, "edge_rolloff", RealNoBool)
    if float(edge_rolloff) < 0.0:
        raise ValueError("`edge_rolloff` must be positive")

    validate_obj_type(smooth_sigma_grad, "smooth_sigma_grad", (RealNoBool, type(None)))
    if smooth_sigma_grad is not None and float(smooth_sigma_grad) <= 0.0:
        raise ValueError("`smooth_sigma_grad` must be positive")

    validate_obj_type(
        smooth_sigma_field, "smooth_sigma_field", (RealNoBool, type(None))
    )
    if smooth_sigma_field is not None and float(smooth_sigma_field) <= 0.0:
        raise ValueError("`smooth_sigma_field` must be positive")

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    # Meshgrid and deformation field cache for all iterations
    meshgrid = create_meshgrid(image.shape)
    deformation_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(n_iters):
        # Get signed distance fields
        sdf_bound, sdf_pial = compute_distance_fields(
            seg_mask,
            spacing=spacing,
            surface="both",
            label_enum=label_enum,
        )

        # Define region to apply expansion
        # 1. with respect to the cortical ribbon
        spread_field = sdf_pial >= float(pial_lower_bound)
        spread_field &= sdf_bound <= float(gwb_upper_bound)
        if pial_upper_bound is not None:
            spread_field &= sdf_pial <= float(pial_upper_bound)
        if gwb_lower_bound is not None:
            spread_field &= sdf_bound >= float(gwb_lower_bound)
        # 2. with respect to the overall brain
        spread_field &= ~np.isin(
            seg_mask, list(label_enum.get_background_labels())
        )  # restrict to brain
        if hemisphere is not None:  # optionally restrict to hemisphere
            spread_field &= np.isin(
                seg_mask,
                list(label_enum.get_all_labels(hemisphere=hemisphere)),
            )

        # Exit due to invalid user-provided bounds (e.g., lower >= upper bound)
        # or if tissue segmentation masks become empty at any iteration
        if not spread_field.any():
            warnings.warn(
                f"No region to expand within the selected bounds; exiting at iteration {i+1}",
                UserWarning,
            )
            break

        # Get soft effect spread field
        spread_field = generate_application_field(
            mask=spread_field,
            spacing=spacing,
            field_type="inward",
            rolloff_mm=edge_rolloff,
        )
        if app_field is not None:
            spread_field *= app_field

        # Compute a displacement field normal to the GM-WM boundary
        # TODO: come back to check smoothing
        ux, uy, uz = get_deformation_fields_from_sdf(
            sdf=sdf_bound if normal_to == "gm_wm" else sdf_pial,
            spacing=spacing,
            scaling_field=spread_field * mm_per_iter,
            smooth_sigma_grad=smooth_sigma_grad,
            smooth_sigma_field=smooth_sigma_field,
        )

        # Ensure no leakage outside the brain (e.g., due to smoothing)
        ux *= spread_field > 0
        uy *= spread_field > 0
        uz *= spread_field > 0

        # Warp image and masks
        image = warp_image(image, (ux, uy, uz), meshgrid=meshgrid)
        seg_mask = warp_image(seg_mask, (ux, uy, uz), meshgrid=meshgrid, interp_order=0)
        if app_field is not None:
            app_field = warp_image(
                app_field, (ux, uy, uz), meshgrid=meshgrid, interp_order=0
            )
        deformation_fields.append((ux, uy, uz))

    return {
        "out_image": image,
        "out_seg_mask": seg_mask,
        "effect_field": deformation_fields,
    }


def cortical_thickening(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    n_iters: int = 1,
    mm_per_iter: _DistanceType = 1.0,
    pial_lower_bound: _DistanceType = 2.0,
    gwb_lower_bound: _DistanceType | None = None,
    gwb_upper_bound: _DistanceType = 2.0,
    edge_rolloff: _DistanceType = 1.0,
    smooth_sigma_grad: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = SMOOTH_SIGMA_DEFORM,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _DeformationResultType:
    """
    Apply cortical thickening to the image and segmentation mask.

    Notes:
        - This function is a wrapper around ``cortical_expansion`` with safety guarantees
          so that cortical thickening always occurs, by enforcing lower and upper bounds below
          and above the GM-WM boundary, respectively, by preventing movement inward (toward CSF),
          and by setting a lower bound above the pial surface to avoid interference with sulcal
          widening.
        - The upper bound is solely defined w.r.t. the GM-WM boundary.
        - The cortical thickening direction vectors are defined based on the GM-WM boundary.

    Args:
        image (np.ndarray):
            The input image to apply the cortical thickening to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing the thickening.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the thickening to. Defaults to ``None``.
        n_iters (int, optional):
            The number of iterations to apply the thickening. Together with ``mm_per_iter``
            it determines the total cortical thickening in mm. Defaults to ``1``.
        mm_per_iter (float | int, optional):
            The amount of cortical thickening in mm per iteration. Defaults to ``1.0`` mm.
        pial_lower_bound (float | int, optional):
            The lower bound to constrain the cortical thickening, defined in terms of
            the signed distance to the pial surface. Defaults to ``2.0`` mm.
        gwb_lower_bound (float | int, optional):
            The lower bound to constrain the cortical thickening, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``None``. If provided,
            stricter lower bound between this and ``pial_lower_bound`` prevails.
        gwb_upper_bound (float | int, optional):
            The upper bound to constrain the cortical thickening, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``2.0`` mm.
        edge_rolloff (float | int, optional):
            The rolloff distance in mm for smooth decay of the effect near the boundaries,
            defined by the above ``*_bound`` settings. Defaults to ``1.0`` mm.
        smooth_sigma_grad (float | int, optional):
            The sigma used for the smoothing of the vector field normal to the GM-WM boundary,
            prior to computing the deformation field. Defaults to ``1.0`` mm.
        smooth_sigma_field (float | int, optional):
            The sigma used for the smoothing the resulting deformation field.
            Defaults to ``0.5`` mm.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image, segmentation mask, and deformation field.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"out_seg_mask"``: the resulting segmentation mask.
            - ``"effect_field"``: a list of tuples of 3 numpy arrays, each representing the
              x, y, and z components of the deformation field for each iteration.

    Warnings:
        UserWarning:
            If no region to expand within the selected bounds is found at a given iteration,
            the expansion is stopped and a warning is issued.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If invalid values are provided for numerical arguments, as well as any
            shape mismatches in provided arrays.
            If any of the ``*_bound`` settings prevent cortical thickening.
    """
    validate_obj_type(gwb_upper_bound, "gwb_upper_bound", RealNoBool)
    if float(gwb_upper_bound) <= 0.0:
        raise ValueError("`gwb_upper_bound` must be positive")

    validate_obj_type(gwb_lower_bound, "gwb_lower_bound", (RealNoBool, type(None)))
    if gwb_lower_bound is not None and float(gwb_lower_bound) > 0.0:
        raise ValueError("`gwb_lower_bound` must not be greater than 0")

    validate_obj_type(
        mm_per_iter, "mm_per_iter", RealNoBool
    )  # avoid movement inward the CSF
    if float(mm_per_iter) <= 0.0:
        raise ValueError("`mm_per_iter` must be positive")

    validate_obj_type(pial_lower_bound, "pial_lower_bound", RealNoBool)
    if float(pial_lower_bound) <= float(mm_per_iter):
        raise ValueError("`pial_lower_bound` must be greater than `mm_per_iter`")

    return cortical_expansion(
        image=image,
        seg_mask=seg_mask,
        spacing=spacing,
        app_field=app_field,
        hemisphere=hemisphere,
        n_iters=n_iters,
        mm_per_iter=mm_per_iter,
        normal_to="gm_wm",
        pial_lower_bound=pial_lower_bound,
        gwb_lower_bound=gwb_lower_bound,
        gwb_upper_bound=gwb_upper_bound,
        edge_rolloff=edge_rolloff,
        smooth_sigma_grad=smooth_sigma_grad,
        smooth_sigma_field=smooth_sigma_field,
        label_enum=label_enum,
        **kwargs,
    )


def sulcal_widening(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    n_iters: int = 1,
    mm_per_iter: _DistanceType = 1.0,
    pial_lower_bound: _DistanceType = -1.0,
    pial_upper_bound: _DistanceType | None = None,
    gwb_upper_bound: _DistanceType = -3.0,
    edge_rolloff: _DistanceType = 1.0,
    smooth_sigma_grad: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = SMOOTH_SIGMA_DEFORM,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _DeformationResultType:
    """
    Apply sulcal widening to the image and segmentation mask.

    Notes:
        - This function is a wrapper around ``cortical_expansion`` with safety guarantees
          so that sulcal widening always occurs, by enforcing a lower and upper bound below
          and above the pial surface, respectively, and by preventing movement inward (toward CSF).
        - The lower bound is solely defined w.r.t. the pial surface.
        - The sulcal widening direction vectors are defined based on the pial surface.

    Args:
        image (np.ndarray):
            The input image to apply the sulcal widening to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing the widening.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the widening to. Defaults to ``None``.
        n_iters (int, optional):
            The number of iterations to apply the widening. Together with ``mm_per_iter``
            it determines the total sulcal widening in mm. Defaults to ``1``.
        mm_per_iter (float | int, optional):
            The amount of sulcal widening in mm per iteration. Defaults to ``1.0`` mm.
        pial_lower_bound (float | int, optional):
            The lower bound to constrain the sulcal widening, defined in terms of
            the signed distance to the pial surface. Defaults to ``-1.0`` mm.
        pial_upper_bound (float | int, optional):
            The upper bound to constrain the sulcal widening, defined in terms of
            the signed distance to the pial surface. Defaults to ``None``. If provided,
            the stricter upper bound between this and ``gwb_upper_bound`` (see below)
            prevails.
        gwb_upper_bound (float | int, optional):
            The upper bound to constrain the sulcal widening, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``-3.0`` mm. If
            positive, cortical thickening will also occur (allowed).
        edge_rolloff (float | int, optional):
            The rolloff distance in mm for smooth decay of the effect near the boundaries,
            defined by the above ``*_bound`` settings. Defaults to ``1.0`` mm.
        smooth_sigma_grad (float | int, optional):
            The sigma used for the smoothing of the vector field normal to the pial surface,
            prior to computing the deformation field. Defaults to ``1.0`` mm.
        smooth_sigma_field (float | int, optional):
            The sigma used for the smoothing the resulting deformation field.
            Defaults to ``0.5`` mm.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image, segmentation mask, and deformation field.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"out_seg_mask"``: the resulting segmentation mask.
            - ``"effect_field"``: a list of tuples of 3 numpy arrays, each representing the
              x, y, and z components of the deformation field for each iteration.

    Warnings:
        UserWarning:
            If no region to widen within the selected bounds is found at a given iteration,
            the widening is stopped and a warning is issued.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If invalid values are provided for numerical arguments, as well as any
            shape mismatches in provided arrays.
            If any of the ``*_bound`` settings prevent cortical thickening.
    """
    validate_obj_type(pial_lower_bound, "pial_lower_bound", RealNoBool)
    if float(pial_lower_bound) > 0.0:
        raise ValueError("`pial_lower_bound` must not be greater than 0")

    validate_obj_type(pial_upper_bound, "pial_upper_bound", (RealNoBool, type(None)))
    if pial_upper_bound is not None and float(pial_upper_bound) <= 0.0:
        raise ValueError("`pial_upper_bound` must be greater than 0")

    validate_obj_type(mm_per_iter, "mm_per_iter", RealNoBool)
    if float(mm_per_iter) <= 0.0:
        raise ValueError("`mm_per_iter` must be positive")

    return cortical_expansion(
        image=image,
        seg_mask=seg_mask,
        spacing=spacing,
        app_field=app_field,
        hemisphere=hemisphere,
        n_iters=n_iters,
        mm_per_iter=mm_per_iter,
        normal_to="pial",
        pial_lower_bound=pial_lower_bound,
        pial_upper_bound=pial_upper_bound,
        gwb_upper_bound=gwb_upper_bound,
        edge_rolloff=edge_rolloff,
        smooth_sigma_grad=smooth_sigma_grad,
        smooth_sigma_field=smooth_sigma_field,
        label_enum=label_enum,
        **kwargs,
    )


def abnormal_gyration(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    n_iters: int = 1,
    mm_per_iter: _DistanceType = 1.0,
    pial_lower_bound: _DistanceType = -1.0,
    pial_upper_bound: _DistanceType | None = None,
    gwb_lower_bound: _DistanceType | None = None,
    gwb_upper_bound: _DistanceType = 3.0,
    edge_rolloff: _DistanceType = 1.0,
    corr_sigma: _DistanceType | None = 10.0,
    hpf_sigma: _DistanceType | None = None,
    smooth_sigma_field: _DistanceType | None = SMOOTH_SIGMA_DEFORM,
    scaling_and_squaring_steps: int | None = 5,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _DeformationResultType:
    """
    Apply abnormal gyration to the image and segmentation mask using a random 3D
    displacement field.

    If ``scaling_and_squaring_steps`` is provided, the generated field per
    iteration is interpreted as a stationary velocity field and integrated with
    scaling-and-squaring before warping. This produces a numerical approximation
    of the time-1 diffeomorphic flow associated with the velocity field and aims
    to produce a smoother and more invertible deformation, given the free-form
    nature of the effect.

    Args:
        image (np.ndarray):
            The input image to apply the abnormal gyration to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing the gyration.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the gyration to. Defaults to ``None``.
        n_iters (int, optional):
            The number of iterations to apply the gyration. Together with ``mm_per_iter``
            it determines the total abnormal gyration in mm. Defaults to ``1``.
        mm_per_iter (float | int, optional):
            The amount of abnormal gyration in mm per iteration. Defaults to ``1.0`` mm.
        pial_lower_bound (float | int, optional):
            The lower bound to constrain the abnormal gyration, defined in terms of
            the signed distance to the pial surface. Defaults to ``-1.0`` mm.
        pial_upper_bound (float | int, optional):
            The upper bound to constrain the abnormal gyration, defined in terms of
            the signed distance to the pial surface. Defaults to ``None``. If provided,
            the stricter upper bound between this and ``gwb_upper_bound`` (see below)
            prevails.
        gwb_lower_bound (float | int, optional):
            The lower bound to constrain the abnormal gyration, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``None``. If provided,
            stricter lower bound between this and ``pial_lower_bound`` prevails.
        gwb_upper_bound (float | int, optional):
            The upper bound to constrain the abnormal gyration, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``3.0`` mm.
        edge_rolloff (float | int, optional):
            The rolloff distance in mm for smooth decay of the effect near the boundaries,
            defined by the above ``*_bound`` settings. Defaults to ``1.0`` mm.
        corr_sigma (float | None, optional):
            The sigma used for the correlating the deformation field via Gaussian smoothing.
            Defaults to ``10.0`` mm.
        hpf_sigma (float | None, optional):
            The sigma used for the high-pass filtering the deformation field.
            Defaults to ``None``; i.e., no high-pass filtering is applied.
        smooth_sigma_field (float | int, optional):
            The sigma used for the smoothing the resulting deformation field.
            Defaults to ``0.5`` mm.
        scaling_and_squaring_steps (int, optional):
            Number of scaling-and-squaring steps used to numerically integrate the
            generated field as a stationary velocity field. If ``None``, the generated field is
            applied directly as a displacement field.
        random_seed (int | None, optional):
            The random seed for the deformation field. Defaults to ``None``.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            The random state for the deformation field. Defaults to ``None``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image, segmentation mask, and deformation field.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"out_seg_mask"``: the resulting segmentation mask.
            - ``"effect_field"``: a list of tuples of 3 numpy arrays, each representing the
              x, y, and z components of the deformation field for each iteration.

    Warnings:
        UserWarning:
            If no region to gyrate within the selected bounds is found at a given iteration,
            the gyration is stopped and a warning is issued.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If invalid values are provided for numerical arguments, as well as any
            shape mismatches in provided arrays.
    """
    validate_3d_numpy_array(image, "image", dtype=np.floating)
    validate_3d_numpy_array(seg_mask, "seg_mask", dtype=np.integer, shape=image.shape)
    validate_spacing(spacing, "spacing")

    if app_field is not None:
        validate_3d_numpy_array(
            app_field, "app_field", dtype=np.floating, shape=image.shape
        )
        if float(app_field.sum()) <= 1e-8:
            warnings.warn(
                "Application field has no support; returning image unchanged.",
                UserWarning,
            )
            return {
                "out_image": image,
                "out_seg_mask": seg_mask,
                "effect_field": [],
            }

    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    validate_obj_type(n_iters, "n_iters", IntNoBool)
    if n_iters <= 0:
        raise ValueError("`n_iters` must be positive")

    validate_obj_type(mm_per_iter, "mm_per_iter", RealNoBool)
    if float(mm_per_iter) == 0.0:
        raise ValueError("`mm_per_iter` must be non-zero")

    validate_obj_type(pial_lower_bound, "pial_lower_bound", RealNoBool)

    validate_obj_type(pial_upper_bound, "pial_upper_bound", (RealNoBool, type(None)))
    if pial_upper_bound is not None and float(pial_upper_bound) <= float(
        pial_lower_bound
    ):
        raise ValueError("`pial_upper_bound` must be greater than `pial_lower_bound`")

    validate_obj_type(gwb_lower_bound, "gwb_lower_bound", (RealNoBool, type(None)))

    validate_obj_type(gwb_upper_bound, "gwb_upper_bound", RealNoBool)
    if gwb_lower_bound is not None and float(gwb_lower_bound) >= float(gwb_upper_bound):
        raise ValueError("`gwb_lower_bound` must be less than `gwb_upper_bound`")

    validate_obj_type(edge_rolloff, "edge_rolloff", RealNoBool)
    if float(edge_rolloff) < 0.0:
        raise ValueError("`edge_rolloff` must be positive")

    validate_obj_type(
        smooth_sigma_field, "smooth_sigma_field", (RealNoBool, type(None))
    )
    if smooth_sigma_field is not None and float(smooth_sigma_field) <= 0.0:
        raise ValueError("`smooth_sigma_field` must be positive")

    if scaling_and_squaring_steps is not None:
        validate_obj_type(
            scaling_and_squaring_steps, "scaling_and_squaring_steps", IntNoBool
        )
        if scaling_and_squaring_steps <= 0:
            raise ValueError("`scaling_and_squaring_steps` must be positive")

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    rng = get_rng(
        seed=random_seed, random_state=random_state, offset=SEED_OFFSET_GYRATION
    )

    meshgrid = create_meshgrid(image.shape)
    deformation_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(n_iters):
        # Get signed distance fields
        sdf_bound, sdf_pial = compute_distance_fields(
            seg_mask,
            spacing=spacing,
            surface="both",
            label_enum=label_enum,
        )

        # Define region to apply expansion
        # 1. with respect to the cortical ribbon
        spread_field = sdf_pial >= float(pial_lower_bound)
        spread_field &= sdf_bound <= float(gwb_upper_bound)
        if pial_upper_bound is not None:
            spread_field &= sdf_pial <= float(pial_upper_bound)
        if gwb_lower_bound is not None:
            spread_field &= sdf_bound >= float(gwb_lower_bound)
        # 2. with respect to the overall brain
        spread_field &= ~np.isin(
            seg_mask, list(label_enum.get_background_labels())
        )  # restrict to brain
        if hemisphere is not None:  # optionally restrict to hemisphere
            spread_field &= np.isin(
                seg_mask,
                list(label_enum.get_all_labels(hemisphere=hemisphere)),
            )

        # Exit due to invalid user-provided bounds (e.g., lower >= upper bound)
        # or if tissue segmentation masks become empty at any iteration
        if not spread_field.any():
            warnings.warn(
                f"No region to expand within the selected bounds; exiting at iteration {i+1}",
                UserWarning,
            )
            break

        # Get soft effect spread field
        spread_field = generate_application_field(
            mask=spread_field,
            spacing=spacing,
            field_type="inward",
            rolloff_mm=edge_rolloff,
        )
        if app_field is not None:
            spread_field *= app_field

        # Get random deformation field
        ux, uy, uz = get_random_deformation_field(
            shape=image.shape,
            spacing=spacing,
            scaling_field=spread_field * mm_per_iter,
            corr_sigma=corr_sigma,
            hpf_sigma=hpf_sigma,
            smooth_sigma_field=smooth_sigma_field,
            random_state=rng,
        )

        if scaling_and_squaring_steps is not None:
            ux, uy, uz = exponentiate_velocity_field(
                velocity_field=(ux, uy, uz),
                spacing=spacing,
                scaling_steps=scaling_and_squaring_steps,
                meshgrid=meshgrid,
                smooth_sigma=None,
            )

        # Ensure no leakage outside the brain (e.g., due to smoothing)
        ux *= spread_field > 0
        uy *= spread_field > 0
        uz *= spread_field > 0

        # Warp image and masks
        image = warp_image(image, (ux, uy, uz), meshgrid=meshgrid)
        seg_mask = warp_image(seg_mask, (ux, uy, uz), meshgrid=meshgrid, interp_order=0)
        if app_field is not None:
            app_field = warp_image(
                app_field, (ux, uy, uz), meshgrid=meshgrid, interp_order=0
            )
        deformation_fields.append((ux, uy, uz))

    return {
        "out_image": image,
        "out_seg_mask": seg_mask,
        "effect_field": deformation_fields,
    }
