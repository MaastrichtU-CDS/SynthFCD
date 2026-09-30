"""
Helper functions for generating diffusion fields.
"""

from __future__ import annotations

__all__ = [
    "boundary_blurring",
    "compute_conduction_coefficient",
    "compute_divergence",
    "hyperintensity",
    "perona_malik",
]

import warnings
from typing import Any

import numpy as np

from synthfcd.core.masks import get_binary_mask
from synthfcd.core.utils import (
    compute_distance_fields,
    generate_application_field,
    generate_noise_field,
)
from synthfcd.utils._aliases import (
    _DistanceType,
    _HemisphereType,
    _IntensityResultType,
    _LabelEnumType,
    _RNGType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._const import (
    DIFF_UPDATE_RATE,
    GRAD_QUANTILE_KAPPA,
    SEED_OFFSET_HYPERINTENSITY,
    WMH_SEEDS_ALLOWED_FIELD_THRES,
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
from synthfcd.utils.misc import get_rng
from synthfcd.utils.seg_labels import SynthSegLabel


def compute_conduction_coefficient(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    kappa: float = GRAD_QUANTILE_KAPPA,
    mask: np.ndarray | None = None,
    *,
    allow_fixed: bool = True,
) -> np.ndarray:
    r"""Computes the conduction coefficient field for a 3D gradient for a Perona-Malik-like
    anisotropic diffusion equation, as in: https://arxiv.org/pdf/1412.6291 (Eq. 1.7).

    It also supports dynamic computation of the diffusion coefficient `kappa` as
    a quantile of the gradient magnitude distribution for consistent behavior across
    imaging sequences.

    Args:
        dx (np.ndarray):
            Gradient component in the x-axis direction. Must be a 3D floating array.
        dy (np.ndarray):
            Gradient component in the y-axis direction. Must match ``dx`` shape and dtype.
        dz (np.ndarray):
            Gradient component in the z-axis direction. Must match ``dx`` shape and dtype.
        kappa (float | int, optional):
            Values in ``(0, 1]`` are treated as a gradient-magnitude quantile used to
            estimate :math:`\\kappa`. Values ``> 1`` are used directly as a fixed
            threshold. Defaults to ``0.95``.
        mask (np.ndarray | None, optional):
            Optional boolean region-of-interest used to estimate :math:`\\kappa` when
            ``kappa`` is a quantile. Must match ``dx.shape``. Defaults to ``None``.
        allow_fixed (bool, optional):
            Reject fixed-threshold ``kappa`` values (``> 1``) if ``False``.
            Defaults to ``True``.

    Returns:
        np.ndarray:
            Conduction coefficient field :math:`c` with the same shape as the input gradients.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If arrays have incompatible shapes, if quantile ``kappa`` is not in ``(0, 1]``,
            or if no valid samples exist to estimate :math:`\\kappa``.

    """
    validate_3d_numpy_array(dx, "dx", dtype=np.floating)
    validate_3d_numpy_array(dy, "dy", dtype=np.floating, shape=dx.shape)
    validate_3d_numpy_array(dz, "dz", dtype=np.floating, shape=dx.shape)
    validate_obj_type(kappa, "kappa", RealNoBool)

    kappa_f = float(kappa)
    use_quantile = kappa_f <= 1.0
    if use_quantile:
        if not (0.0 < kappa_f <= 1.0):
            raise ValueError(
                "`kappa` must be a quantile in (0, 1] or a fixed threshold > 1, "
                f"got {kappa!r}."
            )
    elif not allow_fixed:
        raise ValueError(
            "`kappa` must be a quantile in (0, 1], got fixed threshold " f"{kappa!r}."
        )

    if mask is not None:
        validate_3d_numpy_array(mask, "mask", dtype=np.bool_, shape=dx.shape)

    norm = np.sqrt(dx**2 + dy**2 + dz**2)

    if use_quantile:
        sel = np.isfinite(norm) & (norm > 0)
        if mask is not None:
            sel &= mask
        samples = norm[sel]
        if samples.size == 0:
            raise ValueError(
                "No valid gradient magnitudes available to estimate kappa."
            )
        kappa = float(np.quantile(samples, kappa))
    else:
        kappa = float(kappa)

    return 1 / (1 + norm**2 / (kappa + 1e-12) ** 2)


def compute_gradients(
    image: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the gradients of an image using first-order differences.

    Args:
        image (np.ndarray):
            The input image to compute the gradients from.
        spacing (tuple[float | int, float | int, float | int], optional):
            The voxel spacing of the image in mm.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The gradients of the image in the x, y, and z directions.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If arrays have incompatible shapes.
    """
    validate_3d_numpy_array(image, "image", dtype=np.floating)
    validate_spacing(spacing, "spacing")

    sx, sy, sz = spacing

    dx = np.zeros_like(image)
    dx[:-1, :, :] = np.diff(image, axis=0) / sx
    dy = np.zeros_like(image)
    dy[:, :-1, :] = np.diff(image, axis=1) / sy
    dz = np.zeros_like(image)
    dz[:, :, :-1] = np.diff(image, axis=2) / sz

    return dx, dy, dz


def compute_divergence(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
) -> np.ndarray:
    """
    Compute the divergence of a vector field using first-order differences.

    Args:
        dx (np.ndarray):
            The gradient along the x-axis.
        dy (np.ndarray):
            The gradient along the y-axis.
        dz (np.ndarray):
            The gradient along the z-axis.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the gradient components in mm.

    Returns:
        np.ndarray:
            The divergence of the vector field.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If arrays have incompatible shapes.
    """
    validate_3d_numpy_array(dx, "dx", dtype=np.floating)
    validate_3d_numpy_array(dy, "dy", dtype=np.floating, shape=dx.shape)
    validate_3d_numpy_array(dz, "dz", dtype=np.floating, shape=dx.shape)
    validate_spacing(spacing, "spacing")

    sx, sy, sz = spacing

    div = np.zeros_like(dx)
    div[:-1, :, :] += dx[:-1, :, :] / sx
    div[1:, :, :] -= dx[:-1, :, :] / sx

    div[:, :-1, :] += dy[:, :-1, :] / sy
    div[:, 1:, :] -= dy[:, :-1, :] / sy

    div[:, :, :-1] += dz[:, :, :-1] / sz
    div[:, :, 1:] -= dz[:, :, :-1] / sz

    return div


def perona_malik(
    image: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    compute_c: bool = True,
    c_params: dict[str, Any] | None = None,
    external_c: np.ndarray | None = None,
) -> np.ndarray:
    r"""Single iteration of the Perona-Malik diffusion equation, as described
    in: https://arxiv.org/pdf/1412.6291 (Eq. 1.6), with optional support for
    external conduction coefficient fields.

    Note:
        The gradients and divergence are computed using first-order differences;
        see ``synthfcd.core.compute_gradients`` for more details.

    Args:
        image (np.ndarray):
            The input 3D image to compute the divergence from.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image in mm.
        compute_c (bool, optional):
            Whether to compute the conduction coefficient. Defaults to ``True``.
        external_c (np.ndarray | None, optional):
            Any external conduction coefficient field to apply to the image. Defaults to ``None``.
        c_params (dict[str, Any] | None, optional):
            Keyword arguments forwarded to
            :func:`~synthfcd.core.diffusion.compute_conduction_coefficient`:
            ``kappa``, ``mask``, and ``allow_fixed``. Defaults to ``None``.

    Returns:
        np.ndarray:
            The divergence of the input image after a single step of the Perona-Malik equation.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If arrays have incompatible shapes or no conduction

    """
    validate_3d_numpy_array(image, "image", dtype=np.floating)
    validate_spacing(spacing, "spacing")
    validate_obj_type(compute_c, "compute_c", bool)
    validate_obj_type(c_params, "c_params", (dict, type(None)))

    c_params = dict(c_params or {})
    c_params.pop("is_dynamic", None)

    if external_c is not None:
        validate_3d_numpy_array(
            external_c, "external_c", dtype=np.floating, shape=image.shape
        )

    dx, dy, dz = compute_gradients(image, spacing)

    c = 1.0
    if compute_c:
        c *= compute_conduction_coefficient(dx, dy, dz, **c_params)
    if external_c is not None:
        c *= external_c

    return compute_divergence(dx * c, dy * c, dz * c, spacing)


def boundary_blurring(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    n_iters: int = 10,
    update_rate: float = DIFF_UPDATE_RATE,
    kappa: float = GRAD_QUANTILE_KAPPA,
    pial_lower_bound: _DistanceType = 2.0,
    gwb_lower_bound: _DistanceType | None = None,
    gwb_upper_bound: _DistanceType = 2.0,
    edge_rolloff: _DistanceType = 1.0,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _IntensityResultType:
    r"""Apply blurring to the GM-WM boundary using the Perona-Malik Anisotropic Diffusion algorithm,
    described in: https://arxiv.org/pdf/1412.6291.

    Notes:
        - The lower bound is defined w.r.t. the pial surface and must be non-negative in
          order to avoid accidentally blurring the CSF-GM interface.
        - For the Perona-Malik implementation, see ``synthfcd.core.perona_malik``.
        - The Perona-Malik gradient-based "anisotropic" conduction coefficient is computed
          only once before the initial iteration (instead of every iteration), as empirical
          findings showed consistent blurring behavior, even in extremely low ``kappa`` values
          (high degree of "anisotropy").

    Args:
        image (np.ndarray):
            The input image to apply the blurring to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing the conduction coefficients
            and blurring boundaries.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the boundary blurring to. Defaults to ``None``.
        n_iters (int, optional):
            The number of iterations to apply the blurring. Defaults to ``10``.
        update_rate (float | int, optional):
            The update rate of the diffusion process. Defaults to ``0.1``.
        kappa (float | int, optional):
            Quantile in ``(0, 1]`` or fixed threshold ``> 1`` for the Perona-Malik
            conduction coefficient. Defaults to ``0.95``.
        pial_lower_bound (float | int, optional):
            The lower bound to constrain the boundary blurring, defined in terms of
            the signed distance to the pial surface. Defaults to ``2.0`` mm.
        pial_upper_bound (float | int, optional):
            The upper bound to constrain the boundary blurring, defined in terms of
            the signed distance to the pial surface. Defaults to ``None``. If provided,
            the stricter upper bound between this and ``gwb_upper_bound`` (see below)
            prevails.
        gwb_upper_bound (float | int, optional):
            The upper bound to constrain the boundary blurring, defined in terms of
            the signed distance to the GM-WM boundary. Defaults to ``2.0`` mm.
        edge_rolloff (float | int, optional):
            The rolloff distance in mm for smooth decay of the effect near the boundaries,
            defined by the above ``*_bound`` settings. Defaults to ``1.0`` mm.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image and effect field, as additive intensity.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"effect_field"``: the resulting effect field.

    Warnings:
        UserWarning:
            If no boundary to blur within the selected bounds is found, a warning is issued
            and the function returns the input image unchanged.

    Raises:
        TypeError:
            If argument types are incorrect.
        ValueError:
            If invalid values are provided for numerical arguments, as well as any
            shape mismatches in provided arrays.
            If any of the ``*_bound`` settings prevent boundary blurring.

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
                "effect_field": np.zeros_like(image),
            }

    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    validate_obj_type(n_iters, "n_iters", IntNoBool)
    if n_iters <= 0:
        raise ValueError("`n_iters` must be positive")

    validate_obj_type(pial_lower_bound, "pial_lower_bound", RealNoBool)
    if float(pial_lower_bound) < 0.0:
        raise ValueError("`pial_lower_bound` must be non-negative")

    validate_obj_type(gwb_lower_bound, "gwb_lower_bound", (RealNoBool, type(None)))

    validate_obj_type(gwb_upper_bound, "gwb_upper_bound", RealNoBool)
    if gwb_lower_bound is not None and float(gwb_lower_bound) >= float(gwb_upper_bound):
        raise ValueError("`gwb_lower_bound` must be less than `gwb_upper_bound`")

    validate_obj_type(update_rate, "update_rate", RealNoBool)
    if float(update_rate) <= 0.0:
        raise ValueError("`update_rate` must be greater than 0")

    validate_obj_type(edge_rolloff, "edge_rolloff", RealNoBool)
    if edge_rolloff <= 0.0:
        raise ValueError("`edge_rolloff` must be positive")

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    # Get GM and WM masks
    wm_mask = get_binary_mask(
        seg_mask,
        mask_type="wm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )
    gm_mask = get_binary_mask(
        seg_mask,
        mask_type="gm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )

    if not wm_mask.any() or not gm_mask.any():
        warnings.warn(
            "No GM or WM voxels found to apply boundary blurring; returning image unchanged.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    # Pre-compute conduction coefficient
    c = compute_conduction_coefficient(
        *compute_gradients(image, spacing),
        kappa=kappa,
        mask=seg_mask > 0,
    )

    # Get signed distance fields from GM-WM boundary and pial surfaces
    sdf_bound, sdf_pial = compute_distance_fields(
        seg_mask,
        spacing=spacing,
        surface="both",
        label_enum=label_enum,
    )

    # Define region to apply blurring
    spread_field = gm_mask | wm_mask
    spread_field &= sdf_pial >= float(pial_lower_bound)
    spread_field &= sdf_bound <= float(gwb_upper_bound)
    if gwb_lower_bound is not None:
        spread_field &= sdf_bound >= float(gwb_lower_bound)

    # Exit due to tight user-provided bounds (e.g., lower_pial >= upper_gwb)
    # or empty segmentation masks (e.g., no GM)
    if not spread_field.any():
        warnings.warn(
            "No boundary to blur within the selected bounds; returning image unchanged.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    del sdf_bound, sdf_pial

    # Get soft effect spread field
    spread_field = generate_application_field(
        mask=spread_field,
        spacing=spacing,
        field_type="inward",
        rolloff_mm=edge_rolloff,
    )
    if app_field is not None:
        spread_field *= app_field

    # Apply diffusion iteratively
    diffused_image = image.copy()
    for _ in range(n_iters):
        diffused_image += (
            update_rate
            * spread_field
            * perona_malik(
                diffused_image,
                spacing=spacing,
                compute_c=False,
                external_c=c,
            )
        )

    return {
        "out_image": diffused_image,
        "effect_field": diffused_image - image,
    }


def hyperintensity(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    wmh_seeds_corr: _DistanceType = 2.0,
    wmh_seeds_hpf: _DistanceType = 3.0,
    gwb_prob_rolloff: _DistanceType = 5.0,
    start_prob: float = 0.5,
    end_prob: float = 0.1,
    scale_factor: float = 0.2,
    n_iters: int = 20,
    update_rate: float = DIFF_UPDATE_RATE,
    kappa: float = GRAD_QUANTILE_KAPPA,
    gm_rolloff: _DistanceType = 3.0,
    edge_rolloff: _DistanceType = 2.0,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _IntensityResultType:
    """
    Apply white matter hyperintensity (WMH) to the image by randomly generating seed
    points and letting them diffuse using a Perona-Malik-like anisotropic diffusion
    equation.

    Notes:
        - The Perona-Malik implementation is as described in ``synthfcd.core.perona_malik``.
        - The diffusion takes place in the WMH field other than the raw image intensities.
        - A Perona-Malik conduction coefficient is computed only once before the initial
          iteration (instead of every iteration) on the raw image intensities to impose
          anatomical constraints. No such conduction coefficient is computed on the WMH field,
          as it would severely restrict WMH diffusion right after the initial seeding phase.

    Args:
        image (np.ndarray):
            The input image to apply the WMH effect to.
        seg_mask (np.ndarray):
            The input segmentation mask to use for anatomically constraining WMH.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        app_field (np.ndarray):
            Optional soft application field to control the extent of the effect.
            Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the WMH effect to. Defaults to ``None``.
        wmh_seeds_corr (float | int, optional):
            The correlation sigma of the noise field to use for WMH seeding.
            Defaults to ``2.0`` mm.
        wmh_seeds_hpf (float | int, optional):
            The high-pass filter sigma of the noise field to use for WMH seeding.
            Defaults to ``3.0`` mm.
        gwb_prob_rolloff (float | int, optional):
            The rolloff distance in mm away from the GM-WM boundary (toward the WM)
            for a Gaussian radial  field to transition from `start_prob` (see below)
            to `end_prob` (see below). Defaults to ``5.0`` mm.
        start_prob (float, optional):
            The probability of the WMH seeding field at the GM-WM boundary.
            Defaults to ``0.5``.
        end_prob (float, optional):
            The probability of the WMH seeding field at the `gwb_prob_rolloff` distance
            from the GM-WM boundary. Remains fixed further outwards. Defaults to ``0.1``.
        scale_factor (float | int, optional):
            The scaling to apply to the unit WMH field w.r.t. the median raw image intensity
            in the WM. Defaults to ``0.2``. Use negative values for hypo-intensity.
        n_iters (int, optional):
            The number of iterations to run the WMH diffusion. Defaults to ``20``.
        update_rate (float | int, optional):
            The update rate of the diffusion process. Defaults to ``0.1``.
        kappa (float | int, optional):
            The quantile of the gradient magnitude distribution to use for the conduction
            coefficient in the Perona-Malik equation. Defaults to ``0.95``.
        gm_rolloff (float | int, optional):
            The rolloff distance in mm away from the GM-WM boundary and inward the GM
            to decay the WMH diffusion. Defaults to ``3.0`` mm.
        edge_rolloff (float | int, optional):
            The rolloff distance in mm to decay the WMH diffusion near the allowed
            anatomical boundaries, defined by the GM and WM masks, and optionally
            the `limit_to` mask (see below). Defaults to ``2.0`` mm.
        random_seed (int | None, optional):
            The random seed to use for the WMH seeding. Defaults to ``None``.
        random_state (np.random.RandomState | None, optional):
            The random state to use for the WMH seeding. Defaults to ``None``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading ``seg_mask``.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the output image and effect field, as additive intensity.
            The keys are:
            - ``"out_image"``: the resulting image.
            - ``"effect_field"``: the resulting effect field.

    Warnings:
        UserWarning:
            If no WMH seeds are found.

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

    if hemisphere is not None:
        validate_literal_str(hemisphere, "hemisphere", ("left", "right"))

    validate_obj_type(wmh_seeds_corr, "wmh_seeds_corr", RealNoBool)
    if float(wmh_seeds_corr) < 0.0:
        raise ValueError("`wmh_seeds_corr` must be non-negative")

    validate_obj_type(wmh_seeds_hpf, "wmh_seeds_hpf", RealNoBool)
    if float(wmh_seeds_hpf) < float(wmh_seeds_corr):
        raise ValueError("`wmh_seeds_hpf` must be no less than `wmh_seeds_corr`")

    validate_obj_type(gwb_prob_rolloff, "gwb_prob_rolloff", RealNoBool)
    if float(gwb_prob_rolloff) <= 0.0:
        raise ValueError("`gwb_prob_rolloff` must be positive")

    validate_obj_type(start_prob, "start_prob", RealNoBool)
    if float(start_prob) < 0.0 or float(start_prob) > 1.0:
        raise ValueError("`start_prob` must be between 0 and 1")

    validate_obj_type(end_prob, "end_prob", RealNoBool)
    if float(end_prob) < 0.0 or float(end_prob) > 1.0:
        raise ValueError("`end_prob` must be between 0 and 1")
    if float(start_prob) == 0.0 and float(end_prob) == 0.0:
        raise ValueError("`start_prob` and `end_prob` cannot both be 0")

    validate_obj_type(scale_factor, "scale_factor", RealNoBool)
    if float(scale_factor) == 0.0:
        raise ValueError("`scale_factor` must be non-zero")

    validate_obj_type(n_iters, "n_iters", IntNoBool)
    if n_iters <= 0:
        raise ValueError("`n_iters` must be positive")

    validate_obj_type(update_rate, "update_rate", RealNoBool)
    if float(update_rate) <= 0.0:
        raise ValueError("`update_rate` must be positive")

    validate_obj_type(gm_rolloff, "gm_rolloff", RealNoBool)
    if float(gm_rolloff) < 0.0:
        raise ValueError("`gm_rolloff` must be non-negative")

    validate_obj_type(edge_rolloff, "edge_rolloff", RealNoBool)
    if float(edge_rolloff) < 0.0:
        raise ValueError("`edge_rolloff` must be non-negative")

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    rng = get_rng(
        seed=random_seed,
        random_state=random_state,
        offset=SEED_OFFSET_HYPERINTENSITY,
    )

    # Get GM and WM masks
    wm_mask = get_binary_mask(
        seg_mask,
        mask_type="wm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )
    gm_mask = get_binary_mask(
        seg_mask,
        mask_type="gm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )

    if not wm_mask.any():
        warnings.warn(
            "No WM voxels found to apply WMH; returning image unchanged.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    # Generate noise field
    noise_field = generate_noise_field(
        shape=image.shape,
        spacing=spacing,
        corr_sigma=wmh_seeds_corr,
        hpf_sigma=wmh_seeds_hpf,
        random_state=rng,
    )

    # Get SDF from GM-WM boundary to compute probabilities
    sdf_bound = compute_distance_fields(
        seg_mask,
        spacing=spacing,
        surface="gm_wm",
        label_enum=label_enum,
    )

    # Gaussian radial decay field that peaks at the GM-WM boundary with value
    # of `start_prob` and decays/climbs up to `end_prob` at `gwb_prob_rolloff` mm
    sigma = float(gwb_prob_rolloff / np.sqrt(-2.0 * np.log(1e-2)))
    threshold_field = np.exp(-((sdf_bound) ** 2) / (2 * sigma**2))
    if start_prob < end_prob:  # flip if start < end
        threshold_field = 1 - threshold_field
    low = min(start_prob, end_prob)
    high = max(start_prob, end_prob)
    threshold_field = (
        low + (high - low) * threshold_field
    )  # scale and shift to [start, end]

    # Limit WMH spread within `edge_rolloff` deep in GM-WM
    allowed_field = wm_mask | gm_mask
    allowed_field = generate_application_field(
        mask=allowed_field,
        spacing=spacing,
        field_type="inward",
        rolloff_mm=edge_rolloff,
    )
    if app_field is not None:
        allowed_field *= app_field

    # Avoid WMH seeds near limited-diffusion regions
    threshold_field *= wm_mask * (allowed_field > WMH_SEEDS_ALLOWED_FIELD_THRES)

    # Seed WMH field
    wmh_field = noise_field < threshold_field
    if not wmh_field.any():
        warnings.warn(
            (
                "No WMH seeds found; returning image unchanged. Consider increasing "
                "`start_prob` and `end_prob` values, or loosening `edge_rolloff`."
            ),
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    del noise_field, threshold_field, sdf_bound

    # Get conduction coefficient
    c = compute_conduction_coefficient(
        *compute_gradients(image, spacing),
        kappa=kappa,
        mask=seg_mask > 0,
        allow_fixed=False,
    )

    # Get soft effect spread field that decays inside GM and non-GM/WM edges
    spread_field = generate_application_field(
        mask=wm_mask,
        spacing=spacing,
        field_type="outward",
        rolloff_mm=gm_rolloff,
    )
    spread_field *= allowed_field

    # Apply diffusion iteratively
    wmh_field = wmh_field.astype(np.float32)
    for _ in range(n_iters):
        wmh_field += (
            update_rate
            * spread_field
            * perona_malik(
                wmh_field,
                spacing=spacing,
                compute_c=False,
                external_c=c,
            )
        )

    # Scale to [0, 1] to get unit WMH field
    wmh_field = (wmh_field - wmh_field.min()) / (wmh_field.max() - wmh_field.min())

    # Scale wrt to typical WM intensity (robust to outliers in the mask)
    ref_wm = np.median(image[wm_mask])
    wmh_field *= scale_factor * ref_wm

    return {
        "out_image": image + wmh_field,
        "effect_field": wmh_field,
    }
