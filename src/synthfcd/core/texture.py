"""
Apply texture restoration to the image.
"""

from __future__ import annotations

__all__ = [
    "texture_restoration",
]

import warnings
from typing import cast

import numpy as np
from scipy import ndimage

from synthfcd.core.utils import (
    compute_distance_fields,
    generate_application_field,
    generate_noise_field,
    get_binary_mask,
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
    SEED_OFFSET_TEXTURE,
    TEXTURE_NOISE_CLIP,
)
from synthfcd.utils._validators import (
    RealNoBool,
    validate_3d_numpy_array,
    validate_class_type,
    validate_literal_str,
    validate_obj_type,
    validate_spacing,
)
from synthfcd.utils.misc import get_rng
from synthfcd.utils.seg_labels import SynthSegLabel


def robust_std(image: np.ndarray) -> float:
    """
    Compute a robust standard deviation estimate from the median absolute deviation
    (MAD) of the image.
    """
    return cast(float, 1.4826 * np.median(np.abs(image - np.median(image))))


def texture_restoration(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    orig: np.ndarray,
    app_field: np.ndarray | None = None,
    hemisphere: _HemisphereType | None = None,
    hpf_img_sigma: _DistanceType = 1.0,
    corr_noise_sigma: _DistanceType = 0.8,
    hpf_noise_sigma: _DistanceType = 2.0,
    noise_clip: float | None = TEXTURE_NOISE_CLIP,
    gm_wm_only: bool = True,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
    label_enum: _LabelEnumType = SynthSegLabel,
    **kwargs,
) -> _IntensityResultType:
    """
    Restore the original texture of the image with a  band-pass filtered white noise
    field, scaled to match the residual texture.

    Args:
        image (np.ndarray):
            The input image to restore the texture of.
        seg_mask (np.ndarray):
            The input segmentation mask to use for computing per-tissue texture
            estimates.
        spacing (tuple[float | int, float | int, float | int]):
            The voxel spacing of the image and segmentation mask in mm.
        orig (np.ndarray):
            The original image to match the texture to.
        app_field (np.ndarray):
            Optional soft application field in ``[0, 1]`` controlling the spatial
            extent and magnitude of the restored texture. Residual texture is
            estimated from the available image (or deep GM/WM when
            ``gm_wm_only=True``); ``app_field`` only scales the final noise and
            does not affect those estimates. Defaults to ``None``.
        hemisphere (Literal["left", "right"] | None, optional):
            The hemisphere to restrict the texture restoration to. Defaults to ``None``.
        hpf_img_sigma (float | int, optional):
            The sigma in mm for high-pass filtering the original and image intensities
            to compute the residual texture. Defaults to ``1.0`` mm.
        corr_noise_sigma (float | int, optional):
            The sigma in mm for correlating the noise field. Defaults to ``0.8`` mm.
        hpf_noise_sigma (float | int, optional):
            The sigma in mm for high-pass filtering the noise field. Defaults to ``2.0`` mm.
        noise_clip (float | int | None, optional):
            The clipping value for the noise field, given as a multiple of its standard
            deviation, to prevent artificially high intensity values. Defaults to ``3.0``.
        gm_wm_only (bool, optional):
            Whether to only restore texture only within the GM and WM tissues.
            Defaults to ``True``.
        random_seed (int | None, optional):
            The random seed to use for generating the noise field. Defaults to ``None``.
        random_state (np.random.RandomState | None, optional):
            The random state to use for generating the noise field. Defaults to ``None``.
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
            If no GM-WM voxels are found to restore texture, if the application
            field has no support, or if the residual texture to restore is
            negligible. In these cases, the function returns the input image
            unchanged.

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

    validate_3d_numpy_array(orig, "orig", dtype=np.floating, shape=image.shape)

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

    validate_obj_type(hpf_img_sigma, "hpf_img_sigma", RealNoBool)
    if float(hpf_img_sigma) <= 0.0:
        raise ValueError("`hpf_img_sigma` must be greater than 0")

    validate_obj_type(corr_noise_sigma, "corr_noise_sigma", RealNoBool)
    if float(corr_noise_sigma) <= 0.0:
        raise ValueError("`corr_noise_sigma` must be greater than 0")

    validate_obj_type(hpf_noise_sigma, "hpf_noise_sigma", RealNoBool)
    if float(hpf_noise_sigma) <= 0.0:
        raise ValueError("`hpf_noise_sigma` must be greater than 0")

    validate_obj_type(noise_clip, "noise_clip", (RealNoBool, type(None)))
    if noise_clip is not None and float(noise_clip) <= 0.0:
        raise ValueError("`noise_cap` must be greater than 0")

    validate_obj_type(gm_wm_only, "gm_wm_only", bool)

    validate_class_type(label_enum, "label_enum", _LabelEnum)

    rng = get_rng(
        seed=random_seed, random_state=random_state, offset=SEED_OFFSET_TEXTURE
    )

    # Get GM and WM masks
    gm_mask = get_binary_mask(
        seg_mask=seg_mask,
        mask_type="gm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )
    wm_mask = get_binary_mask(
        seg_mask=seg_mask,
        mask_type="wm",
        hemisphere=hemisphere,
        label_enum=label_enum,
    )
    gm_wm = gm_mask | wm_mask
    if gm_wm_only and not gm_wm.any():
        warnings.warn(
            "No GM or WM voxels to restore texture; returning image unchanged. "
            "Use `gm_wm_only=False` if intentional.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    # High-pass filter the images to compute residual texture
    hpf_img_sigma_vox = (
        float(hpf_img_sigma / spacing[0]),
        float(hpf_img_sigma / spacing[1]),
        float(hpf_img_sigma / spacing[2]),
    )
    r_orig = orig - ndimage.gaussian_filter(orig, hpf_img_sigma_vox)
    r_image = image - ndimage.gaussian_filter(image, hpf_img_sigma_vox)

    if not gm_wm_only:
        var_orig = max(robust_std(r_orig) ** 2, 0.0)
        var_image = max(robust_std(r_image) ** 2, 0.0)
        s_rest: np.ndarray | float = float(np.sqrt(max(var_orig - var_image, 0.0)))
    else:
        # Find the deep GM and WM voxels to compute tissue-specific residual
        # texture std estimates.
        sdf_bound, sdf_pial = compute_distance_fields(
            seg_mask=seg_mask,
            spacing=spacing,
            surface="both",
            label_enum=label_enum,
        )

        margin = float(hpf_img_sigma)
        deep_gm = (sdf_pial > margin) & (sdf_bound < -margin) & gm_mask
        deep_wm = (sdf_bound > margin) & wm_mask
        del sdf_pial, sdf_bound

        n_gm_vox = int(deep_gm.sum())
        n_wm_vox = int(deep_wm.sum())

        if n_gm_vox == 0 and n_wm_vox == 0:
            warnings.warn(
                "No valid voxels to compute separate stds in deep GM or WM; "
                "returning image unchanged.",
                UserWarning,
            )
            return {
                "out_image": image,
                "effect_field": np.zeros_like(image),
            }

        gm_std_orig = robust_std(r_orig[deep_gm]) if n_gm_vox > 0 else 0.0
        wm_std_orig = robust_std(r_orig[deep_wm]) if n_wm_vox > 0 else 0.0
        gm_std_image = robust_std(r_image[deep_gm]) if n_gm_vox > 0 else 0.0
        wm_std_image = robust_std(r_image[deep_wm]) if n_wm_vox > 0 else 0.0
        gm_var_rest = max(gm_std_orig**2 - gm_std_image**2, 0.0)
        wm_var_rest = max(wm_std_orig**2 - wm_std_image**2, 0.0)

        del deep_gm, deep_wm

        # Smooth blending of GM/WM std estimates around the boundary
        w_gm_raw = generate_application_field(
            mask=gm_mask,
            spacing=spacing,
            field_type="outward",
            rolloff_mm=1.0,
        )
        w_wm_raw = generate_application_field(
            mask=wm_mask,
            spacing=spacing,
            field_type="outward",
            rolloff_mm=1.0,
        )
        w_sum = w_gm_raw + w_wm_raw
        valid = (w_sum > 1e-8) & gm_wm

        w_gm = np.zeros_like(w_gm_raw, dtype=np.float32)
        w_wm = np.zeros_like(w_wm_raw, dtype=np.float32)
        w_gm[gm_wm] = 0.5
        w_wm[gm_wm] = 0.5
        w_gm[valid] = w_gm_raw[valid] / w_sum[valid]
        w_wm[valid] = w_wm_raw[valid] / w_sum[valid]

        del w_gm_raw, w_wm_raw, w_sum, valid, gm_wm, gm_mask, wm_mask

        s_rest = np.sqrt(w_gm * gm_var_rest + w_wm * wm_var_rest)
        del w_gm, w_wm

    if np.all(s_rest <= 1e-8):
        warnings.warn(
            "No residual texture to restore; returning image unchanged.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    # Band-pass filtered white noise (correlate first then remove low-freqs)
    noise = generate_noise_field(
        shape=image.shape,
        spacing=spacing,
        corr_sigma=corr_noise_sigma,
        hpf_sigma=hpf_noise_sigma,
        random_state=rng,
    )

    # Normalize globally; `app_field` is applied as a soft weight after scaling.
    noise -= noise.mean()
    std = float(noise.std())

    if std <= 1e-8:
        warnings.warn(
            "No residual texture to restore; returning image unchanged.",
            UserWarning,
        )
        return {
            "out_image": image,
            "effect_field": np.zeros_like(image),
        }

    noise /= std
    if noise_clip is not None:
        noise = np.clip(noise, -noise_clip, noise_clip)
    noise *= s_rest
    if app_field is not None:
        noise *= app_field

    return {
        "out_image": image + noise,
        "effect_field": noise,
    }
