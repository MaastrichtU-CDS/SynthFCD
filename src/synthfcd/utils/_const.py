"""
Constants for internal use.
"""

from __future__ import annotations

from typing import Literal

# ----------------------------------------------------------------#
# Deterministic behavior constants
# ----------------------------------------------------------------#
SEED_MAX: int = 2**32 - 1

SEED_OFFSET_GROWTH: int = 0x1A2B3C4D

SEED_OFFSET_TEXTURE: int = 0x56789ABC

SEED_OFFSET_HYPERINTENSITY: int = 0x267CFEDA

SEED_OFFSET_GYRATION: int = 0x36F72EAD


# ----------------------------------------------------------------#
# Effect-agnostic constants
# ----------------------------------------------------------------#

APP_FIELD_TYPE: Literal["inward", "outward"] = "outward"

APP_FIELD_ROLLOFF: float = 3.0

APP_FIELD_BBOX_THRES: float = 0.05

TANH_ROLLOFF_TARGET: float = 0.9

GAUSSIAN_ROLLOFF_TARGET: float = 0.1


# ----------------------------------------------------------------#
# Effect-specific constants and parameter defaults
# ----------------------------------------------------------------#

GROW_LESION_NOISE_LAMBDA: float = (
    0.3  # empirically determined, contribution of multiplicative noise
)

GROW_LESION_MASK_THRES: float = (
    0.4  # empirically determined, binarization of smoothed mask
)

SMOOTH_SIGMA_DEFORM: float = 1.0

GRAD_QUANTILE_KAPPA: float = 0.95

DIFF_UPDATE_RATE: float = 0.1

WMH_SEEDS_ALLOWED_FIELD_THRES: float = 0.95

TEXTURE_NOISE_CLIP: float = 3.0

TEXTURE_SUPPORT_THRES_FROM_BLUR: float = 0.01

TEXTURE_SUPPORT_ROLLOFF_FROM_BLUR: float = 1.0
