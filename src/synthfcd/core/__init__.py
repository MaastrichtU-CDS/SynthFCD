"""
Core functionality for growing (lesional) targets and simulating effects on images.

This module should remain pipeline-agnostic.
"""

from __future__ import annotations

from .deformations import abnormal_gyration, cortical_thickening, sulcal_widening
from .diffusion import boundary_blurring, hyperintensity
from .grow_lesion import grow_random_lesion
from .masks import extract_masks, get_binary_mask
from .texture import texture_restoration
from .utils import warp

__all__ = [
    "abnormal_gyration",
    "boundary_blurring",
    "cortical_thickening",
    "extract_masks",
    "get_binary_mask",
    "grow_random_lesion",
    "hyperintensity",
    "sulcal_widening",
    "texture_restoration",
    "warp",
]
