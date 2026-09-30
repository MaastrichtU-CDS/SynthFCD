"""
Helpers for visualizing the results of the synthFCD products.

Will not be a part of the shipped package.
"""

from .histograms import plot_histograms
from .label_intensity import plot_npy_label_intensity_3d
from .label_slices import plot_npy_label_slices
from .overlay_arrows import (
    overlay_vector_field_2d,
    overlay_vector_field_3d,
    overlay_vector_field_on_intensity_3d,
)
from .overlay_intensity import overlay_intensity_field
from .overlay_mask import mask_outline, overlay_mask, overlay_mask_outline
from .utils import DEFAULT_LABELS, resolve_labels, show_plot_handles
from .volumes import plot_npy_volumes

__all__ = [
    "DEFAULT_LABELS",
    "mask_outline",
    "overlay_intensity_field",
    "overlay_mask",
    "overlay_mask_outline",
    "overlay_vector_field_2d",
    "overlay_vector_field_3d",
    "overlay_vector_field_on_intensity_3d",
    "plot_histograms",
    "plot_npy_label_intensity_3d",
    "plot_npy_label_slices",
    "plot_npy_volumes",
    "resolve_labels",
    "show_plot_handles",
]
