"""
Binary mask overlays for 2D orthogonal-slice figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colors as mcolors
from scipy import ndimage

from .utils import iter_panels, slice_panel


def mask_outline(
    mask: np.ndarray, *, thickness: int = 1, outwards: bool = False
) -> np.ndarray:
    """
    Return the boundary voxels of a binary mask.

    By default the outline is ``thickness`` voxels wide, measured inward from the
    mask boundary. With ``outwards=True``, the outline is measured outward via
    dilation instead.

    Args:
        mask: Boolean or integer array of any dimensionality.
        thickness: Outline width in voxels. Must be at least ``1``.
        outwards: If ``True``, grow the outline outward from the mask boundary.

    Returns:
        Boolean array with the same shape as ``mask``.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"mask must be a numpy array, got {mask!r}.")
    if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness < 1:
        raise ValueError(f"thickness must be an integer >= 1, got {thickness}.")
    if not isinstance(outwards, bool):
        raise ValueError(f"outwards must be a boolean, got {outwards!r}.")

    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    if outwards:
        dilated = ndimage.binary_dilation(mask, iterations=thickness).astype(bool)
        return dilated & ~mask

    eroded = ndimage.binary_erosion(mask, iterations=thickness).astype(bool)
    return mask & ~eroded


def overlay_mask(
    handles: dict[str, Any],
    mask: np.ndarray,
    *,
    rows: str | list[str] | None = None,
    color: (
        str | tuple[float, float, float] | tuple[float, float, float, float]
    ) = "tab:red",
    alpha: float = 0.5,
    **imshow_kwargs: Any,
) -> dict[str, Any]:
    """
    Overlay a binary mask on 2D orthogonal-slice figures.

    Pass handles returned by :func:`~synthfcd.visualize.volumes.plot_npy_volumes`.

    Args:
        handles: Dict with ``fig``, ``axes``, and ``panels``.
        mask: 3D binary mask aligned with the plotted volumes.
        rows: Optional row name(s) to annotate; defaults to all rows.
        color: Matplotlib color for masked voxels.
        alpha: Overlay opacity in ``[0, 1]``.
        **imshow_kwargs: Forwarded to ``Axes.imshow`` for the overlay layer.

    Returns:
        The same ``handles`` dict.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 3:
        raise ValueError(f"mask must be a 3D numpy array, got {mask!r}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    rgba = mcolors.to_rgba(color, alpha=alpha)
    imshow_opts = {
        "origin": "lower",
        "aspect": "equal",
        "interpolation": "nearest",
        **imshow_kwargs,
    }

    mask = mask.astype(bool)
    for panel in iter_panels(handles, rows):
        sl_mask = slice_panel(mask, panel)
        if not sl_mask.any():
            continue

        overlay = np.zeros((*sl_mask.shape, 4), dtype=float)
        overlay[sl_mask] = rgba
        handles["axes"][panel["row_idx"], panel["col"]].imshow(overlay, **imshow_opts)

    return handles


def overlay_mask_outline(
    handles: dict[str, Any],
    mask: np.ndarray,
    *,
    rows: str | list[str] | None = None,
    thickness: int = 1,
    outwards: bool = False,
    color: (
        str | tuple[float, float, float] | tuple[float, float, float, float]
    ) = "tab:red",
    alpha: float = 0.9,
    **imshow_kwargs: Any,
) -> dict[str, Any]:
    """
    Overlay the outline of a binary mask on 2D orthogonal-slice figures.

    The outline is computed per displayed slice so contours match the visible
    cross-section. Pass handles returned by
    :func:`~synthfcd.visualize.volumes.plot_npy_volumes`.

    Args:
        handles: Dict with ``fig``, ``axes``, and ``panels``.
        mask: 3D binary mask aligned with the plotted volumes.
        rows: Optional row name(s) to annotate; defaults to all rows.
        thickness: Outline width in voxels. Must be at least ``1``.
        outwards: If ``True``, grow the outline outward from the mask boundary.
        color: Matplotlib color for outline voxels.
        alpha: Overlay opacity in ``[0, 1]``.
        **imshow_kwargs: Forwarded to ``Axes.imshow`` for the overlay layer.

    Returns:
        The same ``handles`` dict.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 3:
        raise ValueError(f"mask must be a 3D numpy array, got {mask!r}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness < 1:
        raise ValueError(f"thickness must be an integer >= 1, got {thickness}.")
    if not isinstance(outwards, bool):
        raise ValueError(f"outwards must be a boolean, got {outwards!r}.")

    rgba = mcolors.to_rgba(color, alpha=alpha)
    imshow_opts = {
        "origin": "lower",
        "aspect": "equal",
        "interpolation": "nearest",
        **imshow_kwargs,
    }

    mask = mask.astype(bool)
    for panel in iter_panels(handles, rows):
        sl_outline = mask_outline(
            slice_panel(mask, panel), thickness=thickness, outwards=outwards
        )
        if not sl_outline.any():
            continue

        overlay = np.zeros((*sl_outline.shape, 4), dtype=float)
        overlay[sl_outline] = rgba
        handles["axes"][panel["row_idx"], panel["col"]].imshow(overlay, **imshow_opts)

    return handles
