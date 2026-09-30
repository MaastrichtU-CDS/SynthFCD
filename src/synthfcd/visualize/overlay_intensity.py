"""
Scalar intensity overlays for 2D orthogonal-slice figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps

from .utils import iter_panels, slice_panel


def _as_limit(name: str, limit: float | None) -> float | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, (int, float, np.number)):
        raise ValueError(f"{name} must be a real number, got {limit!r}.")
    return float(limit)


def _in_mask(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    keep = np.isfinite(field)
    if mask is not None:
        keep &= mask
    return keep


def _color_limits(
    field: np.ndarray,
    mask: np.ndarray | None,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    if vmin is not None and vmax is not None:
        return vmin, vmax

    finite = field[_in_mask(field, mask)]
    if finite.size == 0:
        data_min, data_max = 0.0, 1.0
    else:
        data_min = float(finite.min())
        data_max = float(finite.max())
        if data_min == data_max:
            data_max = data_min + 1.0

    lo = data_min if vmin is None else vmin
    hi = data_max if vmax is None else vmax
    if not lo < hi:
        raise ValueError(f"vmin must be less than vmax, got vmin={lo}, vmax={hi}.")
    return lo, hi


def overlay_intensity_field(
    handles: dict[str, Any],
    field: np.ndarray,
    *,
    rows: str | list[str] | None = None,
    mask: np.ndarray | None = None,
    cmap: str = "plasma",
    alpha: float = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    **imshow_kwargs: Any,
) -> dict[str, Any]:
    """
    Overlay a scalar field on 2D orthogonal-slice figures.

    The underlying image keeps its own colormap. Pass handles returned by
    :func:`~synthfcd.visualize.volumes.plot_npy_volumes` or
    :func:`~synthfcd.visualize.label_slices.plot_npy_label_slices`.

    Args:
        handles: Dict with ``fig``, ``axes``, and ``panels``.
        field: 3D scalar array aligned with the plotted volumes.
        rows: Optional row name(s) to annotate; defaults to all rows.
        mask: Optional 3D boolean mask. Voxels outside the mask stay transparent.
            Zero and non-finite field values stay transparent either way, and
            that threshold does not change the color scale.
        cmap: Matplotlib colormap name for the field.
        alpha: Overlay opacity in ``[0, 1]``.
        vmin: Lower color limit. Defaults to the minimum finite field value.
        vmax: Upper color limit. Defaults to the maximum finite field value.
        **imshow_kwargs: Forwarded to ``Axes.imshow``. ``cmap``, ``alpha``,
            ``vmin``, ``vmax``, and ``norm`` are not accepted.

    Returns:
        The same ``handles`` dict.
    """
    if not isinstance(field, np.ndarray) or field.ndim != 3:
        raise ValueError(f"field must be a 3D numpy array, got {field!r}.")
    if not np.issubdtype(field.dtype, np.number):
        raise ValueError(f"field must be numeric, got dtype {field.dtype}.")
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != field.shape:
            raise ValueError(
                f"mask must be a numpy array of shape {field.shape}, got {mask!r}."
            )
        mask = mask.astype(bool)
    if not isinstance(cmap, str) or cmap not in colormaps:
        raise ValueError(f"cmap must be a matplotlib colormap name, got {cmap!r}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    vmin = _as_limit("vmin", vmin)
    vmax = _as_limit("vmax", vmax)
    if vmin is not None and vmax is not None and not vmin < vmax:
        raise ValueError(f"vmin must be less than vmax, got vmin={vmin}, vmax={vmax}.")

    conflict = {"cmap", "alpha", "vmin", "vmax", "norm"} & imshow_kwargs.keys()
    if conflict:
        names = ", ".join(sorted(conflict))
        raise ValueError(
            f"Cannot pass {names} in imshow_kwargs; use the matching arguments."
        )

    lo, hi = _color_limits(field, mask, vmin, vmax)
    imshow_opts = {
        "cmap": cmap,
        "alpha": alpha,
        "vmin": lo,
        "vmax": hi,
        "origin": "lower",
        "aspect": "equal",
        "interpolation": "nearest",
        **imshow_kwargs,
    }

    for panel in iter_panels(handles, rows):
        sl_field = np.asarray(slice_panel(field, panel), dtype=float)
        sl_mask = slice_panel(mask, panel) if mask is not None else None
        visible = _in_mask(sl_field, sl_mask) & (sl_field != 0)
        if not visible.any():
            continue

        handles["axes"][panel["row_idx"], panel["col"]].imshow(
            np.ma.masked_where(~visible, sl_field),
            **imshow_opts,
        )

    return handles
