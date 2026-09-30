"""
Vector-field arrow overlays for 2D slice and 3D intensity figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from .utils import in_plane_components, iter_panels, slice_panel


def _subsample_quiver_points(
    u_x: np.ndarray,
    u_y: np.ndarray,
    *,
    stride: int,
    z: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    d_row, d_col = u_x.shape
    yy, xx = np.mgrid[0:d_row:stride, 0:d_col:stride]
    x = xx.ravel().astype(float)
    y = yy.ravel().astype(float)
    u = u_x[0:d_row:stride, 0:d_col:stride].ravel().astype(float)
    v = u_y[0:d_row:stride, 0:d_col:stride].ravel().astype(float)

    if z is not None:
        z_out = z[0:d_row:stride, 0:d_col:stride].ravel().astype(float)
    else:
        z_out = None

    if mask is not None:
        sel = mask[0:d_row:stride, 0:d_col:stride].ravel()
        x, y, u, v = x[sel], y[sel], u[sel], v[sel]
        if z_out is not None:
            z_out = z_out[sel]

    if z_out is None:
        return x, y, u, v
    return x, y, z_out, u, v


def _validate_color_from_magnitude(
    color_from_magnitude: bool,
    cmap_magnitude: str,
    quiver_kwargs: dict[str, Any],
    *,
    reserved: frozenset[str],
) -> None:
    if not isinstance(color_from_magnitude, bool):
        raise ValueError(
            "color_from_magnitude must be a boolean, " f"got {color_from_magnitude!r}."
        )
    if not color_from_magnitude:
        return
    if cmap_magnitude not in colormaps:
        raise ValueError(
            "cmap_magnitude must be a matplotlib colormap name, "
            f"got {cmap_magnitude!r}."
        )
    conflict = reserved & quiver_kwargs.keys()
    if conflict:
        names = ", ".join(sorted(conflict))
        raise ValueError(
            f"Cannot pass {names} when color_from_magnitude is True; "
            "set the colormap with cmap_magnitude."
        )


def _shared_magnitude_norm(
    components: list[tuple[np.ndarray, np.ndarray]],
) -> Normalize:
    peak = max(
        (float(np.hypot(u, v).max()) for u, v in components),
        default=0.0,
    )
    return Normalize(vmin=0.0, vmax=peak if peak > 0.0 else 1.0)


def overlay_vector_field_2d(
    handles: dict[str, Any],
    vector_field: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rows: str | list[str] | None = None,
    mask: np.ndarray | None = None,
    stride: int = 4,
    arrow_scale: float = 1.0,
    color: str = "k",
    color_from_magnitude: bool = False,
    cmap_magnitude: str = "viridis",
    **quiver_kwargs: Any,
) -> dict[str, Any]:
    """
    Overlay in-plane displacement arrows on 2D orthogonal-slice figures.

    Pass handles returned by :func:`~synthfcd.visualize.volumes.plot_npy_volumes`
    or :func:`~synthfcd.visualize.label_slices.plot_npy_label_slices`.

    Args:
        handles: Dict with ``fig``, ``axes``, and ``panels``.
        vector_field: ``(ux, uy, uz)`` displacement field.
        rows: Optional row name(s) to annotate; defaults to all rows.
        mask: Optional 3D boolean mask; arrows are drawn only where True.
        stride: Subsample arrows every ``stride`` voxels along each in-plane axis.
        arrow_scale: Multiply in-plane displacement components before plotting.
        color: Arrow color. Ignored when ``color_from_magnitude`` is True.
        color_from_magnitude: If True, color each arrow by its in-plane displacement
            magnitude. The scale is shared across panels, from 0 to the largest
            drawn magnitude, and is computed before ``arrow_scale``.
        cmap_magnitude: Matplotlib colormap name used when
            ``color_from_magnitude`` is True.
        **quiver_kwargs: Forwarded to ``Axes.quiver``. ``color``, ``cmap``, and
            ``norm`` are not accepted when ``color_from_magnitude`` is True.

    Returns:
        The same ``handles`` dict.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    _validate_color_from_magnitude(
        color_from_magnitude,
        cmap_magnitude,
        quiver_kwargs,
        reserved=frozenset({"color", "cmap", "norm"}),
    )

    if (
        not isinstance(vector_field, tuple)
        or len(vector_field) != 3
        or not all(isinstance(field, np.ndarray) for field in vector_field)
        or not all(field.ndim == 3 for field in vector_field)
        or not all(field.shape == vector_field[0].shape for field in vector_field)
    ):
        raise ValueError(
            f"vector_field must be a tuple of three numpy arrays, got {vector_field}."
        )
    ux, uy, uz = vector_field

    if mask is not None and mask.shape != ux.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match vector field shape {ux.shape}."
        )

    prepared: list[
        tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for panel in iter_panels(handles, rows):
        u_x, u_y = in_plane_components(ux, uy, uz, panel)
        sl_mask = slice_panel(mask, panel) if mask is not None else None
        x, y, u, v = _subsample_quiver_points(u_x, u_y, stride=stride, mask=sl_mask)
        if len(x) == 0:
            continue
        prepared.append((panel, x, y, u, v))

    norm = (
        _shared_magnitude_norm([(u, v) for _, _, _, u, v in prepared])
        if color_from_magnitude
        else None
    )

    for panel, x, y, u, v in prepared:
        ax = handles["axes"][panel["row_idx"], panel["col"]]
        if color_from_magnitude:
            ax.quiver(
                x,
                y,
                u * arrow_scale,
                v * arrow_scale,
                np.hypot(u, v),
                cmap=cmap_magnitude,
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                **quiver_kwargs,
            )
        else:
            ax.quiver(
                x,
                y,
                u * arrow_scale,
                v * arrow_scale,
                color=color,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                **quiver_kwargs,
            )

    return handles


def overlay_vector_field_3d(
    handles: dict[str, Any],
    vector_field: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rows: str | list[str] | None = None,
    mask: np.ndarray | None = None,
    stride: int = 4,
    arrow_scale: float = 1.0,
    color: str = "k",
    color_from_magnitude: bool = False,
    cmap_magnitude: str = "viridis",
    **quiver_kwargs: Any,
) -> dict[str, Any]:
    """
    Overlay in-plane displacement arrows on a 3D intensity-landscape figure.

    Arrows lie in the x-y floor of each 3D panel; the vertical axis stays intensity.
    Pass handles returned by
    :func:`~synthfcd.visualize.label_intensity.plot_npy_label_intensity_3d`.

    Args:
        handles: Dict with ``fig``, ``axes``, and ``panels``.
        vector_field: ``(ux, uy, uz)`` displacement field.
        rows: Optional row name(s) to annotate; defaults to all rows.
        mask: Optional 3D boolean mask; arrows are drawn only where True.
        stride: Subsample arrows every ``stride`` voxels along each in-plane axis.
        arrow_scale: Multiply in-plane displacement components before plotting.
        color: Arrow color. Ignored when ``color_from_magnitude`` is True.
        color_from_magnitude: If True, color each arrow by its in-plane displacement
            magnitude. The scale is shared across panels, from 0 to the largest
            drawn magnitude, and is computed before ``arrow_scale``.
        cmap_magnitude: Matplotlib colormap name used when
            ``color_from_magnitude`` is True.
        **quiver_kwargs: Forwarded to ``Axes3D.quiver``. ``color``, ``colors``,
            ``cmap``, and ``norm`` are not accepted when
            ``color_from_magnitude`` is True.

    Returns:
        The same ``handles`` dict.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    _validate_color_from_magnitude(
        color_from_magnitude,
        cmap_magnitude,
        quiver_kwargs,
        reserved=frozenset({"color", "colors", "cmap", "norm"}),
    )

    if (
        not isinstance(vector_field, tuple)
        or len(vector_field) != 3
        or not all(isinstance(field, np.ndarray) for field in vector_field)
        or not all(field.ndim == 3 for field in vector_field)
        or not all(field.shape == vector_field[0].shape for field in vector_field)
    ):
        raise ValueError(
            f"vector_field must be a tuple of three numpy arrays, got {vector_field}."
        )
    ux, uy, uz = vector_field

    if mask is not None and mask.shape != ux.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match vector field shape {ux.shape}."
        )

    prepared: list[
        tuple[
            dict[str, Any],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for panel in iter_panels(handles, rows):
        u_x, u_y = in_plane_components(ux, uy, uz, panel)
        sl_mask = slice_panel(mask, panel) if mask is not None else None
        x, y, z, u, v = _subsample_quiver_points(
            u_x, u_y, stride=stride, z=panel["sl_img"], mask=sl_mask
        )
        if len(x) == 0:
            continue
        prepared.append((panel, x, y, z, u, v))

    norm = (
        _shared_magnitude_norm([(u, v) for _, _, _, _, u, v in prepared])
        if color_from_magnitude
        else None
    )

    for panel, x, y, z, u, v in prepared:
        w = np.zeros_like(u)
        ax = handles["axes"][panel["row_idx"], panel["col"]]
        if color_from_magnitude:
            assert norm is not None
            rgba = colormaps[cmap_magnitude](norm(np.hypot(u, v)))
            # Axes3D.quiver stores the shaft, then each side of the head.
            colors = np.concatenate([rgba, rgba, rgba], axis=0)
            ax.quiver(  # type: ignore[union-attr]
                x,
                y,
                z,  # type: ignore[arg-type]
                u * arrow_scale,
                v * arrow_scale,
                w,  # type: ignore[arg-type]
                colors=colors,
                **quiver_kwargs,
            )
        else:
            ax.quiver(  # type: ignore[union-attr]
                x,
                y,
                z,  # type: ignore[arg-type]
                u * arrow_scale,
                v * arrow_scale,
                w,  # type: ignore[arg-type]
                color=color,
                **quiver_kwargs,
            )

    return handles


overlay_vector_field_on_intensity_3d = overlay_vector_field_3d
