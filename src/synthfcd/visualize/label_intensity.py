"""
3D intensity-landscape label viewers.
"""

from __future__ import annotations

from typing import Any, Literal, cast, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .utils import (
    PLANE_SUBTITLES,
    LabelsInput,
    normalize_volume_dict,
    require_show_or_handles,
    resolve_labels,
    resolve_slice_indices_list,
    resolve_volume_slice_indices,
)


@overload
def plot_npy_label_intensity_3d(
    image: np.ndarray | dict[str, np.ndarray],
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 8,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_grid: bool = False,
    show_axes: bool = True,
    elev: float = 25,
    azim: float = -60,
    zlim: tuple[float, float] | None = None,
    show: bool = True,
    *,
    return_handles: Literal[True],
    save_path: str | None = None,
) -> dict[str, Any]: ...


@overload
def plot_npy_label_intensity_3d(
    image: np.ndarray | dict[str, np.ndarray],
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 8,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_grid: bool = False,
    show_axes: bool = True,
    elev: float = 25,
    azim: float = -60,
    zlim: tuple[float, float] | None = None,
    show: bool = True,
    return_handles: Literal[False] = False,
    save_path: str | None = None,
) -> None: ...


def plot_npy_label_intensity_3d(
    image: np.ndarray | dict[str, np.ndarray],
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 8,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_grid: bool = False,
    show_axes: bool = True,
    elev: float = 25,
    azim: float = -60,
    zlim: tuple[float, float] | None = None,
    show: bool = True,
    return_handles: bool = False,
    save_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Plot labeled voxels as 3D intensity landscapes, three orthogonal planes.

    For each plane the floor uses the same row/column axes as
    :func:`~synthfcd.visualize.label_slices.plot_npy_label_slices`; the vertical
    axis is image intensity.

    Args:
        image: 3D intensity volume or dict of named volumes.
        label_volume: 3D integer label volume or dict with the same keys as ``image``.
        labels: Label groups to plot. Defaults to WM / GM / lesion.
        title: Main figure title.
        figsize: Figure size. Defaults to ``(18, 6 * n_volumes)``.
        slice_indices: Per-volume slice index ``(dim1, dim2, dim3)``, or ``'mid'`` /
            ``'bbox'`` (bbox center of voxels with any configured label).
        marker_size: Scatter marker size.
        marker: Scatter marker style.
        show_background: If True, also plot voxels with label 0.
        background_color: Color for background voxels when ``show_background`` is True.
        show_legend: If True, draw a legend on the rightmost panel of each row.
        legend_alpha: Opacity of the legend frame (0 transparent, 1 opaque).
        show_grid: If True, show the 3D axis grid; if False, hide grid and axis ticks.
        show_axes: If False, hide all axis spines, ticks, labels, and grid.
        elev: 3D elevation angle in degrees.
        azim: 3D azimuth angle in degrees.
        zlim: Optional shared ``(zmin, zmax)`` intensity limits per row.
        show: If True, call ``plt.show()`` before returning.
        return_handles: If True, return a dict with ``fig``, ``axes``, and ``panels``.
        save_path: Optional path to save the figure.

    Returns:
        Plot handles when ``return_handles=True``, otherwise ``None``.

    Raises:
        ValueError: If label configuration, dict keys, or arrays are invalid.
    """
    require_show_or_handles(show, return_handles)
    if not 0 <= legend_alpha <= 1:
        raise ValueError(f"legend_alpha must be in [0, 1], got {legend_alpha}.")
    id_styles, legend = resolve_labels(labels)
    name_to_ids: dict[str, list[int]] = {}
    for label_id, (name, _color) in id_styles.items():
        name_to_ids.setdefault(name, []).append(label_id)

    images, image_names = normalize_volume_dict(image, "image")
    label_volumes, label_names = normalize_volume_dict(label_volume, "label_volume")
    if image_names != label_names:
        raise ValueError(
            f"`image` keys {image_names} do not match `label_volume` keys {label_names}."
        )

    slice_indices_list = resolve_slice_indices_list(slice_indices, len(image_names))

    main_title = title or "Label volumes - Intensity landscapes"
    if figsize is None:
        figsize = (18, 6 * len(image_names))

    fig = plt.figure(figsize=figsize, facecolor="white")
    axes = np.empty((len(image_names), 3), dtype=object)
    panels: list[dict[str, Any]] = []
    mask_ids = list(id_styles)

    for row_idx, name in enumerate(image_names):
        img = images[name]
        label_vol = label_volumes[name]
        if img.shape != label_vol.shape:
            raise ValueError(
                f"`image` shape {img.shape} does not match `label_volume` shape "
                f"{label_vol.shape} for case {name!r}."
            )

        idx_d1, idx_d2, idx_d3 = resolve_volume_slice_indices(
            label_vol,
            slice_indices_list[row_idx],
            mask=np.isin(label_vol, mask_ids),
        )
        planes = (
            (
                img[idx_d1, :, :],
                label_vol[idx_d1, :, :],
                PLANE_SUBTITLES[0],
                idx_d1,
                "D1",
                "D3",
                "D2",
            ),
            (
                img[:, idx_d2, :],
                label_vol[:, idx_d2, :],
                PLANE_SUBTITLES[1],
                idx_d2,
                "D2",
                "D3",
                "D1",
            ),
            (
                img[:, :, idx_d3],
                label_vol[:, :, idx_d3],
                PLANE_SUBTITLES[2],
                idx_d3,
                "D3",
                "D2",
                "D1",
            ),
        )

        for col_idx, (
            sl_img,
            sl_label,
            plane_title,
            idx,
            dim_name,
            x_axis,
            y_axis,
        ) in enumerate(planes):
            ax = cast(
                Axes3D,
                fig.add_subplot(
                    len(image_names), 3, row_idx * 3 + col_idx + 1, projection="3d"
                ),
            )
            axes[row_idx, col_idx] = ax
            panels.append(
                {
                    "row": name,
                    "row_idx": row_idx,
                    "col": col_idx,
                    "dim": dim_name,
                    "index": idx,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "sl_img": sl_img,
                }
            )

            d_row, d_col = sl_img.shape
            yy, xx = np.mgrid[0:d_row, 0:d_col]
            x = xx.ravel().astype(float)
            y = yy.ravel().astype(float)
            intensity = sl_img.ravel().astype(float)
            cls = sl_label.ravel()

            ax.set_facecolor("white")

            if show_background:
                bg = cls == 0
                if bg.any():
                    ax.scatter(  # type: ignore
                        x[bg],
                        y[bg],
                        intensity[bg],  # type: ignore[arg-type]
                        c=background_color,
                        s=marker_size,  # type: ignore[arg-type]
                        marker=marker,
                        linewidths=0,
                        depthshade=False,
                    )

            for entry_name, color in legend:
                sel = np.isin(cls, name_to_ids[entry_name])
                if not sel.any():
                    continue
                ax.scatter(  # type: ignore
                    x[sel],
                    y[sel],
                    intensity[sel],  # type: ignore[arg-type]
                    c=color,
                    s=marker_size,  # type: ignore[arg-type]
                    marker=marker,
                    linewidths=0,
                    depthshade=False,
                )

            if zlim is not None:
                ax.set_zlim(zlim)

            ax.view_init(elev=elev, azim=azim)
            if show_axes:
                if show_grid:
                    ax.grid(visible=True)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])  # type: ignore[attr-defined]
                    ax.grid(visible=False)
            else:
                ax.set_axis_off()

            ax.set_title(
                f"{name}\n{plane_title} ({dim_name}={idx})",
                fontsize=10,
                fontweight="bold",
            )

            if show_legend and col_idx == 2:
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            marker=marker,
                            color="w",
                            markerfacecolor=color,
                            markersize=10,
                            label=entry_name,
                        )
                        for entry_name, color in legend
                    ],
                    loc="upper right",
                    fontsize=10,
                    framealpha=legend_alpha,
                )

    plt.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()

    if return_handles:
        return {"fig": fig, "axes": axes, "panels": panels}
    return None
