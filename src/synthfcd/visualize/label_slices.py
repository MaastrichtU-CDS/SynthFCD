"""
2D label-grid viewers.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
def plot_npy_label_slices(
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 50,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_axes: bool = True,
    show: bool = True,
    *,
    return_handles: Literal[True],
    save_path: str | None = None,
) -> dict[str, Any]: ...


@overload
def plot_npy_label_slices(
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 50,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_axes: bool = True,
    show: bool = True,
    return_handles: Literal[False] = False,
    save_path: str | None = None,
) -> None: ...


def plot_npy_label_slices(
    label_volume: np.ndarray | dict[str, np.ndarray],
    labels: LabelsInput = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "bbox",
    marker_size: float = 50,
    marker: str = "o",
    show_background: bool = False,
    background_color: str = "0.15",
    show_legend: bool = True,
    legend_alpha: float = 1.0,
    show_axes: bool = True,
    show: bool = True,
    return_handles: bool = False,
    save_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Plot integer label volumes as a full-slice voxel grid (scatter), three planes.

    Each voxel in the displayed slice is drawn as a marker. Configure labels as
    either a flat ``{id: (name, color)}`` mapping or a sequence of
    ``(ids, (name, color))`` groups when several ids share one legend entry.

    Orientation matches :func:`~synthfcd.visualize.volumes.plot_npy_volumes`
    (slice row axis vertical, column axis horizontal).

    Args:
        label_volume: 3D integer label volume or dict of named label volumes.
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
        show_axes: If False, hide axis spines, ticks, and labels on all panels.
        show: If True, call ``plt.show()`` before returning.
        return_handles: If True, return a dict with ``fig``, ``axes``, and ``panels``.
        save_path: Optional path to save the figure.

    Raises:
        ValueError: If label configuration or arrays are invalid.
    """
    require_show_or_handles(show, return_handles)
    if not 0 <= legend_alpha <= 1:
        raise ValueError(f"legend_alpha must be in [0, 1], got {legend_alpha}.")
    id_styles, legend = resolve_labels(labels)
    name_to_ids: dict[str, list[int]] = {}
    for label_id, (name, _color) in id_styles.items():
        name_to_ids.setdefault(name, []).append(label_id)

    label_volumes, case_names = normalize_volume_dict(label_volume, "label_volume")
    slice_indices_list = resolve_slice_indices_list(slice_indices, len(case_names))

    main_title = title or "Label volumes - Three Orthogonal Planes"
    if figsize is None:
        figsize = (18, 6 * len(case_names))

    fig, axes = plt.subplots(len(case_names), 3, figsize=figsize)
    if len(case_names) == 1:
        axes = axes.reshape(1, -1)

    mask_ids = list(id_styles)
    panels: list[dict[str, Any]] = []

    for row_idx, name in enumerate(case_names):
        label_vol = label_volumes[name]
        idx_d1, idx_d2, idx_d3 = resolve_volume_slice_indices(
            label_vol,
            slice_indices_list[row_idx],
            mask=np.isin(label_vol, mask_ids),
        )
        planes = (
            (label_vol[idx_d1, :, :], PLANE_SUBTITLES[0], idx_d1, "D1"),
            (label_vol[:, idx_d2, :], PLANE_SUBTITLES[1], idx_d2, "D2"),
            (label_vol[:, :, idx_d3], PLANE_SUBTITLES[2], idx_d3, "D3"),
        )

        for col_idx, (sl_label, plane_title, idx, dim_name) in enumerate(planes):
            panels.append(
                {
                    "row": name,
                    "row_idx": row_idx,
                    "col": col_idx,
                    "dim": dim_name,
                    "index": idx,
                }
            )
            ax = axes[row_idx, col_idx]
            d_row, d_col = sl_label.shape
            yy, xx = np.mgrid[0:d_row, 0:d_col]
            x = xx.ravel()
            y = yy.ravel()
            cls = sl_label.ravel()

            if show_background:
                bg = cls == 0
                if bg.any():
                    ax.scatter(
                        x[bg],
                        y[bg],
                        c=background_color,
                        s=marker_size,
                        marker=marker,
                        linewidths=0,
                    )

            for entry_name, color in legend:
                sel = np.isin(cls, name_to_ids[entry_name])
                if not sel.any():
                    continue
                ax.scatter(
                    x[sel],
                    y[sel],
                    c=color,
                    s=marker_size,
                    marker=marker,
                    linewidths=0,
                )

            ax.set_aspect("equal")
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_axis_off()
            ax.set_title(
                f"{name}\n{plane_title} ({dim_name}={idx})",
                fontsize=10,
                fontweight="bold",
            )

        if show_legend:
            axes[row_idx, 2].legend(
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if return_handles:
        return {"fig": fig, "axes": axes, "panels": panels}
    return None
