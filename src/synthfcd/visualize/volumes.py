"""
Orthogonal slice viewers for raw numpy volumes.
"""

from __future__ import annotations

from typing import Any, Literal, cast, overload

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    PLANE_SUBTITLES,
    get_slices,
    require_show_or_handles,
    resolve_slice_indices_list,
)


@overload
def plot_npy_volumes(
    volumes: dict[str, np.ndarray],
    title: str | None = None,
    figsize: tuple[int, int] = (18, 12),
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "mid",
    cmap: str | list[str] = "gray",
    vmin: float | list[float] | None = None,
    vmax: float | list[float] | None = None,
    show_axes: bool = True,
    show: bool = True,
    *,
    return_handles: Literal[True],
    save_path: str | None = None,
) -> dict[str, Any]: ...


@overload
def plot_npy_volumes(
    volumes: dict[str, np.ndarray],
    title: str | None = None,
    figsize: tuple[int, int] = (18, 12),
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "mid",
    cmap: str | list[str] = "gray",
    vmin: float | list[float] | None = None,
    vmax: float | list[float] | None = None,
    show_axes: bool = True,
    show: bool = True,
    return_handles: Literal[False] = False,
    save_path: str | None = None,
) -> None: ...


def plot_npy_volumes(
    volumes: dict[str, np.ndarray],
    title: str | None = None,
    figsize: tuple[int, int] = (18, 12),
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ) = "mid",
    cmap: str | list[str] = "gray",
    vmin: float | list[float] | None = None,
    vmax: float | list[float] | None = None,
    show_axes: bool = True,
    show: bool = True,
    return_handles: bool = False,
    save_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Plot multiple volumes in a single figure for comparison.

    Args:
        volumes: Dict mapping volume names to volume arrays (C, D1, D2, D3)
        title: Main title for the figure
        figsize: Figure size
        slice_indices: Per-volume slice index ``(dim1, dim2, dim3)``, or ``'mid'`` /
            ``'bbox'`` (bbox center of voxels with any configured label).
        cmap: Colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        show_axes: If False, hide axis spines, ticks, and labels on all panels.
        show: If True, call ``plt.show()`` before returning.
        return_handles: If True, return a dict with ``fig``, ``axes``, and ``panels``.
        save_path: Path to save figure

    Notes:
        Arguments `slice_indices`, `cmap`, `vmin`, and `vmax` can also be provided as
        lists to specify unique settings per volume.

    Raises:
        ValueError: If `slice_indices` is not a tuple of/or 'mid' or 'bbox'
        ValueError: If a volume is not 3D
    """
    require_show_or_handles(show, return_handles)
    volume_names = list(volumes.keys())
    n_volumes = len(volume_names)
    slice_indices_list = resolve_slice_indices_list(slice_indices, n_volumes)

    if isinstance(cmap, list):
        if len(cmap) != n_volumes:
            raise ValueError(
                f"Number of colormaps ({len(cmap)}) does not match number of volumes ({n_volumes})."
            )
    else:
        cmap = [cmap] * n_volumes

    if isinstance(vmin, list):
        if len(vmin) != n_volumes:
            raise ValueError(
                f"Number of minimum values ({len(vmin)}) does not match number of volumes ({n_volumes})."
            )
    else:
        vmin = [vmin] * n_volumes  # type: ignore

    if isinstance(vmax, list):
        if len(vmax) != n_volumes:
            raise ValueError(
                f"Number of maximum values ({len(vmax)}) does not match number of volumes ({n_volumes})."
            )
    else:
        vmax = [vmax] * n_volumes  # type: ignore

    main_title = title or "Multi-Volume Comparison - Three Orthogonal Planes"
    fig, axes = plt.subplots(n_volumes, 3, figsize=figsize)
    if n_volumes == 1:
        axes = axes.reshape(1, -1)

    panels: list[dict[str, Any]] = []

    for row_idx, vol_name in enumerate(volume_names):
        volume = volumes[vol_name]
        if volume.ndim != 3:
            raise ValueError(f"Volume {vol_name} is not 3D")

        curr_indices = slice_indices_list[row_idx]
        if curr_indices in ["mid", "bbox"]:
            curr_indices = get_slices(volume, curr_indices)  # type: ignore[arg-type]

        idx_d1, idx_d2, idx_d3 = cast(tuple[int, int, int], curr_indices)

        panels.append(
            {
                "row": vol_name,
                "row_idx": row_idx,
                "col": 0,
                "dim": "D1",
                "index": idx_d1,
            }
        )
        im1 = axes[row_idx, 0].imshow(
            volume[idx_d1, :, :],
            cmap=cmap[row_idx],
            origin="lower",
            vmin=vmin[row_idx],  # type: ignore
            vmax=vmax[row_idx],  # type: ignore
            aspect="equal",
        )
        axes[row_idx, 0].set_title(
            f"{vol_name}\n{PLANE_SUBTITLES[0]} (D1={idx_d1})",
            fontsize=10,
            fontweight="bold",
        )
        if show_axes:
            axes[row_idx, 0].set_xlabel("Dim3")
            axes[row_idx, 0].set_ylabel("Dim2")
            axes[row_idx, 0].grid(False)
        else:
            axes[row_idx, 0].set_axis_off()
        plt.colorbar(im1, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        panels.append(
            {
                "row": vol_name,
                "row_idx": row_idx,
                "col": 1,
                "dim": "D2",
                "index": idx_d2,
            }
        )
        im2 = axes[row_idx, 1].imshow(
            volume[:, idx_d2, :],
            cmap=cmap[row_idx],  # type: ignore
            origin="lower",
            vmin=vmin[row_idx],  # type: ignore
            vmax=vmax[row_idx],  # type: ignore
            aspect="equal",
        )
        axes[row_idx, 1].set_title(
            f"{vol_name}\n{PLANE_SUBTITLES[1]} (D2={idx_d2})",
            fontsize=10,
            fontweight="bold",
        )
        if show_axes:
            axes[row_idx, 1].set_xlabel("Dim3")
            axes[row_idx, 1].set_ylabel("Dim1")
            axes[row_idx, 1].grid(False)
        else:
            axes[row_idx, 1].set_axis_off()
        plt.colorbar(im2, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        panels.append(
            {
                "row": vol_name,
                "row_idx": row_idx,
                "col": 2,
                "dim": "D3",
                "index": idx_d3,
            }
        )
        im3 = axes[row_idx, 2].imshow(
            volume[:, :, idx_d3],
            cmap=cmap[row_idx],  # type: ignore
            origin="lower",
            vmin=vmin[row_idx],  # type: ignore
            vmax=vmax[row_idx],  # type: ignore
            aspect="equal",
        )
        axes[row_idx, 2].set_title(
            f"{vol_name}\n{PLANE_SUBTITLES[2]} (D3={idx_d3})",
            fontsize=10,
            fontweight="bold",
        )
        if show_axes:
            axes[row_idx, 2].set_xlabel("Dim2")
            axes[row_idx, 2].set_ylabel("Dim1")
            axes[row_idx, 2].grid(False)
        else:
            axes[row_idx, 2].set_axis_off()
        plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

    plt.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if return_handles:
        return {"fig": fig, "axes": axes, "panels": panels}
    return None
