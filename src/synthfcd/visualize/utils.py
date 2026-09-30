"""
Shared helpers for volume visualization.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from synthfcd.core.utils import bbox_from_mask

DEFAULT_LABELS: dict[int, tuple[str, str]] = {
    1: ("WM", "0.7"),
    2: ("GM", "tab:blue"),
    3: ("Lesion", "tab:red"),
}
PLANE_SUBTITLES: tuple[str, str, str] = ("Dim3-Dim2", "Dim3-Dim1", "Dim2-Dim1")

LabelInput = tuple[str, str] | str
LabelGroup = int | Iterable[int]
LabelsInput = Sequence[tuple[LabelGroup, LabelInput]] | Mapping[int, LabelInput] | None


def resolve_labels(
    labels: LabelsInput,
) -> tuple[dict[int, tuple[str, str]], list[tuple[str, str]]]:
    """
    Normalize a label configuration.

    Pass either a flat ``{id: style}`` mapping or a sequence of
    ``(ids, style)`` groups. Each group maps one or more integer label ids to
    the same legend name and color.

    Returns:
        ``id_styles``: label id to ``(name, color)``
        ``legend``: ordered ``(name, color)`` entries, one per group
    """
    if labels is None:
        groups = [((label_id,), style) for label_id, style in DEFAULT_LABELS.items()]
    elif isinstance(labels, Mapping):
        groups = [((int(label_id),), style) for label_id, style in labels.items()]
    else:
        groups = []
        for ids, style in labels:
            if isinstance(ids, (int, np.integer)):
                groups.append(((int(ids),), style))
            else:
                try:
                    groups.append((tuple(int(x) for x in ids), style))
                except TypeError as exc:
                    raise ValueError(
                        f"Label ids must be an int or iterable of ints, got {ids!r}."
                    ) from exc

    if not groups:
        raise ValueError("At least one label entry is required.")

    id_styles: dict[int, tuple[str, str]] = {}
    legend: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    for ids, value in groups:
        if not ids:
            raise ValueError("Each label group must contain at least one id.")

        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"Expected (name, color), got {value!r}.")
            name, color = value
            if not isinstance(name, str) or not isinstance(color, str):
                raise ValueError(f"Name and color must be strings, got {value!r}.")
        elif isinstance(value, str):
            if len(ids) != 1:
                raise ValueError(
                    "Color-only shorthand requires a single label id, not a group."
                )
            name, color = str(ids[0]), value
        else:
            raise ValueError(
                f"Expected (name, color) or color str, got {type(value).__name__}."
            )

        if name in seen_names:
            raise ValueError(f"Duplicate legend name {name!r}.")
        seen_names.add(name)
        legend.append((name, color))

        for label_id in ids:
            if label_id == 0:
                raise ValueError("Label id 0 is reserved for background.")
            if label_id in id_styles:
                raise ValueError(
                    f"Label id {label_id} appears in more than one legend group."
                )
            id_styles[label_id] = (name, color)

    return id_styles, legend


def require_show_or_handles(show: bool, return_handles: bool) -> None:
    if not show and not return_handles:
        raise ValueError(
            "At least one of show=True or return_handles=True is required."
        )


def show_plot_handles(
    handles: dict[str, Any],
    *,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str | None = None,
) -> None:
    """
    Show or save a figure returned by a visualize plot function.

    Args:
        handles: Dict with a ``fig`` key, as returned with ``return_handles=True``.
        show: If True, call ``plt.show()``.
        save_path: Optional path passed to ``fig.savefig``.
        dpi: Save resolution when ``save_path`` is set.
        bbox_inches: ``bbox_inches`` passed to ``fig.savefig``.
        facecolor: Optional figure face color for saving.

    Raises:
        ValueError: If ``handles`` has no ``fig`` key, or both ``show`` and
            ``save_path`` are disabled.
    """
    if not show and save_path is None:
        raise ValueError("At least one of show=True or save_path is required.")
    if "fig" not in handles:
        raise ValueError(f"handles must contain a 'fig' key, got {list(handles)}.")

    fig = handles["fig"]
    if save_path is not None:
        save_kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": bbox_inches}
        if facecolor is not None:
            save_kwargs["facecolor"] = facecolor
        fig.savefig(save_path, **save_kwargs)
    if show:
        plt.show()


def get_slices(
    volume: np.ndarray, slice_mode: Literal["mid", "bbox"] = "mid"
) -> tuple[int, int, int]:
    """
    Get slice indices for each spatial dimension.

    Args:
        volume: 3D volume or boolean mask used to locate the bbox center.
        slice_mode: ``'mid'`` for mid slices or ``'bbox'`` for bounding-box center.

    Returns:
        Slice indices for (dim1, dim2, dim3).
    """
    d1, _, _ = volume.shape

    if slice_mode == "mid":
        return (d1 // 2, volume.shape[1] // 2, volume.shape[2] // 2)

    bbox = bbox_from_mask(volume)
    if bbox[0].start is None:
        raise ValueError(
            "No voxels match the configured labels; cannot pick a bbox slice. "
            "Pass explicit slice_indices, or use a label volume that contains "
            "those label ids (not a boolean mask)."
        )
    return (
        int(bbox[0].start + (bbox[0].stop - bbox[0].start) / 2),
        int(bbox[1].start + (bbox[1].stop - bbox[1].start) / 2),
        int(bbox[2].start + (bbox[2].stop - bbox[2].start) / 2),
    )


def normalize_volume_dict(
    volume: np.ndarray | dict[str, np.ndarray],
    name: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if isinstance(volume, np.ndarray):
        if volume.ndim != 3:
            raise ValueError(f"{name} is not 3D")
        return {"volume": volume}, ["volume"]

    if not isinstance(volume, dict):
        raise ValueError(f"{name} must be a 3D array or a dict of 3D arrays.")

    if len(volume) == 0:
        raise ValueError(f"{name} dict must not be empty.")

    for key, arr in volume.items():
        if arr.ndim != 3:
            raise ValueError(f"{name}[{key!r}] is not 3D")

    names = list(volume.keys())
    return volume, names


def resolve_slice_indices_list(
    slice_indices: (
        tuple[int, int, int]
        | Literal["mid", "bbox"]
        | list[tuple[int, int, int] | Literal["mid", "bbox"]]
    ),
    n_volumes: int,
) -> list[tuple[int, int, int] | Literal["mid", "bbox"]]:
    if isinstance(slice_indices, list):
        if len(slice_indices) != n_volumes:
            raise ValueError(
                f"Number of slice indices ({len(slice_indices)}) does not match "
                f"number of volumes ({n_volumes})."
            )
        for s in slice_indices:
            if s not in ["mid", "bbox"] and not isinstance(s, tuple):
                raise ValueError(f"Invalid slice mode: {s}")
        return slice_indices

    return [slice_indices] * n_volumes  # type: ignore[list-item]


def resolve_volume_slice_indices(
    volume: np.ndarray,
    slice_indices: tuple[int, int, int] | Literal["mid", "bbox"],
    *,
    mask: np.ndarray,
) -> tuple[int, int, int]:
    if slice_indices in ["mid", "bbox"]:
        slice_indices = get_slices(mask, slice_indices)  # type: ignore[assignment]
    return cast(tuple[int, int, int], slice_indices)


def iter_panels(
    handles: dict[str, Any],
    rows: str | list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield figure panels, optionally limited to named rows.

    Args:
        handles: Dict with a ``panels`` list, as returned with
            ``return_handles=True``.
        rows: Row name or names to keep. ``None`` yields every panel.

    Raises:
        ValueError: If ``handles`` has no ``panels`` entry.
    """
    panels = handles.get("panels")
    if panels is None:
        raise ValueError(
            "handles must include 'panels'; pass return_handles=True to the plot "
            "function that created the figure."
        )

    if rows is None:
        row_filter = None
    elif isinstance(rows, str):
        row_filter = {rows}
    else:
        row_filter = set(rows)
    for panel in panels:
        if row_filter is not None and panel["row"] not in row_filter:
            continue
        yield panel


def slice_panel(volume: np.ndarray, panel: dict[str, Any]) -> np.ndarray:
    """
    Return the 2D slice of ``volume`` shown on ``panel``.

    ``D1`` indexes the first axis, ``D2`` the second, and ``D3`` the third.
    """
    index = panel["index"]
    if panel["dim"] == "D1":
        return volume[index, :, :]
    if panel["dim"] == "D2":
        return volume[:, index, :]
    return volume[:, :, index]


def in_plane_components(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    panel: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the horizontal and vertical displacement shown on ``panel``.

    Horizontal follows the displayed x axis and vertical the displayed y axis.
    """
    if panel["dim"] == "D1":
        return slice_panel(uz, panel), slice_panel(uy, panel)
    if panel["dim"] == "D2":
        return slice_panel(uz, panel), slice_panel(ux, panel)
    return slice_panel(uy, panel), slice_panel(ux, panel)
