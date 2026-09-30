"""
Histogram comparison plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .utils import require_show_or_handles


def plot_histograms(
    arr1: np.ndarray,
    arr2: np.ndarray,
    labels: tuple[str, str] = ("Array 1", "Array 2"),
    bins: int = 100,
    alpha: float = 0.6,
    xmin: float | None = None,
    xmax: float | None = None,
    figsize: tuple[float, float] = (8, 4),
    title: str | None = None,
    show_axes: bool = True,
    show: bool = True,
    return_handles: bool = False,
) -> dict[str, Any] | None:
    """
    Plot overlaid histograms of two numpy arrays.

    Args:
        arr1: First array to plot
        arr2: Second array to plot
        labels: Labels for the two arrays
        bins: Number of bins
        alpha: Opacity of the histograms
        xmin: Minimum value for the x-axis
        xmax: Maximum value for the x-axis
        figsize: Figure size
        title: Title of the plot
        show_axes: If False, hide axis spines, ticks, and labels.
        show: If True, call ``plt.show()`` before returning.
        return_handles: If True, return a dict with ``fig`` and ``axes``.
    """
    require_show_or_handles(show, return_handles)
    fig, ax = plt.subplots(figsize=figsize)

    combined = np.concatenate([arr1.ravel(), arr2.ravel()])
    bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

    ax.hist(arr1.ravel(), bins=bin_edges, alpha=alpha, label=labels[0], density=True)
    ax.hist(arr2.ravel(), bins=bin_edges, alpha=alpha, label=labels[1], density=True)

    ax.set_xlim(xmin, xmax)
    if show_axes:
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
    else:
        ax.set_axis_off()
    ax.legend()
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if show:
        plt.show()
    if return_handles:
        return {"fig": fig, "axes": ax}
    return None
