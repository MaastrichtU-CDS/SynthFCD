"""
Animated GIF helpers for side-by-side MRI volume pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image as PILImage

from .utils import get_slices


def create_gif(
    images: Sequence[tuple[np.ndarray, np.ndarray]],
    save_path: str | Path,
    *,
    hold_s: float = 0.75,
    fade_s: float = 0.25,
    fps: int = 20,
    loop_fade: bool = True,
    slice_mode: Literal["mid", "bbox"] = "mid",
    rotation_deg: int = 0,
    vmin: tuple[float, float] | None = None,
    vmax: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    out_width: int = 1200,
    gap_px: int = 16,
    margin_px: int = 0,
    background: int = 0,
    gray_levels: int = 64,
) -> Path:
    """
    Create a grayscale GIF of left/right volume pairs along the dim1-dim2 plane.

    Each entry in ``images`` is shown as a side-by-side pair for ``hold_s`` seconds,
    then crossfaded into the next pair over ``fade_s`` seconds. Slice selection
    matches :func:`~synthfcd.visualize.volumes.plot_npy_volumes`. Axes, ticks, and
    labels are omitted.

    Args:
        images: Sequence of ``(left, right)`` 3D volumes. All volumes must share
            one shape so the same slice index stays aligned.
        save_path: Destination path for the written ``.gif``. Parent directories are
            created when missing.
        hold_s: Seconds to hold each pair at full opacity.
        fade_s: Seconds to crossfade from one pair into the next.
        fps: Frames per second used to discretize ``hold_s`` and ``fade_s``.
        loop_fade: If ``True``, fade the last pair back into the first. If
            ``False``, keep the last pair for the fade interval instead.
        slice_mode: ``'mid'`` or ``'bbox'``, as in ``plot_npy_volumes``.
        rotation_deg: Counterclockwise rotation applied after the
            ``plot_npy_volumes`` orientation. Must be a multiple of ``90``.
        vmin: Optional ``(left, right)`` lower display limits. ``None`` uses
            ``percentiles`` over the extracted planes of that contrast.
        vmax: Optional ``(left, right)`` upper display limits.
        percentiles: Percentile window used when ``vmin`` or ``vmax`` is ``None``.
        out_width: Output canvas width in pixels, including the gap and margins.
        gap_px: Horizontal spacing between the left and right panels.
        margin_px: Outer margin around both panels.
        background: Grayscale fill value for the gap and margins.
        gray_levels: Number of gray levels written to the GIF. Lower values
            shrink file size for upload limits.

    Returns:
        Absolute :class:`~pathlib.Path` of the written GIF.

    Raises:
        ValueError: If ``images`` is empty, volumes are not 3D or mismatched,
            ``rotation_deg`` is not a multiple of ``90``, ``hold_s`` is shorter
            than one frame, or display/layout parameters are invalid.
    """
    if not images:
        raise ValueError("images must contain at least one (left, right) pair.")
    if hold_s < 0 or fade_s < 0:
        raise ValueError(f"hold_s and fade_s must be >= 0, got {hold_s}, {fade_s}.")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}.")
    if rotation_deg % 90 != 0:
        raise ValueError(f"rotation_deg must be a multiple of 90, got {rotation_deg}.")
    if gray_levels < 2 or gray_levels > 256:
        raise ValueError(f"gray_levels must be in [2, 256], got {gray_levels}.")
    if out_width < 1 or gap_px < 0 or margin_px < 0:
        raise ValueError("out_width must be >= 1 and gap/margin must be >= 0.")
    if not 0 <= background <= 255:
        raise ValueError(f"background must be in [0, 255], got {background}.")

    save_path = Path(save_path)

    ref = images[0][0]
    if ref.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got shape {ref.shape}.")
    _, _, slice_index = get_slices(ref, slice_mode)

    left_planes: list[np.ndarray] = []
    right_planes: list[np.ndarray] = []
    for left, right in images:
        if left.shape != ref.shape or right.shape != ref.shape:
            raise ValueError(
                "Every volume must share one shape so the slice stays aligned."
            )
        if left.ndim != 3 or right.ndim != 3:
            raise ValueError("Every volume must be 3D.")
        left_planes.append(_dim12_plane(left, slice_index, rotation_deg))
        right_planes.append(_dim12_plane(right, slice_index, rotation_deg))

    left_lo, left_hi = _contrast_limits(
        left_planes,
        None if vmin is None else vmin[0],
        None if vmax is None else vmax[0],
        percentiles=percentiles,
    )
    right_lo, right_hi = _contrast_limits(
        right_planes,
        None if vmin is None else vmin[1],
        None if vmax is None else vmax[1],
        percentiles=percentiles,
    )

    slice_h, slice_w = left_planes[0].shape
    panel_w = (out_width - 2 * margin_px - gap_px) // 2
    if panel_w < 1:
        raise ValueError(
            f"out_width={out_width} is too small for gap_px={gap_px} and "
            f"margin_px={margin_px}."
        )
    panel_h = max(1, round(panel_w * slice_h / slice_w))
    canvas_w = panel_w * 2 + gap_px + 2 * margin_px
    canvas_h = panel_h + 2 * margin_px
    left_x = margin_px
    right_x = margin_px + panel_w + gap_px

    panels = [
        _compose_pair(
            _to_uint8(left, left_lo, left_hi),
            _to_uint8(right, right_lo, right_hi),
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            panel_h=panel_h,
            panel_w=panel_w,
            left_x=left_x,
            right_x=right_x,
            margin_px=margin_px,
            background=background,
        )
        for left, right in zip(left_planes, right_planes)
    ]

    n_hold = round(hold_s * fps)
    n_fade = round(fade_s * fps)
    if n_hold < 1:
        raise ValueError("hold_s is shorter than one frame; raise hold_s or fps.")
    frame_ms = round(1000 / fps)

    frames: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        frames.extend([panel] * n_hold)
        if n_fade == 0:
            continue
        if i == len(panels) - 1 and not loop_fade:
            frames.extend([panel] * n_fade)
            continue
        nxt = panels[(i + 1) % len(panels)]
        for step in range(1, n_fade + 1):
            # Last fade frame is the next pair; its hold then keeps it for hold_s.
            frames.append(_blend(panel, nxt, step / n_fade))

    pil_frames = [
        PILImage.fromarray(_quantize_gray(frame, gray_levels), mode="L")
        for frame in frames
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return save_path.resolve()


def _dim12_plane(volume: np.ndarray, index: int, rotation_deg: int) -> np.ndarray:
    """
    Dim1-dim2 plane, like plot_npy_volumes, then rotated counterclockwise.
    """
    plane = np.flipud(np.asarray(volume[:, :, index], dtype=np.float32))
    turns = (rotation_deg // 90) % 4
    return np.rot90(plane, k=turns) if turns else plane


def _contrast_limits(
    planes: list[np.ndarray],
    lo: float | None,
    hi: float | None,
    *,
    percentiles: tuple[float, float],
) -> tuple[float, float]:
    if lo is None or hi is None:
        sample = np.concatenate([plane.ravel() for plane in planes])
        plo, phi = np.percentile(sample, percentiles)
        lo = float(plo if lo is None else lo)
        hi = float(phi if hi is None else hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_uint8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    mixed = (1.0 - alpha) * a + alpha * b
    return np.round(mixed).astype(np.uint8)


def _compose_pair(
    left_u8: np.ndarray,
    right_u8: np.ndarray,
    *,
    canvas_h: int,
    canvas_w: int,
    panel_h: int,
    panel_w: int,
    left_x: int,
    right_x: int,
    margin_px: int,
    background: int,
) -> np.ndarray:
    canvas = np.full((canvas_h, canvas_w), background, dtype=np.uint8)
    left_im = PILImage.fromarray(left_u8, mode="L").resize(
        (panel_w, panel_h), PILImage.Resampling.LANCZOS
    )
    right_im = PILImage.fromarray(right_u8, mode="L").resize(
        (panel_w, panel_h), PILImage.Resampling.LANCZOS
    )
    canvas[margin_px : margin_px + panel_h, left_x : left_x + panel_w] = np.asarray(
        left_im
    )
    canvas[margin_px : margin_px + panel_h, right_x : right_x + panel_w] = np.asarray(
        right_im
    )
    return canvas


def _quantize_gray(frame: np.ndarray, levels: int) -> np.ndarray:
    if levels >= 256:
        return frame
    step = 255 / (levels - 1)
    q = np.round(frame.astype(np.float32) / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)
