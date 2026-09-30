"""
Helper functions for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from synthfcd.utils.seg_labels import LabelEnum, SynthSegLabel


def write_config(
    tmp_path: Path, config: dict[str, Any], name: str = "config.yaml"
) -> Path:
    """
    Dump a mapping to a YAML file under ``tmp_path`` and return the file path.
    """
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class DummyLabel(LabelEnum):
    """
    Dummy label enumeration for testing.
    """

    LABEL_0 = 0
    LABEL_1 = 1
    LABEL_2 = 2
    LABEL_3 = 3
    LABEL_4 = 4

    @classmethod
    def is_cortical(cls, label: int) -> bool:
        return True

    @classmethod
    def is_left_hemisphere(cls, label: int) -> bool:
        return True

    @classmethod
    def is_right_hemisphere(cls, label: int) -> bool:
        return True

    @classmethod
    def get_all_labels(
        cls,
        hemisphere: str | None = None,
        include_neutral: bool = False,
    ) -> set[int]:
        return {0, 1, 2, 3, 4}

    @classmethod
    def get_cortical_labels(cls, hemisphere: str | None = None) -> set[int]:
        return {1}

    @classmethod
    def get_subcortical_labels(cls, hemisphere: str | None = None) -> set[int]:
        return {4}

    @classmethod
    def get_white_matter_labels(cls, hemisphere: str | None = None) -> set[int]:
        return {2}

    @classmethod
    def get_background_labels(cls) -> set[int]:
        return {0}

    @classmethod
    def get_background_csf_labels(cls) -> set[int]:
        return cls.get_background_labels() | {3}

    @classmethod
    def get_ventricle_labels(
        cls,
        hemisphere: str | None = None,
        lateral_only: bool = False,
    ) -> set[int]:
        return {3}

    @classmethod
    def get_frontal_lobe_labels(cls, hemisphere: str | None = None) -> set[int]:
        return {1}

    @classmethod
    def get_temporal_lobe_labels(cls, hemisphere: str | None = None) -> set[int]:
        return {1}


def _zero_pad(mask: np.ndarray, pad_size: int = 10) -> np.ndarray:
    """
    Zero pad the mask.
    """
    return np.pad(
        mask,
        ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
        constant_values=0,
    ).astype(bool)


def make_nested_seg_mask(
    shape: tuple[int, int, int] = (32, 32, 32),
    *,
    wm_radius: float = 6.0,
    gm_radius: float = 10.0,
    bilateral: bool = False,
) -> np.ndarray:
    """
    Build a synthetic nested WM / cortex / background segmentation.

    Uses SynthSegLabel values: left WM=2, left cortex=1001 (and right
    counterparts when ``bilateral`` is True). Suitable for SDF and deformation
    effect tests.
    """
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = [s // 2 for s in shape]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

    seg = np.zeros(shape, dtype=np.int32)
    cortex = (dist <= gm_radius) & (dist > wm_radius)
    wm = dist <= wm_radius

    if not bilateral:
        seg[wm] = int(SynthSegLabel.LEFT_CEREBRAL_WHITE_MATTER)
        seg[cortex] = int(SynthSegLabel.LEFT_SUPERIOR_FRONTAL)
        return seg

    left = xx < cx
    right = ~left
    seg[wm & left] = int(SynthSegLabel.LEFT_CEREBRAL_WHITE_MATTER)
    seg[wm & right] = int(SynthSegLabel.RIGHT_CEREBRAL_WHITE_MATTER)
    seg[cortex & left] = int(SynthSegLabel.LEFT_SUPERIOR_FRONTAL)
    seg[cortex & right] = int(SynthSegLabel.RIGHT_SUPERIOR_FRONTAL)
    return seg


def make_dummy_nested_seg_mask(
    shape: tuple[int, int, int] = (32, 32, 32),
    *,
    wm_radius: float = 6.0,
    gm_radius: float = 10.0,
) -> np.ndarray:
    """
    Nested seg using DummyLabel convention: bg=0, cortex=1, WM=2.
    """
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = [s // 2 for s in shape]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

    seg = np.zeros(shape, dtype=np.int32)
    seg[dist <= gm_radius] = 1
    seg[dist <= wm_radius] = 2
    return seg
