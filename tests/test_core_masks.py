"""
Tests for ``synthfcd.core.masks``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synthfcd.core.masks import extract_masks, get_binary_mask, resolve_label
from synthfcd.utils.seg_labels import SynthSegLabel
from tests.utils import DummyLabel, make_dummy_nested_seg_mask, make_nested_seg_mask


class TestResolveLabel:
    """
    Tests for ``resolve_label``.
    """

    @pytest.mark.parametrize(
        "wrong_enum",
        [None, "SynthSegLabel", int, object()],
    )
    def test_invalid_label_enum(self, wrong_enum: Any) -> None:
        with pytest.raises(TypeError):
            resolve_label(wrong_enum, 2)

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            resolve_label(SynthSegLabel, [])

    def test_int_and_enum_and_readable_name(self) -> None:
        assert resolve_label(SynthSegLabel, 17) == {17}
        assert resolve_label(SynthSegLabel, SynthSegLabel.LEFT_HIPPOCAMPUS) == {17}
        assert resolve_label(SynthSegLabel, "left hippocampus") == {17}
        assert resolve_label(SynthSegLabel, "LEFT_HIPPOCAMPUS") == {17}
        assert resolve_label(SynthSegLabel, " 17 ") == {17}

    def test_collection_union(self) -> None:
        got = resolve_label(
            SynthSegLabel,
            ["left hippocampus", SynthSegLabel.CSF, 2],
        )
        assert got == {17, 24, 2}

    def test_wrong_enum_member_type(self) -> None:
        with pytest.raises(TypeError):
            resolve_label(SynthSegLabel, DummyLabel.LABEL_1)

    def test_invalid_int(self) -> None:
        with pytest.raises(ValueError):
            resolve_label(SynthSegLabel, 999999)

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            resolve_label(SynthSegLabel, "not a real label")


class TestGetBinaryMask:
    """
    Tests for ``get_binary_mask``.
    """

    @pytest.fixture
    def seg_mask(self) -> np.ndarray:
        return np.array(
            [
                [
                    [0, 2, 4],
                    [14, 17, 24],
                ],
                [
                    [41, 43, 53],
                    [1003, 2003, 2030],
                ],
            ],
            dtype=int,
        )

    @pytest.mark.parametrize(
        "wrong_seg",
        ["str", None, 1, np.zeros((2, 2, 2), dtype=int).tolist()],
    )
    def test_invalid_seg_type(self, wrong_seg: Any) -> None:
        with pytest.raises(TypeError):
            get_binary_mask(wrong_seg, mask_type="wm")

    def test_invalid_seg_dtype(self) -> None:
        with pytest.raises(ValueError):
            get_binary_mask(np.zeros((4, 4, 4), dtype=float), mask_type="wm")

    def test_requires_exactly_one_selector(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            get_binary_mask(seg_mask, mask_type="wm", labels=2)
        with pytest.raises(ValueError, match="exactly one"):
            get_binary_mask(seg_mask)

    def test_invalid_mask_type(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            get_binary_mask(seg_mask, mask_type="lesion")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            get_binary_mask(seg_mask, mask_type=1)  # type: ignore[arg-type]

    def test_invalid_hemisphere(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            get_binary_mask(
                seg_mask, mask_type="wm", hemisphere="middle"  # type: ignore[arg-type]
            )

    def test_invalid_label_enum(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(TypeError):
            get_binary_mask(seg_mask, mask_type="wm", label_enum=object)  # type: ignore[arg-type]

    def test_mask_type_gm_left(self, seg_mask: np.ndarray) -> None:
        out = get_binary_mask(
            seg_mask,
            mask_type="gm",
            hemisphere="left",
            label_enum=SynthSegLabel,
        )
        expected = np.isin(seg_mask, list(SynthSegLabel.get_cortical_labels("left")))
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(out, np.isin(seg_mask, [1003]))

    def test_mask_type_wm(self, seg_mask: np.ndarray) -> None:
        out = get_binary_mask(seg_mask, mask_type="wm")
        expected = np.isin(seg_mask, list(SynthSegLabel.get_white_matter_labels()))
        np.testing.assert_array_equal(out, expected)

    def test_mask_type_ventricles_extra_kwargs(self, seg_mask: np.ndarray) -> None:
        out = get_binary_mask(
            seg_mask,
            mask_type="ventricles",
            extra_kwargs={"lateral_only": False},
        )
        expected = np.isin(
            seg_mask,
            list(SynthSegLabel.get_ventricle_labels(lateral_only=False)),
        )
        np.testing.assert_array_equal(out, expected)
        assert out.any()

        lateral = get_binary_mask(
            seg_mask,
            mask_type="ventricles",
            extra_kwargs={"lateral_only": True},
        )
        assert not np.array_equal(out, lateral)

    def test_mask_type_subcortical_and_background_csf(
        self, seg_mask: np.ndarray
    ) -> None:
        sub = get_binary_mask(seg_mask, mask_type="subcortical")
        bg_csf = get_binary_mask(seg_mask, mask_type="background-csf")
        np.testing.assert_array_equal(
            sub,
            np.isin(seg_mask, list(SynthSegLabel.get_subcortical_labels())),
        )
        np.testing.assert_array_equal(
            bg_csf,
            np.isin(seg_mask, list(SynthSegLabel.get_background_csf_labels())),
        )

    def test_mask_type_all_hemisphere(self, seg_mask: np.ndarray) -> None:
        left = get_binary_mask(seg_mask, mask_type="all", hemisphere="left")
        expected = np.isin(
            seg_mask, list(SynthSegLabel.get_all_labels(hemisphere="left"))
        )
        np.testing.assert_array_equal(left, expected)

    def test_explicit_labels_multiple_formats(self, seg_mask: np.ndarray) -> None:
        out = get_binary_mask(
            seg_mask,
            labels=[
                "left hippocampus",
                SynthSegLabel.RIGHT_HIPPOCAMPUS,
                24,
            ],
        )
        expected = np.isin(seg_mask, [17, 53, 24])
        np.testing.assert_array_equal(out, expected)

    def test_explicit_labels_with_dummy_enum(self) -> None:
        seg = make_dummy_nested_seg_mask((16, 16, 16))
        out = get_binary_mask(seg, labels=2, label_enum=DummyLabel)
        np.testing.assert_array_equal(out, seg == 2)

    def test_nested_synthseg_volume_gm_wm(self) -> None:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=9)
        wm = get_binary_mask(seg, mask_type="wm")
        gm = get_binary_mask(seg, mask_type="gm")
        assert wm.any() and gm.any()
        assert not np.any(wm & gm)
        assert np.all(seg[wm] == int(SynthSegLabel.LEFT_CEREBRAL_WHITE_MATTER))


class TestExtractMasks:
    """
    Light coverage for deprecated ``extract_masks``.
    """

    def test_deprecation_and_keys(self) -> None:
        seg = make_nested_seg_mask((20, 20, 20), wm_radius=4, gm_radius=7)
        with pytest.warns(DeprecationWarning):
            out = extract_masks(seg)
        assert set(out) == {"seg_mask", "gm_mask", "wm_mask"}
        assert out["gm_mask"].dtype == bool
        assert out["wm_mask"].dtype == bool
        assert out["gm_mask"].any() and out["wm_mask"].any()

    def test_empty_seg_raises(self) -> None:
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="empty"):
                extract_masks(np.zeros((8, 8, 8), dtype=int))

    def test_hemisphere_filter(self) -> None:
        seg = make_nested_seg_mask(
            (24, 24, 24), wm_radius=5, gm_radius=9, bilateral=True
        )
        with pytest.warns(DeprecationWarning):
            left = extract_masks(seg, hemisphere="left")
        with pytest.warns(DeprecationWarning):
            right = extract_masks(seg, hemisphere="right")
        assert left["wm_mask"].any() and right["wm_mask"].any()
        assert not np.array_equal(left["wm_mask"], right["wm_mask"])
