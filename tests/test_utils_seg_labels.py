"""
Tests for ``synthfcd.utils.seg_labels``.
"""

from __future__ import annotations

import pytest

from synthfcd.utils.seg_labels import LabelEnum, SynthSegLabel
from tests.utils import DummyLabel


class TestLabelEnumBase:
    """
    Tests for the public ``LabelEnum`` base class.
    """

    def test_is_subclass_of_label_enum(self) -> None:
        assert issubclass(SynthSegLabel, LabelEnum)
        assert issubclass(DummyLabel, LabelEnum)

    def test_dummy_label_implements_required_helpers(self) -> None:
        assert DummyLabel.get_background_labels() == {0}
        assert DummyLabel.get_white_matter_labels() == {2}
        assert DummyLabel.get_cortical_labels() == {1}


class TestSynthSegLabelLookup:
    """
    Lookup helpers and identity for ``SynthSegLabel``.
    """

    def test_lookup_by_value_and_name(self) -> None:
        label = SynthSegLabel(17)
        assert label is SynthSegLabel.LEFT_HIPPOCAMPUS
        assert label.name == "LEFT_HIPPOCAMPUS"
        assert SynthSegLabel["LEFT_HIPPOCAMPUS"].value == 17

    def test_readable_name_roundtrip(self) -> None:
        assert SynthSegLabel(17).readable_name == "Left Hippocampus"
        assert SynthSegLabel.from_readable_name("left hippocampus") is (
            SynthSegLabel.LEFT_HIPPOCAMPUS
        )
        assert SynthSegLabel.from_readable_name("LEFT HIPPOCAMPUS") is (
            SynthSegLabel.LEFT_HIPPOCAMPUS
        )

    def test_from_readable_name_unknown(self) -> None:
        assert SynthSegLabel.from_readable_name("not a label") is None

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SynthSegLabel(999999)


class TestSynthSegLabelClassification:
    """
    Hemisphere / tissue classification helpers.
    """

    def test_is_cortical(self) -> None:
        assert SynthSegLabel.is_cortical(1003)
        assert SynthSegLabel.is_cortical(2003)
        assert not SynthSegLabel.is_cortical(int(SynthSegLabel.LEFT_HIPPOCAMPUS))
        assert not SynthSegLabel.is_cortical(int(SynthSegLabel.BACKGROUND))

    def test_is_hemisphere(self) -> None:
        assert SynthSegLabel.is_left_hemisphere(int(SynthSegLabel.LEFT_THALAMUS))
        assert SynthSegLabel.is_left_hemisphere(1003)
        assert not SynthSegLabel.is_left_hemisphere(int(SynthSegLabel.RIGHT_THALAMUS))

        assert SynthSegLabel.is_right_hemisphere(int(SynthSegLabel.RIGHT_THALAMUS))
        assert SynthSegLabel.is_right_hemisphere(2003)
        assert not SynthSegLabel.is_right_hemisphere(int(SynthSegLabel.LEFT_THALAMUS))

    def test_midline_not_lateralized(self) -> None:
        for label in (
            SynthSegLabel.BRAIN_STEM,
            SynthSegLabel.CSF,
            SynthSegLabel.THIRD_VENTRICLE,
            SynthSegLabel.BACKGROUND,
        ):
            value = int(label)
            assert not SynthSegLabel.is_left_hemisphere(value)
            assert not SynthSegLabel.is_right_hemisphere(value)


class TestSynthSegLabelCollections:
    """
    Collection helpers used by mask extraction.
    """

    def test_background_and_csf(self) -> None:
        assert SynthSegLabel.get_background_labels() == {0}
        assert SynthSegLabel.get_background_csf_labels() == {
            int(SynthSegLabel.BACKGROUND),
            int(SynthSegLabel.CSF),
        }

    def test_white_matter_hemisphere_filter(self) -> None:
        left = SynthSegLabel.get_white_matter_labels("left")
        right = SynthSegLabel.get_white_matter_labels("right")
        both = SynthSegLabel.get_white_matter_labels()
        assert left == {
            int(SynthSegLabel.LEFT_CEREBRAL_WHITE_MATTER),
            int(SynthSegLabel.LEFT_CEREBELLUM_WHITE_MATTER),
        }
        assert right == {
            int(SynthSegLabel.RIGHT_CEREBRAL_WHITE_MATTER),
            int(SynthSegLabel.RIGHT_CEREBELLUM_WHITE_MATTER),
        }
        assert both == left | right

    def test_cortical_hemisphere_filter(self) -> None:
        left = SynthSegLabel.get_cortical_labels("left")
        right = SynthSegLabel.get_cortical_labels("right")
        assert all(1001 <= v <= 1035 for v in left)
        assert all(2001 <= v <= 2035 for v in right)
        assert left.isdisjoint(right)
        assert SynthSegLabel.get_cortical_labels() == left | right

    def test_subcortical_hemisphere_filter(self) -> None:
        left = SynthSegLabel.get_subcortical_labels("left")
        right = SynthSegLabel.get_subcortical_labels("right")
        assert int(SynthSegLabel.LEFT_HIPPOCAMPUS) in left
        assert int(SynthSegLabel.RIGHT_HIPPOCAMPUS) in right
        assert left.isdisjoint(right)

    def test_ventricles_lateral_only(self) -> None:
        all_vent = SynthSegLabel.get_ventricle_labels(lateral_only=False)
        lateral = SynthSegLabel.get_ventricle_labels(lateral_only=True)
        assert int(SynthSegLabel.THIRD_VENTRICLE) in all_vent
        assert int(SynthSegLabel.THIRD_VENTRICLE) not in lateral
        assert int(SynthSegLabel.LEFT_LATERAL_VENTRICLE) in lateral

        left = SynthSegLabel.get_ventricle_labels("left", lateral_only=True)
        assert left == {
            int(SynthSegLabel.LEFT_LATERAL_VENTRICLE),
            int(SynthSegLabel.LEFT_INFERIOR_LATERAL_VENTRICLE),
        }

    def test_all_labels_hemisphere_and_neutral(self) -> None:
        left = SynthSegLabel.get_all_labels("left")
        left_neutral = SynthSegLabel.get_all_labels("left", include_neutral=True)
        assert int(SynthSegLabel.LEFT_HIPPOCAMPUS) in left
        assert int(SynthSegLabel.RIGHT_HIPPOCAMPUS) not in left
        assert int(SynthSegLabel.CSF) not in left
        assert int(SynthSegLabel.CSF) in left_neutral
        assert int(SynthSegLabel.BACKGROUND) not in left
        assert int(SynthSegLabel.BACKGROUND) not in left_neutral

    def test_lobe_helpers(self) -> None:
        frontal_left = SynthSegLabel.get_frontal_lobe_labels("left")
        temporal_right = SynthSegLabel.get_temporal_lobe_labels("right")
        assert int(SynthSegLabel.LEFT_SUPERIOR_FRONTAL) in frontal_left
        assert int(SynthSegLabel.RIGHT_SUPERIOR_TEMPORAL) in temporal_right
        assert frontal_left.isdisjoint(SynthSegLabel.get_frontal_lobe_labels("right"))
