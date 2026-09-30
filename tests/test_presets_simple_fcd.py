"""
Tests for ``synthfcd.presets.simple_fcd`` parameter drawing.
"""

from __future__ import annotations

import enum
from copy import deepcopy
from typing import Any

import pytest

from synthfcd.pipelines.simple_fcd import SimpleFCD
from synthfcd.presets.draw_params import Draw, draw_params
from synthfcd.presets.simple_fcd import (
    draw_simple_fcd_params,
    get_simple_fcd_presets,
)
from synthfcd.utils.misc import callable_name, get_leaf_paths_dict
from tests.utils import DummyLabel

FCD_TYPES = (
    "FCD_type_Ia",
    "FCD_type_Ib",
    "FCD_type_Ic",
    "FCD_type_IIa",
    "FCD_type_IIb",
)
PRESET_KEYS = ("default", "extreme", "minimal", "wmh_only", "type_II_only")
DEFORM_KEYS = tuple(callable_name(e) for e in SimpleFCD.DEFORMATION_EFFECTS)
INTENSITY_KEYS = tuple(callable_name(e) for e in SimpleFCD.INTENSITY_EFFECTS)


def _draw(
    sequences: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if sequences is None:
        sequences = ["T1like", "T2like"]
    return draw_simple_fcd_params(sequences=sequences, **kwargs)  # type: ignore[arg-type]


class TestDrawSimpleFCDOutput:
    """
    Return contract: keys, per-sequence length, and effect blocks.
    """

    @pytest.mark.parametrize("fcd_type", FCD_TYPES)
    def test_output_keys_and_shapes(self, fcd_type: str) -> None:
        params = _draw(fcd_type=fcd_type, random_seed=0)
        assert set(params) == {
            "FCD_type",
            "growth_params",
            "deformation_params",
            "intensity_params",
        }
        assert params["FCD_type"] == fcd_type
        assert isinstance(params["growth_params"], dict)
        assert set(params["deformation_params"]) == set(DEFORM_KEYS)
        assert all(isinstance(p, dict) for p in params["deformation_params"].values())
        assert isinstance(params["intensity_params"], list)
        assert len(params["intensity_params"]) == 2
        for block in params["intensity_params"]:
            assert set(block) == set(INTENSITY_KEYS)
            assert "T1like_settings" not in block["hyperintensity"]

    def test_intensity_list_matches_sequences(self) -> None:
        one = _draw(sequences=["T1like"], fcd_type="FCD_type_Ia", random_seed=0)
        three = _draw(
            sequences=["T1like", "T2like", "T1like"],
            fcd_type="FCD_type_Ia",
            random_seed=0,
        )
        assert len(one["intensity_params"]) == 1
        assert len(three["intensity_params"]) == 3

    def test_injected_seeds_present(self) -> None:
        params = _draw(fcd_type="FCD_type_Ia", random_seed=1)
        assert "random_seed" in params["growth_params"]
        assert "random_seed" in params["deformation_params"]["abnormal_gyration"]
        for block in params["intensity_params"]:
            assert "random_seed" in block["hyperintensity"]
            assert "random_seed" in block["texture_restoration"]


class TestDrawSimpleFCDDeterminism:
    """
    Seeded draws are stable; unseeded draws vary.
    """

    def test_same_seed_repeats(self) -> None:
        a = _draw(random_seed=42)
        b = _draw(random_seed=42)
        assert a == b

    def test_different_seeds_diverge(self) -> None:
        assert _draw(random_seed=1) != _draw(random_seed=2)

    def test_unseeded_draws_differ(self) -> None:
        assert _draw() != _draw()


class TestDrawSimpleFCDPresetSelection:
    """
    Named preset packs and custom overrides.
    """

    def test_unknown_preset_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_simple_fcd_presets("not-a-preset")
        with pytest.raises(ValueError, match="Unknown"):
            _draw(presets="not-a-preset")

    def test_all_named_presets_draw(self) -> None:
        for key in PRESET_KEYS:
            params = _draw(presets=key, random_seed=0)
            assert params["FCD_type"] in FCD_TYPES
            assert set(params["deformation_params"]) == set(DEFORM_KEYS)

    def test_minimal_disables_gyration(self) -> None:
        params = _draw(presets="minimal", fcd_type="FCD_type_Ia", random_seed=0)
        assert params["deformation_params"]["abnormal_gyration"]["enable"] is False
        default = _draw(presets="default", fcd_type="FCD_type_Ia", random_seed=0)
        assert default["deformation_params"]["abnormal_gyration"]["enable"] is True

    def test_wmh_only_enables_hyperintensity_alone(self) -> None:
        params = _draw(presets="wmh_only", fcd_type="FCD_type_IIa", random_seed=0)
        deform = params["deformation_params"]
        intensity = params["intensity_params"][0]
        assert deform["abnormal_gyration"]["enable"] is False
        assert deform["cortical_thickening"]["enable"] is False
        assert deform["sulcal_widening"]["enable"] is False
        assert intensity["hyperintensity"]["enable"] is True
        assert intensity["boundary_blurring"]["enable"] is False
        assert intensity["texture_restoration"]["enable"] is False

    def test_type_ii_only_draws_type_ii(self) -> None:
        drawn = {
            _draw(presets="type_II_only", random_seed=i)["FCD_type"] for i in range(20)
        }
        assert drawn <= {"FCD_type_IIa", "FCD_type_IIb"}
        assert drawn  # at least one draw succeeded

    def test_extreme_differs_from_default_spec(self) -> None:
        default = get_simple_fcd_presets("default")
        extreme = get_simple_fcd_presets("extreme")
        default_gyr = default["FCD_type_Ia"]["deformation_params"]["abnormal_gyration"]
        extreme_gyr = extreme["FCD_type_Ia"]["deformation_params"]["abnormal_gyration"]
        assert isinstance(default_gyr["mm_per_iter"], Draw)
        assert default_gyr["mm_per_iter"] != extreme_gyr["mm_per_iter"]

    def test_custom_presets_override_named(self) -> None:
        custom = get_simple_fcd_presets("default")
        custom["FCD_type_Ia"]["growth_params"]["volume"] = 123.0
        params = _draw(
            presets="extreme",
            custom_presets=custom,
            fcd_type="FCD_type_Ia",
            random_seed=0,
        )
        assert params["growth_params"]["volume"] == 123.0

    def test_custom_presets_must_match_default_structure(self) -> None:
        with pytest.raises(ValueError, match="same keys and structure"):
            _draw(custom_presets={"FCD_type": "FCD_type_Ia", "growth_params": {}})

        close = get_simple_fcd_presets()
        close["FCD_type_IIa"]["intensity_params"]["hyperintensity"][
            "T1like_settings"
        ] = {
            "scale_factor_shrink": 0.5,
            "signs": 1,
        }
        with pytest.raises(ValueError, match="same keys and structure"):
            _draw(custom_presets=close)

    def test_valid_value_change_is_accepted(self) -> None:
        custom = get_simple_fcd_presets()
        custom["FCD_type_IIa"]["intensity_params"]["hyperintensity"][
            "T1like_settings"
        ] = {
            "scale_factor_shrink": 0.5,
            "sign": 1,
        }
        params = _draw(
            sequences=["T1like", "T1like"],
            custom_presets=custom,
            fcd_type="FCD_type_IIa",
        )
        assert params["intensity_params"][0]["hyperintensity"]["scale_factor"] > 0

    def test_already_materialized_presets(self) -> None:
        concrete = draw_params(get_simple_fcd_presets(), random_seed=42)
        from_concrete = _draw(custom_presets=concrete, random_seed=42)
        from_spec = _draw(custom_presets=get_simple_fcd_presets(), random_seed=42)
        assert get_leaf_paths_dict(from_concrete) == get_leaf_paths_dict(from_spec)

    def test_presets_or_custom_required(self) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            _draw(presets=None, custom_presets=None)


class TestDrawSimpleFCDCrossModalCoupling:
    """
    Shared anatomy params vs sequence-specific intensity translation.
    """

    @pytest.mark.parametrize("fcd_type", FCD_TYPES)
    def test_t1_t2_share_anatomy_differ_in_hyperintensity(self, fcd_type: str) -> None:
        params = _draw(fcd_type=fcd_type, random_seed=3)
        t1, t2 = params["intensity_params"]
        assert t1["boundary_blurring"] == t2["boundary_blurring"]
        assert (
            t1["hyperintensity"]["random_seed"] == t2["hyperintensity"]["random_seed"]
        )
        assert (
            t1["hyperintensity"]["scale_factor"] != t2["hyperintensity"]["scale_factor"]
        )
        t1_tex = deepcopy(t1["texture_restoration"])
        t2_tex = deepcopy(t2["texture_restoration"])
        assert t1_tex.pop("random_seed") != t2_tex.pop("random_seed")
        assert t1_tex == t2_tex

    @pytest.mark.parametrize("fcd_type", FCD_TYPES)
    @pytest.mark.parametrize("sequence", ["T1like", "T2like"])
    def test_same_sequence_differs_only_by_texture_seed(
        self, fcd_type: str, sequence: str
    ) -> None:
        params = _draw(sequences=[sequence, sequence], fcd_type=fcd_type, random_seed=4)
        a, b = params["intensity_params"]
        seed_a = a["texture_restoration"].pop("random_seed")
        seed_b = b["texture_restoration"].pop("random_seed")
        assert seed_a != seed_b
        assert a == b

    def test_t1_scale_factor_is_shrunk_t2_with_sign(self) -> None:
        # Type Ia uses a fixed negative T1 sign, so T1 and T2 scales oppose.
        params = _draw(fcd_type="FCD_type_Ia", random_seed=5)
        t1_sf = params["intensity_params"][0]["hyperintensity"]["scale_factor"]
        t2_sf = params["intensity_params"][1]["hyperintensity"]["scale_factor"]
        assert t2_sf > 0
        assert t1_sf < 0
        assert 0.4 <= abs(t1_sf) / t2_sf <= 0.8

    def test_texture_rolloff_coupled_to_blurring(self) -> None:
        params = _draw(fcd_type="FCD_type_Ia", random_seed=6)
        intensity = params["intensity_params"][0]
        assert (
            intensity["texture_restoration"]["app_rolloff"]
            == intensity["boundary_blurring"]["app_rolloff"]
        )
        assert isinstance(intensity["texture_restoration"]["app_rolloff"], float)

    def test_sulcal_widening_enable_follows_bottom_of_sulcus(self) -> None:
        params = _draw(fcd_type="FCD_type_IIa", random_seed=7)
        assert (
            params["deformation_params"]["sulcal_widening"]["enable"]
            == params["growth_params"]["bottom_of_sulcus"]
        )

    def test_wmh_hpf_is_additive_to_corr(self) -> None:
        params = _draw(fcd_type="FCD_type_Ia", random_seed=8)
        hyper = params["intensity_params"][0]["hyperintensity"]
        assert hyper["wmh_seeds_hpf"] > hyper["wmh_seeds_corr"]

    def test_label_enum_only_changes_lobe(self) -> None:
        dummy = _draw(fcd_type="FCD_type_Ia", label_enum=DummyLabel, random_seed=42)
        default = _draw(fcd_type="FCD_type_Ia", random_seed=42)
        lobe_dummy = dummy["growth_params"].pop("lobe")
        lobe_default = default["growth_params"].pop("lobe")
        assert lobe_dummy != lobe_default
        # Type Ia `lobe` is a concrete set, not a Draw, so the rest of the
        # growth spec (including the post-hoc injected seed) sees the same RNG.
        assert dummy["growth_params"] == default["growth_params"]


class TestDrawSimpleFCDValidation:
    """
    Input validation for sequences, FCD type, and label enum.
    """

    @pytest.mark.parametrize("wrong_fcd_type", ("invalid", "FCD_type_I", "FCD_type_II"))
    def test_invalid_fcd_type(self, wrong_fcd_type: str) -> None:
        with pytest.raises(ValueError):
            _draw(fcd_type=wrong_fcd_type)

    @pytest.mark.parametrize("wrong_sequences", (["invalid"], [], "T1like"))
    def test_invalid_sequences(self, wrong_sequences: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            draw_simple_fcd_params(sequences=wrong_sequences)

    def test_invalid_label_enum(self) -> None:
        class InvalidLabel(enum.Enum):
            LABEL_0 = 0

        with pytest.raises(TypeError):
            _draw(label_enum=InvalidLabel)  # type: ignore[arg-type]
