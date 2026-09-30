"""
Tests for generic preset drawing, seeding, and sampler helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synthfcd.presets.draw_params import Draw, draw_params
from synthfcd.presets.utils import (
    sample_neg_trunc_lognormal,
    sample_trunc_lognormal,
    seed_params,
)
from synthfcd.utils._const import SEED_MAX


class TestDraw:
    """
    Validation and sampling for ``Draw``.
    """

    @pytest.mark.parametrize("sampler", [1, None, 3.0, ("choice",)])
    def test_invalid_sampler_type(self, sampler: Any) -> None:
        with pytest.raises(TypeError):
            Draw(sampler)  # type: ignore[arg-type]

    def test_invalid_args_type(self) -> None:
        with pytest.raises(TypeError):
            Draw("choice", args=[(0, 1)])  # type: ignore[arg-type]

    def test_invalid_kwargs_type(self) -> None:
        with pytest.raises(TypeError):
            Draw("choice", kwargs=[])  # type: ignore[arg-type]

    def test_unknown_rng_method(self) -> None:
        with pytest.raises(AttributeError, match="does not have a method"):
            Draw("not_a_method")(np.random.default_rng(0))

    def test_invalid_rng_type(self) -> None:
        with pytest.raises(TypeError):
            Draw("integers", (0, 3))("not-an-rng")  # type: ignore[arg-type]

    def test_rng_method_and_callable(self) -> None:
        rng = np.random.default_rng(0)
        n = Draw("integers", (2, 8))(rng)
        assert isinstance(n, int) and not isinstance(n, np.generic)
        assert 2 <= n < 8

        drawn = Draw(sample_trunc_lognormal, ((1.0, 2.0),))(np.random.default_rng(1))
        assert isinstance(drawn, float)
        assert 1.0 <= drawn <= 2.0

    def test_integers_routes_random_state_randint(self) -> None:
        n = Draw("integers", (0, 10))(np.random.RandomState(0))
        assert isinstance(n, int)
        assert 0 <= n < 10

    def test_choice_normalizes_numpy_scalar(self) -> None:
        value = Draw("choice", ((True, False),))(np.random.default_rng(0))
        assert value in (True, False)
        assert type(value) is bool


class TestDrawParams:
    """
    Recursive materialization of nested ``Draw`` specs.
    """

    def test_recurses_containers_and_copies_literals(self) -> None:
        literal = {"fixed": 3}
        spec = {
            "n": Draw("integers", (1, 5)),
            "nested": {"x": Draw("uniform", (0.0, 1.0))},
            "items": [Draw("choice", ((10, 20),)), literal],
            "pair": (Draw("integers", (0, 2)), 7),
        }
        out = draw_params(spec, random_seed=0)
        assert isinstance(out["n"], int)
        assert 0.0 <= out["nested"]["x"] <= 1.0
        assert out["items"][0] in (10, 20)
        assert out["items"][1] == literal
        assert out["items"][1] is not literal
        assert out["pair"][1] == 7
        assert isinstance(out["pair"], tuple)

    def test_seed_determinism(self) -> None:
        spec = {"a": Draw("uniform", (0.0, 1.0)), "b": Draw("integers", (0, 100))}
        assert draw_params(spec, random_seed=7) == draw_params(spec, random_seed=7)
        assert draw_params(spec, random_seed=7) != draw_params(spec, random_seed=8)

    def test_failed_draw_wraps_path(self) -> None:
        spec = {"bad": Draw("integers", ("x", "y"))}
        with pytest.raises(
            RuntimeError, match=r"Failed to draw parameter `params.bad`"
        ):
            draw_params(spec, random_seed=0)


class TestSampleTruncLognormal:
    """
    Bounds, validation, and seeding for truncated log-normal samplers.
    """

    @pytest.mark.parametrize("bad_range", [(1.0,), (1.0, 2.0, 3.0), "1-2"])
    def test_invalid_range(self, bad_range: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            sample_trunc_lognormal(bad_range)  # type: ignore[arg-type]

    def test_unordered_or_zero_width_range(self) -> None:
        with pytest.raises(ValueError, match="range"):
            sample_trunc_lognormal((2.0, 1.0))
        with pytest.raises(ValueError, match="range"):
            sample_trunc_lognormal((1.0, 1.0))

    def test_samples_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = sample_trunc_lognormal((0.5, 2.0), random_state=rng)
            assert 0.5 <= x <= 2.0

    def test_seed_determinism(self) -> None:
        a = sample_trunc_lognormal((1.0, 4.0), random_seed=11)
        b = sample_trunc_lognormal((1.0, 4.0), random_seed=11)
        c = sample_trunc_lognormal((1.0, 4.0), random_seed=12)
        assert a == b
        assert a != c

    def test_negative_variant_stays_in_range(self) -> None:
        x = sample_neg_trunc_lognormal((-3.0, -1.0), random_seed=0)
        assert -3.0 <= x <= -1.0
        assert x == -sample_trunc_lognormal((1.0, 3.0), random_seed=0)


class TestSeedParams:
    """
    In-place seed injection into nested parameter dicts.
    """

    def test_all_seeds_terminal_dicts(self) -> None:
        params = {
            "growth": {"volume": 10.0},
            "effects": {"gyration": {"enable": True}, "blur": {"enable": True}},
        }
        seed_params(np.random.default_rng(0), params, paths_to_seed="all")
        seeds = [
            params["growth"]["random_seed"],
            params["effects"]["gyration"]["random_seed"],
            params["effects"]["blur"]["random_seed"],
        ]
        assert all(isinstance(s, int) and 0 <= s < SEED_MAX for s in seeds)
        assert len(set(seeds)) == 3

    def test_explicit_paths_only(self) -> None:
        params = {
            "growth": {"volume": 1},
            "intensity": {"texture": {"enable": True}},
        }
        seed_params(
            np.random.default_rng(1),
            params,
            paths_to_seed=["intensity.texture"],
        )
        assert "random_seed" in params["intensity"]["texture"]
        assert "random_seed" not in params["growth"]

    def test_skips_existing_seed_with_warning(self) -> None:
        params = {"growth": {"random_seed": 99}}
        with pytest.warns(UserWarning, match="already provided"):
            seed_params(np.random.default_rng(0), params, paths_to_seed=["growth"])
        assert params["growth"]["random_seed"] == 99

    def test_skips_existing_random_state(self) -> None:
        params = {"growth": {"random_state": np.random.default_rng(0)}}
        with pytest.warns(UserWarning, match="random_state"):
            seed_params(np.random.default_rng(1), params, paths_to_seed=["growth"])
        assert "random_seed" not in params["growth"]

    def test_empty_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            seed_params(np.random.default_rng(0), {}, paths_to_seed=[])

    def test_path_must_point_to_dict(self) -> None:
        with pytest.raises(ValueError, match="dictionary"):
            seed_params(
                np.random.default_rng(0),
                {"growth": {"volume": 1}},
                paths_to_seed=["growth.volume"],
            )

    def test_random_state_randint_path(self) -> None:
        params = {"growth": {"volume": 1}}
        seed_params(np.random.RandomState(0), params, paths_to_seed=["growth"])
        assert isinstance(params["growth"]["random_seed"], int)
