"""
Tests for ``synthfcd.pipelines.apply_effects``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from synthfcd.pipelines.apply_effects import AppliesEffects, apply_effects
from synthfcd.utils._aliases import (
    _DeformationResultType,
    _DistanceType,
    _EffectFnType,
    _IntensityResultType,
)


def _stub_intensity(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    delta: float = 1.0,
    **kwargs: Any,
) -> _IntensityResultType:
    effect = np.full(image.shape, delta, dtype=np.float64)
    return {"out_image": image + effect, "effect_field": effect}


def _stub_deformation(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    shift: float = 0.0,
    **kwargs: Any,
) -> _DeformationResultType:
    z = np.zeros(image.shape, dtype=np.float64)
    field = [(np.full_like(z, shift), z.copy(), z.copy())] if shift != 0.0 else []
    # Identity / no-op for image when shift is 0; for non-zero still return image
    # unchanged so tests focus on field aggregation/caching rather than warping.
    return {
        "out_image": image.copy(),
        "out_seg_mask": seg_mask.copy(),
        "effect_field": field,
    }


def _stub_bad_return(
    image: np.ndarray,
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    **kwargs: Any,
) -> dict[str, Any]:
    return {"out_image": image}


class TestApplyEffects:
    """
    Tests for the functional ``apply_effects`` helper.
    """

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        image = np.zeros((10, 10, 10), dtype=np.float64)
        seg = np.zeros((10, 10, 10), dtype=np.int32)
        seg[3:7, 3:7, 3:7] = 2
        return image, seg

    def test_empty_effects_raises(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError, match="non-empty"):
            apply_effects(image, seg, (1.0, 1.0, 1.0), effects=[], effect_params=[])

    def test_length_mismatch_raises(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError, match="same length"):
            apply_effects(
                image,
                seg,
                (1.0, 1.0, 1.0),
                effects=[_stub_intensity],
                effect_params=[],
            )

    def test_non_callable_effects_raises(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError, match="callable"):
            apply_effects(
                image,
                seg,
                (1.0, 1.0, 1.0),
                effects=["not-callable"],  # type: ignore[list-item]
                effect_params=[{}],
            )

    def test_disabled_without_allow_raises(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError, match="disabled"):
            apply_effects(
                image,
                seg,
                (1.0, 1.0, 1.0),
                effects=[_stub_intensity],
                effect_params=[{"enable": False}],
                allow_disabled=False,
            )

    def test_disabled_with_allow_skips(self, volume: tuple) -> None:
        image, seg = volume
        out = apply_effects(
            image,
            seg,
            (1.0, 1.0, 1.0),
            effects=[_stub_intensity, _stub_intensity],
            effect_params=[{"enable": False}, {"delta": 2.0}],
            allow_disabled=True,
        )
        np.testing.assert_allclose(out["out_image"], image + 2.0)
        assert len(out["effect_fields"]) == 1

    def test_sequential_intensity_aggregation(self, volume: tuple) -> None:
        image, seg = volume
        out = apply_effects(
            image,
            seg,
            (1.0, 1.0, 1.0),
            effects=[_stub_intensity, _stub_intensity],
            effect_params=[{"delta": 1.0}, {"delta": 3.0}],
        )
        np.testing.assert_allclose(out["out_image"], image + 4.0)
        assert len(out["effect_fields"]) == 2
        np.testing.assert_allclose(out["effect_fields"][0], 1.0)
        np.testing.assert_allclose(out["effect_fields"][1], 3.0)

    def test_deformation_fields_aggregated(self, volume: tuple) -> None:
        image, seg = volume
        out = apply_effects(
            image,
            seg,
            (1.0, 1.0, 1.0),
            effects=[_stub_deformation],
            effect_params=[{"shift": 0.25}],
        )
        assert set(out) == {"out_image", "out_seg_mask", "effect_fields"}
        assert len(out["effect_fields"]) == 1
        assert isinstance(out["effect_fields"][0], list)
        assert len(out["effect_fields"][0]) == 1
        assert len(out["effect_fields"][0][0]) == 3

    def test_invalid_effect_return_keys(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            apply_effects(
                image,
                seg,
                (1.0, 1.0, 1.0),
                effects=[cast(_EffectFnType, _stub_bad_return)],
                effect_params=[{}],
            )

    def test_does_not_mutate_inputs(self, volume: tuple) -> None:
        image, seg = volume
        image_before = image.copy()
        seg_before = seg.copy()
        apply_effects(
            image,
            seg,
            (1.0, 1.0, 1.0),
            effects=[_stub_intensity],
            effect_params=[{"delta": 1.0}],
        )
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(seg, seg_before)


class _StubAppliesEffects(AppliesEffects):
    """
    Minimal concrete pipeline for cache / routing tests.
    """

    DEFORMATION_EFFECTS = (_stub_deformation,)
    INTENSITY_EFFECTS = (_stub_intensity,)

    def __init__(self) -> None:
        super().__init__(
            deformation_params=[{"enable": True, "shift": 0.0}],
            intensity_params=[{"enable": True, "delta": 1.5}],
        )


class TestAppliesEffectsMixin:
    """
    Tests for ``AppliesEffects`` caching and param routing.
    """

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        image = np.zeros((8, 8, 8), dtype=np.float64)
        seg = np.ones((8, 8, 8), dtype=np.int32)
        return image, seg

    def test_subclass_requires_effect_tuples(self) -> None:
        with pytest.raises(TypeError):

            class _Bad(AppliesEffects):  # type: ignore[misc]
                pass

            _Bad(  # pragma: no cover - construction should fail in __init_subclass__
                deformation_params=[],
                intensity_params=[],
            )

    def test_intensity_cache_roundtrip(self, volume: tuple) -> None:
        image, seg = volume
        calls = {"n": 0}

        def counting_intensity(
            image: np.ndarray,
            seg_mask: np.ndarray,
            spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
            *,
            delta: float = 1.0,
            **kwargs: Any,
        ) -> _IntensityResultType:
            calls["n"] += 1
            return _stub_intensity(image, seg_mask, spacing, delta=delta, **kwargs)

        class _CountingIntensity(AppliesEffects):
            DEFORMATION_EFFECTS = (_stub_deformation,)
            INTENSITY_EFFECTS = (counting_intensity,)

            def __init__(self) -> None:
                super().__init__(
                    deformation_params=[{"enable": True, "shift": 0.0}],
                    intensity_params=[{"enable": True, "delta": 1.5}],
                )

        pipe = _CountingIntensity()
        first = pipe.apply_intensity(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k"
        )
        assert calls["n"] == 1
        assert "k" in pipe._cached_intensity_fields

        second = pipe.apply_intensity(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k"
        )
        assert calls["n"] == 1  # cache hit: effect not re-run
        np.testing.assert_allclose(first["effect_field"], second["effect_field"])
        np.testing.assert_allclose(second["out_image"], image + 1.5)

    def test_use_cache_true_missing_key_raises(self, volume: tuple) -> None:
        image, seg = volume
        pipe = _StubAppliesEffects()
        with pytest.raises(ValueError, match="No cached"):
            pipe.apply_intensity(
                image, seg, (1.0, 1.0, 1.0), use_cache=True, cache_key="missing"
            )

    def test_deformation_cache_noop_fields(self, volume: tuple) -> None:
        image, seg = volume
        calls = {"n": 0}

        def counting_deformation(
            image: np.ndarray,
            seg_mask: np.ndarray,
            spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
            *,
            shift: float = 0.0,
            **kwargs: Any,
        ) -> _DeformationResultType:
            calls["n"] += 1
            return _stub_deformation(image, seg_mask, spacing, shift=shift, **kwargs)

        class _CountingDeform(AppliesEffects):
            DEFORMATION_EFFECTS = (counting_deformation,)
            INTENSITY_EFFECTS = (_stub_intensity,)

            def __init__(self) -> None:
                super().__init__(
                    deformation_params=[{"enable": True, "shift": 0.0}],
                    intensity_params=[{"enable": True, "delta": 1.0}],
                )

        pipe = _CountingDeform()
        first = pipe.apply_deformations(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="d"
        )
        assert calls["n"] == 1
        assert first["effect_field"] == []

        second = pipe.apply_deformations(
            image + 1.0, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="d"
        )
        assert calls["n"] == 1  # cache hit: effect not re-run
        # Cached empty field is a no-op: returns the *new* image unchanged by warps
        np.testing.assert_array_equal(second["out_image"], image + 1.0)

    def test_param_setter_clears_cache(self, volume: tuple) -> None:
        image, seg = volume
        pipe = _StubAppliesEffects()
        pipe.apply_intensity(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k"
        )
        assert pipe._cached_intensity_fields
        pipe.intensity_params = [{"enable": True, "delta": 2.0}]
        assert pipe._cached_intensity_fields == {}

    def test_clear_cached_fields(self, volume: tuple) -> None:
        image, seg = volume
        pipe = _StubAppliesEffects()
        pipe.apply_intensity(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k"
        )
        pipe.apply_deformations(
            image, seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="d"
        )
        pipe.clear_cached_fields()
        assert pipe._cached_intensity_fields == {}
        assert pipe._cached_deformation_fields == {}

    def test_group_routing(self, volume: tuple) -> None:
        class _TwoIntensity(AppliesEffects):
            DEFORMATION_EFFECTS = (_stub_deformation,)
            INTENSITY_EFFECTS = (_stub_intensity, _stub_intensity)

            def __init__(self) -> None:
                super().__init__(
                    deformation_params=[{"enable": True, "shift": 0.0}],
                    intensity_params=[
                        {"enable": True, "delta": 1.0},
                        {"enable": True, "delta": 10.0},
                    ],
                )

        image, seg = volume
        pipe = _TwoIntensity()
        out = pipe.apply_intensity(
            image, seg, (1.0, 1.0, 1.0), group=(1,), use_cache=False
        )
        np.testing.assert_allclose(out["effect_field"], 10.0)
