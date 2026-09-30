"""
Tests for ``LesionSimulationPipeline.apply_effects_locally`` and helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synthfcd.pipelines.base import LesionSimulationPipeline
from synthfcd.utils._aliases import (
    _DeformationResultType,
    _DistanceType,
    _GrowTargetResultType,
    _IntensityResultType,
)


def _stub_growth(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    **kwargs: Any,
) -> _GrowTargetResultType:
    target = np.zeros(seg_mask.shape, dtype=bool)
    mid = tuple(s // 2 for s in seg_mask.shape)
    target[
        mid[0] - 2 : mid[0] + 2,
        mid[1] - 2 : mid[1] + 2,
        mid[2] - 2 : mid[2] + 2,
    ] = True
    return {"target": target, "growth_stats": {"hemisphere": "left"}}


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
    return {
        "out_image": image.copy(),
        "out_seg_mask": seg_mask.copy(),
        "effect_field": field,
    }


def _local_app_params(**extra: Any) -> dict[str, Any]:
    """
    Tight app-field settings so the bbox stays well inside a 24^3 volume.
    """
    params: dict[str, Any] = {
        "enable": True,
        "app_field_type": "outward",
        "app_rolloff": 1.0,
        "bbox_thres": 0.25,
    }
    params.update(extra)
    return params


class _StubLesionPipeline(LesionSimulationPipeline):
    """
    Minimal lesion pipeline for local-application tests.
    """

    TARGET_GROWTH = (_stub_growth,)
    DEFORMATION_EFFECTS = (_stub_deformation,)
    INTENSITY_EFFECTS = (_stub_intensity, _stub_intensity)

    def __init__(
        self,
        *,
        deformation_params: list[dict[str, Any]] | None = None,
        intensity_params: list[dict[str, Any]] | None = None,
    ) -> None:
        if deformation_params is None:
            deformation_params = [_local_app_params(shift=0.0)]
        if intensity_params is None:
            intensity_params = [
                _local_app_params(delta=1.5),
                _local_app_params(delta=10.0),
            ]
        super().__init__(
            growth_params=[{"enable": True}],
            deformation_params=deformation_params,
            intensity_params=intensity_params,
        )

    def __call__(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError("Not used in local-application unit tests")


class TestApplyEffectsLocally:
    """
    Local crop / stitch / cache behavior of ``apply_effects_locally``.
    """

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.zeros((24, 24, 24), dtype=np.float64)
        seg = np.ones((24, 24, 24), dtype=np.int32)
        target = np.zeros((24, 24, 24), dtype=bool)
        target[10:14, 10:14, 10:14] = True
        return image, seg, target

    def test_intensity_changes_only_near_target(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline()
        out = pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            group=(0,),
            use_cache=False,
        )
        assert out["out_image"][0, 0, 0] == image[0, 0, 0]
        assert out["out_image"][12, 12, 12] == pytest.approx(image[12, 12, 12] + 1.5)
        assert set(out) >= {"out_image", "out_seg_mask", "out_target", "effect_field"}

    def test_does_not_mutate_inputs(self, volume: tuple) -> None:
        image, seg, target = volume
        image_b, seg_b, target_b = image.copy(), seg.copy(), target.copy()
        pipe = _StubLesionPipeline()
        pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            group=(0,),
            use_cache=False,
        )
        np.testing.assert_array_equal(image, image_b)
        np.testing.assert_array_equal(seg, seg_b)
        np.testing.assert_array_equal(target, target_b)

    def test_all_disabled_early_exit(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline(
            intensity_params=[
                _local_app_params(enable=False),
                _local_app_params(enable=False),
            ]
        )
        out = pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            allow_disabled=True,
            use_cache=False,
        )
        np.testing.assert_array_equal(out["out_image"], image)
        np.testing.assert_array_equal(out["out_seg_mask"], seg)
        np.testing.assert_array_equal(out["out_target"], target)
        np.testing.assert_array_equal(out["effect_field"], 0.0)

    def test_disabled_without_allow_raises(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline(
            intensity_params=[
                _local_app_params(enable=False),
                _local_app_params(delta=1.0),
            ]
        )
        with pytest.raises(ValueError, match="disabled"):
            pipe.apply_effects_locally(
                image,
                seg,
                target,
                (1.0, 1.0, 1.0),
                effects_type="intensity",
                allow_disabled=False,
                use_cache=False,
            )

    def test_group_selects_intensity_effect(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline()
        out = pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            group=(1,),
            use_cache=False,
        )
        assert out["out_image"][12, 12, 12] == pytest.approx(10.0)

    def test_effect_sees_cropped_volume(self, volume: tuple) -> None:
        """
        Effects must run on the bbox crop, not the full volume.
        """
        image, seg, target = volume
        seen: dict[str, tuple[int, ...]] = {}

        def shape_spy_intensity(
            image: np.ndarray,
            seg_mask: np.ndarray,
            spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
            *,
            delta: float = 1.0,
            **kwargs: Any,
        ) -> _IntensityResultType:
            seen["shape"] = image.shape
            return _stub_intensity(image, seg_mask, spacing, delta=delta, **kwargs)

        class _SpyPipe(LesionSimulationPipeline):
            TARGET_GROWTH = (_stub_growth,)
            DEFORMATION_EFFECTS = (_stub_deformation,)
            INTENSITY_EFFECTS = (shape_spy_intensity,)

            def __init__(self) -> None:
                super().__init__(
                    growth_params=[{"enable": True}],
                    deformation_params=[_local_app_params(shift=0.0)],
                    intensity_params=[_local_app_params(delta=1.0)],
                )

            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise NotImplementedError

        pipe = _SpyPipe()
        pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            use_cache=False,
        )
        assert "shape" in seen
        assert seen["shape"] != image.shape
        assert all(s < image.shape[i] for i, s in enumerate(seen["shape"]))

    def test_app_field_cache_shared_across_calls(self, volume: tuple) -> None:
        image, seg, target = volume
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

        class _Counting(LesionSimulationPipeline):
            TARGET_GROWTH = (_stub_growth,)
            DEFORMATION_EFFECTS = (_stub_deformation,)
            INTENSITY_EFFECTS = (counting_intensity,)

            def __init__(self) -> None:
                super().__init__(
                    growth_params=[{"enable": True}],
                    deformation_params=[_local_app_params(shift=0.0)],
                    intensity_params=[_local_app_params(delta=2.0)],
                )

            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise NotImplementedError

        pipe = _Counting()
        pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            use_cache="auto",
            cache_key="local",
        )
        assert calls["n"] == 1
        assert pipe._cached_app_fields
        assert pipe._cached_intensity_fields

        pipe.apply_effects_locally(
            image + 1.0,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="intensity",
            use_cache="auto",
            cache_key="local",
        )
        assert calls["n"] == 1  # intensity field + app field reused

    def test_inconsistent_app_field_params_raise(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline(
            intensity_params=[
                _local_app_params(delta=1.0, app_field_type="outward"),
                _local_app_params(delta=1.0, app_field_type="inward"),
            ]
        )
        with pytest.raises(ValueError, match="Inconsistent"):
            pipe.apply_effects_locally(
                image,
                seg,
                target,
                (1.0, 1.0, 1.0),
                effects_type="intensity",
                use_cache=False,
            )

    def test_crop_kwargs_mismatch_raises(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline()
        bad_orig = np.zeros((4, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="Cannot crop"):
            pipe.apply_effects_locally(
                image,
                seg,
                target,
                (1.0, 1.0, 1.0),
                effects_type="intensity",
                group=(0,),
                use_cache=False,
                crop_kwargs=("orig",),
                orig=bad_orig,
            )

    def test_deformation_noop_preserves_target(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline()
        out = pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="deformations",
            use_cache=False,
        )
        np.testing.assert_array_equal(out["out_target"], target)
        assert out["effect_field"] == []

    def test_deformation_field_warps_target(self, volume: tuple) -> None:
        image, seg, target = volume
        pipe = _StubLesionPipeline(deformation_params=[_local_app_params(shift=1.0)])
        out = pipe.apply_effects_locally(
            image,
            seg,
            target,
            (1.0, 1.0, 1.0),
            effects_type="deformations",
            use_cache=False,
        )
        assert len(out["effect_field"]) == 1
        assert out["out_target"].any()
        assert (
            np.issubdtype(out["out_target"].dtype, np.bool_)
            or out["out_target"].dtype == target.dtype
        )
        assert not np.array_equal(out["out_target"], target)
