"""
Tests for ``synthfcd.pipelines.grow_target`` (executor + ``ToTarget`` mixin).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from synthfcd.pipelines.grow_target import ToTarget, grow_target
from synthfcd.utils._aliases import (
    _DistanceType,
    _GrowTargetFnType,
    _GrowTargetResultType,
)


def _stub_growth(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    *,
    center: tuple[int, int, int] = (5, 5, 5),
    radius: int = 1,
    tag: str = "a",
    **kwargs: Any,
) -> _GrowTargetResultType:
    target = np.zeros(seg_mask.shape, dtype=bool)
    cz, cy, cx = center
    target[
        cz - radius : cz + radius + 1,
        cy - radius : cy + radius + 1,
        cx - radius : cx + radius + 1,
    ] = True
    return {"target": target, "growth_stats": {"tag": tag, "voxels": int(target.sum())}}


def _stub_growth_empty(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    **kwargs: Any,
) -> _GrowTargetResultType:
    return {
        "target": np.zeros(seg_mask.shape, dtype=bool),
        "growth_stats": {"empty": True},
    }


def _stub_bad_return(
    seg_mask: np.ndarray,
    spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
    **kwargs: Any,
) -> dict[str, Any]:
    return {"target": np.zeros(seg_mask.shape, dtype=bool)}


class TestGrowTarget:
    """
    Tests for the functional ``grow_target`` helper.
    """

    @pytest.fixture
    def seg(self) -> np.ndarray:
        return np.ones((12, 12, 12), dtype=np.int32)

    def test_empty_growth_fns_raises(self, seg: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            grow_target(seg, (1.0, 1.0, 1.0), growth_fns=[], growth_params=[])

    def test_length_mismatch_raises(self, seg: np.ndarray) -> None:
        with pytest.raises(ValueError, match="same length"):
            grow_target(
                seg, (1.0, 1.0, 1.0), growth_fns=[_stub_growth], growth_params=[]
            )

    def test_non_callable_raises(self, seg: np.ndarray) -> None:
        with pytest.raises(ValueError, match="callable"):
            grow_target(
                seg,
                (1.0, 1.0, 1.0),
                growth_fns=["not-callable"],  # type: ignore[list-item]
                growth_params=[{}],
            )

    def test_disabled_without_allow_raises(self, seg: np.ndarray) -> None:
        with pytest.raises(ValueError, match="disabled"):
            grow_target(
                seg,
                (1.0, 1.0, 1.0),
                growth_fns=[_stub_growth],
                growth_params=[{"enable": False}],
                allow_disabled=False,
            )

    def test_disabled_with_allow_skips(self, seg: np.ndarray) -> None:
        out = grow_target(
            seg,
            (1.0, 1.0, 1.0),
            growth_fns=[_stub_growth, _stub_growth],
            growth_params=[
                {"enable": False},
                {"center": (8, 8, 8), "radius": 1, "tag": "kept"},
            ],
            allow_disabled=True,
        )
        assert set(out["growth_stats"]) == {"_stub_growth"}
        assert out["growth_stats"]["_stub_growth"]["tag"] == "kept"
        assert out["target"][8, 8, 8]
        assert not out["target"][5, 5, 5]

    def test_sequential_union(self, seg: np.ndarray) -> None:
        out = grow_target(
            seg,
            (1.0, 1.0, 1.0),
            growth_fns=[_stub_growth, _stub_growth],
            growth_params=[
                {"center": (3, 3, 3), "radius": 1, "tag": "a"},
                {"center": (9, 9, 9), "radius": 1, "tag": "b"},
            ],
        )
        assert out["target"][3, 3, 3] and out["target"][9, 9, 9]
        # Stats keyed once by callable name — second overwrites first in dict
        assert out["growth_stats"]["_stub_growth"]["tag"] == "b"

    def test_invalid_return_keys(self, seg: np.ndarray) -> None:
        with pytest.raises(ValueError):
            grow_target(
                seg,
                (1.0, 1.0, 1.0),
                growth_fns=[cast(_GrowTargetFnType, _stub_bad_return)],
                growth_params=[{}],
            )

    def test_does_not_mutate_seg(self, seg: np.ndarray) -> None:
        before = seg.copy()
        grow_target(
            seg,
            (1.0, 1.0, 1.0),
            growth_fns=[_stub_growth],
            growth_params=[{"center": (5, 5, 5)}],
        )
        np.testing.assert_array_equal(seg, before)


class _StubToTarget(ToTarget):
    """
    Minimal concrete pipeline for cache / routing tests.
    """

    TARGET_GROWTH = (_stub_growth,)

    def __init__(self) -> None:
        super().__init__(growth_params=[{"center": (5, 5, 5), "radius": 1, "tag": "x"}])


class TestToTargetMixin:
    """
    Tests for ``ToTarget`` caching, routing, and application fields.
    """

    @pytest.fixture
    def seg(self) -> np.ndarray:
        return np.ones((12, 12, 12), dtype=np.int32)

    def test_subclass_requires_target_growth(self) -> None:
        with pytest.raises(TypeError):

            class _Bad(ToTarget):  # type: ignore[misc]
                pass

            _Bad(growth_params=[])  # pragma: no cover

    def test_target_cache_roundtrip(self, seg: np.ndarray) -> None:
        calls = {"n": 0}

        def counting_growth(
            seg_mask: np.ndarray,
            spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
            **kwargs: Any,
        ) -> _GrowTargetResultType:
            calls["n"] += 1
            return _stub_growth(seg_mask, spacing, **kwargs)

        class _Counting(ToTarget):
            TARGET_GROWTH = (counting_growth,)

            def __init__(self) -> None:
                super().__init__(
                    growth_params=[{"center": (4, 4, 4), "radius": 1, "tag": "c"}]
                )

        pipe = _Counting()
        first = pipe.grow_target(seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k")
        assert calls["n"] == 1
        assert "k" in pipe._cached_target

        second = pipe.grow_target(seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k")
        assert calls["n"] == 1
        np.testing.assert_array_equal(first["target"], second["target"])
        assert first["growth_stats"] == second["growth_stats"]

    def test_use_cache_true_missing_key_raises(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        with pytest.raises(ValueError, match="No cached"):
            pipe.grow_target(seg, (1.0, 1.0, 1.0), use_cache=True, cache_key="missing")

    def test_param_setter_clears_cache(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        pipe.grow_target(seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k")
        assert pipe._cached_target
        pipe.growth_params = [{"center": (6, 6, 6), "radius": 1, "tag": "y"}]
        assert pipe._cached_target == {}
        assert pipe._cached_growth_stats == {}
        assert pipe._cached_app_fields == {}

    def test_clear_cached_target(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        pipe.grow_target(seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k")
        target = pipe._cached_target["k"]
        pipe.compute_application_field(
            target,
            field_type="outward",
            rolloff=1.0,
            bbox_thres=0.05,
            spacing=(1.0, 1.0, 1.0),
            use_cache="auto",
            cache_key="app",
        )
        assert pipe._cached_app_fields
        pipe.clear_cached_target()
        assert pipe._cached_target == {}
        assert pipe._cached_growth_stats == {}
        assert pipe._cached_app_fields == {}

    def test_group_routing(self, seg: np.ndarray) -> None:
        class _TwoGrowth(ToTarget):
            TARGET_GROWTH = (_stub_growth, _stub_growth)

            def __init__(self) -> None:
                super().__init__(
                    growth_params=[
                        {"center": (3, 3, 3), "radius": 1, "tag": "first"},
                        {"center": (9, 9, 9), "radius": 1, "tag": "second"},
                    ]
                )

        pipe = _TwoGrowth()
        out = pipe.grow_target(seg, (1.0, 1.0, 1.0), group=(1,), use_cache=False)
        assert out["target"][9, 9, 9]
        assert not out["target"][3, 3, 3]
        assert out["growth_stats"]["_stub_growth"]["tag"] == "second"

    def test_all_disabled_returns_empty_without_cache(self, seg: np.ndarray) -> None:
        class _Disabled(ToTarget):
            TARGET_GROWTH = (_stub_growth,)

            def __init__(self) -> None:
                super().__init__(growth_params=[{"enable": False}])

        pipe = _Disabled()
        out = pipe.grow_target(
            seg, (1.0, 1.0, 1.0), allow_disabled=True, use_cache="auto", cache_key="k"
        )
        assert not out["target"].any()
        assert out["growth_stats"] == {}
        assert pipe._cached_target == {}

    def test_empty_grown_target_warns_and_caches(self, seg: np.ndarray) -> None:
        class _Empty(ToTarget):
            TARGET_GROWTH = (_stub_growth_empty,)

            def __init__(self) -> None:
                super().__init__(growth_params=[{}])

        pipe = _Empty()
        with pytest.warns(UserWarning, match="empty"):
            out = pipe.grow_target(
                seg, (1.0, 1.0, 1.0), use_cache="auto", cache_key="k"
            )
        assert not out["target"].any()
        assert "k" in pipe._cached_target

    def test_application_field_cache_roundtrip(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        target = np.zeros(seg.shape, dtype=bool)
        target[5:8, 5:8, 5:8] = True

        bbox1, field1 = pipe.compute_application_field(
            target,
            field_type="outward",
            rolloff=2.0,
            bbox_thres=0.05,
            spacing=(1.0, 1.0, 1.0),
            use_cache="auto",
            cache_key="app",
        )
        assert "app" in pipe._cached_app_fields
        assert field1.shape == tuple(s.stop - s.start for s in bbox1)
        assert 0.0 <= float(field1.min()) <= float(field1.max()) <= 1.0

        # Corrupt underlying cache entry shape check via second read
        bbox2, field2 = pipe.compute_application_field(
            target,
            field_type="outward",
            rolloff=2.0,
            bbox_thres=0.05,
            spacing=(1.0, 1.0, 1.0),
            use_cache="auto",
            cache_key="app",
        )
        assert bbox1 == bbox2
        np.testing.assert_allclose(field1, field2)

    def test_application_field_empty_target_raises(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        with pytest.raises(ValueError, match="empty"):
            pipe.compute_application_field(
                np.zeros(seg.shape, dtype=bool),
                field_type="outward",
                rolloff=1.0,
                bbox_thres=0.05,
                spacing=(1.0, 1.0, 1.0),
            )

    def test_application_field_float_target_as_field(self, seg: np.ndarray) -> None:
        pipe = _StubToTarget()
        soft = np.zeros(seg.shape, dtype=np.float32)
        soft[4:9, 4:9, 4:9] = 0.8
        bbox, field = pipe.compute_application_field(
            soft,
            field_type="inward",
            rolloff=1.0,
            bbox_thres=0.5,
            spacing=(1.0, 1.0, 1.0),
            use_cache=False,
        )
        assert field.shape == tuple(s.stop - s.start for s in bbox)
        np.testing.assert_allclose(field, soft[bbox])
