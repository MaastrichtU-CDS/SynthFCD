"""
Tests for ``synthfcd.pipelines.utils``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest

from synthfcd.pipelines.utils import (
    readable_cache_key,
    resolve_cache_settings,
    texture_app_field_from_blur_effect,
)
from tests.base_tests import RequiresSpacing, TestSpacingValidation


class TestTextureAppFieldFromBlurEffect(TestSpacingValidation):
    """
    Tests for ``texture_app_field_from_blur_effect``.
    """

    def op_under_test(self) -> RequiresSpacing:
        blur = np.zeros((12, 12, 12), dtype=np.float64)
        blur[4:8, 4:8, 4:8] = 1.0
        return partial(texture_app_field_from_blur_effect, blur_effect=blur)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_invalid_threshold(self, threshold: float) -> None:
        blur = np.ones((8, 8, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="threshold"):
            texture_app_field_from_blur_effect(
                blur, spacing=(1.0, 1.0, 1.0), threshold=threshold
            )

    def test_zero_blur_returns_zeros(self) -> None:
        blur = np.zeros((10, 10, 10), dtype=np.float64)
        out = texture_app_field_from_blur_effect(blur, spacing=(1.0, 1.0, 1.0))
        np.testing.assert_array_equal(out, 0)
        assert out.dtype == np.float32

    def test_support_inside_blur_region(self) -> None:
        blur = np.zeros((16, 16, 16), dtype=np.float64)
        blur[5:11, 5:11, 5:11] = 2.0
        out = texture_app_field_from_blur_effect(
            blur, spacing=(1.0, 1.0, 1.0), threshold=0.01, rolloff_mm=1.0
        )
        assert out.shape == blur.shape
        assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
        assert out[7, 7, 7] > out[0, 0, 0]
        assert out[0, 0, 0] == 0.0

    def test_anisotropic_spacing_changes_field(self) -> None:
        blur = np.zeros((14, 14, 14), dtype=np.float64)
        blur[4:10, 4:10, 4:10] = 1.0
        # Larger rolloff keeps the field in the soft ramp where spacing matters.
        iso = texture_app_field_from_blur_effect(
            blur, spacing=(1.0, 1.0, 1.0), rolloff_mm=5.0
        )
        aniso = texture_app_field_from_blur_effect(
            blur, spacing=(1.0, 2.0, 3.0), rolloff_mm=5.0
        )
        assert not np.allclose(iso, aniso)


class TestResolveCacheSettings:
    """
    Tests for ``resolve_cache_settings``.
    """

    def test_use_cache_false(self) -> None:
        key, read, write = resolve_cache_settings(
            use_cache=False, cache_key="auto", cache={}
        )
        assert key is None
        assert read is False
        assert write is False

    def test_use_cache_true_requires_str_key_or_auto(self) -> None:
        key, read, write = resolve_cache_settings(
            use_cache=True, cache_key="mykey", cache={}
        )
        assert key == "mykey"
        assert read is True
        assert write is True

        key, read, write = resolve_cache_settings(
            use_cache=True, cache_key="auto", cache={}, spacing=(1.0, 1.0, 1.0)
        )
        assert isinstance(key, str) and key
        assert read is True
        assert write is True

    def test_use_cache_auto_read_depends_on_presence(self) -> None:
        cache: dict[str, Any] = {}
        key, read, write = resolve_cache_settings(
            use_cache="auto",
            cache_key="k1",
            cache=cache,
        )
        assert key == "k1"
        assert read is False
        assert write is True

        cache["k1"] = object()
        key, read, write = resolve_cache_settings(
            use_cache="auto",
            cache_key="k1",
            cache=cache,
        )
        assert read is True
        assert write is True

    def test_auto_key_is_stable_for_same_kwargs(self) -> None:
        a, _, _ = resolve_cache_settings(
            use_cache="auto",
            cache_key="auto",
            cache={},
            spacing=(1.0, 1.0, 1.0),
            group=(0, 1),
        )
        b, _, _ = resolve_cache_settings(
            use_cache="auto",
            cache_key="auto",
            cache={},
            spacing=(1.0, 1.0, 1.0),
            group=(0, 1),
        )
        c, _, _ = resolve_cache_settings(
            use_cache="auto",
            cache_key="auto",
            cache={},
            spacing=(1.0, 2.0, 3.0),
            group=(0, 1),
        )
        assert a == b
        assert a != c

    def test_invalid_use_cache_type(self) -> None:
        with pytest.raises(TypeError):
            resolve_cache_settings(
                use_cache="maybe",  # type: ignore[arg-type]
                cache_key="k",
                cache={},
            )

    def test_invalid_cache_key_type_when_not_auto(self) -> None:
        with pytest.raises(TypeError):
            resolve_cache_settings(
                use_cache=True,
                cache_key=None,  # type: ignore[arg-type]
                cache={},
            )


class TestReadableCacheKey:
    """
    Tests for ``readable_cache_key``.
    """

    def test_auto_message(self) -> None:
        msg = readable_cache_key("auto", ["foo", "bar"])
        assert "foo" in msg and "bar" in msg
        assert "spacing" in msg

    def test_explicit_key_message(self) -> None:
        msg = readable_cache_key("abc123", ["foo"])
        assert "abc123" in msg
