"""
Tests for ``synthfcd.core.texture``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from scipy import ndimage

from synthfcd.core.masks import get_binary_mask
from synthfcd.core.texture import robust_std, texture_restoration
from tests.base_tests import (
    RequiresImages,
    RequiresSpacing,
    TestImageValidation,
    TestSpacingValidation,
)
from tests.utils import make_nested_seg_mask


def _make_textured_pair(
    shape: tuple[int, int, int],
    *,
    seed: int = 0,
    mean: float = 100.0,
    texture_std: float = 8.0,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ``(image, orig)`` where ``image`` is a smoothed copy of ``orig``.
    """
    rng = np.random.default_rng(seed)
    orig = rng.normal(mean, texture_std, shape).astype(np.float64)
    image = ndimage.gaussian_filter(orig, sigma=smooth_sigma)
    return image, orig


class TestRobustStd:
    """
    Tests for the ``robust_std`` helper.
    """

    def test_constant_image_is_zero(self) -> None:
        assert robust_std(np.ones((8, 8, 8), dtype=np.float64)) == 0.0

    def test_matches_mad_formula(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.normal(0.0, 2.0, (32, 32, 32))
        expected = 1.4826 * float(np.median(np.abs(image - np.median(image))))
        assert robust_std(image) == pytest.approx(expected)


class TestTextureRestoration(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``texture_restoration``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=9)
        image, orig = _make_textured_pair(seg.shape, seed=1)
        return partial(
            texture_restoration,
            image=image,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            random_seed=0,
        )  # type: ignore[return-value]

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((28, 28, 28), wm_radius=6, gm_radius=10)
        image, orig = _make_textured_pair(seg.shape, seed=2)
        return image, orig, seg

    def test_invalid_seg_mask_type(self, volume: tuple) -> None:
        image, orig, _ = volume
        with pytest.raises(TypeError):
            texture_restoration(
                image,
                seg_mask="bad",  # type: ignore[arg-type]
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
            )

    def test_seg_shape_mismatch(self, volume: tuple) -> None:
        image, orig, _ = volume
        with pytest.raises(ValueError):
            texture_restoration(
                image,
                seg_mask=np.zeros((4, 4, 4), dtype=int),
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
            )

    def test_invalid_orig_type(self, volume: tuple) -> None:
        image, _, seg = volume
        with pytest.raises(TypeError):
            texture_restoration(
                image, seg, spacing=(1.0, 1.0, 1.0), orig="bad"  # type: ignore[arg-type]
            )

    def test_orig_shape_mismatch(self, volume: tuple) -> None:
        image, _, seg = volume
        with pytest.raises(ValueError):
            texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=np.zeros((4, 4, 4), dtype=float),
            )

    @pytest.mark.parametrize(
        "name",
        ["hpf_img_sigma", "corr_noise_sigma", "hpf_noise_sigma"],
    )
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_invalid_positive_sigmas(
        self, volume: tuple, name: str, value: float
    ) -> None:
        image, orig, seg = volume
        with pytest.raises(ValueError):
            texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                **{name: value},  # type: ignore[arg-type]
            )

    def test_invalid_noise_clip(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.raises(ValueError):
            texture_restoration(
                image, seg, spacing=(1.0, 1.0, 1.0), orig=orig, noise_clip=0.0
            )

    def test_invalid_gm_wm_only_type(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.raises(TypeError):
            texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                gm_wm_only="yes",  # type: ignore[arg-type]
            )

    def test_invalid_hemisphere(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.raises(ValueError):
            texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                hemisphere="middle",  # type: ignore[arg-type]
            )

    def test_invalid_label_enum(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.raises(TypeError):
            texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                label_enum=object,  # type: ignore[arg-type]
            )

    def test_empty_app_field_warns_and_returns_unchanged(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.warns(UserWarning, match="Application field"):
            out = texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                app_field=np.zeros_like(image),
                random_seed=0,
            )
        np.testing.assert_array_equal(out["out_image"], image)
        np.testing.assert_array_equal(out["effect_field"], 0)

    def test_no_gm_wm_warns(self, volume: tuple) -> None:
        image, orig, _ = volume
        seg = np.zeros(image.shape, dtype=np.int32)
        with pytest.warns(UserWarning, match="No GM or WM"):
            out = texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                gm_wm_only=True,
                random_seed=0,
            )
        np.testing.assert_array_equal(out["out_image"], image)

    def test_no_deep_tissue_for_std_warns(self, volume: tuple) -> None:
        image, orig, seg = volume
        with pytest.warns(UserWarning, match="No valid voxels"):
            out = texture_restoration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=orig,
                hpf_img_sigma=100.0,
                gm_wm_only=True,
                random_seed=0,
            )
        np.testing.assert_array_equal(out["out_image"], image)

    def test_no_residual_texture_warns(self, volume: tuple) -> None:
        _, _, seg = volume
        # Identical orig/image => no residual high-frequency energy to restore
        flat = np.full(seg.shape, 50.0, dtype=np.float64)
        with pytest.warns(UserWarning, match="No residual texture"):
            out = texture_restoration(
                flat,
                seg,
                spacing=(1.0, 1.0, 1.0),
                orig=flat.copy(),
                random_seed=0,
            )
        np.testing.assert_array_equal(out["out_image"], flat)

    def test_return_contract_and_smoke(self, volume: tuple) -> None:
        image, orig, seg = volume
        out = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            random_seed=3,
        )
        assert set(out) == {"out_image", "effect_field"}
        assert out["out_image"].shape == image.shape
        np.testing.assert_allclose(
            out["out_image"], image + out["effect_field"], atol=1e-12
        )
        tissue = get_binary_mask(seg, mask_type="wm") | get_binary_mask(
            seg, mask_type="gm"
        )
        assert np.any(np.abs(out["effect_field"][tissue]) > 0)
        # Background should stay near zero when gm_wm_only blends away outside tissue
        assert abs(float(out["effect_field"][0, 0, 0])) < 1e-8

    def test_seed_determinism(self, volume: tuple) -> None:
        image, orig, seg = volume
        kwargs = dict(
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            gm_wm_only=True,
        )
        a = texture_restoration(image.copy(), seg.copy(), random_seed=11, **kwargs)  # type: ignore[arg-type]
        b = texture_restoration(image.copy(), seg.copy(), random_seed=11, **kwargs)  # type: ignore[arg-type]
        c = texture_restoration(image.copy(), seg.copy(), random_seed=12, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(a["effect_field"], b["effect_field"])
        assert not np.allclose(a["effect_field"], c["effect_field"])

    def test_soft_app_field_scales_effect(self, volume: tuple) -> None:
        image, orig, seg = volume
        app = np.zeros_like(image)
        app[8:20, 8:20, 8:20] = 0.5
        app[14, 14, 14] = 1.0
        out = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            app_field=app,
            random_seed=1,
        )
        effect = out["effect_field"]
        mid = abs(float(effect[12, 12, 12]))
        peak = abs(float(effect[14, 14, 14]))
        assert mid > 0.0
        assert peak > mid
        assert effect[0, 0, 0] == 0.0

    def test_app_field_scales_effect_not_calibration(self, volume: tuple) -> None:
        image, orig, seg = volume
        app_low = np.zeros_like(image)
        app_low[8:20, 8:20, 8:20] = 0.1
        app_high = np.zeros_like(image)
        app_high[8:20, 8:20, 8:20] = 0.5

        low = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            app_field=app_low,
            random_seed=4,
        )
        high = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            app_field=app_high,
            random_seed=4,
        )
        low_peak = float(np.max(np.abs(low["effect_field"])))
        high_peak = float(np.max(np.abs(high["effect_field"])))
        assert low_peak > 0.0
        assert high_peak > low_peak
        assert low_peak / high_peak == pytest.approx(0.2, rel=1e-5)

    def test_gm_wm_only_false_restores_globally(self, volume: tuple) -> None:
        image, orig, seg = volume
        out = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            gm_wm_only=False,
            random_seed=3,
        )
        assert np.any(out["effect_field"] != 0.0)
        # Without tissue restriction, background can receive texture
        assert abs(float(out["effect_field"][0, 0, 0])) > 0.0

    def test_noise_clip_bounds_effect(self, volume: tuple) -> None:
        image, orig, seg = volume
        clipped = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            noise_clip=0.5,
            gm_wm_only=False,
            random_seed=7,
        )
        wide = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            orig=orig,
            noise_clip=None,
            gm_wm_only=False,
            random_seed=7,
        )
        assert (
            float(np.max(np.abs(clipped["effect_field"])))
            <= float(np.max(np.abs(wide["effect_field"]))) + 1e-12
        )

    def test_anisotropic_spacing_runs(self, volume: tuple) -> None:
        image, orig, seg = volume
        out = texture_restoration(
            image,
            seg,
            spacing=(1.0, 1.5, 2.0),
            orig=orig,
            random_seed=1,
        )
        assert out["out_image"].shape == image.shape
        assert out["effect_field"].any()

    def test_does_not_mutate_inputs(self, volume: tuple) -> None:
        image, orig, seg = volume
        image_before = image.copy()
        orig_before = orig.copy()
        seg_before = seg.copy()
        texture_restoration(
            image, seg, spacing=(1.0, 1.0, 1.0), orig=orig, random_seed=0
        )
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(orig, orig_before)
        np.testing.assert_array_equal(seg, seg_before)
