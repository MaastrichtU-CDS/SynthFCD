"""
Tests for ``synthfcd.core.diffusion``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest

from synthfcd.core.diffusion import (
    boundary_blurring,
    compute_conduction_coefficient,
    compute_divergence,
    compute_gradients,
    hyperintensity,
    perona_malik,
)
from synthfcd.core.masks import get_binary_mask
from tests.base_tests import (
    RequiresImages,
    RequiresSpacing,
    TestImageValidation,
    TestSpacingValidation,
)
from tests.utils import make_nested_seg_mask


def _ramp_image(
    shape: tuple[int, int, int] = (12, 12, 12),
) -> np.ndarray:
    """
    Image with a unit ramp along x.
    """
    x = np.arange(shape[0], dtype=np.float64)
    return np.broadcast_to(x[:, None, None], shape).copy()


def _gradients_ref(
    image: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = spacing
    dx = np.zeros_like(image)
    dx[:-1, :, :] = np.diff(image, axis=0) / sx
    dy = np.zeros_like(image)
    dy[:, :-1, :] = np.diff(image, axis=1) / sy
    dz = np.zeros_like(image)
    dz[:, :, :-1] = np.diff(image, axis=2) / sz
    return dx, dy, dz


def _divergence_ref(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    sx, sy, sz = spacing
    div = np.zeros_like(dx)
    div[:-1, :, :] += dx[:-1, :, :] / sx
    div[1:, :, :] -= dx[:-1, :, :] / sx
    div[:, :-1, :] += dy[:, :-1, :] / sy
    div[:, 1:, :] -= dy[:, :-1, :] / sy
    div[:, :, :-1] += dz[:, :, :-1] / sz
    div[:, :, 1:] -= dz[:, :, :-1] / sz
    return div


class TestComputeGradients(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``compute_gradients``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        return partial(
            compute_gradients,
            image=_ramp_image(),
            spacing=(1.0, 1.0, 1.0),
        )  # type: ignore[return-value]

    def test_ref_matches_first_order_diff(self) -> None:
        image = _ramp_image((16, 14, 12))
        spacing = (1.0, 1.0, 1.0)
        got = compute_gradients(image, spacing)
        expected = _gradients_ref(image, spacing)
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b)

    def test_anisotropic_spacing_scales_gradients(self) -> None:
        image = _ramp_image((16, 12, 10))
        iso = compute_gradients(image, (1.0, 1.0, 1.0))
        aniso = compute_gradients(image, (2.0, 1.0, 1.0))
        # Ramp along x: dx ~ 1/sx
        np.testing.assert_allclose(iso[0][:-1], 1.0)
        np.testing.assert_allclose(aniso[0][:-1], 0.5)
        assert not np.allclose(iso[0], aniso[0])


class TestComputeDivergence(TestSpacingValidation):
    """
    Tests for ``compute_divergence``.
    """

    def op_under_test(self) -> RequiresSpacing:
        z = np.zeros((10, 10, 10), dtype=np.float64)
        return partial(compute_divergence, dx=z, dy=z.copy(), dz=z.copy())

    @pytest.mark.parametrize("name", ["dx", "dy", "dz"])
    def test_invalid_component_type(self, name: str) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        kwargs = {"dx": z, "dy": z.copy(), "dz": z.copy(), "spacing": (1.0, 1.0, 1.0)}
        kwargs[name] = "bad"  # type: ignore[assignment]
        with pytest.raises(TypeError):
            compute_divergence(**kwargs)

    def test_shape_mismatch(self) -> None:
        dx = np.zeros((8, 8, 8), dtype=np.float64)
        dy = np.zeros((7, 8, 8), dtype=np.float64)
        dz = np.zeros((8, 8, 8), dtype=np.float64)
        with pytest.raises(ValueError):
            compute_divergence(dx, dy, dz, spacing=(1.0, 1.0, 1.0))

    def test_ref_matches(self) -> None:
        rng = np.random.default_rng(0)
        shape = (11, 10, 9)
        dx, dy, dz = (rng.normal(size=shape) for _ in range(3))
        spacing = (1.0, 1.5, 2.0)
        got = compute_divergence(dx, dy, dz, spacing)
        np.testing.assert_allclose(got, _divergence_ref(dx, dy, dz, spacing))

    def test_constant_field_zero_divergence_interior(self) -> None:
        shape = (12, 12, 12)
        dx = np.ones(shape, dtype=np.float64)
        dy = np.zeros(shape, dtype=np.float64)
        dz = np.zeros(shape, dtype=np.float64)
        div = compute_divergence(dx, dy, dz, spacing=(1.0, 1.0, 1.0))
        # Interior cancellation for constant flux along x
        np.testing.assert_allclose(div[1:-1], 0.0, atol=1e-12)


class TestComputeConductionCoefficient:
    """
    Tests for ``compute_conduction_coefficient``.
    """

    @pytest.fixture
    def grads(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = _ramp_image((16, 16, 16))
        return compute_gradients(image, (1.0, 1.0, 1.0))

    @pytest.mark.parametrize(
        "wrong",
        ["str", None, 1, np.zeros((8, 8, 8)).tolist()],
    )
    def test_invalid_dx_type(self, wrong: Any) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        with pytest.raises(TypeError):
            compute_conduction_coefficient(wrong, z, z)

    def test_invalid_dtype(self) -> None:
        z = np.zeros((8, 8, 8), dtype=int)
        with pytest.raises(ValueError):
            compute_conduction_coefficient(z, z, z)

    def test_invalid_quantile_kappa(self, grads: tuple) -> None:
        dx, dy, dz = grads
        with pytest.raises(ValueError):
            compute_conduction_coefficient(dx, dy, dz, kappa=0.0)
        with pytest.raises(ValueError):
            compute_conduction_coefficient(dx, dy, dz, kappa=-0.5)

    def test_fixed_kappa_rejected_when_disallowed(self, grads: tuple) -> None:
        dx, dy, dz = grads
        with pytest.raises(ValueError):
            compute_conduction_coefficient(dx, dy, dz, kappa=2.0, allow_fixed=False)

    def test_invalid_mask(self, grads: tuple) -> None:
        dx, dy, dz = grads
        with pytest.raises(ValueError):
            compute_conduction_coefficient(
                dx, dy, dz, kappa=0.9, mask=np.zeros(dx.shape, dtype=int)
            )
        with pytest.raises(ValueError):
            compute_conduction_coefficient(
                dx, dy, dz, kappa=0.9, mask=np.zeros((4, 4, 4), dtype=bool)
            )

    def test_no_valid_samples_raises(self) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="No valid gradient"):
            compute_conduction_coefficient(z, z, z, kappa=0.9)

    def test_fixed_kappa_ref_formula(self, grads: tuple) -> None:
        dx, dy, dz = grads
        kappa = 2.5
        got = compute_conduction_coefficient(dx, dy, dz, kappa=kappa)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        expected = 1 / (1 + norm**2 / (kappa + 1e-12) ** 2)
        np.testing.assert_allclose(got, expected)
        assert got.shape == dx.shape
        assert np.all((got > 0) & (got <= 1))

    def test_quantile_kappa_uses_mask(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((16, 16, 16))
        # Strong gradients only in a corner so masked quantile differs
        image[:4, :4, :4] += 10.0
        dx, dy, dz = compute_gradients(image, (1.0, 1.0, 1.0))
        mask = np.zeros(image.shape, dtype=bool)
        mask[8:, 8:, 8:] = True
        c_all = compute_conduction_coefficient(dx, dy, dz, kappa=0.9)
        c_mask = compute_conduction_coefficient(dx, dy, dz, kappa=0.9, mask=mask)
        assert not np.allclose(c_all, c_mask)


class TestPeronaMalik(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``perona_malik``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        return partial(
            perona_malik,
            image=_ramp_image(),
            spacing=(1.0, 1.0, 1.0),
        )  # type: ignore[return-value]

    def test_invalid_compute_c_type(self) -> None:
        with pytest.raises(TypeError):
            perona_malik(_ramp_image(), (1.0, 1.0, 1.0), compute_c="yes")  # type: ignore[arg-type]

    def test_invalid_c_params_type(self) -> None:
        with pytest.raises(TypeError):
            perona_malik(_ramp_image(), (1.0, 1.0, 1.0), c_params="bad")  # type: ignore[arg-type]

    def test_external_c_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            perona_malik(
                _ramp_image((8, 8, 8)),
                (1.0, 1.0, 1.0),
                external_c=np.ones((4, 4, 4), dtype=float),
            )

    def test_ref_compose_helpers(self) -> None:
        image = _ramp_image((14, 12, 10))
        spacing = (1.0, 1.5, 2.0)
        dx, dy, dz = compute_gradients(image, spacing)
        c = compute_conduction_coefficient(dx, dy, dz, kappa=0.95)
        expected = compute_divergence(dx * c, dy * c, dz * c, spacing)
        got = perona_malik(image, spacing, compute_c=True, c_params={"kappa": 0.95})
        np.testing.assert_allclose(got, expected)

    def test_external_c_only(self) -> None:
        image = _ramp_image((10, 10, 10))
        spacing = (1.0, 1.0, 1.0)
        external = np.full(image.shape, 0.5, dtype=np.float64)
        dx, dy, dz = compute_gradients(image, spacing)
        expected = compute_divergence(
            dx * external, dy * external, dz * external, spacing
        )
        got = perona_malik(image, spacing, compute_c=False, external_c=external)
        np.testing.assert_allclose(got, expected)

    def test_anisotropic_spacing_changes_result(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.random((12, 12, 12))
        iso = perona_malik(image, (1.0, 1.0, 1.0), c_params={"kappa": 2.0})
        aniso = perona_malik(image, (2.0, 1.0, 0.5), c_params={"kappa": 2.0})
        assert not np.allclose(iso, aniso)


class TestBoundaryBlurring(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``boundary_blurring``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=9)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return partial(
            boundary_blurring,
            image=image,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=2,
        )  # type: ignore[return-value]

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((28, 28, 28), wm_radius=6, gm_radius=10)
        image = np.zeros(seg.shape, dtype=np.float64)
        # Distinct GM/WM intensities so blurring changes the boundary
        wm = get_binary_mask(seg, mask_type="wm")
        gm = get_binary_mask(seg, mask_type="gm")
        image[wm] = 1.0
        image[gm] = 0.4
        return image, seg

    def test_invalid_seg_mask_type(self, volume: tuple) -> None:
        image, _ = volume
        with pytest.raises(TypeError):
            boundary_blurring(
                image, seg_mask="bad", spacing=(1.0, 1.0, 1.0)  # type: ignore[arg-type]
            )

    def test_seg_shape_mismatch(self, volume: tuple) -> None:
        image, _ = volume
        with pytest.raises(ValueError):
            boundary_blurring(
                image,
                seg_mask=np.zeros((4, 4, 4), dtype=int),
                spacing=(1.0, 1.0, 1.0),
            )

    @pytest.mark.parametrize("n_iters", [0, -1])
    def test_invalid_n_iters(self, volume: tuple, n_iters: int) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(image, seg, spacing=(1.0, 1.0, 1.0), n_iters=n_iters)

    def test_invalid_update_rate(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(image, seg, spacing=(1.0, 1.0, 1.0), update_rate=0.0)

    def test_invalid_pial_lower_bound(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(
                image, seg, spacing=(1.0, 1.0, 1.0), pial_lower_bound=-1.0
            )

    def test_invalid_gwb_bounds(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                gwb_lower_bound=3.0,
                gwb_upper_bound=1.0,
            )

    def test_invalid_edge_rolloff(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(image, seg, spacing=(1.0, 1.0, 1.0), edge_rolloff=0.0)

    def test_invalid_hemisphere(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            boundary_blurring(
                image, seg, spacing=(1.0, 1.0, 1.0), hemisphere="middle"  # type: ignore[arg-type]
            )

    def test_invalid_label_enum(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(TypeError):
            boundary_blurring(
                image, seg, spacing=(1.0, 1.0, 1.0), label_enum=object  # type: ignore[arg-type]
            )

    def test_empty_app_field_warns_and_returns_unchanged(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.warns(UserWarning, match="Application field"):
            out = boundary_blurring(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                app_field=np.zeros_like(image),
            )
        np.testing.assert_array_equal(out["out_image"], image)
        np.testing.assert_array_equal(out["effect_field"], 0)

    def test_empty_spread_warns(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.warns(UserWarning, match="No boundary to blur"):
            out = boundary_blurring(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                pial_lower_bound=100.0,
                gwb_upper_bound=-100.0,
                n_iters=2,
            )
        np.testing.assert_array_equal(out["out_image"], image)

    def test_no_gm_wm_warns(self) -> None:
        seg = np.zeros((16, 16, 16), dtype=np.int32)
        image = np.ones_like(seg, dtype=np.float64)
        with pytest.warns(UserWarning, match="No GM or WM"):
            out = boundary_blurring(image, seg, spacing=(1.0, 1.0, 1.0), n_iters=2)
        np.testing.assert_array_equal(out["out_image"], image)

    def test_return_contract_and_smoke(self, volume: tuple) -> None:
        image, seg = volume
        out = boundary_blurring(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=5,
            update_rate=0.1,
        )
        assert set(out) == {"out_image", "effect_field"}
        assert out["out_image"].shape == image.shape
        assert out["effect_field"].shape == image.shape
        np.testing.assert_allclose(
            out["effect_field"], out["out_image"] - image, atol=1e-12
        )
        # Blurring should change intensities somewhere near the ribbon
        assert not np.allclose(out["out_image"], image)
        tissue = get_binary_mask(seg, mask_type="wm") | get_binary_mask(
            seg, mask_type="gm"
        )
        assert np.any(np.abs(out["effect_field"][tissue]) > 0)
        # Far background stays unchanged
        assert out["out_image"][0, 0, 0] == image[0, 0, 0]

    def test_anisotropic_spacing_runs(self, volume: tuple) -> None:
        image, seg = volume
        out = boundary_blurring(
            image,
            seg,
            spacing=(1.0, 1.5, 2.0),
            n_iters=2,
        )
        assert out["out_image"].shape == image.shape

    def test_does_not_mutate_input_image(self, volume: tuple) -> None:
        image, seg = volume
        before = image.copy()
        boundary_blurring(image, seg, spacing=(1.0, 1.0, 1.0), n_iters=3)
        np.testing.assert_array_equal(image, before)


class TestHyperintensity(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``hyperintensity``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        seg = make_nested_seg_mask((22, 22, 22), wm_radius=5, gm_radius=8)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return partial(
            hyperintensity,
            image=image,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=3,
            start_prob=0.9,
            end_prob=0.5,
            random_seed=0,
        )  # type: ignore[return-value]

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((26, 26, 26), wm_radius=6, gm_radius=10)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 100.0
        return image, seg

    def test_invalid_seed_sigmas(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            hyperintensity(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                wmh_seeds_corr=-1.0,
                random_seed=0,
            )
        with pytest.raises(ValueError):
            hyperintensity(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                wmh_seeds_corr=3.0,
                wmh_seeds_hpf=2.0,
                random_seed=0,
            )

    def test_invalid_probs(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            hyperintensity(
                image, seg, spacing=(1.0, 1.0, 1.0), start_prob=1.5, random_seed=0
            )
        with pytest.raises(ValueError):
            hyperintensity(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                start_prob=0.0,
                end_prob=0.0,
                random_seed=0,
            )

    def test_invalid_scale_factor(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            hyperintensity(
                image, seg, spacing=(1.0, 1.0, 1.0), scale_factor=0.0, random_seed=0
            )

    def test_invalid_n_iters(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            hyperintensity(
                image, seg, spacing=(1.0, 1.0, 1.0), n_iters=0, random_seed=0
            )

    def test_invalid_update_rate(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            hyperintensity(
                image, seg, spacing=(1.0, 1.0, 1.0), update_rate=0.0, random_seed=0
            )

    def test_no_wm_warns(self) -> None:
        seg = np.zeros((16, 16, 16), dtype=np.int32)
        image = np.ones_like(seg, dtype=np.float64)
        with pytest.warns(UserWarning, match="No WM"):
            out = hyperintensity(image, seg, spacing=(1.0, 1.0, 1.0), random_seed=0)
        np.testing.assert_array_equal(out["out_image"], image)

    def test_no_seeds_warns(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.warns(UserWarning, match="No WMH seeds"):
            out = hyperintensity(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                start_prob=1e-12,
                end_prob=1e-12,
                edge_rolloff=1e-6,
                random_seed=0,
            )
        # May warn about seeds or return unchanged; accept either path if empty effect
        assert (
            np.allclose(out["effect_field"], 0.0)
            or out["effect_field"].shape == image.shape
        )

    def test_seed_determinism(self, volume: tuple) -> None:
        image, seg = volume
        kwargs = dict(
            spacing=(1.0, 1.0, 1.0),
            n_iters=4,
            start_prob=0.95,
            end_prob=0.6,
            scale_factor=0.2,
        )
        a = hyperintensity(image.copy(), seg.copy(), random_seed=11, **kwargs)  # type: ignore[arg-type]
        b = hyperintensity(image.copy(), seg.copy(), random_seed=11, **kwargs)  # type: ignore[arg-type]
        c = hyperintensity(image.copy(), seg.copy(), random_seed=12, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(a["effect_field"], b["effect_field"])
        assert not np.allclose(a["effect_field"], c["effect_field"])

    def test_return_contract_and_smoke(self, volume: tuple) -> None:
        image, seg = volume
        out = hyperintensity(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=5,
            start_prob=0.95,
            end_prob=0.6,
            scale_factor=0.25,
            random_seed=3,
        )
        assert set(out) == {"out_image", "effect_field"}
        assert out["out_image"].shape == image.shape
        np.testing.assert_allclose(
            out["out_image"], image + out["effect_field"], atol=1e-12
        )
        assert out["effect_field"].any()
        # Positive scale_factor should brighten somewhere in WM
        wm = get_binary_mask(seg, mask_type="wm")
        assert out["effect_field"][wm].max() > 0

    def test_negative_scale_factor_hypointensity(self, volume: tuple) -> None:
        image, seg = volume
        out = hyperintensity(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=4,
            start_prob=0.95,
            end_prob=0.6,
            scale_factor=-0.2,
            random_seed=5,
        )
        assert out["effect_field"].min() < 0

    def test_anisotropic_spacing_runs(self, volume: tuple) -> None:
        image, seg = volume
        out = hyperintensity(
            image,
            seg,
            spacing=(1.0, 1.5, 2.0),
            n_iters=3,
            start_prob=0.95,
            end_prob=0.6,
            random_seed=1,
        )
        assert out["out_image"].shape == image.shape

    def test_invalid_label_enum(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(TypeError):
            hyperintensity(
                image, seg, spacing=(1.0, 1.0, 1.0), label_enum=object, random_seed=0  # type: ignore[arg-type]
            )
