"""
Tests for ``synthfcd.core.deformations``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from synthfcd.core.deformations import (
    abnormal_gyration,
    compose_deformation_fields,
    cortical_expansion,
    cortical_thickening,
    exponentiate_velocity_field,
    get_deformation_fields_from_sdf,
    get_random_deformation_field,
    sulcal_widening,
)
from synthfcd.core.utils import create_meshgrid, warp_image
from tests.base_tests import (
    RequiresImages,
    RequiresSpacing,
    TestImageValidation,
    TestSpacingValidation,
)
from tests.utils import make_nested_seg_mask


def _sdf_plane(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> np.ndarray:
    """
    SDF increasing along x with physical spacing.
    """
    x = np.arange(shape[0], dtype=np.float64) * spacing[0]
    return np.broadcast_to(x[:, None, None], shape).copy()


def _deformation_ref(
    sdf: np.ndarray,
    spacing: tuple[float, float, float],
    scaling_field: np.ndarray | float | None = None,
    smooth_sigma_grad: float | None = None,
    smooth_sigma_field: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = spacing
    ux, uy, uz = np.gradient(sdf, sx, sy, sz)
    ux, uy, uz = -ux, -uy, -uz
    if smooth_sigma_grad is not None:
        s = (smooth_sigma_grad / sx, smooth_sigma_grad / sy, smooth_sigma_grad / sz)
        ux = ndimage.gaussian_filter(ux, s)
        uy = ndimage.gaussian_filter(uy, s)
        uz = ndimage.gaussian_filter(uz, s)
    mag = np.sqrt(ux**2 + uy**2 + uz**2) + 1e-8
    ux, uy, uz = ux / mag, uy / mag, uz / mag
    if scaling_field is not None:
        ux, uy, uz = ux * scaling_field, uy * scaling_field, uz * scaling_field
    if smooth_sigma_field is not None:
        s = (smooth_sigma_field / sx, smooth_sigma_field / sy, smooth_sigma_field / sz)
        ux = ndimage.gaussian_filter(ux, s)
        uy = ndimage.gaussian_filter(uy, s)
        uz = ndimage.gaussian_filter(uz, s)
    return ux / sx, uy / sy, uz / sz


class TestGetDeformationFieldsFromSDF(TestSpacingValidation):
    """
    Tests for ``get_deformation_fields_from_sdf``.
    """

    def op_under_test(self) -> RequiresSpacing:
        sdf = _sdf_plane((12, 12, 12), (1.0, 1.0, 1.0))
        return partial(get_deformation_fields_from_sdf, sdf=sdf)  # type: ignore[return-value]

    @pytest.mark.parametrize(
        "wrong_sdf",
        ["str", None, 1, np.zeros((8, 8, 8)).tolist()],
    )
    def test_invalid_sdf_type(self, wrong_sdf: Any) -> None:
        with pytest.raises(TypeError):
            get_deformation_fields_from_sdf(wrong_sdf, spacing=(1.0, 1.0, 1.0))

    def test_invalid_sdf_dtype(self) -> None:
        with pytest.raises(ValueError):
            get_deformation_fields_from_sdf(
                np.zeros((8, 8, 8), dtype=int), spacing=(1.0, 1.0, 1.0)
            )

    @pytest.mark.parametrize("sigma_name", ["smooth_sigma_grad", "smooth_sigma_field"])
    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_invalid_smooth_sigma(self, sigma_name: str, sigma: float) -> None:
        sdf = _sdf_plane((8, 8, 8), (1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            get_deformation_fields_from_sdf(
                sdf, spacing=(1.0, 1.0, 1.0), **{sigma_name: sigma}  # type: ignore[arg-type]
            )

    def test_scaling_field_shape_mismatch(self) -> None:
        sdf = _sdf_plane((8, 8, 8), (1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            get_deformation_fields_from_sdf(
                sdf,
                spacing=(1.0, 1.0, 1.0),
                scaling_field=np.ones((4, 4, 4), dtype=float),
            )

    def test_ref_plane_sdf(self) -> None:
        spacing = (1.0, 1.0, 1.0)
        sdf = _sdf_plane((16, 12, 10), spacing)
        ux, uy, uz = get_deformation_fields_from_sdf(sdf, spacing=spacing)
        ex, ey, ez = _deformation_ref(sdf, spacing)
        np.testing.assert_allclose(ux, ex)
        np.testing.assert_allclose(uy, ey)
        np.testing.assert_allclose(uz, ez)
        # Negated unit gradient of increasing-x SDF points toward -x in mm,
        # then converted to voxels with spacing=1.
        np.testing.assert_allclose(ux, -1.0, atol=1e-6)
        np.testing.assert_allclose(uy, 0.0, atol=1e-6)
        np.testing.assert_allclose(uz, 0.0, atol=1e-6)

    def test_anisotropic_spacing_voxel_magnitudes(self) -> None:
        spacing = (2.0, 1.0, 0.5)
        sdf = _sdf_plane((16, 12, 10), spacing)
        ux, uy, uz = get_deformation_fields_from_sdf(sdf, spacing=spacing)
        # 1 mm along -x -> 1/sx voxels
        np.testing.assert_allclose(ux, -1.0 / spacing[0], atol=1e-5)
        np.testing.assert_allclose(uy, 0.0, atol=1e-5)
        np.testing.assert_allclose(uz, 0.0, atol=1e-5)

    def test_scaling_and_smooth_match_ref(self) -> None:
        spacing = (1.0, 1.5, 2.0)
        sdf = _sdf_plane((14, 14, 14), spacing)
        scale = np.full(sdf.shape, 0.5, dtype=np.float64)
        got = get_deformation_fields_from_sdf(
            sdf,
            spacing=spacing,
            scaling_field=scale,
            smooth_sigma_grad=1.0,
            smooth_sigma_field=0.5,
        )
        expected = _deformation_ref(
            sdf,
            spacing,
            scaling_field=scale,
            smooth_sigma_grad=1.0,
            smooth_sigma_field=0.5,
        )
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b)


class TestGetRandomDeformationField(TestSpacingValidation):
    """
    Tests for ``get_random_deformation_field``.
    """

    def op_under_test(self) -> RequiresSpacing:
        return partial(
            get_random_deformation_field,
            shape=(10, 10, 10),
            random_seed=0,
        )  # type: ignore[return-value]

    @pytest.mark.parametrize(
        "wrong_shape",
        ["str", None, [8, 8, 8], (8, 8), (8, 8, 0)],
    )
    def test_invalid_shape(self, wrong_shape: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            get_random_deformation_field(wrong_shape, spacing=(1.0, 1.0, 1.0))

    def test_invalid_smooth_sigma(self) -> None:
        with pytest.raises(ValueError):
            get_random_deformation_field(
                (8, 8, 8), spacing=(1.0, 1.0, 1.0), smooth_sigma_field=0.0
            )

    def test_output_shape_dtype(self) -> None:
        ux, uy, uz = get_random_deformation_field(
            (9, 10, 11), spacing=(1.0, 1.0, 1.0), random_seed=1
        )
        assert ux.shape == uy.shape == uz.shape == (9, 10, 11)
        assert all(np.issubdtype(a.dtype, np.floating) for a in (ux, uy, uz))

    def test_seed_determinism(self) -> None:
        a = get_random_deformation_field(
            (8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=11
        )
        b = get_random_deformation_field(
            (8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=11
        )
        c = get_random_deformation_field(
            (8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=12
        )
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        assert any(not np.allclose(x, y) for x, y in zip(a, c))

    def test_random_state_determinism(self) -> None:
        rng_a = np.random.default_rng(5)
        rng_b = np.random.default_rng(5)
        a = get_random_deformation_field(
            (8, 8, 8), spacing=(1.0, 1.0, 1.0), random_state=rng_a
        )
        b = get_random_deformation_field(
            (8, 8, 8), spacing=(1.0, 1.0, 1.0), random_state=rng_b
        )
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_anisotropic_spacing_changes_field(self) -> None:
        iso = get_random_deformation_field(
            (10, 10, 10), spacing=(1.0, 1.0, 1.0), random_seed=3
        )
        aniso = get_random_deformation_field(
            (10, 10, 10), spacing=(1.0, 2.0, 3.0), random_seed=3
        )
        assert any(not np.allclose(a, b) for a, b in zip(iso, aniso))


class TestComposeDeformationFields:
    """
    Tests for ``compose_deformation_fields``.
    """

    @pytest.fixture
    def zero_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros((10, 10, 10), dtype=np.float64)
        return z, z.copy(), z.copy()

    @pytest.fixture
    def shift_x(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros((10, 10, 10), dtype=np.float64)
        return np.full_like(z, 0.3), z.copy(), z.copy()

    def test_invalid_field_type(self, zero_field: tuple) -> None:
        with pytest.raises(TypeError):
            compose_deformation_fields("bad", zero_field)  # type: ignore[arg-type]

    def test_length_mismatch(self, zero_field: tuple) -> None:
        with pytest.raises(ValueError):
            compose_deformation_fields((zero_field[0], zero_field[1]), zero_field)  # type: ignore[arg-type]

    def test_compose_with_zero_is_identity(
        self, zero_field: tuple, shift_x: tuple
    ) -> None:
        composed = compose_deformation_fields(shift_x, zero_field)
        for a, b in zip(composed, shift_x):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_compose_formula(self, shift_x: tuple) -> None:
        z = np.zeros_like(shift_x[0])
        second = (z.copy(), np.full_like(z, -0.2), z.copy())
        composed = compose_deformation_fields(shift_x, second)
        mesh = create_meshgrid(shift_x[0].shape)
        ax_w = warp_image(shift_x[0], second, meshgrid=mesh, interp_order=1)
        ay_w = warp_image(shift_x[1], second, meshgrid=mesh, interp_order=1)
        az_w = warp_image(shift_x[2], second, meshgrid=mesh, interp_order=1)
        expected = (second[0] + ax_w, second[1] + ay_w, second[2] + az_w)
        for a, b in zip(composed, expected):
            np.testing.assert_allclose(a, b)

    def test_warp_equivalence(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((12, 12, 12))
        z = np.zeros_like(image)
        a = (np.full_like(z, 0.15), z.copy(), z.copy())
        b = (z.copy(), np.full_like(z, 0.1), z.copy())
        composed = compose_deformation_fields(a, b)
        sequential = warp_image(
            warp_image(image, a, interp_order=1, preserve_zero_displacement=False),
            b,
            interp_order=1,
            preserve_zero_displacement=False,
        )
        single = warp_image(
            image, composed, interp_order=1, preserve_zero_displacement=False
        )
        np.testing.assert_allclose(sequential, single, atol=1e-5)


class TestExponentiateVelocityField(TestSpacingValidation):
    """
    Tests for ``exponentiate_velocity_field``.
    """

    def op_under_test(self) -> RequiresSpacing:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        v = (np.full_like(z, 0.1), z.copy(), z.copy())
        return partial(exponentiate_velocity_field, velocity_field=v)  # type: ignore[return-value]

    def test_invalid_scaling_steps(self) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        v = (z, z.copy(), z.copy())
        with pytest.raises(ValueError):
            exponentiate_velocity_field(v, spacing=(1.0, 1.0, 1.0), scaling_steps=-1)

    def test_invalid_smooth_sigma(self) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        v = (z, z.copy(), z.copy())
        with pytest.raises(ValueError):
            exponentiate_velocity_field(v, spacing=(1.0, 1.0, 1.0), smooth_sigma=0.0)

    def test_scaling_steps_zero_returns_velocity(self) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        v = (np.full_like(z, 0.25), z.copy(), z.copy())
        out = exponentiate_velocity_field(
            v, spacing=(1.0, 1.0, 1.0), scaling_steps=0, smooth_sigma=None
        )
        for a, b in zip(out, v):
            np.testing.assert_allclose(a, b)

    def test_does_not_mutate_input(self) -> None:
        z = np.zeros((8, 8, 8), dtype=np.float64)
        vx = np.full_like(z, 0.2)
        v = (vx, z.copy(), z.copy())
        before = vx.copy()
        exponentiate_velocity_field(v, spacing=(1.0, 1.0, 1.0), scaling_steps=3)
        np.testing.assert_array_equal(vx, before)

    def test_small_velocity_near_identity_warp(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.random((10, 10, 10))
        z = np.zeros_like(image)
        v = (np.full_like(z, 1e-4), z.copy(), z.copy())
        phi = exponentiate_velocity_field(
            v, spacing=(1.0, 1.0, 1.0), scaling_steps=4, smooth_sigma=None
        )
        warped = warp_image(
            image, phi, interp_order=1, preserve_zero_displacement=False
        )
        np.testing.assert_allclose(warped, image, atol=1e-3)


class TestCorticalExpansion(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``cortical_expansion``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=8)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return partial(
            cortical_expansion,
            image=image,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=1,
            mm_per_iter=0.5,
            smooth_sigma_field=None,
        )  # type: ignore[return-value]

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((28, 28, 28), wm_radius=6, gm_radius=10)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return image, seg

    def test_invalid_seg_mask_type(self, volume: tuple) -> None:
        image, _ = volume
        with pytest.raises(TypeError):
            cortical_expansion(
                image, seg_mask="bad", spacing=(1.0, 1.0, 1.0)  # type: ignore[arg-type]
            )

    def test_seg_shape_mismatch(self, volume: tuple) -> None:
        image, _ = volume
        with pytest.raises(ValueError):
            cortical_expansion(
                image,
                seg_mask=np.zeros((4, 4, 4), dtype=int),
                spacing=(1.0, 1.0, 1.0),
            )

    @pytest.mark.parametrize("n_iters", [0, -1])
    def test_invalid_n_iters(self, volume: tuple, n_iters: int) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(image, seg, spacing=(1.0, 1.0, 1.0), n_iters=n_iters)

    def test_invalid_mm_per_iter_zero(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(image, seg, spacing=(1.0, 1.0, 1.0), mm_per_iter=0.0)

    def test_invalid_normal_to(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(
                image, seg, spacing=(1.0, 1.0, 1.0), normal_to="both"  # type: ignore[arg-type]
            )

    def test_invalid_bounds(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                pial_lower_bound=2.0,
                pial_upper_bound=1.0,
            )
        with pytest.raises(ValueError):
            cortical_expansion(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                gwb_lower_bound=3.0,
                gwb_upper_bound=1.0,
            )

    @pytest.mark.parametrize("sigma_name", ["smooth_sigma_grad", "smooth_sigma_field"])
    def test_invalid_smooth_sigma(self, volume: tuple, sigma_name: str) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(
                image, seg, spacing=(1.0, 1.0, 1.0), **{sigma_name: 0.0}  # type: ignore[arg-type]
            )

    def test_invalid_label_enum(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(TypeError):
            cortical_expansion(
                image, seg, spacing=(1.0, 1.0, 1.0), label_enum=object  # type: ignore[arg-type]
            )

    def test_invalid_hemisphere(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_expansion(
                image, seg, spacing=(1.0, 1.0, 1.0), hemisphere="middle"  # type: ignore[arg-type]
            )

    def test_empty_app_field_warns_and_returns_unchanged(self, volume: tuple) -> None:
        image, seg = volume
        app = np.zeros_like(image)
        with pytest.warns(UserWarning, match="Application field"):
            out = cortical_expansion(image, seg, spacing=(1.0, 1.0, 1.0), app_field=app)
        np.testing.assert_array_equal(out["out_image"], image)
        np.testing.assert_array_equal(out["out_seg_mask"], seg)
        assert out["effect_field"] == []

    def test_empty_spread_warns(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.warns(UserWarning, match="No region to expand"):
            out = cortical_expansion(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                pial_lower_bound=100.0,
                gwb_upper_bound=-100.0,
                n_iters=1,
            )
        assert out["effect_field"] == []

    def test_return_contract_and_smoke(self, volume: tuple) -> None:
        image, seg = volume
        out = cortical_expansion(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=1,
            mm_per_iter=0.5,
            smooth_sigma_field=None,
            smooth_sigma_grad=None,
        )
        assert set(out) == {"out_image", "out_seg_mask", "effect_field"}
        assert out["out_image"].shape == image.shape
        assert out["out_seg_mask"].shape == seg.shape
        assert np.issubdtype(out["out_seg_mask"].dtype, np.integer)
        assert len(out["effect_field"]) == 1
        ux, uy, uz = out["effect_field"][0]
        support = (np.abs(ux) + np.abs(uy) + np.abs(uz)) > 0
        assert support.any()
        # Background far from brain should stay fixed
        assert not support[0, 0, 0]

    def test_anisotropic_spacing_runs(self, volume: tuple) -> None:
        image, seg = volume
        out = cortical_expansion(
            image,
            seg,
            spacing=(1.0, 1.5, 2.0),
            n_iters=1,
            mm_per_iter=0.5,
            smooth_sigma_field=None,
        )
        assert len(out["effect_field"]) <= 1
        assert out["out_image"].shape == image.shape


class TestCorticalThickening:
    """
    Tests for ``cortical_thickening`` wrapper guards and forwarding.
    """

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=9)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return image, seg

    def test_invalid_gwb_upper_bound(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_thickening(
                image, seg, spacing=(1.0, 1.0, 1.0), gwb_upper_bound=0.0
            )

    def test_invalid_gwb_lower_bound(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_thickening(
                image, seg, spacing=(1.0, 1.0, 1.0), gwb_lower_bound=1.0
            )

    def test_invalid_mm_per_iter(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_thickening(image, seg, spacing=(1.0, 1.0, 1.0), mm_per_iter=-1.0)

    def test_pial_lower_bound_vs_mm_per_iter(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            cortical_thickening(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                mm_per_iter=1.0,
                pial_lower_bound=0.5,
            )

    def test_forwards_normal_to_gm_wm(
        self, volume: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image, seg = volume
        captured: dict[str, Any] = {}

        def _spy(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {
                "out_image": image,
                "out_seg_mask": seg,
                "effect_field": [],
            }

        monkeypatch.setattr(
            "synthfcd.core.deformations.cortical_expansion",
            _spy,
        )
        cortical_thickening(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            mm_per_iter=0.5,
            pial_lower_bound=2.0,
        )
        assert captured["normal_to"] == "gm_wm"


class TestSulcalWidening:
    """
    Tests for ``sulcal_widening`` wrapper guards and forwarding.
    """

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((24, 24, 24), wm_radius=5, gm_radius=9)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return image, seg

    def test_invalid_pial_lower_bound(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            sulcal_widening(image, seg, spacing=(1.0, 1.0, 1.0), pial_lower_bound=1.0)

    def test_invalid_pial_upper_bound(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            sulcal_widening(image, seg, spacing=(1.0, 1.0, 1.0), pial_upper_bound=-0.1)

    def test_invalid_mm_per_iter(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            sulcal_widening(image, seg, spacing=(1.0, 1.0, 1.0), mm_per_iter=0.0)

    def test_forwards_normal_to_pial(
        self, volume: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image, seg = volume
        captured: dict[str, Any] = {}

        def _spy(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {
                "out_image": image,
                "out_seg_mask": seg,
                "effect_field": [],
            }

        monkeypatch.setattr(
            "synthfcd.core.deformations.cortical_expansion",
            _spy,
        )
        sulcal_widening(image, seg, spacing=(1.0, 1.0, 1.0), mm_per_iter=0.5)
        assert captured["normal_to"] == "pial"


class TestAbnormalGyration(TestImageValidation, TestSpacingValidation):
    """
    Tests for ``abnormal_gyration``.
    """

    def op_under_test(self) -> RequiresImages | RequiresSpacing:
        seg = make_nested_seg_mask((20, 20, 20), wm_radius=4, gm_radius=7)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return partial(
            abnormal_gyration,
            image=image,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=1,
            mm_per_iter=0.5,
            corr_sigma=None,
            hpf_sigma=None,
            smooth_sigma_field=None,
            scaling_and_squaring_steps=None,
            random_seed=0,
        )  # type: ignore[return-value]

    @pytest.fixture
    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        seg = make_nested_seg_mask((22, 22, 22), wm_radius=5, gm_radius=8)
        image = np.zeros(seg.shape, dtype=np.float64)
        image[seg > 0] = 1.0
        return image, seg

    def test_empty_app_field(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.warns(UserWarning, match="Application field"):
            out = abnormal_gyration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                app_field=np.zeros_like(image),
                random_seed=0,
            )
        assert out["effect_field"] == []

    def test_invalid_scaling_and_squaring_steps(self, volume: tuple) -> None:
        image, seg = volume
        with pytest.raises(ValueError):
            abnormal_gyration(
                image,
                seg,
                spacing=(1.0, 1.0, 1.0),
                scaling_and_squaring_steps=0,
                random_seed=0,
            )

    def test_seed_determinism(self, volume: tuple) -> None:
        image, seg = volume
        kwargs = dict(
            spacing=(1.0, 1.0, 1.0),
            n_iters=1,
            mm_per_iter=0.5,
            corr_sigma=None,
            smooth_sigma_field=None,
            scaling_and_squaring_steps=None,
        )
        a = abnormal_gyration(image.copy(), seg.copy(), random_seed=21, **kwargs)  # type: ignore[arg-type]
        b = abnormal_gyration(image.copy(), seg.copy(), random_seed=21, **kwargs)  # type: ignore[arg-type]
        c = abnormal_gyration(image.copy(), seg.copy(), random_seed=22, **kwargs)  # type: ignore[arg-type]
        if a["effect_field"] and b["effect_field"]:
            for x, y in zip(a["effect_field"][0], b["effect_field"][0]):
                np.testing.assert_array_equal(x, y)
        if a["effect_field"] and c["effect_field"]:
            assert any(
                not np.allclose(x, y)
                for x, y in zip(a["effect_field"][0], c["effect_field"][0])
            )

    def test_smoke_return_contract(self, volume: tuple) -> None:
        image, seg = volume
        out = abnormal_gyration(
            image,
            seg,
            spacing=(1.0, 1.0, 1.0),
            n_iters=1,
            mm_per_iter=0.5,
            corr_sigma=None,
            smooth_sigma_field=None,
            scaling_and_squaring_steps=2,
            random_seed=3,
        )
        assert set(out) == {"out_image", "out_seg_mask", "effect_field"}
        assert out["out_image"].shape == image.shape
        assert np.issubdtype(out["out_seg_mask"].dtype, np.integer)
