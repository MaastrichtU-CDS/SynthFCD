"""
Tests for ``synthfcd.core.utils``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from synthfcd.core.utils import (
    bbox_from_mask,
    compute_distance_fields,
    create_meshgrid,
    generate_application_field,
    generate_noise_field,
    get_hemisphere_at_point,
    get_largest_component,
    postprocess_mask,
    warp,
    warp_image,
)
from synthfcd.utils.seg_labels import SynthSegLabel
from tests.base_tests import (
    RequiresMask,
    RequiresSpacing,
    TestMaskValidation,
    TestSpacingValidation,
)
from tests.utils import DummyLabel, _zero_pad, make_dummy_nested_seg_mask


class TestComputeDistanceFields(TestSpacingValidation):
    """
    Tests for ``compute_distance_fields``.
    """

    def op_under_test(self) -> RequiresSpacing:
        return partial(
            compute_distance_fields,
            seg_mask=make_dummy_nested_seg_mask(),
            surface="both",
            label_enum=DummyLabel,
        )  # type: ignore[return-value]

    @pytest.fixture
    def seg_mask(self) -> np.ndarray:
        return make_dummy_nested_seg_mask()

    @pytest.mark.parametrize(
        "wrong_seg_mask",
        [
            "str",
            1,
            None,
            (1, 2, 3),
            np.zeros((32, 32, 32), dtype=int).tolist(),
        ],
    )
    def test_invalid_seg_mask_type(self, wrong_seg_mask: Any) -> None:
        with pytest.raises(TypeError):
            compute_distance_fields(
                seg_mask=wrong_seg_mask,
                spacing=(1.0, 1.0, 1.0),
                label_enum=DummyLabel,
            )

    @pytest.mark.parametrize(
        "wrong_seg_mask",
        [
            np.zeros((32, 32, 32), dtype=float),
            np.zeros((32, 32, 32), dtype=bool),
            np.zeros((32, 32), dtype=int),
        ],
    )
    def test_invalid_seg_mask_value(self, wrong_seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            compute_distance_fields(
                seg_mask=wrong_seg_mask,
                spacing=(1.0, 1.0, 1.0),
                label_enum=DummyLabel,
            )

    @pytest.mark.parametrize("wrong_surface", [1, None, (1, 2, 3)])
    def test_invalid_surface_type(
        self, seg_mask: np.ndarray, wrong_surface: Any
    ) -> None:
        with pytest.raises(TypeError):
            compute_distance_fields(
                seg_mask=seg_mask,
                spacing=(1.0, 1.0, 1.0),
                surface=wrong_surface,
                label_enum=DummyLabel,
            )

    def test_invalid_surface_value(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            compute_distance_fields(
                seg_mask=seg_mask,
                spacing=(1.0, 1.0, 1.0),
                surface="gm_only",  # type: ignore[arg-type]
                label_enum=DummyLabel,
            )

    @pytest.mark.parametrize("wrong_label_enum", [None, "SynthSegLabel", int, object()])
    def test_invalid_label_enum_type(
        self, seg_mask: np.ndarray, wrong_label_enum: Any
    ) -> None:
        with pytest.raises(TypeError):
            compute_distance_fields(
                seg_mask=seg_mask,
                spacing=(1.0, 1.0, 1.0),
                label_enum=wrong_label_enum,
            )

    def test_return_types_for_surface(self, seg_mask: np.ndarray) -> None:
        both = compute_distance_fields(
            seg_mask, spacing=(1.0, 1.0, 1.0), surface="both", label_enum=DummyLabel
        )
        assert isinstance(both, tuple) and len(both) == 2

        gwb = compute_distance_fields(
            seg_mask, spacing=(1.0, 1.0, 1.0), surface="gm_wm", label_enum=DummyLabel
        )
        pial = compute_distance_fields(
            seg_mask, spacing=(1.0, 1.0, 1.0), surface="pial", label_enum=DummyLabel
        )
        assert isinstance(gwb, np.ndarray) and isinstance(pial, np.ndarray)
        np.testing.assert_allclose(both[0], gwb)
        np.testing.assert_allclose(both[1], pial)

    def test_signs_on_nested_labels(self, seg_mask: np.ndarray) -> None:
        gwb, pial = compute_distance_fields(
            seg_mask, spacing=(1.0, 1.0, 1.0), surface="both", label_enum=DummyLabel
        )
        center = tuple(s // 2 for s in seg_mask.shape)
        # Inside WM: positive GWB and pial distances
        assert gwb[center] > 0
        assert pial[center] > 0
        # Far outside: negative pial
        assert pial[0, 0, 0] < 0

    def test_anisotropic_spacing_scales_distances(self, seg_mask: np.ndarray) -> None:
        iso = compute_distance_fields(
            seg_mask, spacing=(1.0, 1.0, 1.0), surface="pial", label_enum=DummyLabel
        )
        aniso = compute_distance_fields(
            seg_mask, spacing=(2.0, 1.0, 1.0), surface="pial", label_enum=DummyLabel
        )
        assert not np.allclose(iso, aniso)

    def test_ref_matches_edt(self, seg_mask: np.ndarray) -> None:
        spacing = (1.0, 1.5, 2.0)
        inner = np.isin(
            seg_mask,
            list(
                DummyLabel.get_white_matter_labels()
                | DummyLabel.get_subcortical_labels()
                | DummyLabel.get_ventricle_labels()
            ),
        )
        expected = ndimage.distance_transform_edt(
            inner, sampling=spacing
        ) - ndimage.distance_transform_edt(
            ~inner, sampling=spacing
        )  # type: ignore
        got = compute_distance_fields(
            seg_mask, spacing=spacing, surface="gm_wm", label_enum=DummyLabel
        )
        np.testing.assert_allclose(got, expected)


class TestCreateMeshgrid:
    """
    Tests for ``create_meshgrid``.
    """

    @pytest.mark.parametrize(
        "wrong_shape",
        ["not_a_tuple", None, 3, [8, 8, 8], {"shape": (8, 8, 8)}],
    )
    def test_invalid_shape_type(self, wrong_shape: Any) -> None:
        with pytest.raises(TypeError):
            create_meshgrid(wrong_shape)

    @pytest.mark.parametrize(
        "wrong_shape",
        [(8, 8), (8, 8, 8, 8), (8, 8, 1.5), (8, 8, True)],
    )
    def test_invalid_shape_value(self, wrong_shape: tuple[Any, ...]) -> None:
        with pytest.raises(ValueError):
            create_meshgrid(wrong_shape)  # type: ignore[arg-type]

    def test_matches_np_meshgrid(self) -> None:
        shape = (5, 6, 7)
        X, Y, Z = create_meshgrid(shape)
        Xe, Ye, Ze = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        np.testing.assert_array_equal(X, Xe)
        np.testing.assert_array_equal(Y, Ye)
        np.testing.assert_array_equal(Z, Ze)
        assert X.shape == shape


class TestBboxFromMask:
    """
    Tests for ``bbox_from_mask``.
    """

    @pytest.fixture
    def mask(self) -> np.ndarray:
        m = np.zeros((20, 20, 20), dtype=bool)
        m[5:10, 6:12, 7:13] = True
        return m

    @pytest.mark.parametrize(
        "wrong_mask",
        ["str", None, 1, np.zeros((20, 20, 20), dtype=bool).tolist()],
    )
    def test_invalid_mask_type(self, wrong_mask: Any) -> None:
        with pytest.raises(TypeError):
            bbox_from_mask(wrong_mask)

    @pytest.mark.parametrize(
        "wrong_mask",
        [np.zeros((20, 20, 20), dtype=float), np.zeros((20, 20), dtype=bool)],
    )
    def test_invalid_mask_value(self, wrong_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            bbox_from_mask(wrong_mask)

    @pytest.mark.parametrize("wrong_pad", ["5", None, 1.5, [1, 2, 3]])
    def test_invalid_pad_type(self, mask: np.ndarray, wrong_pad: Any) -> None:
        with pytest.raises(TypeError):
            bbox_from_mask(mask, pad=wrong_pad)

    @pytest.mark.parametrize("wrong_pad", [(1, 2), (1, 2, 3, 4), (1, 2, 1.5)])
    def test_invalid_pad_value(
        self, mask: np.ndarray, wrong_pad: tuple[Any, ...]
    ) -> None:
        with pytest.raises(ValueError):
            bbox_from_mask(mask, pad=wrong_pad)  # type: ignore[arg-type]

    def test_empty_mask_warns_and_returns_full(self) -> None:
        with pytest.warns(UserWarning):
            result = bbox_from_mask(np.zeros((10, 10, 10), dtype=bool))
        assert result == (slice(None), slice(None), slice(None))

    def test_isotropic_pad(self, mask: np.ndarray) -> None:
        sl = bbox_from_mask(mask, pad=2)
        assert sl[0] == slice(3, 12)
        assert sl[1] == slice(4, 14)
        assert sl[2] == slice(5, 15)

    def test_anisotropic_pad_and_clip(self, mask: np.ndarray) -> None:
        sl = bbox_from_mask(mask, pad=(100, 1, 0))
        assert sl[0].start == 0
        assert sl[0].stop == 20
        assert sl[1] == slice(5, 13)
        assert sl[2] == slice(7, 13)

    def test_ensure_square(self, mask: np.ndarray) -> None:
        sl = bbox_from_mask(mask, pad=1, ensure_square=True)
        sizes = [s.stop - s.start for s in sl]
        assert sizes[0] == sizes[1] == sizes[2]

    def test_ensure_square_too_large_warns(self) -> None:
        m = np.zeros((5, 20, 20), dtype=bool)
        m[1:4, 2:18, 2:18] = True
        with pytest.warns(UserWarning, match="Square bounding box"):
            sl = bbox_from_mask(m, pad=0, ensure_square=True)
        sizes = [s.stop - s.start for s in sl]
        assert len(set(sizes)) > 1

    def test_does_not_mutate_input(self, mask: np.ndarray) -> None:
        before = mask.copy()
        bbox_from_mask(mask, pad=2)
        np.testing.assert_array_equal(mask, before)


class TestGetLargestComponent:
    """
    Tests for ``get_largest_component``.
    """

    @pytest.mark.parametrize(
        "wrong_mask",
        ["str", None, 1, np.zeros((10, 10, 10), dtype=bool).tolist()],
    )
    def test_invalid_mask_type(self, wrong_mask: Any) -> None:
        with pytest.raises(TypeError):
            get_largest_component(wrong_mask)

    @pytest.mark.parametrize(
        "wrong_mask",
        [
            np.zeros((10, 10, 10), dtype=int),
            np.zeros((10, 10), dtype=bool),
        ],
    )
    def test_invalid_mask_value(self, wrong_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            get_largest_component(wrong_mask)

    def test_empty_mask_warns(self) -> None:
        with pytest.warns(UserWarning):
            out = get_largest_component(np.zeros((8, 8, 8), dtype=bool))
        assert not out.any()

    def test_keeps_largest(self) -> None:
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        mask[10:16, 10:16, 10:16] = True
        out = get_largest_component(mask)
        assert out.sum() == (6 * 6 * 6)
        assert not out[2:4, 2:4, 2:4].any()

    def test_equal_size_components_keeps_one(self) -> None:
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        mask[10:13, 10:13, 10:13] = True
        out = get_largest_component(mask)
        assert out.sum() == (3 * 3 * 3)


class TestGenerateApplicationField(TestSpacingValidation):
    """
    Tests for ``generate_application_field``.
    """

    def op_under_test(self) -> RequiresSpacing:
        mask = _zero_pad(np.ones((8, 8, 8), dtype=bool), pad_size=4)
        return partial(
            generate_application_field,
            mask=mask,
            spacing=(1.0, 1.0, 1.0),
            field_type="inward",
        )  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "wrong_mask",
        ["str", None, 1, np.zeros((10, 10, 10), dtype=bool).tolist()],
    )
    def test_invalid_mask_type(self, wrong_mask: Any) -> None:
        with pytest.raises(TypeError):
            generate_application_field(wrong_mask, spacing=(1.0, 1.0, 1.0))

    @pytest.mark.parametrize(
        "wrong_mask",
        [
            np.zeros((10, 10, 10), dtype=float),
            np.zeros((10, 10), dtype=bool),
        ],
    )
    def test_invalid_mask_value(self, wrong_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            generate_application_field(wrong_mask, spacing=(1.0, 1.0, 1.0))

    def test_warning_empty_mask(self) -> None:
        with pytest.warns(UserWarning):
            out = generate_application_field(
                np.zeros((10, 10, 10), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
            )
        np.testing.assert_array_equal(out, 0)

    @pytest.mark.parametrize("wrong_field_type", [1, None, ["inward"]])
    def test_invalid_field_type_type(self, wrong_field_type: Any) -> None:
        mask = _zero_pad(np.ones((6, 6, 6), dtype=bool), pad_size=3)
        with pytest.raises(TypeError):
            generate_application_field(
                mask, spacing=(1.0, 1.0, 1.0), field_type=wrong_field_type
            )

    def test_invalid_field_type_value(self) -> None:
        mask = _zero_pad(np.ones((6, 6, 6), dtype=bool), pad_size=3)
        with pytest.raises(ValueError):
            generate_application_field(
                mask, spacing=(1.0, 1.0, 1.0), field_type="sideways"  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("rolloff_mm", [0.0, -1.0])
    def test_invalid_rolloff(self, rolloff_mm: float) -> None:
        mask = _zero_pad(np.ones((6, 6, 6), dtype=bool), pad_size=3)
        with pytest.raises(ValueError):
            generate_application_field(
                mask, spacing=(1.0, 1.0, 1.0), rolloff_mm=rolloff_mm
            )

    @pytest.mark.parametrize("tanh_target", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_tanh_target(self, tanh_target: float) -> None:
        mask = _zero_pad(np.ones((6, 6, 6), dtype=bool), pad_size=3)
        with pytest.raises(ValueError):
            generate_application_field(
                mask,
                spacing=(1.0, 1.0, 1.0),
                field_type="inward",
                tanh_target=tanh_target,
            )

    @pytest.mark.parametrize("gaussian_target", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_gaussian_target(self, gaussian_target: float) -> None:
        mask = _zero_pad(np.ones((6, 6, 6), dtype=bool), pad_size=3)
        with pytest.raises(ValueError):
            generate_application_field(
                mask,
                spacing=(1.0, 1.0, 1.0),
                field_type="outward",
                gaussian_target=gaussian_target,
            )

    def test_inward_peaks_inside(self) -> None:
        mask = _zero_pad(np.ones((10, 10, 10), dtype=bool), pad_size=5)
        field = generate_application_field(
            mask, spacing=(1.0, 1.0, 1.0), field_type="inward", rolloff_mm=2.0
        )
        assert 0.0 <= field.min() <= field.max() <= 1.0
        center = tuple(s // 2 for s in mask.shape)
        edge = np.argwhere(mask)[0]
        assert field[center] > field[tuple(edge)]

    def test_outward_peaks_at_boundary(self) -> None:
        mask = _zero_pad(np.ones((10, 10, 10), dtype=bool), pad_size=5)
        field = generate_application_field(
            mask, spacing=(1.0, 1.0, 1.0), field_type="outward", rolloff_mm=2.0
        )
        assert field[mask].min() == pytest.approx(1.0)
        assert field[~mask].max() < 1.0

    def test_anisotropic_spacing_changes_field(self) -> None:
        mask = _zero_pad(np.ones((10, 10, 10), dtype=bool), pad_size=5)
        iso = generate_application_field(
            mask, spacing=(1.0, 1.0, 1.0), field_type="inward"
        )
        aniso = generate_application_field(
            mask, spacing=(1.0, 2.0, 3.0), field_type="inward"
        )
        assert not np.allclose(iso, aniso)


class TestWarpImage:
    """
    Tests for ``warp_image``.
    """

    @pytest.fixture
    def image(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((16, 16, 16), dtype=np.float64)

    @pytest.fixture
    def zero_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros((16, 16, 16), dtype=np.float64)
        return z, z.copy(), z.copy()

    @pytest.mark.parametrize(
        "wrong_image",
        ["str", None, 1, np.zeros((16, 16, 16)).tolist()],
    )
    def test_invalid_image_type(
        self, wrong_image: Any, zero_field: tuple[np.ndarray, ...]
    ) -> None:
        with pytest.raises(TypeError):
            warp_image(wrong_image, zero_field)  # type: ignore[arg-type]

    def test_invalid_image_ndim(self, zero_field: tuple[np.ndarray, ...]) -> None:
        with pytest.raises(ValueError):
            warp_image(np.zeros((16, 16), dtype=float), zero_field)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "wrong_field",
        ["str", None, [1, 2, 3], (1, 2)],
    )
    def test_invalid_field_type(self, image: np.ndarray, wrong_field: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            warp_image(image, wrong_field)

    def test_mismatched_field_shape(self, image: np.ndarray) -> None:
        bad = np.zeros((8, 8, 8), dtype=np.float64)
        with pytest.raises(ValueError):
            warp_image(image, (bad, bad, bad))

    def test_zero_field_is_identity(
        self, image: np.ndarray, zero_field: tuple[np.ndarray, ...]
    ) -> None:
        out = warp_image(image, zero_field, interp_order=1)  # type: ignore[arg-type]
        np.testing.assert_allclose(out, image)

    def test_preserve_zero_displacement(self, image: np.ndarray) -> None:
        ux = np.zeros_like(image)
        uy = np.zeros_like(image)
        uz = np.zeros_like(image)
        ux[8, 8, 8] = 0.5
        field = (ux, uy, uz)
        preserved = warp_image(
            image, field, interp_order=3, preserve_zero_displacement=True
        )
        not_preserved = warp_image(
            image, field, interp_order=3, preserve_zero_displacement=False
        )
        zero = (ux == 0) & (uy == 0) & (uz == 0)
        np.testing.assert_array_equal(preserved[zero], image[zero])
        # At least some zero-displacement voxel may change without preservation
        assert not np.array_equal(not_preserved[zero], image[zero]) or np.allclose(
            preserved, not_preserved
        )

    def test_pull_ref_map_coordinates(self, image: np.ndarray) -> None:
        ux = np.full(image.shape, 0.5, dtype=np.float64)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)
        X, Y, Z = create_meshgrid(image.shape)
        Xs = np.clip(X + ux, 0, image.shape[0] - 1)
        Ys = np.clip(Y + uy, 0, image.shape[1] - 1)
        Zs = np.clip(Z + uz, 0, image.shape[2] - 1)
        expected = ndimage.map_coordinates(image, [Xs, Ys, Zs], order=1, mode="nearest")
        got = warp_image(
            image,
            (ux, uy, uz),
            meshgrid=(X, Y, Z),
            interp_order=1,
            preserve_zero_displacement=False,
        )
        np.testing.assert_allclose(got, expected)

    def test_invalid_preserve_flag_type(
        self, image: np.ndarray, zero_field: tuple[np.ndarray, ...]
    ) -> None:
        with pytest.raises(TypeError):
            warp_image(image, zero_field, preserve_zero_displacement="yes")  # type: ignore[arg-type]


class TestWarp:
    """
    Tests for ``warp``.
    """

    @pytest.fixture
    def images(self) -> list[np.ndarray]:
        rng = np.random.default_rng(1)
        return [rng.random((12, 12, 12)) for _ in range(2)]

    @pytest.fixture
    def fields(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        z = np.zeros((12, 12, 12), dtype=np.float64)
        shift = z.copy()
        shift[...] = 0.25
        return [(shift, z.copy(), z.copy())]

    def test_empty_images_raises(self, fields: list) -> None:
        with pytest.raises(ValueError):
            warp([], fields)

    def test_empty_fields_raises(self, images: list[np.ndarray]) -> None:
        with pytest.raises(ValueError):
            warp(images, [])

    def test_broadcast_params(self, images: list[np.ndarray], fields: list) -> None:
        out = warp(images, fields, interp_order=1, mode="nearest", clip=True)
        assert len(out) == 2
        assert all(o.shape == images[0].shape for o in out)

    def test_per_image_param_length_mismatch(
        self, images: list[np.ndarray], fields: list
    ) -> None:
        with pytest.raises(ValueError):
            warp(images, fields, interp_order=[1])

    def test_sequential_matches_repeated_warp_image(
        self, images: list[np.ndarray]
    ) -> None:
        z = np.zeros_like(images[0])
        f1 = (np.full_like(z, 0.2), z.copy(), z.copy())
        f2 = (z.copy(), np.full_like(z, -0.1), z.copy())
        fields = [f1, f2]
        out = warp(
            [images[0]], fields, interp_order=1, preserve_zero_displacement=False
        )[0]
        manual = images[0]
        for f in fields:
            manual = warp_image(
                manual, f, interp_order=1, preserve_zero_displacement=False
            )
        np.testing.assert_allclose(out, manual)


class TestGetHemisphereAtPoint:
    """
    Tests for ``get_hemisphere_at_point``.
    """

    @pytest.fixture
    def seg_mask(self) -> np.ndarray:
        seg = np.zeros((10, 10, 10), dtype=np.int32)
        seg[2, 2, 2] = int(SynthSegLabel.LEFT_HIPPOCAMPUS)
        seg[7, 7, 7] = int(SynthSegLabel.RIGHT_HIPPOCAMPUS)
        seg[5, 5, 5] = int(SynthSegLabel.BRAIN_STEM)
        return seg

    @pytest.mark.parametrize(
        "wrong_seg",
        ["str", None, np.zeros((10, 10, 10), dtype=int).tolist()],
    )
    def test_invalid_seg_type(self, wrong_seg: Any) -> None:
        with pytest.raises(TypeError):
            get_hemisphere_at_point(wrong_seg, (0, 0, 0))

    def test_invalid_seg_dtype(self) -> None:
        with pytest.raises(ValueError):
            get_hemisphere_at_point(np.zeros((5, 5, 5), dtype=float), (0, 0, 0))

    @pytest.mark.parametrize(
        "wrong_point",
        ["str", None, 1, [1, 2, 3], (1, 2), (1, 2, 3, 4), (1.0, 2, 3)],
    )
    def test_invalid_point(self, seg_mask: np.ndarray, wrong_point: Any) -> None:
        with pytest.raises(TypeError):
            get_hemisphere_at_point(seg_mask, wrong_point)

    def test_out_of_bounds(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            get_hemisphere_at_point(seg_mask, (0, 0, 100))

    def test_left_right_neutral(self, seg_mask: np.ndarray) -> None:
        assert get_hemisphere_at_point(seg_mask, (2, 2, 2)) == "left"
        assert get_hemisphere_at_point(seg_mask, (7, 7, 7)) == "right"
        assert get_hemisphere_at_point(seg_mask, (5, 5, 5)) is None
        assert get_hemisphere_at_point(seg_mask, (0, 0, 0)) is None

    def test_invalid_label_enum(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(TypeError):
            get_hemisphere_at_point(seg_mask, (2, 2, 2), label_enum=object)  # type: ignore[arg-type]


class TestPostprocessMask(TestMaskValidation, TestSpacingValidation):
    """
    Tests for ``postprocess_mask``.
    """

    def op_under_test(self) -> RequiresMask | RequiresSpacing:
        return partial(
            postprocess_mask,
            mask=np.ones((16, 16, 16), dtype=bool),
            spacing=(1.0, 1.0, 1.0),
            smooth_sigma=None,
            fill_holes=False,
            keep_largest_component=False,
        )

    def test_warning_empty_mask(self) -> None:
        with pytest.warns(UserWarning):
            out = postprocess_mask(
                np.zeros((8, 8, 8), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
            )
        assert not out.any()

    @pytest.mark.parametrize("smooth_sigma", [0.0, -1.0])
    def test_invalid_smooth_sigma(self, smooth_sigma: float) -> None:
        with pytest.raises(ValueError):
            postprocess_mask(
                np.ones((8, 8, 8), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
                smooth_sigma=smooth_sigma,
            )

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_invalid_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            postprocess_mask(
                np.ones((8, 8, 8), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
                threshold=threshold,
            )

    @pytest.mark.parametrize("closing", [0.0, -1.0])
    def test_invalid_closing(self, closing: float) -> None:
        with pytest.raises(ValueError):
            postprocess_mask(
                np.ones((8, 8, 8), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
                smooth_sigma=None,
                closing=closing,
            )

    def test_empty_limit_to_warns(self) -> None:
        with pytest.warns(UserWarning, match="limit_to"):
            out = postprocess_mask(
                np.ones((8, 8, 8), dtype=bool),
                spacing=(1.0, 1.0, 1.0),
                smooth_sigma=None,
                limit_to=np.zeros((8, 8, 8), dtype=bool),
            )
        assert not out.any()

    def test_empty_after_threshold_warns(self) -> None:
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[5, 5, 5] = True
        with pytest.warns(UserWarning, match="thresholding"):
            out = postprocess_mask(
                mask,
                spacing=(1.0, 1.0, 1.0),
                smooth_sigma=2.0,
                threshold=0.99,
                fill_holes=False,
                keep_largest_component=False,
            )
        assert not out.any()

    def test_closing_keeps_support(self) -> None:
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[6:10, 6:10, 6:10] = True
        mask[8, 8, 8] = False
        out = postprocess_mask(
            mask,
            spacing=(1.0, 1.0, 1.0),
            smooth_sigma=None,
            closing=1.0,
            fill_holes=True,
            keep_largest_component=True,
        )
        assert out.dtype == bool
        assert out.any()

    def test_anisotropic_auto_sigma(self) -> None:
        mask = _zero_pad(np.ones((8, 8, 8), dtype=bool), pad_size=4)
        out = postprocess_mask(
            mask,
            spacing=(1.0, 2.0, 3.0),
            smooth_sigma="auto",
            fill_holes=False,
        )
        assert out.shape == mask.shape
        assert out.any()


class TestGenerateNoiseField(TestSpacingValidation):
    """
    Tests for ``generate_noise_field``.
    """

    def op_under_test(self) -> RequiresSpacing:
        return partial(
            generate_noise_field,
            shape=(12, 12, 12),
            random_seed=0,
        )

    @pytest.mark.parametrize(
        "wrong_shape",
        ["str", None, 3, [8, 8, 8]],
    )
    def test_invalid_shape_type(self, wrong_shape: Any) -> None:
        with pytest.raises(TypeError):
            generate_noise_field(wrong_shape, spacing=(1.0, 1.0, 1.0))

    @pytest.mark.parametrize(
        "wrong_shape",
        [(8, 8), (8, 8, 8, 8), (8, 8, 0), (8, 8, -1), (8, 8, 1.5)],
    )
    def test_invalid_shape_value(self, wrong_shape: tuple[Any, ...]) -> None:
        with pytest.raises(ValueError):
            generate_noise_field(wrong_shape, spacing=(1.0, 1.0, 1.0))  # type: ignore[arg-type]

    def test_invalid_corr_sigma(self) -> None:
        with pytest.raises(ValueError):
            generate_noise_field(
                (8, 8, 8), spacing=(1.0, 1.0, 1.0), corr_sigma=-1.0, random_seed=0
            )

    def test_invalid_hpf_sigma(self) -> None:
        with pytest.raises(ValueError):
            generate_noise_field(
                (8, 8, 8), spacing=(1.0, 1.0, 1.0), hpf_sigma=0.0, random_seed=0
            )

    def test_hpf_must_exceed_corr(self) -> None:
        with pytest.raises(ValueError):
            generate_noise_field(
                (8, 8, 8),
                spacing=(1.0, 1.0, 1.0),
                corr_sigma=2.0,
                hpf_sigma=1.0,
                random_seed=0,
            )

    def test_output_range_and_shape(self) -> None:
        out = generate_noise_field((10, 11, 12), spacing=(1.0, 1.0, 1.0), random_seed=1)
        assert out.shape == (10, 11, 12)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_determinism_seed(self) -> None:
        a = generate_noise_field((8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=7)
        b = generate_noise_field((8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=7)
        c = generate_noise_field((8, 8, 8), spacing=(1.0, 1.0, 1.0), random_seed=8)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_anisotropic_spacing_changes_correlated_noise(self) -> None:
        iso = generate_noise_field(
            (12, 12, 12),
            spacing=(1.0, 1.0, 1.0),
            corr_sigma=2.0,
            random_seed=3,
        )
        aniso = generate_noise_field(
            (12, 12, 12),
            spacing=(1.0, 2.0, 3.0),
            corr_sigma=2.0,
            random_seed=3,
        )
        assert not np.allclose(iso, aniso)

    def test_hpf_changes_field(self) -> None:
        base = generate_noise_field(
            (12, 12, 12),
            spacing=(1.0, 1.0, 1.0),
            corr_sigma=2.0,
            random_seed=4,
        )
        hpf = generate_noise_field(
            (12, 12, 12),
            spacing=(1.0, 1.0, 1.0),
            corr_sigma=2.0,
            hpf_sigma=4.0,
            random_seed=4,
        )
        assert not np.allclose(base, hpf)
