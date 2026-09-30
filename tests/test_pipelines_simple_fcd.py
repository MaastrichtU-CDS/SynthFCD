"""
Tests for ``synthfcd.pipelines.simple_fcd`` (published ``simple_fcd_simulator``).
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any

import numpy as np
import pytest

from synthfcd.core.masks import get_binary_mask
from synthfcd.pipelines import simple_fcd as simple_fcd_mod
from synthfcd.pipelines.simple_fcd import simple_fcd_simulator
from tests.utils import make_nested_seg_mask


def _disabled_deformation_params() -> dict[str, dict[str, Any]]:
    return {
        "abnormal_gyration": {"enable": False},
        "cortical_thickening": {"enable": False},
        "sulcal_widening": {"enable": False},
    }


def _disabled_intensity_params() -> dict[str, dict[str, Any]]:
    return {
        "boundary_blurring": {"enable": False},
        "texture_restoration": {"enable": False},
        "hyperintensity": {"enable": False},
    }


def _light_deformation_params() -> dict[str, dict[str, Any]]:
    return {
        "abnormal_gyration": {"enable": False},
        "cortical_thickening": {
            "enable": True,
            "n_iters": 1,
            "mm_per_iter": 0.5,
            "smooth_sigma_field": None,
            "app_field_type": "outward",
            "app_rolloff": 2.0,
            "bbox_thres": 0.05,
        },
        "sulcal_widening": {"enable": False},
    }


def _stochastic_gyration_params() -> dict[str, dict[str, Any]]:
    """
    Unseeded gyration strong enough to move labels across draws.

    Multimodal runs therefore share anatomy only when deformation fields are cached.
    """
    return {
        "abnormal_gyration": {
            "enable": True,
            "n_iters": 2,
            "mm_per_iter": 2.0,
            "corr_sigma": 2.0,
            "hpf_sigma": None,
            "smooth_sigma_field": None,
            "scaling_and_squaring_steps": 5,
            "app_field_type": "outward",
            "app_rolloff": 2.0,
            "bbox_thres": 0.05,
        },
        "cortical_thickening": {"enable": False},
        "sulcal_widening": {"enable": False},
    }


def _light_intensity_params() -> dict[str, dict[str, Any]]:
    return {
        "boundary_blurring": {
            "enable": True,
            "n_iters": 2,
            "random_seed": 1,
            "app_field_type": "outward",
            "app_rolloff": 2.0,
            "bbox_thres": 0.05,
        },
        "texture_restoration": {"enable": False},
        "hyperintensity": {"enable": False},
    }


@pytest.fixture
def volume() -> tuple[list[np.ndarray], np.ndarray]:
    seg = make_nested_seg_mask((32, 32, 32), wm_radius=6, gm_radius=11)
    rng = np.random.default_rng(0)
    images = [rng.random(seg.shape), rng.random(seg.shape) + 0.25]
    return images, seg


@pytest.fixture
def growth_params() -> dict[str, Any]:
    return {
        "volume": 40.0,
        "gm_prob": 0.5,
        "growth": "distance",
        "random_seed": 0,
        "skip_postprocess": True,
    }


class TestSimpleFCDSimulatorValidation:
    """
    Input validation for ``simple_fcd_simulator``.
    """

    def test_empty_images_raises(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        _, seg = volume
        with pytest.raises(ValueError, match="non-empty"):
            simple_fcd_simulator(
                images=[],
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=_disabled_deformation_params(),
                intensity_params=[],
                verbose=False,
            )

    def test_image_shape_mismatch(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        with pytest.raises(ValueError):
            simple_fcd_simulator(
                images=[np.zeros((4, 4, 4), dtype=float)],
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=_disabled_deformation_params(),
                intensity_params=[_disabled_intensity_params()],
                verbose=False,
            )

    def test_missing_deformation_keys(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        bad = {"cortical_thickening": {"enable": False}}
        with pytest.raises(ValueError):
            simple_fcd_simulator(
                images=[images[0]],
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=bad,
                intensity_params=[_disabled_intensity_params()],
                verbose=False,
            )

    def test_intensity_params_length_mismatch(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        with pytest.raises(ValueError, match="same length"):
            simple_fcd_simulator(
                images=images,
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=_disabled_deformation_params(),
                intensity_params=[_disabled_intensity_params()],
                verbose=False,
            )

    def test_missing_intensity_keys(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        with pytest.raises(ValueError):
            simple_fcd_simulator(
                images=[images[0]],
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=_disabled_deformation_params(),
                intensity_params=[{"boundary_blurring": {"enable": False}}],
                verbose=False,
            )


class TestSimpleFCDSimulatorContract:
    """
    Return contract and basic behavior.
    """

    def test_return_keys_and_shapes(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        out = simple_fcd_simulator(
            images=[images[0]],
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            growth_params=growth_params,
            deformation_params=_disabled_deformation_params(),
            intensity_params=[_disabled_intensity_params()],
            allow_disabled=True,
            verbose=False,
            use_cache="auto",
        )
        assert set(out) == {"out_images", "out_seg_mask", "stats", "extras"}
        assert set(out["extras"]) == {"orig_target", "out_target"}
        assert len(out["out_images"]) == 1
        assert out["out_images"][0].shape == seg.shape
        assert out["out_seg_mask"].shape == seg.shape
        assert out["extras"]["orig_target"].dtype == bool
        assert out["extras"]["out_target"].dtype == bool
        assert out["extras"]["orig_target"].any()
        tissue = get_binary_mask(seg, mask_type="gm") | get_binary_mask(
            seg, mask_type="wm"
        )
        assert np.all(out["extras"]["orig_target"] <= tissue)

    def test_all_effects_disabled_preserves_image(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        out = simple_fcd_simulator(
            images=[images[0]],
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            growth_params=growth_params,
            deformation_params=_disabled_deformation_params(),
            intensity_params=[_disabled_intensity_params()],
            allow_disabled=True,
            verbose=False,
        )
        np.testing.assert_allclose(out["out_images"][0], images[0])

    def test_effects_change_image(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        out = simple_fcd_simulator(
            images=[images[0]],
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            growth_params=growth_params,
            deformation_params=_light_deformation_params(),
            intensity_params=[_light_intensity_params()],
            allow_disabled=True,
            verbose=False,
            use_cache="auto",
        )
        assert not np.allclose(out["out_images"][0], images[0])
        assert out["extras"]["out_target"].any()


class TestSimpleFCDSimulatorCachingAndDeterminism:
    """
    Multimodal cache behavior and seed determinism.
    """

    def test_multimodal_cache_keeps_anatomy(
        self,
        volume: tuple,
        growth_params: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        images, seg = volume
        intensity = _disabled_intensity_params()
        calls = {"n": 0}
        real_gyration = simple_fcd_mod.SimpleFCD.DEFORMATION_EFFECTS[0]

        @wraps(real_gyration)
        def counting_gyration(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return real_gyration(*args, **kwargs)

        # Effects are bound on the class at import time; swap the tuple entry.
        monkeypatch.setattr(
            simple_fcd_mod.SimpleFCD,
            "DEFORMATION_EFFECTS",
            (
                counting_gyration,
                simple_fcd_mod.SimpleFCD.DEFORMATION_EFFECTS[1],
                simple_fcd_mod.SimpleFCD.DEFORMATION_EFFECTS[2],
            ),
        )

        out = simple_fcd_simulator(
            images=images,
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            growth_params=growth_params,
            deformation_params=_stochastic_gyration_params(),
            intensity_params=[intensity, deepcopy(intensity)],
            allow_disabled=True,
            verbose=False,
            use_cache="auto",
        )
        assert len(out["out_images"]) == 2
        assert calls["n"] == 1  # second modality reuses cached deformation fields
        assert not np.allclose(out["out_images"][0], images[0])
        assert not np.allclose(out["out_images"][1], images[1])
        # Gyration moved labels; both modalities share that warped anatomy
        assert not np.array_equal(out["out_seg_mask"], seg)
        assert out["extras"]["out_target"].any()

    def test_multimodal_without_cache_or_seeds_can_mismatch(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        intensity = _disabled_intensity_params()
        # Unseeded gyration redraws a new field per modality when cache is off
        with pytest.raises(ValueError, match="Mismatch in output anatomy"):
            simple_fcd_simulator(
                images=images,
                seg_mask=seg,
                spacing=(1.0, 1.0, 1.0),
                growth_params=growth_params,
                deformation_params=_stochastic_gyration_params(),
                intensity_params=[intensity, deepcopy(intensity)],
                allow_disabled=True,
                verbose=False,
                use_cache=False,
            )

    def test_fixed_seed_determinism(
        self, volume: tuple, growth_params: dict[str, Any]
    ) -> None:
        images, seg = volume
        kwargs = dict(
            images=[images[0]],
            seg_mask=seg,
            spacing=(1.0, 1.0, 1.0),
            growth_params=growth_params,
            deformation_params=_light_deformation_params(),
            intensity_params=[_light_intensity_params()],
            allow_disabled=True,
            verbose=False,
            use_cache=False,
        )
        a = simple_fcd_simulator(**kwargs)  # type: ignore[arg-type]
        b = simple_fcd_simulator(**kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(
            a["extras"]["orig_target"], b["extras"]["orig_target"]
        )
        np.testing.assert_allclose(a["out_images"][0], b["out_images"][0])
