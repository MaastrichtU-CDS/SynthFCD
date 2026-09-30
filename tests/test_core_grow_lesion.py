"""
Tests for ``synthfcd.core.grow_lesion``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from synthfcd.core.grow_lesion import get_bottom_of_sulci, grow_random_lesion
from synthfcd.core.masks import get_binary_mask
from synthfcd.utils.seg_labels import SynthSegLabel
from tests.base_tests import RequiresSpacing, TestSpacingValidation
from tests.utils import make_nested_seg_mask


def _nested_seg(shape: tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    return make_nested_seg_mask(shape, wm_radius=8, gm_radius=14)


class TestGrowRandomLesion(TestSpacingValidation):
    """
    Tests for ``grow_random_lesion``.
    """

    def op_under_test(self) -> RequiresSpacing:
        return partial(
            grow_random_lesion,
            seg_mask=_nested_seg(),
            volume=40.0,
            random_seed=0,
            skip_postprocess=True,
        )  # type: ignore[return-value]

    @pytest.fixture
    def seg_mask(self) -> np.ndarray:
        return _nested_seg()

    @pytest.mark.parametrize(
        "wrong_seg",
        ["str", None, 1, np.zeros((8, 8, 8), dtype=int).tolist()],
    )
    def test_invalid_seg_type(self, wrong_seg: Any) -> None:
        with pytest.raises(TypeError):
            grow_random_lesion(wrong_seg, (1.0, 1.0, 1.0), volume=20.0)

    def test_invalid_seg_dtype(self) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(
                np.zeros((8, 8, 8), dtype=float), (1.0, 1.0, 1.0), volume=20.0
            )

    def test_empty_gm_or_wm_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            grow_random_lesion(
                np.zeros((16, 16, 16), dtype=int),
                (1.0, 1.0, 1.0),
                volume=20.0,
            )

    @pytest.mark.parametrize("volume", [0.0, -10.0])
    def test_invalid_volume(self, seg_mask: np.ndarray, volume: float) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(seg_mask, (1.0, 1.0, 1.0), volume=volume)

    @pytest.mark.parametrize("gm_prob", [-0.1, 1.1])
    def test_invalid_gm_prob(self, seg_mask: np.ndarray, gm_prob: float) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(seg_mask, (1.0, 1.0, 1.0), volume=20.0, gm_prob=gm_prob)

    def test_invalid_growth_mode(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                growth="isotropic",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("distance_noise_ratio", [-0.1, 1.1])
    def test_invalid_distance_noise_ratio(
        self, seg_mask: np.ndarray, distance_noise_ratio: float
    ) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                distance_noise_ratio=distance_noise_ratio,
            )

    def test_invalid_lobe_non_cortical(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError, match="cortical"):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                lobe=int(SynthSegLabel.LEFT_HIPPOCAMPUS),
            )

    def test_invalid_lobe_list(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                lobe=["not-an-int"],  # type: ignore[list-item]
            )

    def test_invalid_label_enum(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(TypeError):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                label_enum=object,  # type: ignore[arg-type]
            )

    def test_invalid_max_attempts(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(ValueError):
            grow_random_lesion(seg_mask, (1.0, 1.0, 1.0), volume=20.0, max_attempts=0)

    def test_attempt_no_exceeds_max_attempts(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(RuntimeError, match="max_attempts"):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                attempt_no=2,
                max_attempts=2,
            )

    def test_lobe_without_candidates_raises(self, seg_mask: np.ndarray) -> None:
        # Cortical label present in enum but absent from this nested volume
        with pytest.raises(RuntimeError, match="lobe"):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                lobe=int(SynthSegLabel.LEFT_ENTORHINAL),
                random_seed=0,
            )

    def test_bottom_of_sulcus_on_sphere_raises(self, seg_mask: np.ndarray) -> None:
        with pytest.raises(RuntimeError, match="bottom of sulcus"):
            grow_random_lesion(
                seg_mask,
                (1.0, 1.0, 1.0),
                volume=20.0,
                bottom_of_sulcus=True,
                random_seed=0,
                max_attempts=2,
            )

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_return_contract_and_volume(
        self, seg_mask: np.ndarray, growth: str
    ) -> None:
        target_volume = 80.0
        out = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=target_volume,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        assert set(out) == {"target", "growth_stats"}
        assert out["target"].shape == seg_mask.shape
        assert out["target"].dtype == bool
        assert out["target"].any()

        stats = out["growth_stats"]
        assert set(stats) >= {"volume", "hemisphere", "DKT_labels"}
        assert stats["hemisphere"] == "left"
        assert isinstance(stats["DKT_labels"], list)
        assert stats["volume"] == pytest.approx(target_volume, abs=1e-6)
        assert int(out["target"].sum()) == int(target_volume)

        tissue = get_binary_mask(seg_mask, mask_type="gm") | get_binary_mask(
            seg_mask, mask_type="wm"
        )
        assert np.all(out["target"] <= tissue)

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_seed_determinism(self, seg_mask: np.ndarray, growth: str) -> None:
        kwargs = dict(
            spacing=(1.0, 1.0, 1.0),
            volume=60.0,
            growth=growth,
            skip_postprocess=True,
        )
        a = grow_random_lesion(seg_mask.copy(), random_seed=5, **kwargs)  # type: ignore[arg-type]
        b = grow_random_lesion(seg_mask.copy(), random_seed=5, **kwargs)  # type: ignore[arg-type]
        c = grow_random_lesion(seg_mask.copy(), random_seed=6, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(a["target"], b["target"])
        assert not np.array_equal(a["target"], c["target"])

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_larger_volume_grows_more(self, seg_mask: np.ndarray, growth: str) -> None:
        small = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=30.0,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        large = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=120.0,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        assert small["growth_stats"]["volume"] < large["growth_stats"]["volume"]
        assert small["target"].sum() < large["target"].sum()

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_lobe_filter_uses_requested_cortex(
        self, seg_mask: np.ndarray, growth: str
    ) -> None:
        lobe = int(SynthSegLabel.LEFT_SUPERIOR_FRONTAL)
        out = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=50.0,
            lobe=lobe,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        assert (
            SynthSegLabel.LEFT_SUPERIOR_FRONTAL.readable_name
            in out["growth_stats"]["DKT_labels"]
        )
        # Seed neighborhood constrained by lobe candidates; lesion intersects that label
        assert np.any(seg_mask[out["target"]] == lobe)

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_gm_prob_shifts_tissue_composition(
        self, seg_mask: np.ndarray, growth: str
    ) -> None:
        gm = get_binary_mask(seg_mask, mask_type="gm")
        low = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=100.0,
            gm_prob=0.1,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        high = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=100.0,
            gm_prob=0.9,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        assert (low["target"] & gm).sum() <= (high["target"] & gm).sum()

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_postprocess_keeps_support_in_tissue(
        self, seg_mask: np.ndarray, growth: str
    ) -> None:
        out = grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=80.0,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=False,
        )
        tissue = get_binary_mask(seg_mask, mask_type="gm") | get_binary_mask(
            seg_mask, mask_type="wm"
        )
        assert out["target"].any()
        assert np.all(out["target"] <= tissue)
        assert out["growth_stats"]["volume"] > 0.0

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_anisotropic_spacing_runs(self, seg_mask: np.ndarray, growth: str) -> None:
        spacing = (1.0, 1.5, 2.0)
        out = grow_random_lesion(
            seg_mask,
            spacing,
            volume=100.0,
            growth=growth,  # type: ignore[arg-type]
            random_seed=1,
            skip_postprocess=True,
        )
        assert out["target"].any()
        voxel_vol = float(np.prod(spacing))
        assert out["growth_stats"]["volume"] == pytest.approx(
            out["target"].sum() * voxel_vol, rel=1e-6
        )

    @pytest.mark.parametrize("growth", ["distance", "irregular"])
    def test_does_not_mutate_seg_mask(self, seg_mask: np.ndarray, growth: str) -> None:
        before = seg_mask.copy()
        grow_random_lesion(
            seg_mask,
            (1.0, 1.0, 1.0),
            volume=40.0,
            growth=growth,  # type: ignore[arg-type]
            random_seed=0,
            skip_postprocess=True,
        )
        np.testing.assert_array_equal(seg_mask, before)


class TestGetBottomOfSulci:
    """
    Tests for ``get_bottom_of_sulci``.
    """

    @pytest.fixture
    def nested(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seg = _nested_seg((36, 36, 36))
        wm = get_binary_mask(seg, mask_type="wm")
        gm = get_binary_mask(seg, mask_type="gm")
        candidates = ndimage.binary_dilation(wm, iterations=1) & gm
        return seg, wm, candidates

    def test_empty_wm_raises(self, nested: tuple) -> None:
        seg, _, candidates = nested
        with pytest.raises(ValueError, match="No points"):
            get_bottom_of_sulci(
                seg.shape,
                candidates,
                np.zeros_like(candidates),
                (1.0, 1.0, 1.0),
            )

    def test_sphere_returns_empty_or_sparse(self, nested: tuple) -> None:
        seg, wm, candidates = nested
        out = get_bottom_of_sulci(seg.shape, candidates, wm, (1.0, 1.0, 1.0))
        assert out.shape == seg.shape
        assert out.dtype == bool
        # Smooth nested spheres have no true sulcal bottoms
        assert out.sum() == 0

    def test_output_subset_of_candidates(self, nested: tuple) -> None:
        seg, wm, candidates = nested
        # Add a shallow indentation in WM to create depth variation
        indented = wm.copy()
        cz, cy, cx = [s // 2 for s in seg.shape]
        indented[cz - 2 : cz + 3, cy - 6 : cy - 2, cx - 2 : cx + 3] = False
        out = get_bottom_of_sulci(seg.shape, candidates, indented, (1.0, 1.0, 1.0))
        assert np.all(out <= candidates)
