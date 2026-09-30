"""
Tests for ``RandSimpleFCDd``: preset drawing, pipeline call, types, MONAI loading.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord

from synthfcd.pipelines import simple_fcd_simulator
from synthfcd.presets import draw_simple_fcd_params, get_simple_fcd_presets
from synthfcd.transforms import RandSimpleFCDd
from tests.utils import make_nested_seg_mask

SHAPE = (16, 16, 16)
SPACING = (1.0, 1.0, 1.0)


def _fast_presets() -> dict[str, Any]:
    """
    Default preset tree with cheap growth/effects so full pipeline tests stay fast.
    """
    presets = get_simple_fcd_presets()
    for key, block in presets.items():
        if key == "FCD_type":
            continue
        growth = block["growth_params"]
        growth["lobe"] = None
        growth["volume"] = 40.0
        growth["gm_prob"] = 0.5
        growth["bottom_of_sulcus"] = False
        deform = block["deformation_params"]
        for params in deform.values():
            if "n_iters" in params:
                params["n_iters"] = 1
            if "mm_per_iter" in params:
                params["mm_per_iter"] = 0.5
            if "scaling_and_squaring_steps" in params:
                params["scaling_and_squaring_steps"] = 2
        intensity = block["intensity_params"]
        for params in intensity.values():
            if "n_iters" in params:
                params["n_iters"] = 1
    return presets


def _fake_simulator(
    images: list[np.ndarray],
    seg_mask: np.ndarray,
    spacing: tuple[float, float, float],
    **kwargs: Any,
) -> dict[str, Any]:
    target = (seg_mask > 0).astype(bool)
    return {
        "out_images": [img + 1.0 for img in images],
        "out_seg_mask": seg_mask.copy(),
        "stats": {"n_images": len(images)},
        "extras": {"out_target": target, "orig_target": target.copy()},
    }


@pytest.fixture
def volume() -> dict[str, Any]:
    seg = make_nested_seg_mask(SHAPE, wm_radius=4, gm_radius=7)
    rng = np.random.default_rng(0)
    return {
        "image1": rng.random(SHAPE).astype(np.float32),
        "image2": (rng.random(SHAPE) + 0.25).astype(np.float32),
        "seg_mask": seg,
        "subject_id": "sub-01",
    }


@pytest.fixture
def presets() -> dict[str, Any]:
    return _fast_presets()


def _transform(presets: dict[str, Any], **kwargs: Any) -> RandSimpleFCDd:
    defaults: dict[str, Any] = dict(
        keys=["image1", "image2"],
        sequences=["T1like", "T2like"],
        seg_mask_key="seg_mask",
        spacing=SPACING,
        custom_presets=presets,
        out_stats_key="synthfcd_stats",
        out_lesion_mask_key="synthfcd_lesion",
        orig_lesion_mask_key="synthfcd_orig_target",
    )
    defaults.update(kwargs)
    return RandSimpleFCDd(**defaults)


class TestRandSimpleFCDInit:
    """
    Construction-time validation.
    """

    def test_sequences_length_must_match_keys(self, presets: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="same length"):
            _transform(presets, sequences=["T1like"])

    def test_seg_mask_key_cannot_be_an_image_key(self, presets: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="cannot be one of"):
            _transform(
                presets, keys=["image1", "seg_mask"], sequences=["T1like", "T2like"]
            )

    def test_custom_presets_must_match_default_structure(self) -> None:
        with pytest.raises(ValueError, match="same keys and structure"):
            RandSimpleFCDd(
                keys=["image1"],
                sequences=["T1like"],
                custom_presets={"FCD_type": "FCD_type_Ia"},
                spacing=SPACING,
            )


class TestRandSimpleFCDPipeline:
    """
    Drawn presets are forwarded to ``simple_fcd_simulator``.
    """

    def test_randomize_matches_draw_simple_fcd_params(
        self, presets: dict[str, Any]
    ) -> None:
        tfm = _transform(presets)
        tfm.set_random_state(state=np.random.RandomState(42))
        tfm.randomize()
        reference = draw_simple_fcd_params(
            sequences=["T1like", "T2like"],
            custom_presets=presets,
            random_state=np.random.RandomState(42),
        )
        assert tfm._params == reference

    def test_output_matches_simple_fcd_simulator(
        self, volume: dict, presets: dict[str, Any]
    ) -> None:
        tfm = _transform(presets)
        tfm.set_random_state(seed=0)
        out = tfm(volume)
        ref = simple_fcd_simulator(
            images=[volume["image1"], volume["image2"]],
            seg_mask=volume["seg_mask"],
            spacing=SPACING,
            growth_params=tfm._params["growth_params"],  # type: ignore[index]
            deformation_params=tfm._params["deformation_params"],  # type: ignore[index]
            intensity_params=tfm._params["intensity_params"],  # type: ignore[index]
            verbose=False,
        )
        np.testing.assert_allclose(out["image1"], ref["out_images"][0])
        np.testing.assert_allclose(out["image2"], ref["out_images"][1])
        np.testing.assert_array_equal(out["seg_mask"], ref["out_seg_mask"])
        np.testing.assert_array_equal(
            out["synthfcd_lesion"], ref["extras"]["out_target"]
        )
        np.testing.assert_array_equal(
            out["synthfcd_orig_target"], ref["extras"]["orig_target"]
        )
        assert out["synthfcd_stats"] == ref["stats"]

    def test_simulator_called_with_drawn_params(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        received: dict[str, Any] = {}

        def _capturing(images, seg_mask, spacing, **kwargs: Any) -> dict[str, Any]:
            received["n_images"] = len(images)
            received["kwargs"] = kwargs
            return _fake_simulator(images, seg_mask, spacing, **kwargs)

        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _capturing
        )
        tfm = _transform(presets)
        tfm.set_random_state(seed=1)
        tfm(volume)
        assert received["n_images"] == 2
        assert "FCD_type" not in received["kwargs"]
        assert set(received["kwargs"]) >= {
            "growth_params",
            "deformation_params",
            "intensity_params",
        }
        assert received["kwargs"]["growth_params"] == tfm._params["growth_params"]  # type: ignore[index]


class TestRandSimpleFCDDeterminism:
    """
    MONAI RNG seeding makes successive draws reproducible.
    """

    def test_same_seed_repeats(self, volume: dict, presets: dict[str, Any]) -> None:
        tfm = _transform(presets)

        tfm.set_random_state(seed=7)
        tfm.randomize()
        params_a = deepcopy(tfm._params)
        tfm.set_random_state(seed=7)
        tfm.randomize()
        assert tfm._params == params_a

        tfm.set_random_state(seed=7)
        out_a = deepcopy(tfm(volume))
        tfm.set_random_state(seed=7)
        out_b = tfm(volume)
        np.testing.assert_array_equal(out_a["image1"], out_b["image1"])
        np.testing.assert_array_equal(out_a["image2"], out_b["image2"])
        np.testing.assert_array_equal(out_a["seg_mask"], out_b["seg_mask"])
        np.testing.assert_array_equal(
            out_a["synthfcd_lesion"], out_b["synthfcd_lesion"]
        )
        np.testing.assert_array_equal(
            out_a["synthfcd_orig_target"], out_b["synthfcd_orig_target"]
        )
        assert out_a["synthfcd_stats"] == out_b["synthfcd_stats"]

    def test_rng_is_consumed(self, volume: dict, presets: dict[str, Any]) -> None:
        tfm = _transform(presets)

        tfm.set_random_state(seed=3)
        tfm.randomize()
        params_a = deepcopy(tfm._params)
        tfm.randomize()
        assert tfm._params != params_a

        tfm.set_random_state(seed=3)
        out_a = deepcopy(tfm(volume))
        out_b = tfm(volume)
        assert not np.array_equal(out_a["image1"], out_b["image1"])
        assert not np.array_equal(out_a["image2"], out_b["image2"])
        assert not np.array_equal(out_a["synthfcd_lesion"], out_b["synthfcd_lesion"])
        assert not np.array_equal(
            out_a["synthfcd_orig_target"], out_b["synthfcd_orig_target"]
        )


class TestRandSimpleFCDPreservationAndDtypes:
    """
    Unrelated keys, optional extras, and container dtypes survive the call.
    """

    def test_preserves_unrelated_keys_and_writes_extras(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        tfm = _transform(presets)
        out = tfm(volume)
        assert out["subject_id"] == "sub-01"
        assert set(out) >= {
            "image1",
            "image2",
            "seg_mask",
            "subject_id",
            "synthfcd_stats",
            "synthfcd_lesion",
            "synthfcd_orig_target",
        }

    def test_optional_keys_can_be_disabled(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        tfm = _transform(
            presets,
            out_stats_key=None,
            out_lesion_mask_key=None,
            orig_lesion_mask_key=None,
        )
        out = tfm(volume)
        assert "synthfcd_stats" not in out
        assert "synthfcd_lesion" not in out
        assert "synthfcd_orig_target" not in out

    def test_numpy_inputs_stay_numpy(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        out = _transform(presets)(volume)
        assert isinstance(out["image1"], np.ndarray)
        assert isinstance(out["seg_mask"], np.ndarray)
        assert np.issubdtype(out["seg_mask"].dtype, np.integer)

    def test_tensor_inputs_stay_tensors(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        data = {
            "image1": torch.from_numpy(volume["image1"]),
            "image2": torch.from_numpy(volume["image2"]),
            "seg_mask": torch.from_numpy(volume["seg_mask"]),
        }
        out = _transform(presets)(data)  # type: ignore[arg-type]
        assert type(out["image1"]) is torch.Tensor
        assert type(out["synthfcd_lesion"]) is torch.Tensor
        assert out["seg_mask"].dtype == torch.int32

    def test_metatensor_type_and_affine_preserved(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        data = {
            "image1": MetaTensor(volume["image1"], affine=affine),
            "image2": MetaTensor(volume["image2"], affine=affine),
            "seg_mask": MetaTensor(volume["seg_mask"], affine=affine),
        }
        meta = dict(data["image1"].meta)
        out = _transform(presets, spacing=None)(data)  # type: ignore[arg-type]
        assert isinstance(out["image1"], MetaTensor)
        assert out["image1"].meta == meta
        assert isinstance(out["synthfcd_lesion"], MetaTensor)


class TestRandSimpleFCDMONAIPipeline:
    """
    ``Compose`` + ``Dataset``/``DataLoader`` integration.
    """

    def test_compose_and_dataloader(
        self,
        volume: dict,
        presets: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "synthfcd.transforms.simple_fcd.simple_fcd_simulator", _fake_simulator
        )
        keys = ["image1", "image2", "seg_mask"]
        pipe = Compose(
            [
                ToTensord(keys=keys),
                _transform(presets),
                EnsureChannelFirstd(
                    keys=["image1", "image2", "seg_mask", "synthfcd_lesion"],
                    channel_dim="no_channel",
                ),
            ]
        )
        sample_b = deepcopy(volume)
        sample_b["subject_id"] = "sub-02"
        ds = Dataset([volume, sample_b], transform=pipe)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        assert set(batch) >= {
            "synthfcd_stats",
            "synthfcd_lesion",
            "synthfcd_orig_target",
        }
        assert batch["image1"].shape == (2, 1, *SHAPE)
        assert batch["synthfcd_lesion"].shape == (2, 1, *SHAPE)
        assert batch["seg_mask"].shape[0] == 2
        assert list(batch["subject_id"]) == ["sub-01", "sub-02"]
