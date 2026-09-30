"""
Tests for ``MONAIBase``: numpy/tensor conversion and metadata/spacing handling.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pytest
import torch
from monai.data.meta_tensor import MetaTensor

from synthfcd.transforms.base import MONAIBase


class _StubMONAI(MONAIBase):
    """
    Concrete ``MONAIBase`` so conversion helpers can be unit-tested.
    """

    def __call__(self, data: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
        return data


def _affine(spacing: tuple[float, float, float]) -> torch.Tensor:
    return torch.diag(torch.tensor([*spacing, 1.0], dtype=torch.float64))


def _base(**kwargs: Any) -> _StubMONAI:
    defaults: dict[str, Any] = dict(
        keys=["image"],
        seg_mask_key="seg_mask",
        spacing=(1.0, 1.0, 1.0),
    )
    defaults.update(kwargs)
    return _StubMONAI(**defaults)


class TestMONAIBaseInit:
    """
    Construction-time validation.
    """

    def test_seg_mask_key_cannot_overlap_image_keys(self) -> None:
        with pytest.raises(ValueError, match="cannot be one of"):
            _StubMONAI(keys=["image", "seg_mask"], seg_mask_key="seg_mask")

    def test_invalid_spacing(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            _StubMONAI(keys=["image"], seg_mask_key="seg", spacing=(1.0, 1.0))  # type: ignore[arg-type]


class TestNumpyTensorRoundtrip:
    """
    ``image_to_numpy`` / ``numpy_to_image`` restore type, device, and values.
    """

    def test_numpy_passthrough(self) -> None:
        tfm = _base()
        arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        out = tfm.image_to_numpy(arr, "image")
        np.testing.assert_array_equal(out, arr)
        back = tfm.numpy_to_image(out, "image")
        assert isinstance(back, np.ndarray)
        np.testing.assert_array_equal(back, arr)

    def test_torch_tensor_cpu_roundtrip(self) -> None:
        tfm = _base()
        tensor = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        arr = tfm.image_to_numpy(tensor, "image")
        assert isinstance(arr, np.ndarray)
        back = tfm.numpy_to_image(arr, "image")
        assert isinstance(back, torch.Tensor)
        assert not isinstance(back, MetaTensor)
        assert back.device == tensor.device
        assert back.dtype == tensor.dtype
        torch.testing.assert_close(back, tensor)

    def test_metatensor_preserves_meta(self) -> None:
        tfm = _base(spacing=None)
        affine = _affine((1.0, 2.0, 3.0))
        mt = MetaTensor(
            np.ones((3, 3, 3), dtype=np.float32),
            affine=affine,
        )
        meta_before = dict(mt.meta)
        arr = tfm.image_to_numpy(mt, "image")
        assert isinstance(arr, np.ndarray)
        assert tfm._meta["image"]["spacing"] == (1.0, 2.0, 3.0)
        back = tfm.numpy_to_image(arr, "image")
        assert isinstance(back, MetaTensor)
        assert back.meta == meta_before

    def test_second_convert_without_clear_raises(self) -> None:
        tfm = _base()
        tfm.image_to_numpy(np.zeros((2, 2, 2)), "image")
        with pytest.raises(ValueError, match="already exists"):
            tfm.image_to_numpy(np.zeros((2, 2, 2)), "image")

    def test_numpy_to_image_requires_prior_convert(self) -> None:
        tfm = _base()
        with pytest.raises(ValueError, match="No metadata"):
            tfm.numpy_to_image(np.zeros((2, 2, 2)), "image")

    def test_clear_metadata(self) -> None:
        tfm = _base()
        tfm.image_to_numpy(np.zeros((2, 2, 2)), "image")
        tfm.clear_metadata()
        assert tfm._meta == {}
        # can convert again after clear
        tfm.image_to_numpy(np.ones((2, 2, 2)), "image")

    def test_unsupported_type_raises(self) -> None:
        tfm = _base()
        with pytest.raises(TypeError, match="Unsupported"):
            tfm.image_to_numpy([1, 2, 3], "image")  # type: ignore[arg-type]


class TestResolveInputs:
    """
    Spacing resolution from init, dict key, and MetaTensor metadata.
    """

    @pytest.fixture
    def volume(self) -> dict[str, np.ndarray]:
        image = np.zeros((4, 4, 4), dtype=np.float32)
        seg = np.ones((4, 4, 4), dtype=np.int32)
        return {"image": image, "seg_mask": seg}

    def test_uses_init_spacing(self, volume: dict) -> None:
        tfm = _base(spacing=(1.0, 1.0, 1.5))
        images, seg, spacing = tfm.resolve_inputs(volume)
        assert len(images) == 1
        assert images[0].shape == (4, 4, 4)
        assert seg.dtype == np.int32
        assert spacing == (1.0, 1.0, 1.5)

    def test_uses_spacing_key(self, volume: dict) -> None:
        tfm = _base(spacing=None, spacing_key="spacing")
        volume["spacing"] = (0.8, 0.8, 1.0)
        _, _, spacing = tfm.resolve_inputs(volume)
        assert spacing == (0.8, 0.8, 1.0)

    def test_infers_spacing_from_metatensor(self) -> None:
        tfm = _base(spacing=None)
        affine = _affine((1.0, 2.0, 3.0))
        data = {
            "image": MetaTensor(np.zeros((4, 4, 4), np.float32), affine=affine),
            "seg_mask": MetaTensor(np.ones((4, 4, 4), np.int32), affine=affine),
        }
        _, _, spacing = tfm.resolve_inputs(data)  # type: ignore[arg-type]
        assert spacing == (1.0, 2.0, 3.0)

    def test_no_spacing_raises(self, volume: dict) -> None:
        tfm = _base(spacing=None)
        with pytest.raises(ValueError, match="No spacing"):
            tfm.resolve_inputs(volume)

    def test_missing_seg_mask_raises(self) -> None:
        tfm = _base()
        with pytest.raises(ValueError, match="Segmentation mask key"):
            tfm.resolve_inputs({"image": np.zeros((2, 2, 2))})

    def test_missing_spacing_key_raises(self, volume: dict) -> None:
        tfm = _base(spacing_key="spacing")
        with pytest.raises(ValueError, match="Spacing key"):
            tfm.resolve_inputs(volume)

    def test_spacing_key_mismatch_with_init(self, volume: dict) -> None:
        tfm = _base(spacing=(1.0, 1.0, 1.0), spacing_key="spacing")
        volume["spacing"] = (0.5, 0.5, 0.5)
        with pytest.raises(ValueError, match="does not match predefined"):
            tfm.resolve_inputs(volume)

    def test_metatensor_mismatch_with_init(self) -> None:
        tfm = _base(spacing=(1.0, 1.0, 1.0))
        affine = _affine((2.0, 2.0, 2.0))
        data = {
            "image": MetaTensor(np.zeros((3, 3, 3), np.float32), affine=affine),
            "seg_mask": np.ones((3, 3, 3), dtype=np.int32),
        }
        with pytest.raises(ValueError, match="does not match predefined"):
            tfm.resolve_inputs(data)  # type: ignore[arg-type]

    def test_metatensor_mismatch_across_keys(self) -> None:
        tfm = _base(spacing=None)
        data = {
            "image": MetaTensor(
                np.zeros((3, 3, 3), np.float32), affine=_affine((1.0, 1.0, 1.0))
            ),
            "seg_mask": MetaTensor(
                np.ones((3, 3, 3), np.int32), affine=_affine((2.0, 2.0, 2.0))
            ),
        }
        with pytest.raises(ValueError, match="Spacing mismatch"):
            tfm.resolve_inputs(data)  # type: ignore[arg-type]

    def test_resolve_clears_stale_metadata(self, volume: dict) -> None:
        tfm = _base()
        tfm.image_to_numpy(volume["image"], "image")
        tfm.resolve_inputs(volume)
        assert set(tfm._meta) == {"image", "seg_mask"}
