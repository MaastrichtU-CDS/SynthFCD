"""
Utility functions for transforms.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing

from synthfcd.utils._aliases import _DistanceType, _MONAIArrayType


def _arr_to_numpy(arr: _MONAIArrayType, /) -> tuple[np.ndarray, dict[str, Any] | None]:
    """
    Convert a PyTorch or MONAI MetaTensor to a NumPy array.

    Args:
        arr (MetaTensor | torch.Tensor | np.ndarray):
            The array to convert.

    Returns:
        A tuple containing the NumPy array and the metadata
        (if originally a MONAI MetaTensor, otherwise ``None``).

    Raises:
        TypeError: If the array is not a PyTorch tensor, a NumPy array,
        or a MONAI MetaTensor.
    """
    if isinstance(arr, MetaTensor):
        return arr.array, arr.meta
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy(), None
    elif isinstance(arr, np.ndarray):
        return arr, None
    else:
        raise TypeError(f"Unsupported type {type(arr)}")


def _arr_from_numpy(
    arr: np.ndarray,
    /,
    *,
    meta: dict[Hashable, Any] | None = None,
    orig_type: type | None = None,
    device: torch.device | None = None,
    key: Hashable | None = None,
) -> _MONAIArrayType:
    """
    Convert a NumPy array back to a PyTorch or MONAI MetaTensor.

    Args:
        arr (np.ndarray):
            The NumPy array.
        meta (dict[Hashable, Any] | None):
            The metadata of the array (if originally a MONAI MetaTensor).
        orig_type (type | None):
            The original type of the array
            (e.g., ``MetaTensor``, ``torch.Tensor``, ``np.ndarray``).
        device (torch.device | None):
            The device to move the tensor to (if originally a PyTorch tensor).

    Returns:
        A PyTorch tensor, a NumPy array, or a MONAI MetaTensor
        (depending on the original).
    """
    if orig_type == MetaTensor and meta is not None:
        return MetaTensor(arr, meta=meta)
    elif orig_type == torch.Tensor:
        tensor = torch.from_numpy(arr)
        return tensor.to(device) if device else tensor
    return arr


def _maybe_spacing_from_metatensors(
    arr: _MONAIArrayType,
    /,
) -> tuple[float, float, float] | None:
    """
    Infer the voxel spacing from the metadata of a MONAI MetaTensor.

    Args:
        arr (MetaTensor | torch.Tensor | np.ndarray):
            The MONAI MetaTensor to infer the voxel spacing from.

    Returns:
        The voxel spacing as a tuple of 3 floats.

    Raises:
        ValueError: If the voxel spacing is not 3D.
    """
    if not isinstance(arr, MetaTensor):
        return None

    spacing = tuple(float(s) for s in affine_to_spacing(arr.affine))
    if len(spacing) != 3:
        raise ValueError(f"Expected 3D spacing, got {len(spacing)}D: {spacing}.")
    return spacing


def _matches_spacing(
    a: tuple[_DistanceType, _DistanceType, _DistanceType],
    b: tuple[_DistanceType, _DistanceType, _DistanceType],
    /,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Check if two spacing tuples are equal within floating-point tolerance.
    """
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))
