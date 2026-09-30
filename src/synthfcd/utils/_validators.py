"""
Helper functions for validating user-provided arguments.
"""

from __future__ import annotations

import abc
import numbers
from pathlib import Path
from typing import Any

import numpy as np


class ShapeMismatchError(ValueError):
    """
    Exception raised when shapes do not match.
    """


class _PredicateMeta(abc.ABCMeta):
    """
    Metaclass for predicate-based type validation.

    `_pred` gets modified by subclasses to define membership criteria.
    """

    def __instancecheck__(cls, obj: object) -> bool:
        pred = getattr(cls, "_pred", None)
        if pred is None:
            return False
        return bool(pred(obj))


class IntNoBool(metaclass=_PredicateMeta):
    """
    Instance-check type: All integers except booleans.
    """

    _pred = staticmethod(
        lambda obj: isinstance(obj, numbers.Integral)
        and not isinstance(obj, (bool, np.bool_))
    )


class RealNoBool(metaclass=_PredicateMeta):
    """
    Instance-check type: All real numbers except booleans.
    """

    _pred = staticmethod(
        lambda obj: isinstance(obj, numbers.Real)
        and not isinstance(obj, (bool, np.bool_))
    )


def validate_dict_and_keys(
    obj: Any,
    /,
    name: str,
    keys: tuple[str, ...],
    *,
    strict_match: bool = False,
) -> None:
    """
    Validate that an object is a dictionary and contains all the required keys.

    Args:
        obj: The object to validate.
        name: The name of the object.
        keys: The required keys.
        strict_match: Whether to require an exact match of keys.

    Raises:
        TypeError: If the object is not a dictionary.
        ValueError: If the object does not contain all the required keys.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be a dictionary. Got {type(obj).__name__}")

    required = set(keys)
    present = set(obj.keys())

    missing = required - present
    extra = present - required

    if missing:
        missing_s = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required keys: {missing_s}")

    if strict_match and extra:
        extra_s = ", ".join(sorted(extra))
        required_s = ", ".join(keys)
        raise ValueError(
            f"{name} must contain the exact keys: {required_s}. "
            f"Unexpected keys: {extra_s}"
        )


def validate_3d_numpy_array(
    obj: Any,
    /,
    name: str,
    dtype: type | tuple[type, ...] | None = None,
    shape: tuple[int, ...] | None = None,
) -> None:
    """
    Validate that an object is a 3D numpy array.

    Args:
        obj: The object to validate.
        name: The name of the object.
        dtype: The target dtype(s) of the numpy array.
        shape: The target shape of the numpy array.

    Raises:
        TypeError: If the object is not a numpy array.
        ValueError: If the array is not 3D.
        ValueError: If the array is not of the specified `dtype`.
        ShapeMismatchError: If the array does not have the specified shape.
    """
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"{name} must be a numpy array. Got {type(obj).__name__}.")

    if obj.ndim != 3:
        raise ValueError(f"{name} must be a 3D array. Got {obj.ndim}D.")

    if dtype is not None:
        if not isinstance(dtype, tuple):
            dtype = (dtype,)

        if not any(np.issubdtype(obj.dtype, d) for d in dtype):
            raise ValueError(
                f"{name} must be a 3D array of dtype(s) "
                f"{', '.join(str(d) for d in dtype)}. "
                f"Got {obj.dtype}."
            )

    if shape is not None and obj.shape != shape:
        raise ShapeMismatchError(f"{name} must have shape {shape}. Got {obj.shape}.")


def validate_masks_dict(
    masks: Any,
    /,
    keys: tuple[str, ...],
    dtype: type | tuple[type, ...] | None = None,
) -> None:
    """
    Perform validation checks on the `masks` entry:

    - Must be a dictionary.
    - Must contain all the required keys.
    - Must be 3D boolean numpy arrays.
    - Array elements must be of the specified dtype(s).
    - All masks must have the same shape.

    Args:
        masks: The masks to validate.
        keys: The required keys.
        dtype: The target dtype(s) of the numpy arrays.

    Raises:
        TypeError: If the `masks` entry is not a dictionary.
        ValueError: If the `masks` entry does not contain all the required keys.
        TypeError: If values are not 3D numpy arrays.
        ValueError: If array elements are not of the specified dtype(s).
        ValueError: If arrays have different shapes.
    """
    validate_dict_and_keys(masks, "masks", keys)

    for k in keys:
        if k == keys[0]:
            validate_3d_numpy_array(masks[k], k, dtype)
            continue
        try:
            validate_3d_numpy_array(masks[k], k, dtype, masks[keys[0]].shape)
        except ShapeMismatchError:
            raise ValueError(
                f"All masks must have the same shape. "
                f"Got {masks[k].shape} for {k} and {masks[keys[0]].shape} "
                f"for {keys[0]}."
            )


def validate_obj_type(
    obj: Any,
    /,
    name: str,
    target_type: type | tuple[type, ...],
) -> None:
    """
    Validate that an object is of a specific type.

    Args:
        obj: The object to validate.
        name: The name of the object.
        target_type: The target type(s) of the object.

    Raises:
        TypeError: If the object is not of the specified type(s).
    """
    if not isinstance(target_type, tuple):
        target_type = (target_type,)

    if not any(isinstance(obj, t) for t in target_type):
        raise TypeError(
            f"{name} must be of type(s) "
            f"{', '.join(str(t) for t in target_type)}. "
            f"Got {type(obj).__name__}."
        )


def validate_literal_str(
    obj: Any,
    /,
    name: str,
    target_literals: tuple[str, ...],
) -> None:
    """
    Validate that an object is a literal.

    Args:
        obj: The object to validate.
        name: The name of the object.
        target_literals: The target literal(s) of the object.

    Raises:
        TypeError: If the object is not a string.
        ValueError: If the object is not one of the target literals.
    """
    validate_obj_type(obj, name, str)

    if obj not in target_literals:
        raise ValueError(f"{name} must be one of {target_literals}. Got {obj}.")


def validate_class_type(
    obj: Any,
    /,
    name: str,
    target_type: type | tuple[type, ...],
) -> None:
    """
    Validate that an object is a class and is a subclass of the target type(s).

    Args:
        obj: The object to validate.
        name: The name of the object.
        target_type: The target base class or classes.

    Raises:
        TypeError: If the object is not a class or not a subclass of the target type(s).
    """
    if not isinstance(target_type, tuple):
        target_type = (target_type,)

    if not isinstance(obj, type):
        raise TypeError(f"{name} must be a class. Got {type(obj).__name__}.")

    if not any(issubclass(obj, t) for t in target_type):
        raise TypeError(
            f"{name} must be a subclass of "
            f"{', '.join(str(t) for t in target_type)}. "
            f"Got {obj.__name__}."
        )


def validate_spacing(
    obj: Any,
    /,
    name: str,
) -> None:
    """
    Validate that a spacing tuple is valid.

    Args:
        obj: The object to validate.
        name: The name of the object.

    Raises:
        TypeError: If `obj` is not a tuple.
        ValueError: If `obj` does not contain 3 elements.
        ValueError: If `obj` elements are not positive floats/integers.
    """
    validate_obj_type(obj, name, tuple)
    if len(obj) != 3:
        raise ValueError(f"Spacing from {name} must contain 3 elements")
    if not all(isinstance(s, RealNoBool) for s in obj) or not all(
        float(s) > 0.0 for s in obj
    ):
        raise ValueError(
            f"Spacing from {name} must be a tuple of positive floats or integers"
        )


def validate_file_exists(
    filepath: str | Path,
    /,
    name: str,
) -> None:
    """
    Validate that a file exists.

    Args:
        filepath: The filepath to validate.
        name: The name of the entity that is represented by the file.

    Raises:
        TypeError: If `filepath` is not a string or Path.
        FileNotFoundError: If `filepath` does not exist.
    """
    validate_obj_type(filepath, name, (str, Path))
    if not Path(filepath).exists():
        raise FileNotFoundError(f"file path for {name} does not exist: {filepath}.")
