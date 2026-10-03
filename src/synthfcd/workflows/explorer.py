"""
Factory for instantiating a data explorer with user-provided settings.
"""

from __future__ import annotations

__all__ = [
    "get_data_explorer",
]

import importlib
from collections.abc import Sequence
from typing import Any, cast

from nifti_finder.explorers import FileFinder
from nifti_finder.filters import Filter

from synthfcd.utils._aliases import (
    _ComposeFilterKwargsType,
    _DataExplorerType,
    _FilterConfigType,
)

_FILTER_MODULES: tuple[str, ...] = ("nifti_finder.filters",)


def _get_filter_name_and_kwargs(entry: _FilterConfigType) -> dict[str, Any]:
    if "name" not in entry:
        raise ValueError("`name` key is required in filter entry")
    if "kwargs" not in entry:
        raise ValueError("`kwargs` key is required in filter entry")
    return {"filter_name": entry["name"], "kwargs": entry["kwargs"]}


def _get_filter_obj(filter_name: str, kwargs: dict[str, Any]) -> Filter:
    for module_name in _FILTER_MODULES:
        try:
            module = importlib.import_module(module_name)
            filter_cls = getattr(module, filter_name)
            return filter_cls(**kwargs)
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ImportError(
        f"Filter '{filter_name}' not found in any known modules: {list(_FILTER_MODULES)}"
    )


def _build_filter(filters: _FilterConfigType | None) -> Filter | None:
    """
    Recursively instantiate filters from a nested mapping.

    Accepts the same shape your existing factory uses:
      - ``None``                                       -> no filter
      - ``{name: "<ClassName>", kwargs: {...}}``       -> single filter
      - ``{name: "ComposeFilter", kwargs: {filters: <single|list>, logic: "AND"|"OR"}}``
    """
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ValueError(f"Invalid type for `filters`: {type(filters).__name__}")

    name_and_kwargs = _get_filter_name_and_kwargs(filters)

    if name_and_kwargs["filter_name"] == "ComposeFilter":
        inner = cast(_ComposeFilterKwargsType, name_and_kwargs["kwargs"])["filters"]
        if isinstance(inner, dict):
            return _build_filter(inner)
        if isinstance(inner, list):
            new_kwargs = dict(name_and_kwargs["kwargs"])
            composed: list[Filter] = []
            for entry in inner:
                if entry is None:
                    continue
                composed.append(cast(Filter, _build_filter(entry)))
            new_kwargs["filters"] = composed
            return _get_filter_obj("ComposeFilter", new_kwargs)
        raise ValueError(
            f"Invalid type for `filters` in `ComposeFilter`: {type(inner).__name__}"
        )

    return _get_filter_obj(**name_and_kwargs)


def _validate_patterns(patterns: str | Sequence[str]) -> str | list[str]:
    if isinstance(patterns, str):
        return patterns
    if isinstance(patterns, Sequence) and not isinstance(patterns, (bytes, bytearray)):
        items = list(patterns)
        if not items:
            raise ValueError("`patterns` must be a non-empty string or sequence")
        if any(not isinstance(item, str) for item in items):
            raise ValueError("`patterns` must be a string or sequence of strings")
        return items
    raise ValueError(
        f"`patterns` must be a string or sequence of strings, got "
        f"{type(patterns).__name__}"
    )


def _validate_levels(
    levels: dict[str, str | Sequence[str]] | None,
) -> dict[str, str | list[str]] | None:
    """
    Validate ``levels``; ``None`` means flat recursive search (FileFinder default).
    """
    if levels is None:
        return None
    if not isinstance(levels, dict):
        raise ValueError(
            f"`levels` must be a dictionary or None, got {type(levels).__name__}"
        )
    if not levels:
        raise ValueError(
            "`levels` must not be empty; omit `levels` (or pass None) for flat "
            "recursive search"
        )

    normalized: dict[str, str | list[str]] = {}
    for key, value in levels.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"`levels` keys must be non-empty strings, got {key!r}")
        if isinstance(value, str):
            normalized[key] = value
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items = list(value)
            if not items or any(not isinstance(item, str) for item in items):
                raise ValueError(
                    f"`levels[{key!r}]` must be a string or non-empty sequence of strings"
                )
            normalized[key] = items
            continue
        raise ValueError(
            f"`levels[{key!r}]` must be a string or sequence of strings, got "
            f"{type(value).__name__}"
        )
    return normalized


def get_data_explorer(
    patterns: str | Sequence[str] = "*.nii*",
    levels: dict[str, str | Sequence[str]] | None = None,
    filters: _FilterConfigType | None = None,
) -> _DataExplorerType:
    r"""
    Instantiate nifti-finder's :class:`~nifti_finder.explorers.FileFinder`.

    Args:
        patterns: Glob pattern or list of patterns (default ``"*.nii*"``).
        levels: Optional named directory-traversal levels. ``None`` (default)
            enables flat recursive search. An empty mapping is rejected.
        filters: Optional filter config mapping (``{name, kwargs}``). A list of
            filters is not accepted here; combine filters via ``ComposeFilter``
            and its own ``logic`` kwarg.

    Returns:
        A :class:`~nifti_finder.explorers.FileFinder` instance.
    """
    validated_patterns = _validate_patterns(patterns)
    validated_levels = _validate_levels(levels)
    built_filters = _build_filter(filters) if filters is not None else None

    return FileFinder(
        patterns=validated_patterns,
        levels=validated_levels,
        filters=built_filters,
    )
