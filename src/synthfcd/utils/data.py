"""
Helper functions for data handling.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "get_ext",
    "resolve_path",
]


def resolve_path(path: Path | str) -> Path:
    """
    Expand user and resolve path.
    """
    return Path(path).expanduser().resolve()


def get_ext(path: Path | str) -> str:
    """
    Get file extension from filepath with leading dot.
    """
    p = Path(path)
    suffixes = p.suffixes
    full_ext = "".join(suffixes)
    return full_ext
