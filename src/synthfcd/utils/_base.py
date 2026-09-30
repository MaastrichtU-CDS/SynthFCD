"""
Base classes and interfaces for internal use.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import EnumMeta, IntEnum


class ABCEnumMeta(EnumMeta, ABCMeta):
    """
    Metaclass for enumerations.
    """


class _LabelEnum(IntEnum, metaclass=ABCEnumMeta):
    """
    Base class for label enumerations.
    """

    @classmethod
    @abstractmethod
    def get_all_labels(
        cls,
        hemisphere: str | None = None,
        include_neutral: bool = False,
    ) -> set[int]:
        """
        Return all label values; can limit to specific hemisphere.

        Can optionally include neutral (hemisphere-agnostic) labels when choosing a
        hemisphere via the flag `include_neutral`, except for the background label.
        """

    @classmethod
    @abstractmethod
    def get_cortical_labels(
        cls,
        hemisphere: str | None = None,
    ) -> set[int]:
        """
        Return all cortical labels; can limit to specific hemisphere.
        """

    @classmethod
    @abstractmethod
    def get_subcortical_labels(
        cls,
        hemisphere: str | None = None,
    ) -> set[int]:
        """
        Return all labels of subcortical structures; can limit to specific hemisphere.
        """

    @classmethod
    @abstractmethod
    def get_white_matter_labels(
        cls,
        hemisphere: str | None = None,
    ) -> set[int]:
        """
        Return all white matter labels; can limit to specific hemisphere.
        """

    @classmethod
    @abstractmethod
    def get_background_labels(cls) -> set[int]:
        """
        Return background label values (typically outside the segmentation).
        """

    @classmethod
    @abstractmethod
    def get_background_csf_labels(
        cls,
    ) -> set[int]:
        """
        Return all background and CSF labels.
        """

    @classmethod
    @abstractmethod
    def get_ventricle_labels(
        cls,
        hemisphere: str | None = None,
        lateral_only: bool = False,
    ) -> set[int]:
        """
        Return all lateral ventricle labels; can limit to specific hemisphere.

        Can optionally include 3rd and 4th ventricle labels by setting `lateral_only` to
        False.
        """

    @classmethod
    @abstractmethod
    def get_frontal_lobe_labels(
        cls,
        hemisphere: str | None = None,
    ) -> set[int]:
        """
        Return all frontal lobe labels; can limit to specific hemisphere.
        """

    @classmethod
    @abstractmethod
    def get_temporal_lobe_labels(
        cls,
        hemisphere: str | None = None,
    ) -> set[int]:
        """
        Return all temporal lobe labels; can limit to specific hemisphere.
        """

    @property
    def readable_name(self) -> str:
        """
        Return a human-readable version of the label name.
        """
        return self.name.replace("_", " ").title()

    @classmethod
    def from_readable_name(cls, name: str) -> _LabelEnum | None:
        """
        Get a label from a human-readable name (case-insensitive).
        """
        enum_name = name.strip().upper().replace(" ", "_")
        try:
            return cls[enum_name]
        except KeyError:
            return None
