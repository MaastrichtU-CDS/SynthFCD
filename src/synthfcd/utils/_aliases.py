"""
Type aliases for internal use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
)

from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._validators import IntNoBool, RealNoBool

if TYPE_CHECKING:
    import numpy as np
    from monai.data.meta_tensor import MetaTensor
    from nifti_finder.explorers import FileFinder
    from torch import Tensor


# ----------------------------------------------------------------#
# Numbers
# ----------------------------------------------------------------#
_IntNoBoolType: TypeAlias = IntNoBool

_RealNoBoolType: TypeAlias = RealNoBool

# ----------------------------------------------------------------#
# Parameters
# ----------------------------------------------------------------#
_HemisphereType: TypeAlias = Literal["left", "right"]

_LabelType: TypeAlias = (
    int
    | str
    | _LabelEnum
    | list[int]
    | list[str]
    | list[_LabelEnum]
    | list[int | str | _LabelEnum]
    | tuple[int, ...]
    | tuple[str, ...]
    | tuple[_LabelEnum, ...]
    | tuple[int | str | _LabelEnum, ...]
    | set[int]
    | set[str]
    | set[_LabelEnum]
    | set[int | str | _LabelEnum]
)

_LabelEnumType: TypeAlias = type[_LabelEnum]

_LobeType: TypeAlias = _LabelType

_LesionType: TypeAlias = Literal["type_I", "type_IIa", "type_IIb", "type_bottom_sulcus"]

_MaskType: TypeAlias = Literal[
    "all", "gm", "wm", "ventricles", "subcortical", "background-csf"
]

_GrowthModeType: TypeAlias = Literal["distance", "irregular"]

_DistanceType: TypeAlias = int | float

_RNGType: TypeAlias = "np.random.Generator | np.random.RandomState"

_CacheAccessType: TypeAlias = bool | Literal["auto"]

_CacheKeyType: TypeAlias = str

_sMRIType: TypeAlias = Literal["T1like", "T2like"]

_FCDType: TypeAlias = Literal[
    "FCD_type_Ia",
    "FCD_type_Ib",
    "FCD_type_Ic",
    "FCD_type_IIa",
    "FCD_type_IIb",
]

_MONAIArrayType: TypeAlias = "MetaTensor | Tensor | np.ndarray"


# ----------------------------------------------------------------#
# Return types & protocols
# Level 1: Core growth functions & effects
# ----------------------------------------------------------------#
class _GrowTargetResultType(TypedDict):
    """
    Result of a target growth function.
    """

    target: np.ndarray
    growth_stats: dict[str, Any]


class _DeformationResultType(TypedDict):
    """
    Result of a deformation effect.
    """

    out_image: np.ndarray
    out_seg_mask: np.ndarray
    effect_field: list[tuple[np.ndarray, np.ndarray, np.ndarray]]


class _IntensityResultType(TypedDict):
    """
    Result of an intensity effect.
    """

    out_image: np.ndarray
    effect_field: np.ndarray


class _GrowTargetFnType(Protocol):
    """
    A function growing a lesion mask based on a segmentation mask.
    """

    def __call__(
        self,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        label_enum: _LabelEnumType,
        **kwargs: Any,
    ) -> _GrowTargetResultType:
        """
        Grow a lesion mask.
        """
        ...


class _EffectFnType(Protocol):
    """
    A function applying an effect to an image.
    """

    def __call__(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        *,
        label_enum: _LabelEnumType,
        **kwargs: Any,
    ) -> _DeformationResultType | _IntensityResultType:
        """
        Apply an effect to an image.
        """
        ...


# ----------------------------------------------------------------#
# Level 2: Simulation pipelines
# ----------------------------------------------------------------#
class _SimulationResultType(TypedDict):
    """
    Result of a simulation pipeline.

    Lesion simulation pipelines and in general any pipeline that grows a target could
    return the generated or/and output target in the ``"extras"`` dictionary.
    """

    out_images: list[np.ndarray]
    out_seg_mask: np.ndarray
    stats: dict[str, Any]
    extras: dict[str, Any]


class _SimulationFnType(Protocol):
    """
    A function simulating effects on a list of images.
    """

    def __call__(
        self,
        images: list[np.ndarray],
        seg_mask: np.ndarray,
        spacing: tuple[_DistanceType, _DistanceType, _DistanceType],
        **kwargs: Any,
    ) -> _SimulationResultType:
        """
        Simulate effects on a list of images.

        **kwargs could be any target growth, deformation, or intensity parameters, as
        well as verbosity and caching settings.
        """
        ...


# ----------------------------------------------------------------#
# Level 3: Parameter sampling
# ----------------------------------------------------------------#
class _ParamsDrawFnType(Protocol):
    """
    A function drawing parameters for a simulation function.
    """

    def __call__(
        self,
        *,
        random_seed: int | None,
        random_state: _RNGType | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Draw parameters for a simulation function using a random seed or state.

        The output dictionary is left free, as the simulation pipeline parameters are
        also free (other than input-specific parameters like ``images``, ``seg_mask``,
        and ``spacing``).

        Users developing such functions should make sure that the keys of the returned
        dictionary match the expected parameters of the intended simulation function.
        """
        ...


# ----------------------------------------------------------------#
# Level 4: Workflows
# ----------------------------------------------------------------#


class _WorkflowConfigType(TypedDict):
    """
    A parsed CLI workflow configuration.

    ``settings`` are keyword arguments forwarded to the workflow constructor (empty if
    omitted from the YAML). ``inputs`` is a path string or a list of path strings
    forwarded to ``workflow.run(inputs=...)``.
    """

    workflow: str
    settings: dict[str, Any]
    inputs: str | list[str]


class _SimpleFilterConfigType(TypedDict):
    name: str
    kwargs: dict[str, Any]


class _ComposeFilterKwargsType(TypedDict):
    filters: _FilterConfigType | list[_FilterConfigType | None]
    logic: Literal["AND", "OR"]


class _ComposeFilterConfigType(TypedDict):
    name: Literal["ComposeFilter"]
    kwargs: _ComposeFilterKwargsType


_FilterConfigType: TypeAlias = _SimpleFilterConfigType | _ComposeFilterConfigType

_GlobPatternType: TypeAlias = str | Sequence[str]


class _NiftiFinderConfigType(TypedDict, total=False):
    patterns: _GlobPatternType
    levels: dict[str, _GlobPatternType] | None
    filters: _FilterConfigType | None


_DataExplorerType: TypeAlias = "FileFinder"


_LogLevelType: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
