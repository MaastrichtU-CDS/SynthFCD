"""
Workflows for simulating effects on images and segmentation masks.

Support flexible file exploration using nifti-finder's explorers and parallel processing
execution, as well as parameter tracking and deterministic control.
"""

from __future__ import annotations

__all__ = [
    "SimulationWorkflow",
]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# likely for some polymorphism in the future; ignore for now
_ALLOWED_PIPELINE_FNS: tuple[str, ...] = ("simple_fcd_simulator",)

_ALLOWED_PARAMS_DRAW_FNS: tuple[str, ...] = ()


class SimulationWorkflow(ABC):
    """
    Base class for all simulation workflows; empty now but shows intent.
    """

    @abstractmethod
    def run(
        self,
        inputs: str | Path | list[str | Path],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run the workflow end-to-end over the provided inputs.
        """

    def __call__(
        self,
        inputs: str | Path | list[str | Path],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Invoke :meth:`run` so the workflow can be used as a callable.
        """
        return self.run(inputs, dry_run)
