"""
Presets for parameter sampling and simulation pipelines.

The main "products" of this module are functions that generate parameter values for the
simulation pipelines.

As each simulation pipeline accepts its own set of parameters, no strict contract is
imposed on the outputs of these functions except that it must be a dictionary.

Most workflows will benefit from the use of random parameter values, drawn from
predefined distributions. For this, helper functions on specifying and consuming such
distributions are provided under the ``draw_params`` module.
"""

from .draw_params import *
from .simple_fcd import *

__all__ = [
    "draw_simple_fcd_params",
]
