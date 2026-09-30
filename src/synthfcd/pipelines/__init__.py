"""
Pipelines for sequential application of (lesional) target growth and effect functions on
images.

The main "ingredients" of this module are target growth and effect functions from the
`synthfcd.core` module.

The main "products" of this module are functional pipelines that accept a list of input
images, a segmentation mask, spacing, and additional parameters, to return a list of
output image, the output segmentation mask, and any additional statistics or metadata.

Most pipelines share common features, such as effect contracts, sequential application
of effects, restricting effects to a local region, caching of effect fields, etc. For
this, useful helpers, base classes, and mixins can be found in ``base.py``,
``apply_effects.py``, and ``grow_target.py``, and re-used for creating concrete
pipelines.
"""

from .apply_effects import AppliesEffects, apply_effects
from .grow_target import ToTarget, grow_target
from .simple_fcd import SimpleFCD, simple_fcd_simulator

__all__ = [
    "apply_effects",
    "grow_target",
    "simple_fcd_simulator",
]
