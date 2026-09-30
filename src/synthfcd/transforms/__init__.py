"""
MONAI transforms for real-time execution of the simulation pipelines during training
(and/or inference) of deep learning models.
"""

from .simple_fcd import RandSimpleFCDd

__all__ = [
    "RandSimpleFCDd",
]
