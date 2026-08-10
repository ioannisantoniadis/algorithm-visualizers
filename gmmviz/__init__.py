"""Gaussian mixture model (EM) educational implementation + visualiser."""

from .algorithm import Snapshot, fit
from .data import SHAPE_DEFAULTS, SHAPE_KEYS, SHAPE_NAMES, make_dataset

__all__ = [
    "Snapshot",
    "fit",
    "SHAPE_KEYS",
    "SHAPE_NAMES",
    "SHAPE_DEFAULTS",
    "make_dataset",
]
