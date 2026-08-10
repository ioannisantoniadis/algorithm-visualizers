"""Educational Principal Component Analysis (2-D) with step-by-step snapshots."""

from .algorithm import Snapshot, fit
from .data import SHAPE_KEYS, SHAPE_NAMES, make_dataset

__all__ = [
    "Snapshot",
    "SHAPE_KEYS",
    "SHAPE_NAMES",
    "fit",
    "make_dataset",
]
