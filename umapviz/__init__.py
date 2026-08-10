"""UMAP educational implementation — high-D → 2-D embedding."""

from .algorithm import Snapshot, fit
from .data import (
    DATASET_KEYS,
    DATASET_NAMES,
    has_full_latent,
    latent_intrinsic_dim,
    make_dataset,
)

__all__ = [
    "Snapshot",
    "fit",
    "DATASET_KEYS",
    "DATASET_NAMES",
    "has_full_latent",
    "latent_intrinsic_dim",
    "make_dataset",
]
