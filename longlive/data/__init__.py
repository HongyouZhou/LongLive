"""Shared data loaders for method-independent training utilities."""

from .motion_refs import (
    GeneralPromptDataset,
    ReferenceVideoDataset,
    SkateboardingLatentDataset,
    make_reference_dataset,
)

__all__ = [
    "GeneralPromptDataset",
    "ReferenceVideoDataset",
    "SkateboardingLatentDataset",
    "make_reference_dataset",
]
