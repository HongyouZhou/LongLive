"""Shared data loaders for method-independent training utilities."""

from .motion_refs import GeneralPromptDataset, SkateboardingLatentDataset

__all__ = [
    "GeneralPromptDataset",
    "SkateboardingLatentDataset",
]
