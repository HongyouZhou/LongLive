"""Lazy NVFP4 dependency loaders."""

from __future__ import annotations

from types import ModuleType

from .quantization import (
    BACKEND_NVFP4_FOUROVERSIX,
    BACKEND_NVFP4_TE,
    resolve_quant_backend,
)


def require_fouroversix() -> ModuleType:
    try:
        import fouroversix
    except ImportError as exc:
        raise RuntimeError(
            "FourOverSix NVFP4 backend requested, but package 'fouroversix' is not installed. "
            "On RTX Blackwell use CUDA_ARCHS=120 when building it."
        ) from exc
    return fouroversix


def require_transformer_engine() -> ModuleType:
    try:
        import transformer_engine
    except ImportError as exc:
        raise RuntimeError(
            "TransformerEngine NVFP4 backend requested, but package 'transformer_engine' is not installed."
        ) from exc
    return transformer_engine


def ensure_nvfp4_backend(config_or_backend, *, device_index: int | None = None):
    selection = resolve_quant_backend(
        config_or_backend,
        require=True,
        device_index=device_index,
    )
    if selection.backend == BACKEND_NVFP4_FOUROVERSIX:
        return selection, require_fouroversix()
    if selection.backend == BACKEND_NVFP4_TE:
        return selection, require_transformer_engine()
    raise RuntimeError(f"Expected an NVFP4 backend, got {selection.backend!r}")
