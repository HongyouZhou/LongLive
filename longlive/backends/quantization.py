"""Backend selection for optional quantized execution paths.

This module only probes dependency availability. It does not import heavy CUDA
extensions, so it is safe on HPC environments that only provide Hopper/H200
BF16 training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hardware import GPUInfo, get_primary_gpu, has_module


BACKEND_NONE = "none"
BACKEND_AUTO = "auto"
BACKEND_NVFP4_FOUROVERSIX = "nvfp4_fouroversix"
BACKEND_NVFP4_TE = "nvfp4_te"

_ALIASES = {
    None: BACKEND_NONE,
    "": BACKEND_NONE,
    "false": BACKEND_NONE,
    "off": BACKEND_NONE,
    "disabled": BACKEND_NONE,
    "none": BACKEND_NONE,
    "bf16": BACKEND_NONE,
    "auto": BACKEND_AUTO,
    "nvfp4": BACKEND_AUTO,
    "fp4": BACKEND_AUTO,
    "fouroversix": BACKEND_NVFP4_FOUROVERSIX,
    "4o6": BACKEND_NVFP4_FOUROVERSIX,
    "nvfp4_4o6": BACKEND_NVFP4_FOUROVERSIX,
    "nvfp4_fouroversix": BACKEND_NVFP4_FOUROVERSIX,
    "te": BACKEND_NVFP4_TE,
    "transformer_engine": BACKEND_NVFP4_TE,
    "nvfp4_te": BACKEND_NVFP4_TE,
}


@dataclass(frozen=True)
class QuantizationProbe:
    gpu: GPUInfo | None
    fouroversix_available: bool
    transformer_engine_available: bool

    @property
    def is_blackwell(self) -> bool:
        return bool(self.gpu and self.gpu.capability[0] >= 10)


@dataclass(frozen=True)
class QuantizationSelection:
    backend: str
    enabled: bool
    reason: str
    probe: QuantizationProbe


def normalize_quant_backend(value: Any) -> str:
    key = str(value).strip().lower() if value is not None else None
    if key not in _ALIASES:
        valid = ", ".join(
            [
                BACKEND_NONE,
                BACKEND_AUTO,
                BACKEND_NVFP4_FOUROVERSIX,
                BACKEND_NVFP4_TE,
            ]
        )
        raise ValueError(f"Unknown quant backend {value!r}. Expected one of: {valid}")
    return _ALIASES[key]


def probe_quantization(device_index: int | None = None) -> QuantizationProbe:
    return QuantizationProbe(
        gpu=get_primary_gpu(device_index),
        fouroversix_available=has_module("fouroversix"),
        transformer_engine_available=has_module("transformer_engine"),
    )


def get_config_quant_backend(config: Any, default: str = BACKEND_NONE) -> str:
    """Read quant_backend from flat configs or nested config.infra mappings."""

    if config is None:
        return default
    if isinstance(config, str):
        return config

    infra = _get_value(config, "infra", None)
    if infra is not None:
        value = _get_value(infra, "quant_backend", None)
        if value is not None:
            return value

    for key in ("quant_backend", "model_quant_backend"):
        value = _get_value(config, key, None)
        if value is not None:
            return value

    # Compatibility with LongLive 2.0 style booleans. Keep this conservative:
    # model_quant=true without an explicit backend means "auto".
    model_quant = _get_value(config, "model_quant", None)
    if model_quant is None and infra is not None:
        model_quant = _get_value(infra, "model_quant", None)
    if bool(model_quant):
        return BACKEND_AUTO

    return default


def resolve_quant_backend(
    config_or_backend: Any = None,
    *,
    require: bool = False,
    device_index: int | None = None,
    probe: QuantizationProbe | None = None,
) -> QuantizationSelection:
    requested = normalize_quant_backend(get_config_quant_backend(config_or_backend))
    probe = probe or probe_quantization(device_index)

    if requested == BACKEND_NONE:
        return QuantizationSelection(
            backend=BACKEND_NONE,
            enabled=False,
            reason="quantization disabled",
            probe=probe,
        )

    if not probe.gpu:
        return _unavailable(requested, probe, require, "CUDA GPU is not available")

    if not probe.is_blackwell:
        arch = f"{probe.gpu.name} ({probe.gpu.sm}, {probe.gpu.architecture})"
        return _unavailable(
            requested,
            probe,
            require or requested != BACKEND_AUTO,
            f"NVFP4 backends require Blackwell; detected {arch}",
        )

    if requested == BACKEND_AUTO:
        if probe.fouroversix_available:
            return QuantizationSelection(
                backend=BACKEND_NVFP4_FOUROVERSIX,
                enabled=True,
                reason="auto selected FourOverSix NVFP4 on Blackwell",
                probe=probe,
            )
        if probe.transformer_engine_available:
            return QuantizationSelection(
                backend=BACKEND_NVFP4_TE,
                enabled=True,
                reason="auto selected TransformerEngine NVFP4 on Blackwell",
                probe=probe,
            )
        return _unavailable(
            requested,
            probe,
            require,
            "Blackwell GPU detected but neither fouroversix nor transformer_engine is installed",
        )

    if requested == BACKEND_NVFP4_FOUROVERSIX:
        if probe.fouroversix_available:
            return QuantizationSelection(
                backend=BACKEND_NVFP4_FOUROVERSIX,
                enabled=True,
                reason="FourOverSix NVFP4 requested and available",
                probe=probe,
            )
        return _unavailable(requested, probe, True, "fouroversix is not installed")

    if requested == BACKEND_NVFP4_TE:
        if probe.transformer_engine_available:
            return QuantizationSelection(
                backend=BACKEND_NVFP4_TE,
                enabled=True,
                reason="TransformerEngine NVFP4 requested and available",
                probe=probe,
            )
        return _unavailable(requested, probe, True, "transformer_engine is not installed")

    raise AssertionError(f"unhandled quant backend: {requested}")


def _get_value(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _unavailable(
    requested: str,
    probe: QuantizationProbe,
    strict: bool,
    reason: str,
) -> QuantizationSelection:
    if strict:
        raise RuntimeError(f"Quant backend {requested!r} is unavailable: {reason}")
    return QuantizationSelection(
        backend=BACKEND_NONE,
        enabled=False,
        reason=reason,
        probe=probe,
    )
