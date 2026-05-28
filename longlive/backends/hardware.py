"""Hardware and dependency probes for optional acceleration backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Optional


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_memory_gib: float | None = None

    @property
    def architecture(self) -> str:
        major, _ = self.capability
        if major >= 10:
            return "blackwell"
        if major == 9:
            return "hopper"
        if major == 8:
            return "ampere"
        if major == 7:
            return "volta/turing"
        return f"sm{major}"

    @property
    def sm(self) -> str:
        major, minor = self.capability
        return f"sm_{major}{minor}"


@dataclass(frozen=True)
class CudaInfo:
    torch_available: bool
    cuda_available: bool
    torch_version: str | None = None
    cuda_version: str | None = None
    devices: tuple[GPUInfo, ...] = ()
    current_device: int | None = None
    error: str | None = None


def has_module(module_name: str) -> bool:
    return find_spec(module_name) is not None


def get_cuda_info() -> CudaInfo:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return CudaInfo(
            torch_available=False,
            cuda_available=False,
            error=f"torch import failed: {type(exc).__name__}: {exc}",
        )

    torch_version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on local env
        return CudaInfo(
            torch_available=True,
            cuda_available=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            error=f"torch.cuda.is_available failed: {type(exc).__name__}: {exc}",
        )

    devices: list[GPUInfo] = []
    current_device: Optional[int] = None
    if cuda_available:
        try:
            current_device = int(torch.cuda.current_device())
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    GPUInfo(
                        index=index,
                        name=torch.cuda.get_device_name(index),
                        capability=tuple(torch.cuda.get_device_capability(index)),
                        total_memory_gib=props.total_memory / (1024**3),
                    )
                )
        except Exception as exc:  # pragma: no cover - depends on local env
            return CudaInfo(
                torch_available=True,
                cuda_available=False,
                torch_version=torch_version,
                cuda_version=cuda_version,
                error=f"CUDA device probe failed: {type(exc).__name__}: {exc}",
            )

    return CudaInfo(
        torch_available=True,
        cuda_available=cuda_available,
        torch_version=torch_version,
        cuda_version=cuda_version,
        devices=tuple(devices),
        current_device=current_device,
    )


def get_primary_gpu(device_index: int | None = None) -> GPUInfo | None:
    info = get_cuda_info()
    if not info.cuda_available or not info.devices:
        return None

    index = info.current_device if device_index is None else device_index
    if index is None:
        index = 0
    for gpu in info.devices:
        if gpu.index == index:
            return gpu
    return None


def is_blackwell(device_index: int | None = None) -> bool:
    gpu = get_primary_gpu(device_index)
    return bool(gpu and gpu.capability[0] >= 10)
