"""Optional hardware backend helpers.

The package is intentionally lightweight: importing it must not require CUDA
extensions such as FourOverSix or TransformerEngine. Those dependencies are
loaded lazily by the backend that actually needs them.
"""

from .hardware import CudaInfo, GPUInfo, get_cuda_info, get_primary_gpu, is_blackwell
from .quantization import (
    QuantizationProbe,
    QuantizationSelection,
    resolve_quant_backend,
)

__all__ = [
    "CudaInfo",
    "GPUInfo",
    "QuantizationProbe",
    "QuantizationSelection",
    "get_cuda_info",
    "get_primary_gpu",
    "is_blackwell",
    "resolve_quant_backend",
]
