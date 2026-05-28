#!/usr/bin/env python
"""Print optional backend availability for the current machine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from longlive.backends.hardware import get_cuda_info
from longlive.backends.quantization import resolve_quant_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()

    info = get_cuda_info()
    print(f"torch_available: {info.torch_available}")
    print(f"cuda_available:  {info.cuda_available}")
    print(f"torch_version:   {info.torch_version}")
    print(f"cuda_version:    {info.cuda_version}")
    if info.error:
        print(f"probe_error:     {info.error}")
    for gpu in info.devices:
        mem = "unknown" if gpu.total_memory_gib is None else f"{gpu.total_memory_gib:.1f} GiB"
        print(
            f"gpu[{gpu.index}]:       {gpu.name} "
            f"capability={gpu.capability[0]}.{gpu.capability[1]} "
            f"{gpu.sm} arch={gpu.architecture} memory={mem}"
        )

    try:
        selection = resolve_quant_backend(
            args.backend,
            require=args.require,
            device_index=args.device,
        )
    except RuntimeError as exc:
        print(f"quant_backend:   ERROR: {exc}")
        return 2

    print(f"fouroversix:     {selection.probe.fouroversix_available}")
    print(f"transformer_eng: {selection.probe.transformer_engine_available}")
    print(f"quant_backend:   {selection.backend}")
    print(f"quant_enabled:   {selection.enabled}")
    print(f"reason:          {selection.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
