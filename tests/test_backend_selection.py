"""CPU-only backend selection tests. Self-running like the other repo tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.backends.hardware import GPUInfo
from longlive.backends.quantization import (
    BACKEND_NONE,
    BACKEND_NVFP4_FOUROVERSIX,
    BACKEND_NVFP4_TE,
    QuantizationProbe,
    resolve_quant_backend,
)


def _probe(major, *, fouroversix=False, te=False):
    gpu = None
    if major is not None:
        gpu = GPUInfo(index=0, name=f"fake-sm{major}0", capability=(major, 0))
    return QuantizationProbe(
        gpu=gpu,
        fouroversix_available=fouroversix,
        transformer_engine_available=te,
    )


def test_default_is_disabled():
    selection = resolve_quant_backend(None, probe=_probe(9, fouroversix=True, te=True))
    assert selection.backend == BACKEND_NONE
    assert selection.enabled is False


def test_auto_hopper_falls_back_to_bf16():
    selection = resolve_quant_backend("auto", probe=_probe(9, fouroversix=True, te=True))
    assert selection.backend == BACKEND_NONE
    assert selection.enabled is False
    assert "Blackwell" in selection.reason


def test_auto_blackwell_prefers_fouroversix():
    selection = resolve_quant_backend("auto", probe=_probe(12, fouroversix=True, te=True))
    assert selection.backend == BACKEND_NVFP4_FOUROVERSIX
    assert selection.enabled is True


def test_auto_blackwell_can_select_te():
    selection = resolve_quant_backend("auto", probe=_probe(10, fouroversix=False, te=True))
    assert selection.backend == BACKEND_NVFP4_TE
    assert selection.enabled is True


def test_explicit_nvfp4_rejects_hopper():
    try:
        resolve_quant_backend("nvfp4_fouroversix", probe=_probe(9, fouroversix=True))
    except RuntimeError as exc:
        assert "require Blackwell" in str(exc)
        return
    raise AssertionError("expected RuntimeError for explicit NVFP4 on Hopper")


def test_explicit_nvfp4_requires_dependency():
    try:
        resolve_quant_backend("nvfp4_fouroversix", probe=_probe(12, fouroversix=False))
    except RuntimeError as exc:
        assert "fouroversix" in str(exc)
        return
    raise AssertionError("expected RuntimeError when fouroversix is missing")


def main():
    tests = [
        test_default_is_disabled,
        test_auto_hopper_falls_back_to_bf16,
        test_auto_blackwell_prefers_fouroversix,
        test_auto_blackwell_can_select_te,
        test_explicit_nvfp4_rejects_hopper,
        test_explicit_nvfp4_requires_dependency,
    ]
    print("running backend_selection tests:")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  PASS {test.__name__}")
    if failed:
        print(f"{failed} / {len(tests)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
