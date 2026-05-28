"""CPU-only tests for checkpoint/adapter mechanical helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.utils.checkpoints import (
    cast_fp32_params_to_bf16,
    clean_fsdp_key,
    clean_state_dict_keys,
    find_adapter_params,
    select_generator_state_dict,
    select_lora_state_dict,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.ModuleDict({
            "default": nn.Linear(2, 2),
            "anchor": nn.Linear(2, 2),
        })


def test_clean_fsdp_key():
    name = "_fsdp_wrapped_module._checkpoint_wrapped_module._orig_mod.model.q.weight"
    assert clean_fsdp_key(name) == "model.q.weight"


def test_clean_state_dict_keys():
    sd = {"_fsdp_wrapped_module.model.q.weight": torch.ones(1)}
    assert list(clean_state_dict_keys(sd)) == ["model.q.weight"]


def test_select_generator_state_dict_prefers_ema_when_requested():
    ckpt = {"generator": {"a": 1}, "generator_ema": {"b": 2}}
    assert select_generator_state_dict(ckpt, use_ema=False) == {"a": 1}
    assert select_generator_state_dict(ckpt, use_ema=True) == {"b": 2}


def test_select_lora_state_dict_accepts_wrapped_or_raw():
    raw = {"x": torch.ones(1)}
    assert select_lora_state_dict({"generator_lora": raw}) is raw
    assert select_lora_state_dict(raw) is raw


def test_cast_fp32_params_to_bf16():
    model = nn.Linear(2, 2)
    n = cast_fp32_params_to_bf16(model)
    assert n == 2
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_find_adapter_params():
    model = Tiny()
    params = find_adapter_params(model, "default")
    assert len(params) == 2
    assert {id(p) for p in params} == {id(p) for p in model.lora["default"].parameters()}


def main():
    tests = [
        test_clean_fsdp_key,
        test_clean_state_dict_keys,
        test_select_generator_state_dict_prefers_ema_when_requested,
        test_select_lora_state_dict_accepts_wrapped_or_raw,
        test_cast_fp32_params_to_bf16,
        test_find_adapter_params,
    ]
    print("running checkpoint_utils tests:")
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
