"""Checkpoint and adapter utilities shared by entry points.

These helpers intentionally avoid owning any training loop. They only cover
stable mechanics that otherwise drift across methods: state-dict key cleanup,
baseline LoRA merge, PEFT state loading, dtype cleanup, and LoRA checkpoint
save/prune.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import peft
import torch
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)


def expand_path(path: str | os.PathLike[str]) -> str:
    return os.path.expandvars(os.path.expanduser(str(path)))


def clean_fsdp_key(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {clean_fsdp_key(k): v for k, v in state_dict.items()}


def select_generator_state_dict(
    checkpoint: dict[str, Any],
    *,
    use_ema: bool = False,
) -> dict[str, torch.Tensor]:
    if "generator" in checkpoint or "generator_ema" in checkpoint:
        key = "generator_ema" if use_ema and "generator_ema" in checkpoint else "generator"
        return checkpoint[key]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_generator_checkpoint(
    generator: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    use_ema: bool = False,
    strict: bool = True,
    clean_keys: bool = False,
):
    checkpoint = torch.load(expand_path(checkpoint_path), map_location="cpu")
    state = select_generator_state_dict(checkpoint, use_ema=use_ema)
    if clean_keys:
        state = clean_state_dict_keys(state)
    return generator.load_state_dict(state, strict=strict)


def select_lora_state_dict(
    checkpoint: dict[str, Any],
    *,
    key: str = "generator_lora",
):
    if isinstance(checkpoint, dict) and key in checkpoint:
        return checkpoint[key]
    return checkpoint


def load_lora_state_dict(
    peft_model: torch.nn.Module,
    checkpoint_or_path,
    *,
    key: str = "generator_lora",
) -> None:
    checkpoint = (
        torch.load(expand_path(checkpoint_or_path), map_location="cpu")
        if isinstance(checkpoint_or_path, (str, os.PathLike))
        else checkpoint_or_path
    )
    peft.set_peft_model_state_dict(
        peft_model,
        select_lora_state_dict(checkpoint, key=key),
    )


def merge_lora_into_transformer(
    transformer: torch.nn.Module,
    *,
    model_name: str,
    adapter_config,
    lora_ckpt: str | os.PathLike[str],
    is_main_process: bool = True,
) -> torch.nn.Module:
    from longlive.utils.lora_utils import configure_adapter_for_model

    wrapped = configure_adapter_for_model(
        transformer,
        model_name=model_name,
        adapter_config=adapter_config,
        is_main_process=is_main_process,
    )
    load_lora_state_dict(wrapped, lora_ckpt)
    return wrapped.merge_and_unload()


def cast_fp32_params_to_bf16(module: torch.nn.Module) -> int:
    n_cast = 0
    for param in module.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            n_cast += 1
    return n_cast


def find_adapter_params(
    model: torch.nn.Module,
    adapter_tag: str,
) -> list[torch.nn.Parameter]:
    needle = f".{adapter_tag}."
    return [p for n, p in model.named_parameters() if needle in n]


def save_lora_checkpoint(
    fsdp_peft_model: torch.nn.Module,
    out_dir: str | os.PathLike[str],
    tag: str,
    *,
    rank0: bool,
    adapter_name: str = "default",
) -> Path | None:
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_peft_model, StateDictType.FULL_STATE_DICT, save_policy):
        full = fsdp_peft_model.state_dict()
    if not rank0:
        return None
    lora_state = get_peft_model_state_dict(
        fsdp_peft_model,
        state_dict=full,
        adapter_name=adapter_name,
    )
    lora_state = {k: v.detach().cpu() for k, v in lora_state.items()}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"lora_{tag}.pt"
    torch.save(lora_state, path)
    return path


def prune_old_lora_checkpoints(out_dir: str | os.PathLike[str], keep_last: int) -> None:
    out_path = Path(out_dir)
    ckpts = sorted(
        (p for p in out_path.glob("lora_*.pt") if "final" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    while len(ckpts) > keep_last:
        ckpts[0].unlink()
        ckpts.pop(0)
