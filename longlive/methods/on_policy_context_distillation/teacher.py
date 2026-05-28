"""Teacher-adapter helpers for on-policy context distillation."""

from __future__ import annotations

import os
from pathlib import Path

import peft
import torch

from longlive.utils.checkpoints import select_lora_state_dict


def optional_path(value) -> str | None:
    if value is None:
        return None
    text = os.path.expandvars(os.path.expanduser(str(value))).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def load_teacher_adapter(
    peft_model: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    adapter_name: str = "teacher",
    state_key: str = "generator_lora",
) -> None:
    """Load a frozen teacher adapter into an existing PEFT model."""
    path_text = optional_path(checkpoint_path)
    if path_text is None:
        raise ValueError("teacher checkpoint path is empty")
    ckpt_path = Path(path_text)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = select_lora_state_dict(checkpoint, key=state_key)
    peft.set_peft_model_state_dict(
        peft_model,
        state,
        adapter_name=adapter_name,
    )


def freeze_adapter_params(model: torch.nn.Module, adapter_name: str) -> int:
    """Freeze parameters whose PEFT-qualified name belongs to adapter_name."""
    needle = f".{adapter_name}."
    n = 0
    for name, param in model.named_parameters():
        if needle in name:
            param.requires_grad_(False)
            n += param.numel()
    return n
