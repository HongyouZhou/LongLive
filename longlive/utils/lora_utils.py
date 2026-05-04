# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import peft
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig
)


# Adapter registry — maps adapter_type string to a factory that builds a
# peft.PeftConfig from (adapter_cfg, target_modules). Methods register
# additional types via side-effect import; see longlive/methods/<idea>/__init__.py
# and the auto-scan at the bottom of this module.
_ADAPTER_REGISTRY = {}


def register_adapter(name, peft_config_factory):
    """Register an adapter type under `name` with a factory callable.

    factory(adapter_cfg, target_modules) -> peft.PeftConfig
    """
    if name in _ADAPTER_REGISTRY:
        raise KeyError(f"Adapter '{name}' already registered")
    _ADAPTER_REGISTRY[name] = peft_config_factory


# Built-in LoRA. Lives here (rather than under longlive/methods/) because LoRA
# is the default adapter and is the single dependency every entry-point already
# imports indirectly.
register_adapter("lora", lambda cfg, targets: peft.LoraConfig(
    r=cfg.get('rank', 16),
    lora_alpha=cfg.get('alpha', None) or cfg.get('rank', 16),
    lora_dropout=cfg.get('dropout', 0.0),
    target_modules=targets,
))


def configure_adapter_for_model(transformer, model_name, adapter_config, is_main_process=True):
    """Wrap a WanDiffusionWrapper transformer with the adapter selected by
    adapter_config.type (default 'lora'). LoRA / OFT / future adapters all
    flow through PEFT.

    Args:
        transformer: The transformer model to wrap
        model_name: 'generator' or 'fake_score'
        adapter_config: dict-like with .type and adapter-specific fields
        is_main_process: Whether this is the main process (for logging)

    Returns:
        peft-wrapped model with adapter parameters trainable, base frozen
    """
    target_linear_modules = set()

    if model_name == 'generator':
        adapter_target_modules = ['CausalWanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    target_linear_modules = list(target_linear_modules)

    adapter_type = adapter_config.get('type', 'lora')

    if is_main_process:
        print(f"Adapter '{adapter_type}' target modules for {model_name}: "
              f"{len(target_linear_modules)} Linear layers")
        if adapter_config.get('verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")

    factory = _ADAPTER_REGISTRY.get(adapter_type)
    if factory is None:
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Registered: {sorted(_ADAPTER_REGISTRY)}"
        )

    peft_config = factory(adapter_config, target_linear_modules)
    adapter_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        adapter_model.print_trainable_parameters()

    return adapter_model


# Backward-compat alias — call sites in trainer / inference / vbench import
# this name. Function is now polymorphic (lora / oft / ...) but name kept to
# avoid touching ~6 call sites in this PR.
configure_lora_for_model = configure_adapter_for_model


def gather_lora_state_dict(lora_model):
    with FSDP.state_dict_type(
        lora_model,                  
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full = lora_model.state_dict()
    return get_peft_model_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict
    
    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")
    
    peft.set_peft_model_state_dict(lora_model, lora_state_dict)

    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully")


def _autoload_methods():
    """Walk longlive/methods/* and import each subpackage so its __init__.py
    can side-effect-register adapters / losses / DMD score variants.

    Per docs/01.md "auto-scan + side-effect registration" convention. Run once
    at module import — any caller of lora_utils gets the full registry.
    """
    import importlib
    import pkgutil
    import longlive.methods as _m
    for _, name, ispkg in pkgutil.iter_modules(_m.__path__, _m.__name__ + "."):
        if ispkg:
            importlib.import_module(name)


_autoload_methods()
