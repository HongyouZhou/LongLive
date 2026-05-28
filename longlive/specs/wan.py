"""Wan model/cache specifications.

This module centralizes the dimensions that used to be spread across the Wan
wrapper and inference pipelines.  It is intentionally dependency-light so
config validation and tests can import it without loading CUDA/Wan modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WAN_MODEL_NAME = "Wan2.1-T2V-1.3B"


@dataclass(frozen=True)
class WanCacheSpec:
    transformer_blocks: int
    frame_seq_length: int
    kv_heads: int
    head_dim: int
    crossattn_tokens: int
    global_attention_tokens: int
    global_context_frames: int

    def kv_tokens_for_frames(self, num_output_frames: int, local_attn_size: int) -> int:
        """Cache token count used by inference for a generated clip."""
        if int(local_attn_size) != -1:
            return int(local_attn_size) * self.frame_seq_length
        return int(num_output_frames) * self.frame_seq_length

    def default_kv_tokens(self, local_attn_size: int) -> int:
        """Backward-compatible default cache size when clip length is unknown."""
        if int(local_attn_size) != -1:
            return int(local_attn_size) * self.frame_seq_length
        return self.global_attention_tokens

    def attention_tokens(self, local_attn_size: int) -> int:
        if int(local_attn_size) == -1:
            return self.global_attention_tokens
        return int(local_attn_size) * self.frame_seq_length

    def wrapper_seq_len(self, local_attn_size: int) -> int:
        """Legacy WanDiffusionWrapper seq_len policy."""
        local_attn_size = int(local_attn_size)
        if local_attn_size > self.global_context_frames:
            return local_attn_size * self.frame_seq_length
        return self.global_attention_tokens

    def kv_shape(self, batch_size: int, tokens: int) -> tuple[int, int, int, int]:
        return (int(batch_size), int(tokens), self.kv_heads, self.head_dim)

    def crossattn_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        return (int(batch_size), self.crossattn_tokens, self.kv_heads, self.head_dim)


@dataclass(frozen=True)
class WanModelSpec:
    name: str
    model_dirname: str
    t5_checkpoint_rel: str
    tokenizer_rel: str
    vae_checkpoint_rel: str
    vae_z_dim: int
    cache: WanCacheSpec

    def model_dir(self, root: str | Path) -> Path:
        return Path(root) / self.model_dirname

    def t5_checkpoint(self, root: str | Path) -> Path:
        return self.model_dir(root) / self.t5_checkpoint_rel

    def tokenizer_dir(self, root: str | Path) -> Path:
        return self.model_dir(root) / self.tokenizer_rel

    def vae_checkpoint(self, root: str | Path) -> Path:
        return self.model_dir(root) / self.vae_checkpoint_rel


WAN_21_T2V_13B = WanModelSpec(
    name=DEFAULT_WAN_MODEL_NAME,
    model_dirname=DEFAULT_WAN_MODEL_NAME,
    t5_checkpoint_rel="models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_rel="google/umt5-xxl/",
    vae_checkpoint_rel="Wan2.1_VAE.pth",
    vae_z_dim=16,
    cache=WanCacheSpec(
        transformer_blocks=30,
        frame_seq_length=1560,
        kv_heads=12,
        head_dim=128,
        crossattn_tokens=512,
        global_attention_tokens=32760,
        global_context_frames=21,
    ),
)


WAN_MODEL_SPECS = {
    WAN_21_T2V_13B.name: WAN_21_T2V_13B,
}


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or attribute-style configs."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def get_wan_model_spec(model_name: str | None = None) -> WanModelSpec:
    name = model_name or DEFAULT_WAN_MODEL_NAME
    name = str(name).strip().rstrip("/")
    try:
        return WAN_MODEL_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(WAN_MODEL_SPECS))
        raise ValueError(f"Unknown Wan model_name={name!r}. Known specs: {known}") from exc


def get_wan_model_spec_from_config(model_kwargs: Any) -> WanModelSpec:
    return get_wan_model_spec(get_config_value(model_kwargs, "model_name", DEFAULT_WAN_MODEL_NAME))
