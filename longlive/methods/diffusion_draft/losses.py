"""DRaFT-K loss terms.

The reward loss is inline in train.py (it's literally `-reward_coef * mf`).
This file only holds the explicit KL anchor regularization that pins the
trainable LoRA to the no-LoRA base (via the zero-init `anchor` PEFT
adapter) — DRaFT has no implicit anchor in its target form, so this term
is required to prevent caption-collapse (NFT-H3 / H3' failure mode).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Anti-drift MSE between v_θ (trainable) and v_ref (frozen base).

    Identical formula to the version in diffusion_nft / diffusion_ram —
    duplicated here so diffusion_draft is self-contained per CLAUDE.md
    `longlive/methods/<idea>/ is self-contained` convention.

    Args:
        v_default: (B, F, C, H, W) — trainable network output at (x_t, t).
        v_anchor:  (B, F, C, H, W) — frozen reference (zero-init "anchor"
            adapter), caller's responsibility to call under no_grad.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(v_default, v_anchor.detach())
