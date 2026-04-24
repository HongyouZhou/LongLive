"""Auxiliary motion-consistency loss for Motion-Recache training.

Operates in VAE latent space: matches the temporal-delta "energy" (mean
squared frame-to-frame difference) between generated video and motion_ref.

The loss is fully self-contained — it depends only on two tensors supplied
by the caller and exposes nothing about how they were produced. Callers
wire it as an additive term on top of the existing DMD generator loss.

Two strengths are supported:
    scalar_energy       — single scalar matching of overall motion magnitude
    per_channel_energy  — per-VAE-channel (16-dim) magnitude matching

Scalar is the gentler default; channel-wise is a drop-in upgrade when the
scalar version converges to "correct total motion, wrong distribution".
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionConsistencyLoss(nn.Module):
    """Temporal-delta energy matching in VAE latent space.

    Args:
        loss_type: one of {"scalar_energy", "per_channel_energy"}.
        weight:    scaling factor applied when the caller combines this loss
                   with DMD. Stored for convenience; the module itself
                   returns an unscaled loss.
    """

    VALID_TYPES = ("scalar_energy", "per_channel_energy")

    def __init__(self, loss_type: str = "scalar_energy", weight: float = 0.05):
        super().__init__()
        if loss_type not in self.VALID_TYPES:
            raise ValueError(
                f"loss_type must be one of {self.VALID_TYPES}, got {loss_type!r}")
        self.loss_type = loss_type
        self.weight = float(weight)

    @staticmethod
    def _delta_energy(latent: torch.Tensor, reduce_dims: Sequence[int]) -> torch.Tensor:
        """Energy of first-order temporal difference, averaged over `reduce_dims`.

        Input latent layout: [B, T, C, h, w].
        """
        if latent.shape[1] < 2:
            return latent.new_zeros(())
        delta = latent[:, 1:] - latent[:, :-1]
        return delta.pow(2).mean(dim=list(reduce_dims))

    def forward(self, gen_latent: torch.Tensor, ref_latent: torch.Tensor) -> torch.Tensor:
        ref_latent = ref_latent.to(device=gen_latent.device, dtype=gen_latent.dtype)

        if self.loss_type == "scalar_energy":
            reduce = [0, 1, 2, 3, 4]   # single scalar
        else:  # per_channel_energy
            reduce = [0, 1, 3, 4]      # preserve channel dim [C]

        gen_e = self._delta_energy(gen_latent, reduce)
        ref_e = self._delta_energy(ref_latent, reduce)
        return F.mse_loss(gen_e, ref_e)
