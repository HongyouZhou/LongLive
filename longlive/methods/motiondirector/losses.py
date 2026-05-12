"""MotionDirector loss terms in epsilon space.

Two terms (paper §3.2, eq 3 / 5):
  * L_temporal_MSE — standard epsilon-MSE; computed at the training site
    (one-liner `F.mse_loss(eps_pred, eps_gt)`), not factored here.
  * L_AD — appearance-debias; subtract a randomly-picked anchor frame
    before MSE so that signal shared across frames (appearance) cancels,
    leaving only frame-to-frame residual (motion).

Per docs/04.md Phase 2:
  * paper alpha = sqrt(2), beta = 1 — verbatim, applicable because B1
    close-form reverse `eps_pred = flow_pred + pred_x0` puts the model
    output back in epsilon space, where the paper's debias derivation holds.
  * L_spatial is dropped — Wan DiT has no Transformer2D / Temporal split
    to anchor a separate spatial LoRA on.
"""
import math

import torch
import torch.nn.functional as F


def appearance_debias_loss(
    eps_pred: torch.Tensor,
    eps_gt: torch.Tensor,
    alpha: float = math.sqrt(2.0),
    beta: float = 1.0,
    ran_idx: int | None = None,
) -> torch.Tensor:
    """L_AD = MSE(alpha * eps_pred - beta * eps_pred[ran_idx],
                  alpha * eps_gt   - beta * eps_gt[ran_idx])

    Args:
        eps_pred / eps_gt: shape (B, F, C, H, W); F is frame count.
        ran_idx: which frame to anchor on. If None, drawn uniformly per call.
    """
    n_frames = eps_pred.shape[1]
    if ran_idx is None:
        ran_idx = int(torch.randint(0, n_frames, (1,)).item())
    pred_dec = alpha * eps_pred - beta * eps_pred[:, ran_idx:ran_idx + 1]
    gt_dec = alpha * eps_gt - beta * eps_gt[:, ran_idx:ran_idx + 1]
    return F.mse_loss(pred_dec, gt_dec)
