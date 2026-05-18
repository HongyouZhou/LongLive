"""MotionDirector finetune loss terms.

Two families used in this project:

(A) Original MotionDirector paper losses (epsilon space):
  * L_temporal_MSE — eps-MSE, computed at the training site (one-liner
    `F.mse_loss(eps_pred, eps_gt)`), not factored here.
  * `appearance_debias_loss` — L_AD; subtract a random anchor frame before
    MSE so that frame-shared appearance signal cancels, leaving frame-to-
    frame residual (motion). Paper alpha=sqrt(2), beta=1.
  L_spatial is dropped — Wan DiT has no Transformer2D / Temporal split.

(B) Latent inter-frame delta cosine (docs/02.md) — replaces (A) entirely
    when `loss_space=trajectory_cosine`:
  * `trajectory_cosine_loss` — supervises frame-to-frame structure in
    student's predicted clean latent against the reference clip's latent,
    via cosine on per-frame deltas. Scale-invariant + appearance-cancelling
    by construction. The fundamental departure from (A): supervision unit
    is *inter-frame relations*, not per-point eps/x0 values, avoiding the
    4-anchor concentration that collapses (A) on a DMD few-step student.
  * `prior_consistency_loss` — anti-drift regularizer; on non-reference
    prompts, the LoRA-on prediction must agree with the LoRA-off base.
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


def trajectory_cosine_loss(
    pred_x0: torch.Tensor,
    z_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_motion — cosine distance between student & reference inter-frame deltas.

    For each pair of adjacent latent frames (f, f+1):
        delta_s[f] = pred_x0[f+1] - pred_x0[f]      student
        delta_r[f] = z_ref[f+1]   - z_ref[f]         reference
        loss[f]    = 1 - cos(delta_s[f], delta_r[f])
    Returned as mean over batch and frame pairs.

    Args:
        pred_x0: (B, F, C, H, W) — student's predicted clean latent at one
            denoising anchor t_k. Caller forwards once per anchor and accumulates
            (or sums) the loss across anchors as desired.
        z_ref:   (B, F, C, H, W) — reference clip's clean VAE-encoded latent
            (anchor-independent).

    Notes:
        * Cosine is computed on the flattened (C*H*W) feature; high-dim cosine
          values are numerically small even for well-aligned deltas. Saturation
          / scaling behavior is documented in docs/02.md §8 Risk 2. If observed
          to be problematic, switch to per-channel-then-mean by editing this
          function (no caller-side change).
        * No magnitude term — pure direction supervision. If motion amplitude
          collapses, the caller layers a `max(0, ||delta_r|| - ||delta_s||)^2`
          term separately (docs/02.md §8 Risk 1).
    """
    delta_s = pred_x0[:, 1:] - pred_x0[:, :-1]
    delta_r = z_ref[:, 1:] - z_ref[:, :-1]
    delta_s = delta_s.flatten(start_dim=2)
    delta_r = delta_r.flatten(start_dim=2)
    cos = F.cosine_similarity(delta_s, delta_r, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def prior_consistency_loss(
    pred_x0_lora: torch.Tensor,
    pred_x0_base: torch.Tensor,
) -> torch.Tensor:
    """L_anchor — anti-drift regularizer on non-reference prompts.

    On a generic (non-reference) prompt the LoRA-on prediction must agree
    with the LoRA-off base prediction. Same mechanism as DiffusionNFT's
    hidden KL term (||v_theta - v_base_no_LoRA||^2) and Noise Consistency
    Regularization's prior-consistency loss.

    Caller is responsible for obtaining `pred_x0_base` from the same model
    with the LoRA disabled (PEFT `disable_adapter()` context). This function
    detaches base internally so the caller doesn't have to.

    Args:
        pred_x0_lora: (B, F, C, H, W) — LoRA-on student prediction.
        pred_x0_base: (B, F, C, H, W) — LoRA-off base prediction (no grad).
    """
    return F.mse_loss(pred_x0_lora, pred_x0_base.detach())
