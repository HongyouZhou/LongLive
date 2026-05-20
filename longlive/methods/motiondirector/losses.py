"""MotionDirector finetune loss terms.

Two families used in this project:

(A) Original MotionDirector paper losses (epsilon space):
  * L_temporal_MSE — eps-MSE, computed at the training site (one-liner
    `F.mse_loss(eps_pred, eps_gt)`), not factored here.
  * `appearance_debias_loss` — L_AD; subtract a random anchor frame before
    MSE so that frame-shared appearance signal cancels, leaving frame-to-
    frame residual (motion). Paper alpha=sqrt(2), beta=1.
  L_spatial is dropped — Wan DiT has no Transformer2D / Temporal split.

(B) Latent inter-frame delta supervision (docs/02.md) — replaces (A) entirely
    when `loss_space=trajectory_cosine`:
  * `trajectory_cosine_loss` (L_dir) — supervises *direction* of frame-to-
    frame change in student's predicted clean latent against reference,
    via cosine on inter-frame deltas. Scale-invariant + appearance-cancelling.
    Fundamental departure from (A): supervision unit is inter-frame
    relations, not per-point eps/x0 values, avoiding the 4-anchor
    concentration that collapses (A) on a DMD few-step student.
  * `amplitude_penalty_loss` (L_amp) — supervises *magnitude* of the same
    inter-frame delta. Companion to L_dir; without it, cosine's scale
    invariance lets the LoRA collapse motion magnitude (Risk 1 in
    docs/02.md §8, observed in 2026-05-19 traj_cos run: dynamic_degree
    29 vs BASE 50).
  * `trajectory_nmse_loss` (L_nmse) — per-frame normalized MSE on the same
    inter-frame deltas. Strict superset of L_dir: penalizes direction AND
    magnitude in one coherent gradient signal. Supersedes the L_dir+L_amp
    pair (2026-05-19 traj_cos+amp run showed the two terms conflict when
    cosine is intermediate — see docs/02.md §12). Do NOT combine with L_amp.
  * `prior_consistency_loss` (L_anchor) — anti-drift regularizer; on
    non-reference prompts, LoRA-on prediction must agree with LoRA-off
    base. Currently disabled (lambda_anchor=0) due to disable_adapter ×
    FSDP × checkpoint incompat (Risk 3); pending option-C fix.
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


def amplitude_penalty_loss(
    pred_x0: torch.Tensor,
    z_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_amp — single-sided amplitude penalty companion to trajectory_cosine_loss.

    For each pair of adjacent latent frames (f, f+1):
        delta_s[f] = pred_x0[f+1] - pred_x0[f]      student
        delta_r[f] = z_ref[f+1]   - z_ref[f]         reference (detached)
        ratio[f]   = ||delta_s[f]|| / ||delta_r[f]||
        loss[f]    = relu(1 - ratio[f])^2
    Mean-reduced over batch and frame pairs.

    Penalty fires only when student's inter-frame delta norm is *smaller*
    than reference's (ratio < 1). Student motion exceeding reference
    amplitude is unpenalized — we don't want to clamp generation magnitude
    just because the LoRA happened to overshoot, only catch the failure
    mode that cosine alone exhibits (Risk 1 in docs/02.md §8: cosine is
    scale-invariant so the model can match direction with arbitrarily
    small magnitude).

    Args:
        pred_x0: (B, F, C, H, W) — student's predicted clean latent at one
            denoising anchor. Same forward output as trajectory_cosine_loss.
        z_ref:   (B, F, C, H, W) — reference clip's clean VAE-encoded latent.

    Notes:
        * Normalized by reference norm (ratio form) so loss is in [0, 1],
          numerically comparable to trajectory_cosine_loss in [0, 2]. Lets
          a single lambda_amp on the order of 1.0 give meaningful weighting.
        * Reference norm is detached. Gradient flows only through student's
          delta_s magnitude.
    """
    delta_s = pred_x0[:, 1:] - pred_x0[:, :-1]
    delta_r = z_ref[:, 1:] - z_ref[:, :-1]
    norm_s = delta_s.flatten(start_dim=2).norm(dim=-1)
    norm_r = delta_r.flatten(start_dim=2).norm(dim=-1).detach()
    ratio = norm_s / (norm_r + eps)
    return F.relu(1.0 - ratio).pow(2).mean()


def trajectory_nmse_loss(
    pred_x0: torch.Tensor,
    z_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_nmse — per-frame normalized MSE on inter-frame deltas.

    For each pair of adjacent latent frames (f, f+1):
        delta_s[f] = pred_x0[f+1] - pred_x0[f]      student
        delta_r[f] = z_ref[f+1]   - z_ref[f]         reference (detached)
        loss[f]    = ||delta_s[f] - delta_r[f]||^2 / ||delta_r[f]||^2
    Mean-reduced over batch and frame pairs.

    Strict superset of `trajectory_cosine_loss`:
      Let r = ||delta_s|| / ||delta_r||, c = cos(delta_s, delta_r).
        nmse = r^2 - 2 r c + 1
      The unique minimum is at delta_s = delta_r (i.e. c=1 AND r=1).
      Cosine's degenerate ray of optima `delta_s = eps * delta_r` (c=1 with
      any r>=0) is no longer flat: nmse = (r-1)^2 there, pulling r->1.

    Why this is not just `MSE on x0 in disguise`:
        Expand ||(z_s[f+1] - z_r[f+1]) - (z_s[f] - z_r[f])||^2
            = ||e[f+1]||^2 + ||e[f]||^2 - 2 <e[f+1], e[f]>
        where e[f] = z_s[f] - z_r[f] is the per-frame x0 error.
        The cross-term `-2 <e[f+1], e[f]>` cancels appearance-shared error
        (background / subject identity, which is static across f and f+1).
        Memorizing reference appearance does not reduce L_nmse — only
        learning the motion pattern does. This is why the loss escapes the
        L_MSE-on-x0 failure mode (5 prior runs collapsed motion_fidelity).

    Args:
        pred_x0: (B, F, C, H, W) — student's predicted clean latent at one
            denoising anchor.
        z_ref:   (B, F, C, H, W) — reference clip's clean VAE-encoded latent.

    Notes:
        * Per-frame normalization (one ratio per (b, f) pair, not per
          pixel). High-motion frames don't dominate the mean.
        * eps floor only protects against fully-static reference frames
          (rare). In normal operation, denominator is O(latent_var * C*H*W)
          which dwarfs eps.
        * DO NOT combine with `amplitude_penalty_loss`. NMSE already
          contains a magnitude term (the r^2 piece); adding L_amp
          unconditionally forces r->1 even when c is poor, re-creating
          the destructive gradient conflict observed in the 2026-05-19
          traj_cos+amp run (motion_fidelity 0.44 -> 0.20).
    """
    delta_s = pred_x0[:, 1:] - pred_x0[:, :-1]
    delta_r = z_ref[:, 1:] - z_ref[:, :-1]
    delta_s = delta_s.flatten(start_dim=2)
    delta_r = delta_r.flatten(start_dim=2)
    num = (delta_s - delta_r).pow(2).sum(dim=-1)
    den = delta_r.pow(2).sum(dim=-1).add(eps)
    return (num / den).mean()


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
