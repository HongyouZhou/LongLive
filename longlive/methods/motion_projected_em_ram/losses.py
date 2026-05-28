"""Motion-Projected EM-RAM losses for reward-tilted few-step distillation.

The E-step is unchanged from EM-RAM: build a reward-tilted endpoint
distribution under a KL(q || uniform) budget.  The M-step changes the update
subspace:

  E-step:  q_i proportional to exp(A_i / eta), with KL(q || uniform) controlled.
  M-step:  alpha_i gates only a configured motion projection of RAM's
           residual; all other velocity components stay anchored to v_ref.

This writes the intended invariance into the optimization problem: reward can
increase motion-consistent residuals, but static/appearance/color directions do
not receive reward credit.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_float_tensor(
    values: list[float] | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().float()
        return tensor.to(device) if device is not None else tensor
    return torch.tensor(values, dtype=torch.float32, device=device)


def _softmax_weights(advantages: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q, importance weights w=Nq, and KL(q||uniform)."""
    n = advantages.numel()
    logits = advantages * float(beta)
    q = torch.softmax(logits, dim=0)
    w = q * float(n)
    kl = (q * (torch.log(q.clamp_min(1e-12)) + math.log(float(n)))).sum()
    return q, w, kl


def em_tilt_weights(
    rewards: list[float] | torch.Tensor,
    *,
    target_kl: float = 0.10,
    eta: float | None = None,
    adv_clip_max: float = 5.0,
    weight_clip: float = 4.0,
    alpha_mode: str = "positive_excess",
    alpha_max: float = 1.0,
    std_floor: float = 1e-4,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute empirical EM weights from rollout rewards.

    Args:
        rewards: one scalar reward per rollout in the gathered cross-rank group.
        target_kl: KL(q || uniform) budget for the E-step when `eta` is None.
        eta: fixed temperature.  If None, solve eta by bisection on inverse
            temperature so the empirical KL is close to `target_kl`.
        adv_clip_max: z-scored advantages are clipped to this range.
        weight_clip: cap on importance weights w=Nq before alpha mapping.
        alpha_mode: currently "positive_excess" or "weight".
        alpha_max: cap on the M-step correction coefficient.
        std_floor: reward std floor for degenerate groups.
        device: optional output device.

    Returns:
        (alpha, diagnostics), where alpha has shape (N,) and mean correction
        mass is zero when all rewards are tied.
    """
    rewards_t = _as_float_tensor(rewards, device=device).view(-1)
    if rewards_t.numel() == 0:
        raise ValueError("em_tilt_weights requires at least one reward")

    raw_mean = rewards_t.mean()
    raw_std = rewards_t.std(unbiased=False)
    raw_std_value = float(raw_std)
    advantages = (rewards_t - raw_mean) / (raw_std + float(std_floor))
    advantages = advantages.clamp(-float(adv_clip_max), float(adv_clip_max))

    n = rewards_t.numel()
    if raw_std_value < float(std_floor) or n == 1:
        q = torch.full_like(rewards_t, 1.0 / float(n))
        w = torch.ones_like(rewards_t)
        kl = torch.zeros((), device=rewards_t.device)
        beta = 0.0
    else:
        if eta is not None and float(eta) > 0.0:
            beta = 1.0 / float(eta)
            q, w, kl = _softmax_weights(advantages, beta)
        else:
            # KL is monotone in beta for fixed advantages.  Solve for the
            # largest beta inside the trust-region budget.
            target = float(max(0.0, min(float(target_kl), math.log(float(n)))))
            lo = 0.0
            hi = 1.0
            _q_hi, _w_hi, kl_hi = _softmax_weights(advantages, hi)
            while float(kl_hi) < target and hi < 1024.0:
                hi *= 2.0
                _q_hi, _w_hi, kl_hi = _softmax_weights(advantages, hi)
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                _q_mid, _w_mid, kl_mid = _softmax_weights(advantages, mid)
                if float(kl_mid) <= target:
                    lo = mid
                else:
                    hi = mid
            beta = lo
            q, w, kl = _softmax_weights(advantages, beta)

    w_clipped = w.clamp(0.0, float(weight_clip))
    if alpha_mode == "positive_excess":
        alpha = (w_clipped - 1.0).clamp_min(0.0)
    elif alpha_mode == "weight":
        alpha = w_clipped
    else:
        raise ValueError(f"unknown alpha_mode: {alpha_mode!r}")
    alpha = alpha.clamp(0.0, float(alpha_max))

    ess = 1.0 / (q.square().sum().clamp_min(1e-12))
    entropy = -(q * torch.log(q.clamp_min(1e-12))).sum()
    diag = {
        "reward/raw_mean": float(raw_mean),
        "reward/raw_std": raw_std_value,
        "reward/group_collapsed": float(raw_std_value < float(std_floor)),
        "em/kl": float(kl),
        "em/target_kl": float(target_kl),
        "em/eta": 1.0e12 if beta == 0.0 else float(1.0 / beta),
        "em/beta": float(beta),
        "em/entropy": float(entropy),
        "em/ess": float(ess),
        "em/weight_min": float(w.min()),
        "em/weight_max": float(w.max()),
        "em/alpha_min": float(alpha.min()),
        "em/alpha_max": float(alpha.max()),
        "em/alpha_mean": float(alpha.mean()),
    }
    return alpha.float(), diag


def motion_project(
    x: torch.Tensor,
    *,
    spatial_pool: int = 2,
    temporal_center: bool = True,
) -> torch.Tensor:
    """Project a velocity/residual tensor onto a coarse motion subspace.

    The temporal centering removes frame-shared/static velocity.  Spatial
    pooling removes texture-scale detail while returning a tensor on the
    original latent grid, so it can be used directly in a v-space target.
    """
    if x.ndim != 5:
        raise ValueError(f"expected (B,F,C,H,W), got shape={tuple(x.shape)}")
    y = x
    if temporal_center:
        y = y - y.mean(dim=1, keepdim=True)
    pool = int(spatial_pool)
    if pool <= 1:
        return y
    b, f, c, h, w = y.shape
    flat = y.reshape(b * f, c, h, w).float()
    pooled = F.avg_pool2d(flat, kernel_size=pool, stride=pool)
    up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
    return up.reshape(b, f, c, h, w).to(dtype=x.dtype)


def _match_reference_shape(reference: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Resize a reference latent/motion tensor to ``like`` as (B,F,C,H,W)."""
    if reference.ndim != 5 or like.ndim != 5:
        raise ValueError(
            f"expected 5D tensors, got reference={tuple(reference.shape)} "
            f"like={tuple(like.shape)}"
        )
    if reference.shape[2] != like.shape[2]:
        raise ValueError(
            f"channel mismatch: reference={tuple(reference.shape)} like={tuple(like.shape)}"
        )
    ref = reference.detach().to(device=like.device, dtype=like.dtype)
    if ref.shape[1:] != like.shape[1:]:
        ref_bcthw = ref.permute(0, 2, 1, 3, 4).float()
        ref = F.interpolate(
            ref_bcthw,
            size=(like.shape[1], like.shape[3], like.shape[4]),
            mode="trilinear",
            align_corners=False,
        ).permute(0, 2, 1, 3, 4).to(dtype=like.dtype)
    if ref.shape[0] == like.shape[0]:
        return ref
    if ref.shape[0] == 1:
        return ref.expand(like.shape[0], -1, -1, -1, -1)
    raise ValueError(
        f"batch mismatch: reference={tuple(reference.shape)} like={tuple(like.shape)}"
    )


def reference_motion_basis(
    x0_reference: torch.Tensor,
    *,
    like: torch.Tensor,
    spatial_pool: int = 2,
    temporal_center: bool = False,
) -> torch.Tensor:
    """Build a framewise reference-motion basis from the encoded reference clip.

    Adjacent-frame differencing removes frame-shared appearance before the
    spatial low-pass.  The returned tensor keeps the original latent shape so
    RAM's residual can be projected in v-space without changing the optimizer.
    """
    ref = _match_reference_shape(x0_reference, like)
    if ref.shape[1] < 2:
        return torch.zeros_like(ref)

    delta = ref[:, 1:] - ref[:, :-1]
    basis = torch.zeros_like(ref)
    basis[:, 0] = delta[:, 0]
    basis[:, -1] = delta[:, -1]
    if ref.shape[1] > 2:
        basis[:, 1:-1] = 0.5 * (delta[:, :-1] + delta[:, 1:])
    return motion_project(
        basis,
        spatial_pool=spatial_pool,
        temporal_center=temporal_center,
    )


def reference_motion_project(
    x: torch.Tensor,
    x0_reference: torch.Tensor,
    *,
    spatial_pool: int = 2,
    temporal_center: bool = True,
    reference_temporal_center: bool = False,
    projection_scope: str = "frame",
    positive_only: bool = False,
    mix: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project ``x`` onto the reference-motion subspace.

    The coarse projector first removes texture/static directions from the RAM
    residual.  The reference projector then keeps only the component aligned
    with the reference clip's latent temporal derivative.  ``mix`` allows a
    controlled interpolation back to the original coarse projector for ablation.
    """
    coarse = motion_project(
        x,
        spatial_pool=spatial_pool,
        temporal_center=temporal_center,
    )
    basis = reference_motion_basis(
        x0_reference,
        like=coarse,
        spatial_pool=spatial_pool,
        temporal_center=reference_temporal_center,
    )

    scope = str(projection_scope)
    coarse_f = coarse.float()
    basis_f = basis.float()
    if scope == "global":
        coarse_vec = coarse_f.flatten(start_dim=1)
        basis_vec = basis_f.flatten(start_dim=1)
        coef = (coarse_vec * basis_vec).sum(dim=-1) / (
            basis_vec.square().sum(dim=-1).clamp_min(float(eps))
        )
        coef_b = coef.view(-1, 1, 1, 1, 1)
    elif scope == "frame":
        coarse_vec = coarse_f.flatten(start_dim=2)
        basis_vec = basis_f.flatten(start_dim=2)
        coef = (coarse_vec * basis_vec).sum(dim=-1) / (
            basis_vec.square().sum(dim=-1).clamp_min(float(eps))
        )
        coef_b = coef.view(coarse.shape[0], coarse.shape[1], 1, 1, 1)
    else:
        raise ValueError(f"unknown reference projection scope: {projection_scope!r}")

    if positive_only:
        coef_b = coef_b.clamp_min(0.0)
    aligned = coef_b.to(dtype=coarse.dtype) * basis
    mix_f = float(max(0.0, min(1.0, float(mix))))
    projected = (1.0 - mix_f) * coarse + mix_f * aligned

    with torch.no_grad():
        aligned_f = aligned.float()
        projected_f = projected.float()
        coarse_norm = coarse_f.flatten(1).norm(dim=1).mean()
        aligned_norm = aligned_f.flatten(1).norm(dim=1).mean()
        denom = (
            coarse_f.flatten(1).norm(dim=1)
            * aligned_f.flatten(1).norm(dim=1)
        ).clamp_min(float(eps))
        align_cos = (
            coarse_f.flatten(1) * aligned_f.flatten(1)
        ).sum(dim=1).div(denom).mean()
        diag = {
            "mpem/coarse_shift_norm": coarse_norm,
            "mpem/reference_basis_norm": basis_f.flatten(1).norm(dim=1).mean(),
            "mpem/reference_aligned_shift_norm": aligned_norm,
            "mpem/reference_projection_residual_norm": (
                coarse_f - aligned_f
            ).flatten(1).norm(dim=1).mean(),
            "mpem/reference_alignment_cos": align_cos,
            "mpem/reference_coef_mean": coef_b.float().mean(),
            "mpem/reference_coef_abs_mean": coef_b.float().abs().mean(),
            "mpem/reference_mix": torch.tensor(mix_f, device=x.device),
            "mpem/projected_shift_norm": projected_f.flatten(1).norm(dim=1).mean(),
        }
    return projected.to(dtype=x.dtype), diag


def motion_projected_em_ram_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
    noise: torch.Tensor,
    x0_ref: torch.Tensor,
    alpha: torch.Tensor,
    x0_reference: torch.Tensor | None = None,
    reward_coef: float = 1.0,
    lambda_motion: float = 1.0,
    lambda_static: float = 0.05,
    subspace_mode: str = "coarse_motion",
    motion_pool: int = 2,
    motion_temporal_center: bool = True,
    reference_motion_scope: str = "frame",
    reference_motion_positive: bool = False,
    reference_motion_mix: float = 1.0,
    reference_motion_temporal_center: bool = False,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """EM-gated RAM residual, projected onto the configured motion subspace.

    Target construction:
        raw_shift    = (eps - x0) - stopgrad(v_theta)
        motion_shift = P_subspace(raw_shift)
        target       = v_ref + reward_coef * alpha * motion_shift

    `alpha=0` makes the step a pure anchor update.  Positive alpha applies the
    RAM residual only in the projected motion subspace selected by the E-step.
    """
    v_anchor_d = v_anchor.detach()
    alpha_b = alpha.to(v_default.dtype).view(-1, *([1] * (v_default.ndim - 1)))
    alpha_eff = float(reward_coef) * alpha_b

    raw_shift = (noise - x0_ref) - v_default.detach()
    subspace = str(subspace_mode)
    projector_diag: dict[str, torch.Tensor] = {}
    if subspace == "coarse_motion":
        motion_shift = motion_project(
            raw_shift,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
        )
    elif subspace == "reference_motion":
        if x0_reference is None:
            raise ValueError("subspace_mode='reference_motion' requires x0_reference")
        motion_shift, projector_diag = reference_motion_project(
            raw_shift,
            x0_reference,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
            reference_temporal_center=reference_motion_temporal_center,
            projection_scope=reference_motion_scope,
            positive_only=reference_motion_positive,
            mix=reference_motion_mix,
        )
    else:
        raise ValueError(f"unknown subspace_mode: {subspace_mode!r}")
    target = v_anchor_d + alpha_eff * motion_shift
    motion_loss = F.mse_loss(v_default, target.detach())

    static_default = v_default.mean(dim=1, keepdim=True)
    static_anchor = v_anchor_d.mean(dim=1, keepdim=True)
    static_loss = F.mse_loss(static_default, static_anchor)

    loss = float(lambda_motion) * motion_loss + float(lambda_static) * static_loss

    with torch.no_grad():
        diag = {
            "loss/mpem": loss.detach(),
            "mpem/motion_loss": motion_loss.detach(),
            "mpem/static_loss": static_loss.detach(),
            "mpem/raw_shift_norm": raw_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/motion_shift_norm": motion_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "mpem/alpha": alpha_b.float().mean(),
            "mpem/alpha_eff": alpha_eff.float().mean(),
            "mpem/anchor_idx": torch.tensor(float(anchor_idx)),
            "mpem/subspace_reference": torch.tensor(float(subspace == "reference_motion")),
        }
        diag.update(projector_diag)
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional direct velocity anchor on top of the projected EM-RAM target."""
    return F.mse_loss(v_default, v_anchor.detach())
