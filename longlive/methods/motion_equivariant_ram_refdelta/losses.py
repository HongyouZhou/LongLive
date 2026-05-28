"""Motion-equivalent RAM RefDelta losses.

The method keeps RAM's rollout/reward loop but changes the optimization
geometry.  The scalar reward is allowed to move only a coarse temporal-motion
subspace; frame-shared/static velocity stays anchored to the frozen base.

This writes the desired invariance into the loss:

    same motion, different pixels/colors/textures = acceptable
    high reward should not copy full endpoint appearance into the student

RefDelta differs from the first Motion-Equivariant RAM variant in one place:
the rewarded motion target is the encoded reference clip, not the sampled
on-policy rollout endpoint.  Rollouts still provide the noised training states
and the reward-ranking signal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _broadcast_scalar(x: torch.Tensor, ndim: int, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).view(-1, *([1] * (ndim - 1)))


def coarse_temporal_delta(
    x: torch.Tensor,
    pool: int = 2,
) -> torch.Tensor:
    """Return coarse adjacent-frame deltas.

    Args:
        x: (B, F, C, H, W) latent or velocity tensor.
        pool: average-pool factor over H/W.  pool=1 keeps native resolution.

    Temporal differencing cancels frame-shared appearance.  Spatial pooling
    removes texture-scale pixel/color detail, leaving a lower-frequency motion
    representation for the reward-gated term.
    """
    if x.ndim != 5:
        raise ValueError(f"expected (B,F,C,H,W), got shape={tuple(x.shape)}")
    delta = x[:, 1:] - x[:, :-1]
    if int(pool) <= 1:
        return delta
    b, f, c, h, w = delta.shape
    flat = delta.reshape(b * f, c, h, w)
    pooled = F.avg_pool2d(flat, kernel_size=int(pool), stride=int(pool))
    return pooled.reshape(b, f, c, pooled.shape[-2], pooled.shape[-1])


def normalized_delta_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-frame normalized MSE on motion deltas.

    This is the delta-space analogue of trajectory NMSE: it preserves both
    direction and magnitude while being invariant to frame-shared appearance.
    """
    pred_f = pred.float().flatten(start_dim=2)
    target_f = target.detach().float().flatten(start_dim=2)
    num = (pred_f - target_f).pow(2).sum(dim=-1)
    den = target_f.pow(2).sum(dim=-1).add(float(eps))
    return (num / den).mean()


def match_delta_grid(delta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize a reference motion delta to the current latent motion grid."""
    if delta.shape == target.shape:
        return delta
    if delta.ndim != 5 or target.ndim != 5:
        raise ValueError(
            f"expected 5D deltas, got delta={tuple(delta.shape)} target={tuple(target.shape)}"
        )
    if delta.shape[0] != target.shape[0] or delta.shape[2] != target.shape[2]:
        raise ValueError(
            f"cannot align batch/channel mismatch: delta={tuple(delta.shape)} "
            f"target={tuple(target.shape)}"
        )
    delta_bcthw = delta.permute(0, 2, 1, 3, 4).float()
    aligned = F.interpolate(
        delta_bcthw,
        size=(target.shape[1], target.shape[3], target.shape[4]),
        mode="trilinear",
        align_corners=False,
    )
    return aligned.permute(0, 2, 1, 3, 4).to(dtype=target.dtype)


def motion_equivariant_ram_refdelta_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
    pred_x0_default: torch.Tensor,
    pred_x0_anchor: torch.Tensor,
    x0_reference: torch.Tensor,
    r: torch.Tensor,
    reward_coef: float = 1.0,
    lambda_motion: float = 1.0,
    lambda_static: float = 0.05,
    motion_pool: int = 2,
    max_motion_weight: float = 1.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Reward-gated motion update with static-space anchoring.

    Motion target:
        w = clamp(reward_coef * r, 0, max_motion_weight)
        target_delta = (1 - w) * P_m(x0_anchor) + w * P_m(x0_reference)

    Static target:
        mean_t(v_default) should stay close to mean_t(v_anchor).

    The high-reward path distills only P_m(x0_reference), not the full rollout
    latent.  Pixel/color/appearance components are therefore not rewarded.
    """
    v_anchor_d = v_anchor.detach()
    pred_x0_anchor_d = pred_x0_anchor.detach()
    x0_reference_d = x0_reference.detach()

    w_b = _broadcast_scalar(r, pred_x0_default.ndim, pred_x0_default.dtype)
    w_b = (float(reward_coef) * w_b).clamp(0.0, float(max_motion_weight))

    delta_default = coarse_temporal_delta(pred_x0_default, pool=motion_pool)
    delta_anchor = coarse_temporal_delta(pred_x0_anchor_d, pool=motion_pool)
    delta_reference = coarse_temporal_delta(x0_reference_d, pool=motion_pool)
    delta_reference = match_delta_grid(delta_reference, delta_anchor)
    target_delta = delta_anchor + w_b * (delta_reference - delta_anchor)

    motion_loss = normalized_delta_mse(delta_default, target_delta)

    static_default = v_default.mean(dim=1, keepdim=True)
    static_anchor = v_anchor_d.mean(dim=1, keepdim=True)
    static_loss = F.mse_loss(static_default, static_anchor)

    loss = float(lambda_motion) * motion_loss + float(lambda_static) * static_loss

    with torch.no_grad():
        diag = {
            "loss/meref": loss.detach(),
            "meref/motion_loss": motion_loss.detach(),
            "meref/static_loss": static_loss.detach(),
            "meref/motion_weight": w_b.float().mean(),
            "meref/delta_default_norm": delta_default.float().flatten(1).norm(dim=1).mean(),
            "meref/delta_target_norm": target_delta.float().flatten(1).norm(dim=1).mean(),
            "meref/delta_anchor_norm": delta_anchor.float().flatten(1).norm(dim=1).mean(),
            "meref/delta_reference_norm": delta_reference.float().flatten(1).norm(dim=1).mean(),
            "meref/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "meref/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "meref/r_scalar": r.float().mean(),
            "meref/anchor_idx": torch.tensor(float(anchor_idx)),
        }
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional full-velocity anchor on top of the projected objective."""
    return F.mse_loss(v_default, v_anchor.detach())
