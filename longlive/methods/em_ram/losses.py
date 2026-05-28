"""EM-RAM losses for reward-tilted few-step distillation.

The method keeps RAM's stable M-step target geometry and changes only the
outer-loop credit assignment.  Each outer epoch performs a small empirical
EM / mirror-descent step:

  E-step:  q_i proportional to exp(A_i / eta), with KL(q || uniform) controlled.
  M-step:  use RAM's residual target, gated by q_i's positive excess mass.

This avoids the RT-fDMD failure mode where reward directly pulled the velocity
toward a self-generated flow target.  Here reward selects which endpoints
deserve a RAM correction; low/average endpoints only anchor back to v_ref.
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


def em_ram_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
    noise: torch.Tensor,
    x0_ref: torch.Tensor,
    alpha: torch.Tensor,
    reward_coef: float = 1.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """RAM M-step with EM-gated positive correction.

    Target construction:
        shift  = (eps - x0) - stopgrad(v_theta)
        target = v_ref + reward_coef * alpha * shift

    `alpha=0` makes the step a pure anchor update; positive alpha applies the
    RAM residual only to endpoints selected by the E-step.
    """
    v_anchor_d = v_anchor.detach()
    alpha_b = alpha.to(v_default.dtype).view(-1, *([1] * (v_default.ndim - 1)))
    alpha_eff = float(reward_coef) * alpha_b

    shift = (noise - x0_ref) - v_default.detach()
    target = v_anchor_d + alpha_eff * shift
    loss = F.mse_loss(v_default, target)

    with torch.no_grad():
        diag = {
            "loss/em_ram": loss.detach(),
            "em_ram/shift_norm": shift.float().flatten(1).norm(dim=1).mean(),
            "em_ram/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "em_ram/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "em_ram/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "em_ram/alpha": alpha_b.float().mean(),
            "em_ram/alpha_eff": alpha_eff.float().mean(),
            "em_ram/anchor_idx": torch.tensor(float(anchor_idx)),
        }
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional direct velocity anchor on top of the EM-RAM target."""
    return F.mse_loss(v_default, v_anchor.detach())
