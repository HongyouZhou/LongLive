"""Reward-tilted f-divergence flow-map losses.

This method keeps RAM's useful engineering shape (on-policy few-step rollouts
+ frozen base adapter), but changes the training objective:

  * reward is interpreted as a bounded density-ratio tilt
    p_tilt(x) proportional to p_base(x) exp(beta R(x)), not as an unconstrained RL target;
  * the flow target is mixed with the frozen-base velocity through that tilt;
  * a short-interval flow-map penalty keeps the student transition near the
    base transition at the same noisy point.

The default "js" tilt is bounded: rho/(1+rho), where rho=exp(beta(r-b)).
It is less aggressive than a raw forward-KL / exponential weight and is meant
to be the first stable step toward mode-covering reward distillation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _tilt_weight(
    r: torch.Tensor,
    *,
    beta: float,
    baseline: float,
    mode: str,
    clip_min: float,
    clip_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded reward tilt and the underlying density-ratio proxy."""
    rho = torch.exp(float(beta) * (r.float() - float(baseline)))
    rho = rho.clamp(float(clip_min), float(clip_max))

    if mode == "js":
        # Bounded JS / GAN-style density-ratio transform.  Values stay in
        # (0, 1), so the target remains a convex blend of base and flow target.
        weight = rho / (1.0 + rho)
    elif mode == "clipped_fkl":
        # Forward-KL-like ratio, clipped and normalized so weight=1 at rho=1.
        weight = rho
    elif mode == "exp":
        # Kept for ablations.  Still clipped by the rho clamp above.
        weight = rho
    else:
        raise ValueError(f"unknown tilt mode: {mode!r}")

    return weight, rho


def _sigma_for_timestep(
    scheduler,
    timestep: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device)
    timesteps = scheduler.timesteps.to(device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    return sigmas[timestep_id].reshape(-1, 1, 1, 1)


def _scheduler_step_5d(
    scheduler,
    flow: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    to_timestep: int | None,
) -> torch.Tensor:
    shape = sample.shape
    flow_f = flow.flatten(0, 1)
    sample_f = sample.flatten(0, 1)
    timestep_f = timestep.flatten(0, 1)

    if to_timestep is None:
        stepped = scheduler.step(
            model_output=flow_f,
            timestep=timestep_f,
            sample=sample_f,
        )
    else:
        sigma_from = _sigma_for_timestep(
            scheduler, timestep_f, device=sample.device
        )
        if int(to_timestep) <= 0:
            sigma_to = torch.zeros_like(sigma_from)
        else:
            to_t = torch.full_like(timestep_f, int(to_timestep))
            sigma_to = _sigma_for_timestep(scheduler, to_t, device=sample.device)
        stepped = sample_f + flow_f * (sigma_to - sigma_from)
    return stepped.unflatten(0, shape[:2])


def reward_tilted_flow_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
    noise: torch.Tensor,
    x0_ref: torch.Tensor,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    to_timestep: int | None,
    scheduler,
    r: torch.Tensor,
    *,
    reward_beta: float = 2.0,
    reward_baseline: float = 0.5,
    reward_coef: float = 1.0,
    tilt_mode: str = "js",
    ratio_clip_min: float = 0.25,
    ratio_clip_max: float = 4.0,
    lambda_flowmap: float = 0.05,
    lambda_anchor: float = 0.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Reward-tilted flow objective with a local flow-map trust region.

    Target:
        rho    = clip(exp(beta * (r - baseline)))
        w      = rho / (1 + rho)                  # default JS-style bounded tilt
        v_fm   = eps - x0
        target = (1 - alpha) * v_anchor + alpha * v_fm
        alpha  = clip(reward_coef * w, 0, 1)

    The optional flow-map term compares the DMD-anchor interval transition
    from the same noisy point under the student velocity and the frozen-base
    velocity.  This is a cheap short-interval constraint: it does not require
    storing or backpropagating through a teacher trajectory.
    """
    v_anchor_d = v_anchor.detach()
    r_b = r.float().view(-1, *([1] * (v_default.ndim - 1)))
    w, rho = _tilt_weight(
        r_b,
        beta=reward_beta,
        baseline=reward_baseline,
        mode=tilt_mode,
        clip_min=ratio_clip_min,
        clip_max=ratio_clip_max,
    )
    alpha = (float(reward_coef) * w).clamp(0.0, 1.0).to(v_default.dtype)

    flow_target = (noise - x0_ref).detach()
    target = v_anchor_d + alpha * (flow_target - v_anchor_d)
    loss_tilt = F.mse_loss(v_default, target)

    if lambda_flowmap > 0.0:
        step_default = _scheduler_step_5d(
            scheduler, v_default, timestep, noisy, to_timestep
        )
        with torch.no_grad():
            step_anchor = _scheduler_step_5d(
                scheduler, v_anchor_d, timestep, noisy, to_timestep
            )
        loss_flowmap = F.mse_loss(step_default, step_anchor)
    else:
        loss_flowmap = torch.zeros((), device=v_default.device, dtype=v_default.dtype)

    if lambda_anchor > 0.0:
        loss_anchor = F.mse_loss(v_default, v_anchor_d)
    else:
        loss_anchor = torch.zeros((), device=v_default.device, dtype=v_default.dtype)

    loss = loss_tilt + float(lambda_flowmap) * loss_flowmap + float(lambda_anchor) * loss_anchor

    with torch.no_grad():
        diag = {
            "loss/tilt": loss_tilt.detach(),
            "loss/flowmap": loss_flowmap.detach(),
            "loss/anchor": loss_anchor.detach(),
            "rt/ratio": rho.float().mean(),
            "rt/tilt_weight": w.float().mean(),
            "rt/alpha": alpha.float().mean(),
            "rt/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "rt/flow_target_norm": flow_target.float().flatten(1).norm(dim=1).mean(),
            "rt/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "rt/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "rt/r_scalar": r_b.float().mean(),
            "rt/anchor_idx": torch.tensor(float(anchor_idx)),
        }
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Explicit velocity anchor, kept for direct RAM-style ablations."""
    return F.mse_loss(v_default, v_anchor.detach())
