"""Motion-Projected EM-RAM losses for reward-tilted few-step distillation.

The E-step is unchanged from EM-RAM: build a reward-tilted endpoint
distribution under a KL(q || uniform) budget.  The M-step changes the update
subspace:

  E-step:  q_i proportional to exp(A_i / eta), with KL(q || uniform) controlled.
  M-step:  alpha_i gates only a configured motion projection of RAM's
           residual; all other velocity components stay anchored to the
           frozen LongLive base.

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


def em_tilt_alpha_and_weights(
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute empirical EM alpha and importance weights from rollout rewards.

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
        (alpha, importance_weight, diagnostics), where both tensors have shape
        (N,).  ``alpha`` is the historical MP-EM-RAM target-shift coefficient.
        ``importance_weight`` is the clipped EM importance weight ``N*q`` for
        reward-weighted M-step objectives.
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
    importance_weight = w_clipped.float()

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
        "em/importance_weight_min": float(importance_weight.min()),
        "em/importance_weight_max": float(importance_weight.max()),
        "em/importance_weight_mean": float(importance_weight.mean()),
    }
    return alpha.float(), importance_weight, diag


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
    """Backward-compatible wrapper returning only historical EM alpha."""
    alpha, _importance_weight, diag = em_tilt_alpha_and_weights(
        rewards,
        target_kl=target_kl,
        eta=eta,
        adv_clip_max=adv_clip_max,
        weight_clip=weight_clip,
        alpha_mode=alpha_mode,
        alpha_max=alpha_max,
        std_floor=std_floor,
        device=device,
    )
    return alpha, diag


def feature_consistency_gates(
    components: list[dict[str, float]],
    *,
    direction_min: float = 0.0,
    speed_penalty_max: float | None = None,
    speed_ratio_min: float | None = None,
    speed_ratio_max: float | None = None,
    fallback_topk: int = 0,
    fallback_speed_penalty_coef: float = 0.25,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return per-rollout gates for feature-consistent motion selection.

    This is deliberately a selector, not a new loss.  EM still ranks endpoints
    by reward, but an endpoint can update the model only when its CoTracker
    direction is compatible with the reference and its speed is not far outside
    the configured band.  Pixel/appearance information is not inspected.
    """
    if not components:
        raise ValueError("feature_consistency_gates requires at least one component dict")

    direction = torch.tensor(
        [float(c.get("direction", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )
    speed_penalty = torch.tensor(
        [float(c.get("speed_penalty", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )
    speed_ratio = torch.tensor(
        [float(c.get("speed_ratio", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )

    gate = direction >= float(direction_min)
    if speed_penalty_max is not None:
        gate = gate & (speed_penalty <= float(speed_penalty_max))
    if speed_ratio_min is not None:
        gate = gate & (speed_ratio >= float(speed_ratio_min))
    if speed_ratio_max is not None:
        gate = gate & (speed_ratio <= float(speed_ratio_max))

    accepted_before_fallback = int(gate.sum().item())
    fallback_used = 0
    topk = int(fallback_topk)
    if accepted_before_fallback == 0 and topk > 0:
        k = min(topk, direction.numel())
        selector_score = direction - float(fallback_speed_penalty_coef) * speed_penalty
        top_idx = torch.topk(selector_score, k=k).indices
        gate[top_idx] = True
        fallback_used = 1

    gate_f = gate.float()
    diag = {
        "feature_selector/accepted": float(gate_f.sum().item()),
        "feature_selector/accept_rate": float(gate_f.mean().item()),
        "feature_selector/fallback_used": float(fallback_used),
        "feature_selector/direction_mean": float(direction.mean().item()),
        "feature_selector/direction_min": float(direction.min().item()),
        "feature_selector/direction_max": float(direction.max().item()),
        "feature_selector/speed_penalty_mean": float(speed_penalty.mean().item()),
        "feature_selector/speed_ratio_mean": float(speed_ratio.mean().item()),
    }
    return gate_f, diag


def feature_consistency_weights(
    components: list[dict[str, float]],
    *,
    direction_center: float = 0.0,
    direction_temperature: float = 0.25,
    speed_penalty_coef: float = 0.25,
    min_weight: float = 0.25,
    max_weight: float = 1.5,
    normalize_mean: bool = True,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return soft per-rollout weights from CoTracker motion components.

    Unlike the hard selector, this keeps every endpoint trainable and only
    redistributes EM's update mass toward feature-consistent motion.  Mean-one
    normalization preserves the coarse MP-EM-RAM update budget, which is useful
    when the whole rollout group is mediocre but still contains a relative best
    endpoint.
    """
    if not components:
        raise ValueError("feature_consistency_weights requires at least one component dict")

    direction = torch.tensor(
        [float(c.get("direction", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )
    speed_penalty = torch.tensor(
        [float(c.get("speed_penalty", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )
    speed_ratio = torch.tensor(
        [float(c.get("speed_ratio", 0.0)) for c in components],
        dtype=torch.float32,
        device=device,
    )

    temp = max(float(direction_temperature), 1e-6)
    direction_weight = torch.sigmoid((direction - float(direction_center)) / temp)
    speed_weight = torch.exp(-float(speed_penalty_coef) * speed_penalty.clamp_min(0.0))
    weights = direction_weight * speed_weight
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(float(min_weight), float(max_weight))

    diag = {
        "feature_selector/weight_mean": float(weights.mean().item()),
        "feature_selector/weight_min": float(weights.min().item()),
        "feature_selector/weight_max": float(weights.max().item()),
        "feature_selector/direction_mean": float(direction.mean().item()),
        "feature_selector/direction_min": float(direction.min().item()),
        "feature_selector/direction_max": float(direction.max().item()),
        "feature_selector/speed_penalty_mean": float(speed_penalty.mean().item()),
        "feature_selector/speed_ratio_mean": float(speed_ratio.mean().item()),
    }
    return weights, diag


def score_consistency_weights(
    rewards: list[float] | torch.Tensor,
    *,
    score_center: float = 0.0,
    score_temperature: float = 0.25,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    normalize_mean: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return soft feasibility weights from absolute rollout reward.

    EM's KL tilt is relative within the sampled group, so a rollout can receive
    positive update mass even when every endpoint is poor in absolute motion
    reward.  This weight is an absolute acceptor: endpoints below the motion
    reward floor still participate in ranking, but their M-step update mass is
    reduced instead of being preserved by mean normalization.
    """
    rewards_t = _as_float_tensor(rewards, device=device).view(-1)
    if rewards_t.numel() == 0:
        raise ValueError("score_consistency_weights requires at least one reward")

    temp = max(float(score_temperature), 1e-6)
    weights = torch.sigmoid((rewards_t - float(score_center)) / temp)
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(float(min_weight), float(max_weight))

    diag = {
        "feature_selector/weight_mean": float(weights.mean().item()),
        "feature_selector/weight_min": float(weights.min().item()),
        "feature_selector/weight_max": float(weights.max().item()),
        "feature_selector/score_mean": float(rewards_t.mean().item()),
        "feature_selector/score_min": float(rewards_t.min().item()),
        "feature_selector/score_max": float(rewards_t.max().item()),
    }
    return weights, diag


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


def _project_ram_shift(
    raw_shift: torch.Tensor,
    *,
    x0_reference: torch.Tensor | None = None,
    subspace_mode: str = "coarse_motion",
    motion_pool: int = 2,
    motion_temporal_center: bool = True,
    reference_motion_scope: str = "frame",
    reference_motion_positive: bool = False,
    reference_motion_mix: float = 1.0,
    reference_motion_temporal_center: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], bool]:
    subspace = str(subspace_mode)
    projector_diag: dict[str, torch.Tensor] = {}
    uses_reference_subspace = subspace in ("reference_motion", "hybrid_reference_motion")
    if subspace == "coarse_motion":
        motion_shift = motion_project(
            raw_shift,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
        )
    elif uses_reference_subspace:
        if x0_reference is None:
            raise ValueError(
                f"subspace_mode={subspace!r} requires x0_reference"
            )
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
    return motion_shift, projector_diag, uses_reference_subspace


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
    lambda_reference_orthogonal: float = 0.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """EM-gated RAM residual, projected onto the configured motion subspace.

    Target construction:
        raw_shift    = (eps - x0) - stopgrad(v_theta)
        motion_shift = P_subspace(raw_shift)
        target       = v_anchor + reward_coef * alpha * motion_shift

    `alpha=0` makes the step a pure anchor update.  Positive alpha applies the
    RAM residual only in the projected motion subspace selected by the E-step.
    """
    v_anchor_d = v_anchor.detach()
    alpha_b = alpha.to(v_default.dtype).view(-1, *([1] * (v_default.ndim - 1)))
    alpha_eff = float(reward_coef) * alpha_b

    raw_shift = (noise - x0_ref) - v_default.detach()
    motion_shift, projector_diag, uses_reference_subspace = _project_ram_shift(
        raw_shift,
        x0_reference=x0_reference,
        subspace_mode=subspace_mode,
        motion_pool=motion_pool,
        motion_temporal_center=motion_temporal_center,
        reference_motion_scope=reference_motion_scope,
        reference_motion_positive=reference_motion_positive,
        reference_motion_mix=reference_motion_mix,
        reference_motion_temporal_center=reference_motion_temporal_center,
    )
    target = v_anchor_d + alpha_eff * motion_shift
    motion_loss = F.mse_loss(v_default, target.detach())

    static_default = v_default.mean(dim=1, keepdim=True)
    static_anchor = v_anchor_d.mean(dim=1, keepdim=True)
    static_loss = F.mse_loss(static_default, static_anchor)

    reference_orthogonal_loss = torch.zeros((), device=v_default.device, dtype=v_default.dtype)
    if uses_reference_subspace and float(lambda_reference_orthogonal) > 0.0:
        assert x0_reference is not None
        delta_v = v_default - v_anchor_d
        coarse_delta = motion_project(
            delta_v,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
        )
        ref_delta, _ref_delta_diag = reference_motion_project(
            delta_v,
            x0_reference,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
            reference_temporal_center=reference_motion_temporal_center,
            projection_scope=reference_motion_scope,
            positive_only=False,
            mix=1.0,
        )
        reference_orthogonal_loss = (coarse_delta - ref_delta).float().square().mean()

    loss = (
        float(lambda_motion) * motion_loss
        + float(lambda_static) * static_loss
        + float(lambda_reference_orthogonal) * reference_orthogonal_loss
    )

    with torch.no_grad():
        diag = {
            "loss/mpem": loss.detach(),
            "mpem/motion_loss": motion_loss.detach(),
            "mpem/static_loss": static_loss.detach(),
            "mpem/reference_orthogonal_loss": reference_orthogonal_loss.detach(),
            "mpem/raw_shift_norm": raw_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/motion_shift_norm": motion_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "mpem/alpha": alpha_b.float().mean(),
            "mpem/alpha_eff": alpha_eff.float().mean(),
            "mpem/anchor_idx": torch.tensor(float(anchor_idx)),
            "mpem/subspace_reference": torch.tensor(float(uses_reference_subspace)),
            "mpem/subspace_hybrid": torch.tensor(float(str(subspace_mode) == "hybrid_reference_motion")),
        }
        diag.update(projector_diag)
    return loss, diag


def reward_weighted_velocity_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
    noise: torch.Tensor,
    x0_ref: torch.Tensor,
    loss_weight: torch.Tensor,
    x0_reference: torch.Tensor | None = None,
    shift_coef: float = 0.25,
    anchor_beta: float = 0.1,
    lambda_motion: float = 1.0,
    lambda_static: float = 0.05,
    subspace_mode: str = "coarse_motion",
    motion_pool: int = 2,
    motion_temporal_center: bool = True,
    reference_motion_scope: str = "frame",
    reference_motion_positive: bool = False,
    reference_motion_mix: float = 1.0,
    reference_motion_temporal_center: bool = False,
    lambda_reference_orthogonal: float = 0.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Reward-weighted RAM residual with fixed target-shift magnitude.

    This objective decouples rollout confidence from target amplitude:

        target = v_anchor + shift_coef * P_motion(raw_shift)
        loss   = w_i * ||v_default - sg(target)||^2
               + anchor_beta * ||v_default - sg(v_anchor)||^2

    The reward/EM information enters only through ``loss_weight``.  The target
    displacement from LongLive is controlled by ``shift_coef``.
    """
    v_anchor_d = v_anchor.detach()
    weight = loss_weight.to(v_default.dtype).view(-1)
    if weight.numel() != v_default.shape[0]:
        if weight.numel() == 1:
            weight = weight.expand(v_default.shape[0])
        else:
            raise ValueError(
                f"loss_weight batch mismatch: weights={tuple(loss_weight.shape)} "
                f"v_default={tuple(v_default.shape)}"
            )

    raw_shift = (noise - x0_ref) - v_default.detach()
    motion_shift, projector_diag, uses_reference_subspace = _project_ram_shift(
        raw_shift,
        x0_reference=x0_reference,
        subspace_mode=subspace_mode,
        motion_pool=motion_pool,
        motion_temporal_center=motion_temporal_center,
        reference_motion_scope=reference_motion_scope,
        reference_motion_positive=reference_motion_positive,
        reference_motion_mix=reference_motion_mix,
        reference_motion_temporal_center=reference_motion_temporal_center,
    )
    target = v_anchor_d + float(shift_coef) * motion_shift

    per_sample_motion = (
        v_default.float() - target.detach().float()
    ).square().flatten(1).mean(dim=1)
    weighted_motion_loss = (weight.float() * per_sample_motion).mean()
    anchor_loss = F.mse_loss(v_default, v_anchor_d)

    static_default = v_default.mean(dim=1, keepdim=True)
    static_anchor = v_anchor_d.mean(dim=1, keepdim=True)
    static_loss = F.mse_loss(static_default, static_anchor)

    reference_orthogonal_loss = torch.zeros((), device=v_default.device, dtype=v_default.dtype)
    if uses_reference_subspace and float(lambda_reference_orthogonal) > 0.0:
        assert x0_reference is not None
        delta_v = v_default - v_anchor_d
        coarse_delta = motion_project(
            delta_v,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
        )
        ref_delta, _ref_delta_diag = reference_motion_project(
            delta_v,
            x0_reference,
            spatial_pool=motion_pool,
            temporal_center=motion_temporal_center,
            reference_temporal_center=reference_motion_temporal_center,
            projection_scope=reference_motion_scope,
            positive_only=False,
            mix=1.0,
        )
        reference_orthogonal_loss = (coarse_delta - ref_delta).float().square().mean()

    loss = (
        float(lambda_motion) * weighted_motion_loss
        + float(anchor_beta) * anchor_loss
        + float(lambda_static) * static_loss
        + float(lambda_reference_orthogonal) * reference_orthogonal_loss
    )

    with torch.no_grad():
        diag = {
            "loss/mpem": loss.detach(),
            "mpem/motion_loss": weighted_motion_loss.detach(),
            "mpem/unweighted_motion_loss": per_sample_motion.mean().detach(),
            "mpem/anchor_loss": anchor_loss.detach(),
            "mpem/static_loss": static_loss.detach(),
            "mpem/reference_orthogonal_loss": reference_orthogonal_loss.detach(),
            "mpem/raw_shift_norm": raw_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/motion_shift_norm": motion_shift.float().flatten(1).norm(dim=1).mean(),
            "mpem/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "mpem/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "mpem/alpha": weight.float().mean(),
            "mpem/alpha_eff": weight.float().mean(),
            "mpem/loss_weight": weight.float().mean(),
            "mpem/loss_weight_min": weight.float().min(),
            "mpem/loss_weight_max": weight.float().max(),
            "mpem/shift_coef": torch.tensor(float(shift_coef), device=v_default.device),
            "mpem/anchor_beta": torch.tensor(float(anchor_beta), device=v_default.device),
            "mpem/anchor_idx": torch.tensor(float(anchor_idx)),
            "mpem/subspace_reference": torch.tensor(float(uses_reference_subspace)),
            "mpem/subspace_hybrid": torch.tensor(float(str(subspace_mode) == "hybrid_reference_motion")),
        }
        diag.update(projector_diag)
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional direct velocity anchor on top of the projected EM-RAM target."""
    return F.mse_loss(v_default, v_anchor.detach())
