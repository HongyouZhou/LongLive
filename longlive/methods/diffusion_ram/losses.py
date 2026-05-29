"""RAM (Reinforce Adjoint Matching) loss — paper Eq. 17, v-space.

Single MSE between v_θ(X_t) and a stop-grad target that pulls toward the
frozen LongLive base anchor when reward is low and toward the rectified-flow
target (ε − X_0) when reward is high.

Conventions (Wan / our `WanDiffusionWrapper`):
  * The generator outputs (flow_pred, pred_x0) = (v_θ, x_t − σ_t · v_θ).
  * `flow_pred = v_θ` is rectified-flow velocity, equal to (ε − x_0) at
    optimal pretraining (paper Eq. 4).
  * `FlowMatchScheduler.add_noise(x_0, ε, t)` returns `(1 − σ_t)·x_0 + σ_t·ε`
    with σ_t = t/1000 — matches paper Eq. 1 with t-axis rescaled by 1000.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ram_loss(
    v_default: torch.Tensor,     # (B, F, C, H, W) bf16, grad ON  — v_θ(X_t)
    v_anchor: torch.Tensor,      # (B, F, C, H, W) bf16, no_grad   — base anchor output
    noise: torch.Tensor,         # (B, F, C, H, W) bf16            — ε used to build X_t
    x0_ref: torch.Tensor,        # (B, F, C, H, W) bf16            — rollout endpoint x_0
    r: torch.Tensor,             # (1,) float32                    — group-normed reward ∈ [0, 1]
    reward_coef: float = 1.0,
    anchor_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """RAM Eq. 17 in v-space, with reward as a multiplicative target-shift coefficient.

    Target construction (all stop-grad):
        shift  = (ε − x_0) − v_θ
        target = v_anchor + reward_coef · r · shift
        loss   = ‖v_θ − sg(target)‖²

    When r → 0, target collapses to v_anchor → loss pulls v_θ back to the
    frozen LongLive base. When r → 1, target equals
    v_anchor + coef · (ε − x_0 − v_θ); gradient direction is set by
    (ε − x_0) − v_θ, which is the rectified-flow pretraining residual.

    Args:
        v_default: trainable network output at (X_t, t).
        v_anchor: frozen LongLive base output at the same (X_t, t).  Caller
            must have produced this under `no_grad` with the trainable LoRA
            disabled (via the zero-init "anchor" adapter in our setup).
        noise: the ε tensor that was used to construct X_t.  RAM uses the raw
            ε − x_0, NOT the rescaled (X_t − x_0)/σ_t form.
        x0_ref: the clean endpoint latent x_0 (from rollout).
        r: scalar group-normed reward in [0, 1].  Cast to v_default.dtype
            inside this function to avoid silent upcast of `target` to fp32.
        reward_coef: paper §D multiplies normalized reward by 100 / 1000 for
            their binary/partial-credit rewards.  Our motion_fidelity is
            continuous in [-1, 1]; we start at 1.0 and sweep.
        anchor_idx: 0..3 of the DMD anchor t used (logging only).

    Returns:
        (loss, diag) — diag has the per-inner-step scalars the trainer logs.
    """
    v_anchor_d = v_anchor.detach()
    # Cast r to v_default.dtype once.  Single scalar, bf16 precision is fine
    # since r ∈ [0, 1]; prevents silent fp32 upcast of `target` (which would
    # double its memory footprint inside the inner forward).
    r_b = r.to(v_default.dtype).view(-1, *([1] * (v_default.ndim - 1)))

    shift = (noise - x0_ref) - v_default.detach()
    target = v_anchor_d + (reward_coef * r_b) * shift
    loss = F.mse_loss(v_default, target)

    with torch.no_grad():
        diag = {
            "loss/ram": loss.detach(),
            "ram/shift_norm": shift.float().flatten(1).norm(dim=1).mean(),
            "ram/target_norm": target.float().flatten(1).norm(dim=1).mean(),
            "ram/v_default_norm": v_default.float().flatten(1).norm(dim=1).mean(),
            "ram/v_anchor_norm": v_anchor_d.float().flatten(1).norm(dim=1).mean(),
            "ram/r_scalar": r_b.float().mean(),
            "ram/anchor_idx": torch.tensor(float(anchor_idx)),
        }
    return loss, diag


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional anti-drift MSE between v_θ and the frozen LongLive anchor.

    RAM's target form already anchors v_θ toward v_anchor when r is low — this
    function is exposed for runs that want an *explicit* β_KL penalty on top
    (gated by `beta_kl > 0` in the yaml; off by default).
    """
    return F.mse_loss(v_default, v_anchor.detach())
