"""Losses for on-policy context distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def context_velocity_distillation_loss(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MSE surrogate for reverse-KL distillation on visited states.

    Both tensors are rectified-flow velocities at the same on-policy noisy
    state `(x_t, t)`. Gradients flow only through `v_student`; callers should
    produce `v_teacher` under `torch.no_grad()`.
    """
    v_teacher_d = v_teacher.detach()
    loss = F.mse_loss(v_student, v_teacher_d)
    with torch.no_grad():
        diff = (v_student - v_teacher_d).float()
        student_f = v_student.float().flatten(1)
        teacher_f = v_teacher_d.float().flatten(1)
        diag = {
            "loss/context_distill": loss.detach(),
            "distill/delta_norm": diff.flatten(1).norm(dim=1).mean(),
            "distill/student_norm": student_f.norm(dim=1).mean(),
            "distill/teacher_norm": teacher_f.norm(dim=1).mean(),
            "distill/cosine": F.cosine_similarity(student_f, teacher_f, dim=1).mean(),
        }
    return loss, diag


def anchor_velocity_loss(
    v_student: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Optional anti-drift loss to the no-context base velocity."""
    return F.mse_loss(v_student, v_anchor.detach())
