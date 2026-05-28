import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.on_policy_context_distillation.losses import (
    anchor_velocity_loss,
    context_velocity_distillation_loss,
)
from longlive.methods.on_policy_context_distillation.teacher import optional_path


def test_context_velocity_distillation_loss_is_student_only() -> None:
    student = torch.zeros(2, 3, 4, requires_grad=True)
    teacher = torch.ones(2, 3, 4, requires_grad=True)

    loss, diag = context_velocity_distillation_loss(student, teacher)
    loss.backward()

    assert torch.isclose(loss.detach(), torch.tensor(1.0))
    assert student.grad is not None
    assert teacher.grad is None
    assert "distill/delta_norm" in diag
    assert "distill/cosine" in diag


def test_anchor_velocity_loss_detaches_anchor() -> None:
    student = torch.zeros(1, 2, requires_grad=True)
    anchor = torch.ones(1, 2, requires_grad=True)

    loss = anchor_velocity_loss(student, anchor)
    loss.backward()

    assert student.grad is not None
    assert anchor.grad is None


def test_optional_path_normalizes_empty_values() -> None:
    assert optional_path(None) is None
    assert optional_path("") is None
    assert optional_path("null") is None
    assert optional_path("None") is None
    assert optional_path("~/x").endswith("/x")


def main() -> None:
    test_context_velocity_distillation_loss_is_student_only()
    test_anchor_velocity_loss_detaches_anchor()
    test_optional_path_normalizes_empty_values()


if __name__ == "__main__":
    main()
