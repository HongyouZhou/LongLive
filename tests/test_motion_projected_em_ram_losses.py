import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.motion_projected_em_ram.losses import (
    motion_projected_em_ram_loss,
    reference_motion_basis,
    reference_motion_project,
)


def test_reference_motion_project_recovers_basis_direction() -> None:
    torch.manual_seed(0)
    reference = torch.randn(1, 5, 3, 4, 4)
    like = torch.zeros_like(reference)
    basis = reference_motion_basis(reference, like=like, spatial_pool=1)

    projected, diag = reference_motion_project(
        basis,
        reference,
        spatial_pool=1,
        temporal_center=False,
        projection_scope="global",
    )

    assert torch.allclose(projected, basis, atol=1e-5, rtol=1e-5)
    assert diag["mpem/reference_alignment_cos"] > 0.999


def test_reference_motion_positive_clamps_opposite_direction() -> None:
    torch.manual_seed(1)
    reference = torch.randn(1, 5, 3, 4, 4)
    like = torch.zeros_like(reference)
    basis = reference_motion_basis(reference, like=like, spatial_pool=1)

    projected, diag = reference_motion_project(
        -basis,
        reference,
        spatial_pool=1,
        temporal_center=False,
        projection_scope="global",
        positive_only=True,
    )

    assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-6)
    assert diag["mpem/reference_coef_mean"] == 0.0


def test_reference_motion_loss_requires_reference_latent() -> None:
    x = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.zeros_like(x)
    noise = torch.randn_like(x)
    x0 = torch.randn_like(x)
    alpha = torch.ones(1)

    try:
        motion_projected_em_ram_loss(
            v_default=x,
            v_anchor=anchor,
            noise=noise,
            x0_ref=x0,
            alpha=alpha,
            subspace_mode="reference_motion",
        )
    except ValueError as exc:
        assert "requires x0_reference" in str(exc)
    else:
        raise AssertionError("reference_motion mode should require x0_reference")


def test_coarse_motion_loss_keeps_anchor_detached() -> None:
    student = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.ones_like(student, requires_grad=True)
    noise = torch.randn_like(student)
    x0 = torch.randn_like(student)
    alpha = torch.ones(1)

    loss, diag = motion_projected_em_ram_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        alpha=alpha,
        subspace_mode="coarse_motion",
    )
    loss.backward()

    assert student.grad is not None
    assert anchor.grad is None
    assert "mpem/motion_shift_norm" in diag


def main() -> None:
    test_reference_motion_project_recovers_basis_direction()
    test_reference_motion_positive_clamps_opposite_direction()
    test_reference_motion_loss_requires_reference_latent()
    test_coarse_motion_loss_keeps_anchor_detached()


if __name__ == "__main__":
    main()
