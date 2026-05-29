import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.motion_projected_em_ram.losses import (
    feature_consistency_gates,
    feature_consistency_weights,
    motion_project,
    motion_projected_em_ram_loss,
    reference_motion_basis,
    reference_motion_project,
    score_consistency_weights,
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


def test_feature_consistency_gates_filter_bad_motion_components() -> None:
    components = [
        {"direction": 0.4, "speed_penalty": 0.1, "speed_ratio": 1.0},
        {"direction": -0.2, "speed_penalty": 0.1, "speed_ratio": 1.0},
        {"direction": 0.6, "speed_penalty": 1.2, "speed_ratio": 4.0},
    ]

    gates, diag = feature_consistency_gates(
        components,
        direction_min=0.0,
        speed_penalty_max=0.75,
    )

    assert torch.equal(gates, torch.tensor([1.0, 0.0, 0.0]))
    assert diag["feature_selector/accepted"] == 1.0
    assert abs(diag["feature_selector/accept_rate"] - 1.0 / 3.0) < 1e-6


def test_feature_consistency_gates_optional_fallback_topk() -> None:
    components = [
        {"direction": -0.3, "speed_penalty": 0.1, "speed_ratio": 1.0},
        {"direction": -0.1, "speed_penalty": 0.4, "speed_ratio": 1.0},
    ]

    gates, diag = feature_consistency_gates(
        components,
        direction_min=0.0,
        speed_penalty_max=0.75,
        fallback_topk=1,
    )

    assert torch.equal(gates, torch.tensor([0.0, 1.0]))
    assert diag["feature_selector/fallback_used"] == 1.0


def test_feature_consistency_weights_are_soft_and_mean_preserving() -> None:
    components = [
        {"direction": 0.6, "speed_penalty": 0.1, "speed_ratio": 1.0},
        {"direction": 0.0, "speed_penalty": 0.3, "speed_ratio": 1.0},
        {"direction": -0.5, "speed_penalty": 1.0, "speed_ratio": 3.0},
    ]

    weights, diag = feature_consistency_weights(
        components,
        direction_center=0.0,
        direction_temperature=0.25,
        speed_penalty_coef=0.25,
        min_weight=0.25,
        max_weight=1.5,
        normalize_mean=True,
    )

    assert torch.all(weights > 0.0)
    assert weights[0] > weights[1] > weights[2]
    assert abs(float(weights.mean()) - diag["feature_selector/weight_mean"]) < 1e-6
    assert abs(float(weights.mean()) - 1.0) < 0.1


def test_score_consistency_weights_downweight_bad_absolute_rewards() -> None:
    rewards = torch.tensor([-0.5, 0.0, 0.5])

    weights, diag = score_consistency_weights(
        rewards,
        score_center=0.0,
        score_temperature=0.25,
        min_weight=0.0,
        max_weight=1.0,
        normalize_mean=False,
    )

    assert torch.all(weights >= 0.0)
    assert weights[0] < weights[1] < weights[2]
    assert abs(float(weights[1]) - 0.5) < 1e-6
    assert diag["feature_selector/score_mean"] == 0.0


def test_hybrid_reference_motion_mix_keeps_coarse_component() -> None:
    torch.manual_seed(2)
    reference = torch.randn(1, 5, 3, 4, 4)
    x = torch.randn_like(reference)
    coarse = motion_project(x, spatial_pool=1, temporal_center=False)
    aligned, _ = reference_motion_project(
        x,
        reference,
        spatial_pool=1,
        temporal_center=False,
        projection_scope="global",
        mix=1.0,
    )

    projected, diag = reference_motion_project(
        x,
        reference,
        spatial_pool=1,
        temporal_center=False,
        projection_scope="global",
        mix=0.25,
    )

    expected = 0.75 * coarse + 0.25 * aligned
    assert torch.allclose(projected, expected, atol=1e-6)
    assert float(diag["mpem/reference_mix"]) == 0.25


def test_hybrid_reference_motion_loss_reports_orthogonal_penalty() -> None:
    torch.manual_seed(3)
    student = torch.zeros(1, 5, 2, 3, 3, requires_grad=True)
    anchor = torch.ones_like(student, requires_grad=True)
    noise = torch.randn_like(student)
    x0 = torch.randn_like(student)
    reference = torch.randn_like(student)
    alpha = torch.ones(1)

    loss, diag = motion_projected_em_ram_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        alpha=alpha,
        x0_reference=reference,
        subspace_mode="hybrid_reference_motion",
        reference_motion_mix=0.25,
        lambda_reference_orthogonal=0.05,
    )
    loss.backward()

    assert student.grad is not None
    assert anchor.grad is None
    assert diag["mpem/subspace_hybrid"] == 1.0
    assert diag["mpem/reference_orthogonal_loss"] >= 0.0


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
    test_feature_consistency_gates_filter_bad_motion_components()
    test_feature_consistency_gates_optional_fallback_topk()
    test_feature_consistency_weights_are_soft_and_mean_preserving()
    test_score_consistency_weights_downweight_bad_absolute_rewards()
    test_hybrid_reference_motion_mix_keeps_coarse_component()
    test_hybrid_reference_motion_loss_reports_orthogonal_penalty()
    test_coarse_motion_loss_keeps_anchor_detached()


if __name__ == "__main__":
    main()
