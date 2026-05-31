import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.motion_projected_em_ram.losses import (
    em_tilt_alpha_and_weights,
    em_tilt_weights,
    feature_consistency_gates,
    feature_consistency_weights,
    mode_cover_velocity_loss,
    motion_project,
    motion_projected_em_ram_loss,
    reference_motion_basis,
    reference_motion_project,
    residual_bucket_time_weights,
    reward_weighted_velocity_loss,
    score_consistency_weights,
    time_local_reward_weighted_velocity_loss,
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


def test_residual_bucket_time_weights_map_bucket_directions_to_frames() -> None:
    components = [
        {
            "bucket0_direction": 1.0,
            "bucket1_direction": 0.0,
            "bucket2_direction": -1.0,
        }
    ]

    weights, diag = residual_bucket_time_weights(
        components,
        latent_frames=6,
        bucket_count=3,
        direction_temperature=0.25,
    )

    assert weights.shape == (1, 6)
    assert torch.all(weights[:, :2] > weights[:, 2:4])
    assert torch.all(weights[:, 2:4] > weights[:, 4:])
    assert diag["time_weight/max"] <= 1.0


def test_residual_bucket_time_weights_requires_bucket_components() -> None:
    try:
        residual_bucket_time_weights(
            [{"direction": 0.5}],
            latent_frames=4,
            bucket_count=2,
        )
    except ValueError as exc:
        assert "residual-bucket" in str(exc)
    else:
        raise AssertionError("missing bucket components should fail")


def test_em_tilt_exposes_importance_weights_without_changing_alpha() -> None:
    rewards = torch.tensor([-1.0, 0.0, 2.0])

    alpha, importance, diag = em_tilt_alpha_and_weights(
        rewards,
        target_kl=0.05,
        weight_clip=4.0,
        alpha_max=1.0,
    )
    alpha_compat, diag_compat = em_tilt_weights(
        rewards,
        target_kl=0.05,
        weight_clip=4.0,
        alpha_max=1.0,
    )

    assert torch.allclose(alpha, alpha_compat)
    assert importance.shape == rewards.shape
    assert importance[-1] > importance[0]
    assert diag["em/importance_weight_mean"] > 0.0
    assert diag_compat["em/alpha_mean"] == diag["em/alpha_mean"]


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


def test_reward_weighted_velocity_loss_weights_loss_not_target_shift() -> None:
    student = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.zeros_like(student, requires_grad=True)
    noise = torch.ones_like(student)
    x0 = torch.zeros_like(student)

    loss_low, diag_low = reward_weighted_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        loss_weight=torch.tensor([0.25]),
        shift_coef=0.5,
        anchor_beta=0.0,
        lambda_static=0.0,
        subspace_mode="coarse_motion",
        motion_pool=1,
        motion_temporal_center=False,
    )
    loss_high, diag_high = reward_weighted_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        loss_weight=torch.tensor([2.0]),
        shift_coef=0.5,
        anchor_beta=0.0,
        lambda_static=0.0,
        subspace_mode="coarse_motion",
        motion_pool=1,
        motion_temporal_center=False,
    )

    assert torch.allclose(diag_low["mpem/target_norm"], diag_high["mpem/target_norm"])
    assert loss_high > loss_low
    assert diag_high["mpem/loss_weight"] > diag_low["mpem/loss_weight"]


def test_reward_weighted_velocity_loss_keeps_anchor_when_weight_zero() -> None:
    student = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.ones_like(student, requires_grad=True)
    noise = torch.zeros_like(student)
    x0 = torch.zeros_like(student)

    loss, diag = reward_weighted_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        loss_weight=torch.tensor([0.0]),
        shift_coef=0.5,
        anchor_beta=0.1,
        lambda_static=0.0,
        subspace_mode="coarse_motion",
    )
    loss.backward()

    assert float(loss) > 0.0
    assert student.grad is not None
    assert anchor.grad is None
    assert diag["mpem/loss_weight"] == 0.0
    assert diag["mpem/anchor_loss"] > 0.0


def test_time_local_reward_weighted_velocity_masks_target_shift() -> None:
    student = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.zeros_like(student, requires_grad=True)
    noise = torch.ones_like(student)
    x0 = torch.zeros_like(student)

    loss_zero, diag_zero = time_local_reward_weighted_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        loss_weight=torch.tensor([1.0]),
        time_weight=torch.zeros(1, 4),
        shift_coef=0.5,
        local_anchor_beta=0.0,
        lambda_static=0.0,
        subspace_mode="coarse_motion",
        motion_pool=1,
        motion_temporal_center=False,
    )
    loss_full, diag_full = time_local_reward_weighted_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        noise=noise,
        x0_ref=x0,
        loss_weight=torch.tensor([1.0]),
        time_weight=torch.ones(1, 4),
        shift_coef=0.5,
        local_anchor_beta=0.0,
        lambda_static=0.0,
        subspace_mode="coarse_motion",
        motion_pool=1,
        motion_temporal_center=False,
    )

    assert loss_full > loss_zero
    assert diag_full["mpem/motion_shift_norm"] > diag_zero["mpem/motion_shift_norm"]
    assert diag_zero["mpem/time_weight_mean"] == 0.0


def test_mode_cover_velocity_loss_detaches_anchor() -> None:
    student = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    anchor = torch.ones_like(student, requires_grad=True)

    loss, diag = mode_cover_velocity_loss(
        v_default=student,
        v_anchor=anchor,
        cover_loss_weight=1.0,
        lambda_static=0.0,
    )
    loss.backward()

    assert student.grad is not None
    assert anchor.grad is None
    assert diag["mpem/cover_stream"] == 1.0
    assert diag["mpem/cover_loss"] > 0.0


def main() -> None:
    test_reference_motion_project_recovers_basis_direction()
    test_reference_motion_positive_clamps_opposite_direction()
    test_reference_motion_loss_requires_reference_latent()
    test_feature_consistency_gates_filter_bad_motion_components()
    test_feature_consistency_gates_optional_fallback_topk()
    test_feature_consistency_weights_are_soft_and_mean_preserving()
    test_score_consistency_weights_downweight_bad_absolute_rewards()
    test_residual_bucket_time_weights_map_bucket_directions_to_frames()
    test_residual_bucket_time_weights_requires_bucket_components()
    test_em_tilt_exposes_importance_weights_without_changing_alpha()
    test_hybrid_reference_motion_mix_keeps_coarse_component()
    test_hybrid_reference_motion_loss_reports_orthogonal_penalty()
    test_coarse_motion_loss_keeps_anchor_detached()
    test_reward_weighted_velocity_loss_weights_loss_not_target_shift()
    test_reward_weighted_velocity_loss_keeps_anchor_when_weight_zero()
    test_time_local_reward_weighted_velocity_masks_target_shift()
    test_mode_cover_velocity_loss_detaches_anchor()


if __name__ == "__main__":
    main()
