import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.motion_projected_em_ram.motion_features import (
    build_reference_motion_descriptor,
    descriptor_pair_metrics,
)


def _synthetic_tracks(
    *,
    residual_sign: float = 1.0,
    global_step: tuple[float, float] = (2.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    t = 6
    n = 8
    base = np.stack([
        np.linspace(20, 80, n),
        np.linspace(30, 90, n),
    ], axis=-1)
    tracks = np.zeros((t, n, 2), dtype=np.float32)
    for i in range(t):
        tracks[i] = base + i * np.asarray(global_step, dtype=np.float32)
        tracks[i, -2:, 1] += residual_sign * i * 5.0
    visibility = np.ones((t, n), dtype=bool)
    return tracks, visibility


def test_descriptor_removes_global_motion_and_selects_residual_tracks() -> None:
    tracks, visibility = _synthetic_tracks()

    desc = build_reference_motion_descriptor(
        tracks,
        visibility,
        frame_height=100,
        frame_width=100,
        moving_percentile=60.0,
        min_moving_tracks=2,
    )

    np.testing.assert_allclose(
        desc.global_motion_xy.mean(axis=0),
        np.asarray([0.02, 0.0]),
        atol=1e-6,
    )
    assert int(desc.moving_mask.sum()) >= 2
    assert desc.summary["residual_speed_mean"] > 0.0
    assert len(desc.bucket_summaries) == 3


def test_descriptor_pair_scores_same_motion_above_opposite_motion() -> None:
    ref_tracks, ref_vis = _synthetic_tracks(residual_sign=1.0)
    same_tracks, same_vis = _synthetic_tracks(residual_sign=1.0)
    opposite_tracks, opposite_vis = _synthetic_tracks(residual_sign=-1.0)

    ref = build_reference_motion_descriptor(
        ref_tracks,
        ref_vis,
        frame_height=100,
        frame_width=100,
        min_moving_tracks=2,
    )
    same = build_reference_motion_descriptor(
        same_tracks,
        same_vis,
        frame_height=100,
        frame_width=100,
        min_moving_tracks=2,
    )
    opposite = build_reference_motion_descriptor(
        opposite_tracks,
        opposite_vis,
        frame_height=100,
        frame_width=100,
        min_moving_tracks=2,
    )

    same_metrics = descriptor_pair_metrics(same, ref)
    opposite_metrics = descriptor_pair_metrics(opposite, ref)

    assert same_metrics["residual_direction"] > 0.99
    assert opposite_metrics["residual_direction"] < -0.99
    assert same_metrics["residual_score"] > opposite_metrics["residual_score"]


def main() -> None:
    test_descriptor_removes_global_motion_and_selects_residual_tracks()
    test_descriptor_pair_scores_same_motion_above_opposite_motion()


if __name__ == "__main__":
    main()
