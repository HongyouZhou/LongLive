"""Motion-only rollout rewards for Motion-Projected EM-RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from longlive.methods.motion_equivariant_ram.reward import motion_equivariant_pair
from longlive.methods.motion_projected_em_ram.motion_features import (
    ReferenceMotionDescriptor,
    build_reference_motion_descriptor,
    descriptor_pair_metrics,
)


MOTION_REWARD_MODES = ("tracklet_scalar", "residual_bucket")


def _frame_size(path: str | Path) -> tuple[int, int]:
    from scripts.motion_eval.metrics.video_io import read_video_frames

    frames = read_video_frames(path, max_frames=1)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected video frames shaped (T, H, W, 3), got {frames.shape}")
    return int(frames.shape[1]), int(frames.shape[2])


def _legacy_residual_components(metrics: dict[str, float]) -> dict[str, float]:
    """Expose residual descriptor metrics through the legacy component keys."""
    components = {
        "score": metrics["residual_score"],
        "direction": metrics["residual_direction"],
        "speed_penalty": metrics["residual_speed_penalty"],
        "speed_ratio": metrics["residual_speed_ratio"],
        **metrics,
    }
    return {key: float(value) for key, value in components.items()}


def motion_projected_reward_components(
    *,
    gen_tracks: np.ndarray,
    gen_visibility: np.ndarray,
    ref_tracks: np.ndarray,
    ref_visibility: np.ndarray,
    motion_mode: str = "tracklet_scalar",
    gen_frame_height: int | float | None = None,
    gen_frame_width: int | float | None = None,
    ref_frame_height: int | float | None = None,
    ref_frame_width: int | float | None = None,
    ref_descriptor: ReferenceMotionDescriptor | None = None,
    bucket_count: int = 3,
    moving_percentile: float = 60.0,
    min_moving_tracks: int = 4,
    moving_speed_floor: float = 1e-5,
    speed_lower: float = 0.5,
    speed_upper: float = 2.0,
    speed_penalty_coef: float = 0.25,
) -> dict[str, float]:
    """Return rollout reward components for the configured motion mode."""
    if motion_mode not in MOTION_REWARD_MODES:
        raise ValueError(
            f"motion_mode must be one of {MOTION_REWARD_MODES}, got {motion_mode!r}"
        )

    if motion_mode == "tracklet_scalar":
        return motion_equivariant_pair(
            gen_tracks=gen_tracks,
            gen_visibility=gen_visibility,
            ref_tracks=ref_tracks,
            ref_visibility=ref_visibility,
            speed_lower=speed_lower,
            speed_upper=speed_upper,
            speed_penalty_coef=speed_penalty_coef,
        )

    if gen_frame_height is None or gen_frame_width is None:
        raise ValueError("residual_bucket reward requires generated frame height/width")
    if ref_descriptor is None and (ref_frame_height is None or ref_frame_width is None):
        raise ValueError("residual_bucket reward requires reference frame height/width")

    if ref_descriptor is None:
        ref_descriptor = build_reference_motion_descriptor(
            ref_tracks,
            ref_visibility,
            frame_height=float(ref_frame_height),
            frame_width=float(ref_frame_width),
            bucket_count=bucket_count,
            moving_percentile=moving_percentile,
            min_moving_tracks=min_moving_tracks,
            moving_speed_floor=moving_speed_floor,
        )
    gen_descriptor = build_reference_motion_descriptor(
        gen_tracks,
        gen_visibility,
        frame_height=float(gen_frame_height),
        frame_width=float(gen_frame_width),
        bucket_count=bucket_count,
        moving_percentile=moving_percentile,
        min_moving_tracks=min_moving_tracks,
        moving_speed_floor=moving_speed_floor,
    )
    metrics = descriptor_pair_metrics(
        gen_descriptor,
        ref_descriptor,
        speed_lower=speed_lower,
        speed_upper=speed_upper,
        speed_penalty_coef=speed_penalty_coef,
    )
    return _legacy_residual_components(metrics)


class MotionProjectedEMReward:
    """Score one generated video against one cached reference video."""

    def __init__(
        self,
        ref_path: str | Path,
        scratch_dir: str | Path,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        n_frames: int = 16,
        grid_size: int = 30,
        fps: int = 16,
        speed_lower: float = 0.5,
        speed_upper: float = 2.0,
        speed_penalty_coef: float = 0.25,
        motion_mode: str = "tracklet_scalar",
        bucket_count: int = 3,
        moving_percentile: float = 60.0,
        min_moving_tracks: int = 4,
        moving_speed_floor: float = 1e-5,
    ):
        from scripts.motion_eval.metrics.motion_fidelity import CoTrackerWrapper

        if motion_mode not in MOTION_REWARD_MODES:
            raise ValueError(
                f"motion_mode must be one of {MOTION_REWARD_MODES}, got {motion_mode!r}"
            )

        self.ref_path = Path(ref_path)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.speed_lower = float(speed_lower)
        self.speed_upper = float(speed_upper)
        self.speed_penalty_coef = float(speed_penalty_coef)
        self.motion_mode = str(motion_mode)
        self.bucket_count = int(bucket_count)
        self.moving_percentile = float(moving_percentile)
        self.min_moving_tracks = int(min_moving_tracks)
        self.moving_speed_floor = float(moving_speed_floor)
        self.last_components: dict[str, float] = {}

        self.tracker = CoTrackerWrapper(
            device=device,
            cache_dir=cache_dir,
            n_frames=n_frames,
            grid_size=grid_size,
        )
        self.ref_tracks, self.ref_visibility = self.tracker.get_tracklets(self.ref_path)
        self.ref_frame_height: int | None = None
        self.ref_frame_width: int | None = None
        self.ref_descriptor: ReferenceMotionDescriptor | None = None

        if self.motion_mode == "residual_bucket":
            self.ref_frame_height, self.ref_frame_width = _frame_size(self.ref_path)
            self.ref_descriptor = build_reference_motion_descriptor(
                self.ref_tracks,
                self.ref_visibility,
                frame_height=self.ref_frame_height,
                frame_width=self.ref_frame_width,
                bucket_count=self.bucket_count,
                moving_percentile=self.moving_percentile,
                min_moving_tracks=self.min_moving_tracks,
                moving_speed_floor=self.moving_speed_floor,
            )

    def _write_mp4(self, video: torch.Tensor, tag: str) -> Path:
        from torchvision.io import write_video

        frames = (video.detach().clamp(0, 1).permute(0, 2, 3, 1) * 255.0).round()
        frames = frames.to(torch.uint8).cpu()
        out_path = self.scratch_dir / f"rollout_{tag}.mp4"
        write_video(str(out_path), frames, fps=self.fps, video_codec="libx264")
        return out_path

    @torch.no_grad()
    def score(self, video: torch.Tensor, tag: str) -> float:
        gen_path = self._write_mp4(video, tag)
        try:
            gen_tracks, gen_vis = self.tracker.get_tracklets(gen_path)
            components = motion_projected_reward_components(
                gen_tracks=gen_tracks,
                gen_visibility=gen_vis,
                ref_tracks=self.ref_tracks,
                ref_visibility=self.ref_visibility,
                motion_mode=self.motion_mode,
                gen_frame_height=int(video.shape[-2]),
                gen_frame_width=int(video.shape[-1]),
                ref_frame_height=self.ref_frame_height,
                ref_frame_width=self.ref_frame_width,
                ref_descriptor=self.ref_descriptor,
                bucket_count=self.bucket_count,
                moving_percentile=self.moving_percentile,
                min_moving_tracks=self.min_moving_tracks,
                moving_speed_floor=self.moving_speed_floor,
                speed_lower=self.speed_lower,
                speed_upper=self.speed_upper,
                speed_penalty_coef=self.speed_penalty_coef,
            )
            self.last_components = components
        finally:
            try:
                gen_path.unlink()
            except FileNotFoundError:
                pass
        return float(components["score"])


__all__ = [
    "MOTION_REWARD_MODES",
    "MotionProjectedEMReward",
    "motion_equivariant_pair",
    "motion_projected_reward_components",
]
