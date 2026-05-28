"""Motion-equivalent rollout reward.

This is deliberately narrower than image/video quality rewards.  It uses
CoTracker tracklets to score motion direction and speed-ratio consistency, and
does not inspect color, texture, caption similarity, or pixel reconstruction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def motion_equivariant_pair(
    gen_tracks: np.ndarray,
    gen_visibility: np.ndarray,
    ref_tracks: np.ndarray,
    ref_visibility: np.ndarray,
    speed_lower: float = 0.5,
    speed_upper: float = 2.0,
    speed_penalty_coef: float = 0.25,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Tracklet direction score with a soft speed-ratio band penalty.

    Direction follows the existing Yatim-style metric: for each reference
    tracklet, pick the generated tracklet with the highest mean velocity cosine.
    Speed is then compared on those same best matches, so over-dynamic motion
    cannot win purely by moving more.
    """
    assert gen_tracks.shape[0] == ref_tracks.shape[0], (
        f"frame-count mismatch: gen T={gen_tracks.shape[0]}, ref T={ref_tracks.shape[0]}"
    )

    gen_visible = gen_visibility.all(axis=0)
    ref_visible = ref_visibility.all(axis=0)
    if not gen_visible.any() or not ref_visible.any():
        return {
            "score": 0.0,
            "direction": 0.0,
            "speed_penalty": 0.0,
            "speed_ratio": 0.0,
        }

    gen_t = gen_tracks[:, gen_visible]
    ref_t = ref_tracks[:, ref_visible]

    gen_disp = np.diff(gen_t, axis=0)
    ref_disp = np.diff(ref_t, axis=0)
    gen_speed = np.linalg.norm(gen_disp, axis=-1)
    ref_speed = np.linalg.norm(ref_disp, axis=-1)

    gen_dir = gen_disp / (gen_speed[..., None] + eps)
    ref_dir = ref_disp / (ref_speed[..., None] + eps)
    per_frame_cos = np.einsum("tnc,tmc->tnm", ref_dir, gen_dir)
    mean_t_cos = per_frame_cos.mean(axis=0)
    best_idx = mean_t_cos.argmax(axis=1)
    best_cos = mean_t_cos[np.arange(mean_t_cos.shape[0]), best_idx]
    direction = float(best_cos.mean())

    ref_speed_mean = ref_speed.mean(axis=0)
    gen_speed_mean = gen_speed.mean(axis=0)
    matched_gen_speed = gen_speed_mean[best_idx]
    ratio = matched_gen_speed / (ref_speed_mean + eps)

    log_ratio = np.log(ratio + eps)
    lower = float(max(speed_lower, eps))
    upper = float(max(speed_upper, lower + eps))
    under = np.maximum(np.log(lower) - log_ratio, 0.0)
    over = np.maximum(log_ratio - np.log(upper), 0.0)
    speed_penalty = float((under * under + over * over).mean())

    score = direction - float(speed_penalty_coef) * speed_penalty
    return {
        "score": float(score),
        "direction": direction,
        "speed_penalty": speed_penalty,
        "speed_ratio": float(ratio.mean()),
    }


class MotionEquivariantReward:
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
    ):
        from scripts.motion_eval.metrics.motion_fidelity import CoTrackerWrapper

        self.ref_path = Path(ref_path)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.speed_lower = float(speed_lower)
        self.speed_upper = float(speed_upper)
        self.speed_penalty_coef = float(speed_penalty_coef)
        self.last_components: dict[str, float] = {}

        self.tracker = CoTrackerWrapper(
            device=device,
            cache_dir=cache_dir,
            n_frames=n_frames,
            grid_size=grid_size,
        )
        self.ref_tracks, self.ref_visibility = self.tracker.get_tracklets(self.ref_path)

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
            components = motion_equivariant_pair(
                gen_tracks=gen_tracks,
                gen_visibility=gen_vis,
                ref_tracks=self.ref_tracks,
                ref_visibility=self.ref_visibility,
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
