"""Online motion_fidelity reward for rollout-based finetuning.

Wraps `scripts.motion_eval.metrics.motion_fidelity.MotionFidelity` for in-process
scoring during the training loop. The reference clip's CoTracker3 tracklets are
cached once at trainer startup; per-rollout cost is just one CoTracker forward
on the generated video (~5-10s on H200).

We mp4-roundtrip the generated video tensor before scoring because the existing
`CoTrackerWrapper.get_tracklets` API is tied to a file path (and so is its
on-disk cache, keyed by ref-path sha10). A future optimization is to add a
tensor-input variant; for now the mp4 write (~50 ms) is in the noise floor
next to CoTracker.
"""
from __future__ import annotations

from pathlib import Path

import torch

# Heavy + env-specific imports (CoTracker, torchvision, PIL via motion_eval
# helpers) are deferred to the method bodies so basic method imports do not
# require scripts/motion_eval/setup_motion_eval_env.sh dependencies.


class MotionFidelityReward:
    """Score a generated video against ONE cached reference.

    Group-normalization is downstream — this class is just the per-rollout
    scoring primitive. Initialize once per trainer; pass the same instance
    every step.
    """

    def __init__(
        self,
        ref_path: str | Path,
        scratch_dir: str | Path,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        n_frames: int = 16,
        grid_size: int = 30,
        fps: int = 16,
    ):
        """Args:
            ref_path: path to the single reference clip (UCF-Skateboarding by
                default). CoTracker tracklets for this clip are computed once
                at init and cached on disk by `CoTrackerWrapper`.
            scratch_dir: where to write per-rollout mp4 files. Cleaned up after
                each scoring call.
            device: cuda by default. CoTracker lives here.
            cache_dir: persistent tracklet cache. None disables caching (only
                relevant for the reference — gen-side never caches since the
                gen path is unique per rollout).
            n_frames / grid_size / fps: pass-through to CoTracker / write_video.
        """
        # Deferred import — keeps top-level package importable without
        # motion_eval optional deps (Pillow, decord, cotracker hub).
        from scripts.motion_eval.metrics.motion_fidelity import CoTrackerWrapper

        self.ref_path = Path(ref_path)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)

        self.tracker = CoTrackerWrapper(
            device=device,
            cache_dir=cache_dir,
            n_frames=n_frames,
            grid_size=grid_size,
        )
        # Pre-extract reference tracklets — populates the cache if cache_dir
        # is provided, otherwise computes once and holds in memory.
        self.ref_tracks, self.ref_visibility = self.tracker.get_tracklets(self.ref_path)

    def _write_mp4(self, video: torch.Tensor, tag: str) -> Path:
        """video: (T, 3, H, W) in [0, 1] float -> mp4 on disk."""
        from torchvision.io import write_video  # deferred — see top of file

        # write_video wants (T, H, W, C) uint8.
        frames = (video.detach().clamp(0, 1).permute(0, 2, 3, 1) * 255.0).round()
        frames = frames.to(torch.uint8).cpu()
        out_path = self.scratch_dir / f"rollout_{tag}.mp4"
        # Older torchvision raises with default video_codec; libx264 is on
        # every node we use. fps=16 matches motion_eval_inference.yaml.
        write_video(str(out_path), frames, fps=self.fps, video_codec="libx264")
        return out_path

    @torch.no_grad()
    def score(self, video: torch.Tensor, tag: str) -> float:
        """Score one rollout video against the cached reference.

        Args:
            video: (T, 3, H, W) in [0, 1] float — pixel-space output of
                CausalInferencePipeline.inference (after the `(v * 0.5 + 0.5)`
                rescale already applied inside the pipeline).
            tag: unique string per rollout (e.g. `f"{step}_{k}"`) so concurrent
                writes don't clobber. The mp4 is removed after scoring.

        Returns:
            scalar motion_fidelity ∈ [-1, 1].
        """
        from scripts.motion_eval.metrics.motion_fidelity import motion_fidelity_pair

        gen_path = self._write_mp4(video, tag)
        try:
            gen_tracks, gen_vis = self.tracker.get_tracklets(gen_path)
            score = motion_fidelity_pair(
                gen_tracks=gen_tracks,
                gen_visibility=gen_vis,
                ref_tracks=self.ref_tracks,
                ref_visibility=self.ref_visibility,
            )
        finally:
            try:
                gen_path.unlink()
            except FileNotFoundError:
                pass
        return float(score)
