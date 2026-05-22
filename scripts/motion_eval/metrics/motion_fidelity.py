"""Motion fidelity via Yatim et al. (CVPR 2024) tracklet-velocity cosine.

Reference: "Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer"
arXiv:2311.17009, code: https://github.com/diffusion-motion-transfer/diffusion-motion-transfer

The field-standard auto metric for cross-subject motion comparison since
CVPR 2024 (MotionMatcher / MoTrans / MotionInversion / MotionShop / CoMo
all cite it). MotionDirector predates this convention by ~4 months and
relied on MTurk; we substitute Yatim's score as the reproducible auto
metric for the same "is the motion similar?" question. Output is in
``[-1, 1]``; subject identity is discarded by construction (only point
trajectories survive).

Algorithm:
  1. Use CoTracker3 to extract dense point tracklets for ``gen`` and
     each ``ref`` video.
  2. Keep only tracklets visible across all frames.
  3. Compute unit per-frame displacement (velocity direction).
  4. For each ref tracklet n, take ``max_m mean_t cos(d_ref[n,t], d_gen[m,t])``
     across all gen tracklets m.
  5. Mean over ref tracklets -> scalar score.
  6. For UCF multi-video categories, compute step 1-5 for each reference
     clip and return the mean (per the design doc — paper is silent on
     the per-task reference selection rule, so we marginalize).

Tracklet caching: per-reference tracklets are written to
``<cache_dir>/<sha10>.npz`` keyed by the absolute reference path.
Recomputing CoTracker3 on a single 16-frame clip is ~5-10s on an H200;
caching is the difference between a 10-minute eval and an hour.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .video_io import read_video_frames


def _video_to_tensor(path: str | Path, n_frames: int) -> torch.Tensor:
    """Read mp4 -> (1, T, 3, H, W) float32 in [0, 255]. Uniformly samples to n_frames."""
    frames = read_video_frames(path)  # (T_native, H, W, 3) uint8
    t_native = frames.shape[0]
    if t_native >= n_frames:
        idxs = np.linspace(0, t_native - 1, n_frames).astype(int)
    else:
        # Pad by repeating the last frame.
        idxs = np.concatenate([
            np.arange(t_native),
            np.full(n_frames - t_native, t_native - 1, dtype=int),
        ])
    frames = frames[idxs]  # (n_frames, H, W, 3)
    t = torch.from_numpy(frames).float()  # (T, H, W, 3) float
    t = t.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    return t


def motion_fidelity_pair(
    gen_tracks: np.ndarray,
    gen_visibility: np.ndarray,
    ref_tracks: np.ndarray,
    ref_visibility: np.ndarray,
) -> float:
    """Compute Yatim's score from pre-extracted tracklets.

    All inputs are numpy arrays of identical T (caller's responsibility).
        gen_tracks      (T, N_gen, 2)
        gen_visibility  (T, N_gen) bool
        ref_tracks      (T, N_ref, 2)
        ref_visibility  (T, N_ref) bool
    """
    assert gen_tracks.shape[0] == ref_tracks.shape[0], (
        f"frame-count mismatch: gen T={gen_tracks.shape[0]}, ref T={ref_tracks.shape[0]}; "
        "callers must resample both videos to a common T before tracklet extraction"
    )

    gen_visible = gen_visibility.all(axis=0)  # (N_gen,)
    ref_visible = ref_visibility.all(axis=0)  # (N_ref,)
    if not gen_visible.any() or not ref_visible.any():
        return 0.0
    gen_t = gen_tracks[:, gen_visible]  # (T, N_gen', 2)
    ref_t = ref_tracks[:, ref_visible]  # (T, N_ref', 2)

    gen_d = np.diff(gen_t, axis=0)  # (T-1, N_gen', 2)
    ref_d = np.diff(ref_t, axis=0)  # (T-1, N_ref', 2)
    gen_d /= np.linalg.norm(gen_d, axis=-1, keepdims=True) + 1e-8
    ref_d /= np.linalg.norm(ref_d, axis=-1, keepdims=True) + 1e-8

    # Per-frame cos(d_ref[t, n], d_gen[t, m]) -> (T-1, N_ref', N_gen')
    per_frame_cos = np.einsum("tnc,tmc->tnm", ref_d, gen_d)
    mean_t_cos = per_frame_cos.mean(axis=0)  # (N_ref', N_gen')
    best_match = mean_t_cos.max(axis=1)  # (N_ref',)
    return float(best_match.mean())


def motion_fidelity_pair_grad(
    gen_tracks: torch.Tensor,        # (T, N_gen, 2)  float, grad-tracked if needed
    gen_visibility: torch.Tensor,    # (T, N_gen)     bool
    ref_tracks: torch.Tensor,        # (T, N_ref, 2)  float
    ref_visibility: torch.Tensor,    # (T, N_ref)     bool
) -> torch.Tensor:
    """Differentiable torch version of `motion_fidelity_pair` (numpy).

    Same algorithm as Yatim et al. (CVPR 2024) tracklet velocity cosine,
    but uses torch ops so gradients flow back through `gen_tracks`.

    Args:
        gen_tracks:     (T, N_gen, 2) generated-video tracklets.
        gen_visibility: (T, N_gen)    boolean visibility mask (treated as
                                       constant — no gradient through it).
        ref_tracks:     (T, N_ref, 2) reference-video tracklets (constant).
        ref_visibility: (T, N_ref)    constant.

    Returns:
        Scalar torch tensor with `requires_grad` matching `gen_tracks`.
        Same value range as numpy version: ∈ [-1, 1].
    """
    assert gen_tracks.shape[0] == ref_tracks.shape[0], (
        f"frame-count mismatch: gen T={gen_tracks.shape[0]} vs ref T={ref_tracks.shape[0]}"
    )

    # Visibility: keep only tracklets visible in ALL frames.
    gen_visible = gen_visibility.all(dim=0)  # (N_gen,)
    ref_visible = ref_visibility.all(dim=0)  # (N_ref,)
    if not gen_visible.any() or not ref_visible.any():
        return torch.zeros((), device=gen_tracks.device, dtype=gen_tracks.dtype)

    gen_t = gen_tracks[:, gen_visible]   # (T, N_gen', 2)
    ref_t = ref_tracks[:, ref_visible]   # (T, N_ref', 2)

    # Per-frame displacement (velocity direction).
    gen_d = gen_t[1:] - gen_t[:-1]       # (T-1, N_gen', 2)
    ref_d = ref_t[1:] - ref_t[:-1]
    gen_d = gen_d / (gen_d.norm(dim=-1, keepdim=True) + 1e-8)
    ref_d = ref_d / (ref_d.norm(dim=-1, keepdim=True) + 1e-8)

    # cos(d_ref[t, n], d_gen[t, m])  →  (T-1, N_ref', N_gen')
    per_frame_cos = torch.einsum("tnc,tmc->tnm", ref_d, gen_d)
    mean_t_cos = per_frame_cos.mean(dim=0)         # (N_ref', N_gen')
    best_match = mean_t_cos.max(dim=1).values      # (N_ref',)
    return best_match.mean()


class CoTrackerWrapper:
    """Lazy-loaded CoTracker3 + optional per-reference tracklet cache."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        cache_dir: Optional[str | Path] = None,
        n_frames: int = 16,
        grid_size: int = 30,
    ):
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_frames = n_frames
        self.grid_size = grid_size
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # CoTracker3 offline: returns tracks for fixed-T input in one pass.
        # Online variant exists but is for long videos; we resample to 16 frames.
        self._model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        ).to(self.device).eval()

    def _cache_path(self, video_path: str | Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Cache key includes n_frames + grid_size so re-running with different
        # hyperparameters doesn't accidentally reuse stale tracks.
        key_str = f"{Path(video_path).resolve()}|T={self.n_frames}|G={self.grid_size}"
        h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{h}.npz"

    @torch.no_grad()
    def get_tracklets(self, video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Returns (tracks (T, N, 2), visibility (T, N) bool)."""
        cache = self._cache_path(video_path)
        if cache is not None and cache.exists():
            data = np.load(cache)
            return data["tracks"], data["visibility"]

        self._ensure_model()
        video = _video_to_tensor(video_path, self.n_frames).to(self.device)
        # CoTracker3 offline API: model(video, grid_size=...) -> (tracks, visibility)
        # tracks: (B, T, N, 2);  visibility: (B, T, N) bool
        pred_tracks, pred_visibility = self._model(video, grid_size=self.grid_size)
        tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
        visibility = pred_visibility[0].cpu().numpy().astype(bool)  # (T, N)

        if cache is not None:
            np.savez_compressed(cache, tracks=tracks, visibility=visibility)
        return tracks, visibility

    def get_tracklets_from_tensor(
        self,
        video: torch.Tensor,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same algorithm as `get_tracklets(video_path)` but takes an
        in-memory torch tensor and optionally keeps gradients.

        Args:
            video: (T, 3, H, W) or (1, T, 3, H, W) float in [0, 1].
            requires_grad: When True, the CoTracker forward is NOT wrapped
                in torch.no_grad() so reward gradients can flow back to
                the input video tensor (used by DRaFT-K reward backprop).

        Returns:
            tracks:     (T, N, 2) float torch tensor (still on device).
            visibility: (T, N) bool  torch tensor.

        NOTE: When `requires_grad=True`, the *visibility* output remains
        boolean and is treated as a constant in the gradient path (we mask
        invisible tracklets but don't differentiate through visibility).
        """
        self._ensure_model()
        # Accept (T, 3, H, W) or (1, T, 3, H, W); CoTracker wants (B, T, 3, H, W).
        if video.ndim == 4:
            video = video.unsqueeze(0)
        # Uniform-sample to self.n_frames if longer.
        T_in = video.shape[1]
        if T_in > self.n_frames:
            idxs = torch.linspace(0, T_in - 1, self.n_frames, device=video.device).long()
            video = video[:, idxs]
        # CoTracker expects [0, 255] float input (matches existing _video_to_tensor).
        video_input = video * 255.0
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            pred_tracks, pred_visibility = self._model(
                video_input, grid_size=self.grid_size
            )
        # Drop batch dim.  Keep on device + dtype unchanged so caller can
        # decide whether to .cpu() / .float() / etc.
        return pred_tracks[0], pred_visibility[0].bool()


class MotionFidelity:
    """Top-level motion-fidelity scorer. Handles UCF multi-ref by averaging."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        cache_dir: Optional[str | Path] = None,
        n_frames: int = 16,
        grid_size: int = 30,
    ):
        self.tracker = CoTrackerWrapper(
            device=device, cache_dir=cache_dir,
            n_frames=n_frames, grid_size=grid_size,
        )

    def score(self, gen_path: str | Path, ref_paths: list[str | Path]) -> float:
        """Score a generated video against one or more reference videos.

        For LOVEU-TGVE (single ref), ``ref_paths`` has length 1.
        For UCF Sports (multi-ref category), returns mean over references.
        """
        if not ref_paths:
            return 0.0
        gen_tracks, gen_vis = self.tracker.get_tracklets(gen_path)
        scores = []
        for ref in ref_paths:
            try:
                ref_tracks, ref_vis = self.tracker.get_tracklets(ref)
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[motion_fidelity] tracklet failure for {ref}: {e}",
                      file=sys.stderr)
                continue
            scores.append(motion_fidelity_pair(
                gen_tracks, gen_vis, ref_tracks, ref_vis))
        if not scores:
            return 0.0
        return float(np.mean(scores))


def _smoke():
    """Standalone: ``python -m scripts.motion_eval.metrics.motion_fidelity --gen X --ref Y``."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ref", action="append", required=True,
                    help="Reference video path. Pass multiple times for multi-ref average.")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--grid_size", type=int, default=30)
    args = ap.parse_args()

    mf = MotionFidelity(
        cache_dir=args.cache_dir, n_frames=args.n_frames, grid_size=args.grid_size,
    )
    score = mf.score(args.gen, args.ref)
    print(f"motion_fidelity: {score:.4f}")


if __name__ == "__main__":
    _smoke()
