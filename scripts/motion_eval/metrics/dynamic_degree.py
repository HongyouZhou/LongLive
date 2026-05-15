"""VBench-aligned Dynamic Degree metric.

Measures whether a generated video actually moves (as opposed to being
temporally smooth but static). Uses RAFT optical flow magnitude with VBench's
top-5% aggregation and threshold logic.

Algorithm (mirrors vbench.dynamic_degree):
  1. For each consecutive frame pair (t, t+1):
       u, v   = RAFT(frame_t, frame_{t+1})          # (H, W, 2) flow field
       rad    = sqrt(u^2 + v^2)                     # (H, W) magnitudes
       max_rad[t] = mean of top-5% values in rad    # per-frame motion score
  2. dynamic_score (continuous) = mean(max_rad)
  3. is_dynamic  (VBench binary) = True iff
        count(max_rad[t] > threshold) >= count_num
     where:
        threshold = 6.0 * (min(H, W) / 256)
        count_num = round(4 * n_pairs / 16)

The continuous ``dynamic_score`` is the amplitude proxy we actually care
about (analogous to our CoTracker mean velocity but on the optical-flow side
and community-standard); ``is_dynamic`` is preserved for direct paper-table
comparison with VBench-reporting baselines.

Tracklet/RAFT caching: per-video flow magnitudes are written to
``<cache_dir>/<sha10>.npz`` keyed by (resolved video path, n_frames). This is
small (one float per frame) and lets re-running the metric with a different
threshold be free.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _video_to_tensor(path: str | Path, n_frames: int | None = None) -> torch.Tensor:
    """Read mp4 → (T, 3, H, W) float tensor in [0, 255]. Native frame rate; if
    n_frames is given, uniformly subsample to that many frames."""
    from torchvision.io import read_video
    frames, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
    t_native = frames.shape[0]
    if t_native == 0:
        raise ValueError(f"empty video: {path}")
    if n_frames is not None and t_native > n_frames:
        idxs = np.linspace(0, t_native - 1, n_frames).astype(int)
        frames = frames[idxs]
    return frames.permute(0, 3, 1, 2).contiguous().float()  # (T, 3, H, W)


class DynamicDegree:
    """RAFT-based per-video dynamic degree metric.

    Lazy-loads torchvision RAFT-Large on first call.
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str | Path] = None,
        n_frames: Optional[int] = None,
        raft_iters: int = 20,
    ):
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_frames = n_frames
        self.raft_iters = raft_iters
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        self._model = raft_large(weights=weights, progress=False).to(self.device).eval()
        self._transform = weights.transforms()

    def _cache_path(self, video_path: str | Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = f"{Path(video_path).resolve()}|T={self.n_frames}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{h}.npz"

    @torch.no_grad()
    def _compute_max_rad(self, video_path: str | Path) -> tuple[np.ndarray, int, int]:
        """Compute per-pair top-5% flow magnitude. Returns (max_rad, H, W)."""
        cache = self._cache_path(video_path)
        if cache is not None and cache.exists():
            z = np.load(cache)
            return z["max_rad"], int(z["H"]), int(z["W"])

        self._ensure_model()
        frames = _video_to_tensor(video_path, self.n_frames)  # (T, 3, H, W) float
        T, _, H, W = frames.shape
        if T < 2:
            raise ValueError(f"need >=2 frames, got {T} from {video_path}")

        # RAFT preprocessing expects pairs of (B, 3, H, W) in [-1, 1].
        max_rad = np.zeros(T - 1, dtype=np.float32)
        cut = max(1, int(H * W * 0.05))  # top-5% pixels per VBench

        for t in range(T - 1):
            img1 = frames[t:t + 1].to(self.device)
            img2 = frames[t + 1:t + 2].to(self.device)
            img1_t, img2_t = self._transform(img1, img2)
            # RAFT returns a list of flow estimates; take the final iterate.
            flow_list = self._model(img1_t, img2_t, num_flow_updates=self.raft_iters)
            flow = flow_list[-1]               # (1, 2, H, W)
            u = flow[0, 0].cpu().numpy()
            v = flow[0, 1].cpu().numpy()
            rad = np.sqrt(u * u + v * v).reshape(-1)
            # Top-5% indices via argpartition (faster than full sort)
            if rad.size > cut:
                topk_idx = np.argpartition(rad, -cut)[-cut:]
                max_rad[t] = float(rad[topk_idx].mean())
            else:
                max_rad[t] = float(rad.mean())

        if cache is not None:
            np.savez_compressed(cache, max_rad=max_rad, H=H, W=W)
        return max_rad, H, W

    def score(self, video_path: str | Path) -> dict:
        """Returns:
            dynamic_score: mean of per-pair top-5% flow magnitudes (continuous)
            is_dynamic:    True iff VBench threshold satisfied (binary)
            n_pairs:       number of consecutive frame pairs evaluated
            threshold:     the resolution-scaled threshold actually used
            count_num:     the minimum-pairs-over-threshold count actually used
        """
        max_rad, H, W = self._compute_max_rad(video_path)
        scale = min(H, W)
        threshold = 6.0 * (scale / 256.0)
        n_pairs = max_rad.size
        count_num = max(1, round(4 * n_pairs / 16.0))
        n_over = int((max_rad > threshold).sum())
        is_dynamic = n_over >= count_num

        return {
            "dynamic_score": float(max_rad.mean()),
            "is_dynamic": bool(is_dynamic),
            "n_pairs": int(n_pairs),
            "threshold": float(threshold),
            "count_num": int(count_num),
            "n_over": int(n_over),
        }
