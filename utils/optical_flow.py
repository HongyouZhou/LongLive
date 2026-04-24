"""Optical flow metrics for evaluating motion fidelity.

Farneback dense flow is used (CPU) so it doesn't compete with inference for
A40 memory. Sufficient as an MVP motion control signal; RAFT can be swapped in
later if precision is insufficient.
"""
from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch


def _to_gray_uint8_frames(video: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a video to a [T, H, W] uint8 grayscale numpy array.

    Accepts:
      - torch.Tensor [T, 3, H, W] in [0,1] float or uint8
      - torch.Tensor [T, H, W, 3] in uint8
      - numpy.ndarray [T, H, W, 3] in uint8
    """
    if isinstance(video, torch.Tensor):
        v = video.detach().cpu()
        if v.ndim == 4 and v.shape[1] == 3:
            # [T, 3, H, W] -> [T, H, W, 3]
            v = v.permute(0, 2, 3, 1)
        if v.dtype.is_floating_point:
            v = (v.clamp(0, 1) * 255).to(torch.uint8)
        v = v.numpy()
    else:
        v = np.asarray(video)
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)
    assert v.ndim == 4 and v.shape[-1] == 3, \
        f"Expected [T,H,W,3] uint8, got shape {v.shape} dtype {v.dtype}"
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v], axis=0)
    return gray


def compute_mean_flow_magnitude(
    video: Union[torch.Tensor, np.ndarray],
    *,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> float:
    """Mean L2 magnitude of Farneback flow across all adjacent-frame pairs.

    Returns a single scalar (higher = more motion).
    """
    gray = _to_gray_uint8_frames(video)
    T = gray.shape[0]
    if T < 2:
        return 0.0
    mags = []
    prev = gray[0]
    for t in range(1, T):
        nxt = gray[t]
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None,
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
        prev = nxt
    return float(np.mean(mags))


def compute_flow_l2_vs_reference(
    generated: Union[torch.Tensor, np.ndarray],
    reference: Union[torch.Tensor, np.ndarray],
) -> dict:
    """Compute motion-fidelity metrics between generated and reference videos.

    Strategy: summarise each video by its mean adjacent-frame flow magnitude
    (a scalar "motion intensity"), then report the absolute and relative
    difference. This avoids the frame-alignment problem that would otherwise
    require resampling to a common T.
    """
    gen_mag = compute_mean_flow_magnitude(generated)
    ref_mag = compute_mean_flow_magnitude(reference)
    abs_diff = abs(gen_mag - ref_mag)
    rel_diff = abs_diff / ref_mag if ref_mag > 1e-6 else float("inf")
    return {
        "gen_flow_mag": gen_mag,
        "ref_flow_mag": ref_mag,
        "flow_abs_diff": abs_diff,
        "flow_rel_diff": rel_diff,
    }
