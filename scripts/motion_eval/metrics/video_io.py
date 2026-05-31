"""Uniform mp4 reading for motion-eval metrics.

decord is preferred (already in the longlive env); falls back to a frame-by-frame
torchvision read if decord is missing. Returns numpy uint8 arrays.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def read_video_frames(
    path: str | Path,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> np.ndarray:
    """Decode an mp4 to a contiguous (T, H, W, 3) uint8 array.

    Parameters
    ----------
    path : str | Path
    max_frames : Optional[int]
        Cap the number of frames returned (post-stride).
    stride : int
        Take every ``stride``-th frame.

    Returns
    -------
    np.ndarray of shape (T, H, W, 3), uint8.
    """
    path = str(path)
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        idxs = list(range(0, n, stride))
        if max_frames is not None:
            idxs = idxs[:max_frames]
        frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3) uint8
        return frames
    except ImportError:
        pass

    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open video: {path}")
        frames = []
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"cv2 read zero frames from video: {path}")
        return np.stack(frames, axis=0).astype(np.uint8)
    except ImportError:
        pass

    # torchvision fallback
    import torch
    from torchvision.io import read_video
    video, _audio, _info = read_video(path, pts_unit="sec")
    # read_video returns (T, H, W, C) uint8 already.
    n = video.shape[0]
    idxs = list(range(0, n, stride))
    if max_frames is not None:
        idxs = idxs[:max_frames]
    video = video[torch.as_tensor(idxs)]
    return video.numpy()


def frames_to_pil(frames: np.ndarray) -> list:
    """Convert (T, H, W, 3) uint8 -> list of PIL.Image. For CLIP/PickScore preprocessing."""
    from PIL import Image
    return [Image.fromarray(f) for f in frames]
