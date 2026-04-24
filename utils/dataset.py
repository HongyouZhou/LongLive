# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import datasets
import torchvision.io as tvio
import torchvision.transforms.functional as TF



class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


def _load_motion_video(path: str, length: int, size) -> torch.Tensor:
    """Load a video and return [T, 3, H, W] in [-1, 1]. Pads or truncates to length."""
    video, _, _ = tvio.read_video(path, pts_unit="sec", output_format="TCHW")
    if video.shape[0] == 0:
        raise ValueError(f"Empty video at {path}")
    T = video.shape[0]
    if T >= length:
        indices = torch.linspace(0, T - 1, length).round().long()
        video = video[indices]
    else:
        pad = video[-1:].expand(length - T, -1, -1, -1)
        video = torch.cat([video, pad], dim=0)
    video = video.float() / 127.5 - 1.0
    if isinstance(size, int):
        size = (size, size)
    video = TF.resize(video, list(size), antialias=True)
    return video.contiguous()


class MotionSwitchDataset(Dataset):
    """Dataset returning two text prompts + two motion reference videos.

    Each line of ``pair_jsonl`` is a JSON object like::

        {"prompt_a": "...", "prompt_b": "...",
         "motion_a": "relative/path.mp4", "motion_b": "relative/path.mp4",
         "switch_frame": 39}

    ``motion_*`` paths are resolved against ``motion_ref_root``. ``switch_frame``
    is optional; when absent the training loop samples from ``switch_choices``.
    When only a single motion / prompt is available, set ``motion_b == motion_a``
    and ``prompt_b == prompt_a`` — downstream code still works because the
    DMDSwitch path tolerates identical segments.
    """

    def __init__(
        self,
        pair_jsonl: str,
        motion_ref_root: str,
        ref_length: int = 16,
        image_size=(480, 832),
        target_video_length: int = 0,
    ):
        """
        target_video_length: when > 0, additionally load a longer clip from
        the SAME source file as motion_a and return it under the 'target_video'
        key. Used by Uni-DAD teacher_turn as the diffusion-MSE target (self-pair
        supervision). When 0 (default), no target_video is yielded and callers
        that need it fall back to motion_ref.
        """
        self.pairs = []
        with open(pair_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))
        self.motion_ref_root = motion_ref_root
        self.ref_length = ref_length
        self.image_size = tuple(image_size)
        self.target_video_length = int(target_video_length)

    def __len__(self):
        return len(self.pairs)

    def _resolve(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.motion_ref_root, rel_path)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        motion_a = _load_motion_video(
            self._resolve(p["motion_a"]), self.ref_length, self.image_size)
        motion_b_path = p.get("motion_b", p["motion_a"])
        motion_b = _load_motion_video(
            self._resolve(motion_b_path), self.ref_length, self.image_size)
        sample = {
            "prompts": p["prompt_a"],
            "switch_prompts": p.get("prompt_b", p["prompt_a"]),
            "motion_ref": motion_a,
            "switch_motion_ref": motion_b,
            "idx": idx,
        }
        if "switch_frame" in p:
            sample["switch_frame_index"] = int(p["switch_frame"])
        if self.target_video_length > 0:
            # Self-pair supervision for teacher_turn: longer clip from the
            # SAME source as motion_a. Truncates/pads to target_video_length
            # pixel frames; downstream VAE compresses temporally by ~4x.
            sample["target_video"] = _load_motion_video(
                self._resolve(p["motion_a"]),
                self.target_video_length,
                self.image_size,
            )
        return sample


def cycle(dl):
    while True:
        for data in dl:
            yield data
