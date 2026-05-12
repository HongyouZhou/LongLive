"""Reference clip loader for MotionDirector teacher-finetune.

Reads UCF Sports Action clips of a chosen category (default Skateboarding),
normalizes to Wan-14B native input shape (81 frame x 480^2), runs them
through the Wan VAE encoder to produce latents. Train caption comes from
``scripts/motion_eval/prompts/ucf_sports.yaml`` (uniform per category).

Per docs/04.md §4.3 — on-the-fly encode for first version; if found to be
the training bottleneck, switch to offline pre-encode.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from longlive.utils.wan_wrapper import WanVAEWrapper


REPO_ROOT = Path(__file__).resolve().parents[3]
UCF_PROMPTS_YAML = REPO_ROOT / "scripts" / "motion_eval" / "prompts" / "ucf_sports.yaml"


class SkateboardingLatentDataset:
    """Yields (latent, train_caption) from UCF clips of one category.

    Default category is ``Skateboarding`` because UCF Sports Action is the
    only one where MotionDirector paper releases all 7 prompts verbatim
    (Fig 8a + Fig 10) — gives paper-anchorable comparison after Phase 3
    distillation.
    """

    def __init__(
        self,
        data_root: str | Path,
        vae: WanVAEWrapper,
        frame_count: int = 81,
        resolution: int = 480,
        category: str = "Skateboarding",
        device: torch.device | str = "cuda",
    ):
        self.data_root = Path(data_root)
        self.vae = vae
        self.frame_count = frame_count
        self.resolution = resolution
        self.category = category
        self.device = torch.device(device)

        # Load clip paths from manifest.
        manifest = self.data_root / "ucf_sports" / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} not found. "
                f"Run scripts/prepare_motion_eval.py --datasets ucf first."
            )
        self.clips: list[Path] = []
        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["category"] != category:
                    continue
                rel = row["path"]
                if rel.startswith("ucf_sports/"):
                    rel = rel[len("ucf_sports/"):]
                self.clips.append(self.data_root / "ucf_sports" / rel)
        if not self.clips:
            raise RuntimeError(
                f"No clips found for category {category!r} in {manifest}"
            )

        # Train caption from ucf_sports.yaml (same for all clips in a category).
        with open(UCF_PROMPTS_YAML) as f:
            spec = yaml.safe_load(f)
        self.train_caption = spec["categories"][category]["train_caption"]

        print(
            f"[SkateboardingLatentDataset] category={category!r} "
            f"{len(self.clips)} clips, caption={self.train_caption!r}"
        )

    def _load_clip_pixels(self, path: Path) -> torch.Tensor:
        """Returns (1, C, T, H, W) in [-1, 1] bf16 on device."""
        import decord  # lazy — only required at training time on HPC, not on orchestration boxes
        vr = decord.VideoReader(str(path), num_threads=1)
        n = len(vr)
        if n >= self.frame_count:
            idxs = np.linspace(0, n - 1, self.frame_count).round().astype(int).tolist()
        else:
            # loop padding for short clips
            idxs = (list(range(n)) * (self.frame_count // n + 1))[: self.frame_count]
        frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3) uint8

        # to torch (T, 3, H, W) float in [0, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # resize so shorter edge == resolution, then center crop
        _, _, H, W = x.shape
        scale = self.resolution / min(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        x = F.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        y0 = (new_h - self.resolution) // 2
        x0 = (new_w - self.resolution) // 2
        x = x[:, :, y0:y0 + self.resolution, x0:x0 + self.resolution]

        # to (1, 3, T, H, W) in [-1, 1] bf16 on device
        x = x.permute(1, 0, 2, 3).unsqueeze(0) * 2.0 - 1.0
        return x.to(self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, str]:
        """Returns (latent (1, F_latent, 16, H/8, W/8), caption)."""
        clip_path = random.choice(self.clips)
        pixels = self._load_clip_pixels(clip_path)  # (1, C, T, H, W)
        latent = self.vae.encode_to_latent(pixels)
        return latent, self.train_caption
