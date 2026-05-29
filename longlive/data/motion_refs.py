"""Reference-video data loaders shared by fast-adaptation methods.

These datasets started in ``longlive.methods.motiondirector`` because the
first use case was MotionDirector-style LoRA finetuning.  They are generic
enough for RAM/NFT/DRaFT-style methods as well, so the canonical import lives
here and the old module re-exports these classes for compatibility.
"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import yaml

if TYPE_CHECKING:
    from longlive.utils.wan_wrapper import WanVAEWrapper


REPO_ROOT = Path(__file__).resolve().parents[2]
UCF_PROMPTS_YAML = REPO_ROOT / "scripts" / "motion_eval" / "prompts" / "ucf_sports.yaml"


def _load_clip_to_pixel_tensor(
    path: Path,
    frame_count: int,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Load a video file as ``(1, 3, T, H, W)`` fp32 in ``[-1, 1]``."""
    import decord  # lazy: only required at training time

    vr = decord.VideoReader(str(path), num_threads=1)
    n = len(vr)
    if n >= frame_count:
        idxs = np.linspace(0, n - 1, frame_count).round().astype(int).tolist()
    else:
        idxs = (list(range(n)) * (frame_count // n + 1))[:frame_count]
    frames = vr.get_batch(idxs).asnumpy()

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    _, _, height, width = x.shape
    scale = resolution / min(height, width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    y0 = (new_h - resolution) // 2
    x0 = (new_w - resolution) // 2
    x = x[:, :, y0:y0 + resolution, x0:x0 + resolution]

    x = x.permute(1, 0, 2, 3).unsqueeze(0) * 2.0 - 1.0
    return x.to(device, dtype=torch.float32)


def _resolve_video_path(data_root: str | Path, path: str | Path) -> Path:
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    if not candidate.is_absolute():
        candidate = Path(data_root) / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"reference video not found: {candidate}")
    return candidate


class ReferenceVideoDataset:
    """Yield one explicit reference video and caption for per-unit adaptation."""

    def __init__(
        self,
        data_root: str | Path,
        vae: WanVAEWrapper,
        reference_video_path: str | Path,
        train_caption: str,
        frame_count: int = 81,
        resolution: int = 480,
        device: torch.device | str = "cuda",
        unit_id: str | None = None,
    ):
        if not str(train_caption).strip():
            raise ValueError("ReferenceVideoDataset requires a non-empty train_caption")
        self.data_root = Path(data_root)
        self.vae = vae
        self.frame_count = frame_count
        self.resolution = resolution
        self.device = torch.device(device)
        self.unit_id = unit_id or Path(reference_video_path).stem
        self.train_clip_path = _resolve_video_path(self.data_root, reference_video_path)
        self.clips = [self.train_clip_path]
        self.train_caption = str(train_caption)

        print(
            f"[ReferenceVideoDataset] unit_id={self.unit_id!r} "
            f"ref={self.train_clip_path}, caption={self.train_caption!r}"
        )

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, str]:
        pixels = _load_clip_to_pixel_tensor(
            self.train_clip_path, self.frame_count, self.resolution, self.device,
        )
        latent = self.vae.encode_to_latent(pixels)
        return latent.to(torch.bfloat16), self.train_caption


class SkateboardingLatentDataset:
    """Yield ``(latent, train_caption)`` from one UCF Sports category."""

    def __init__(
        self,
        data_root: str | Path,
        vae: WanVAEWrapper,
        frame_count: int = 81,
        resolution: int = 480,
        category: str = "Skateboarding",
        device: torch.device | str = "cuda",
        single_video: bool = False,
    ):
        self.data_root = Path(data_root)
        self.vae = vae
        self.frame_count = frame_count
        self.resolution = resolution
        self.category = category
        self.device = torch.device(device)

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
            raise RuntimeError(f"No clips found for category {category!r} in {manifest}")

        self.clips.sort()
        if single_video:
            self.clips = self.clips[:1]
        self.train_clip_path = self.clips[0]

        with open(UCF_PROMPTS_YAML) as f:
            spec = yaml.safe_load(f)
        self.train_caption = spec["categories"][category]["train_caption"]

        print(
            f"[SkateboardingLatentDataset] category={category!r} "
            f"{len(self.clips)} clip{'s' if len(self.clips) > 1 else ''}"
            f"{' (single-video mode)' if single_video else ''}, "
            f"ref={self.train_clip_path.name}, caption={self.train_caption!r}"
        )

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, str]:
        clip_path = random.choice(self.clips)
        pixels = _load_clip_to_pixel_tensor(
            clip_path, self.frame_count, self.resolution, self.device,
        )
        latent = self.vae.encode_to_latent(pixels)
        return latent.to(torch.bfloat16), self.train_caption


def make_reference_dataset(
    cfg,
    *,
    vae: WanVAEWrapper,
    device: torch.device | str,
):
    """Build the reference dataset requested by a method config.

    Per-reference protocol configs provide ``reference_video_path`` and
    ``train_caption``.  Legacy single-category configs omit those fields and
    keep using the deterministic first UCF category clip.
    """
    reference_video_path = getattr(cfg, "reference_video_path", None)
    if reference_video_path:
        return ReferenceVideoDataset(
            data_root=cfg.data_root,
            vae=vae,
            reference_video_path=reference_video_path,
            train_caption=str(getattr(cfg, "train_caption", "")),
            frame_count=int(cfg.frame_count),
            resolution=int(cfg.resolution),
            device=device,
            unit_id=str(getattr(cfg, "unit_id", "")) or None,
        )
    return SkateboardingLatentDataset(
        data_root=cfg.data_root,
        vae=vae,
        frame_count=int(cfg.frame_count),
        resolution=int(cfg.resolution),
        category=str(cfg.category),
        device=device,
        single_video=True,
    )


class GeneralPromptDataset:
    """Yield ``(latent, caption)`` from generic OpenVid motion-pair clips."""

    def __init__(
        self,
        data_root: str | Path,
        vae: WanVAEWrapper,
        frame_count: int = 81,
        resolution: int = 480,
        device: torch.device | str = "cuda",
        manifest_rel: str = "prompts/motion_pairs_train.jsonl",
        max_clips: int = 50,
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.vae = vae
        self.frame_count = frame_count
        self.resolution = resolution
        self.device = torch.device(device)

        manifest = self.data_root / manifest_rel
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} not found. "
                f"Run scripts/prepare_openvid.py first to build motion_pairs_train.jsonl."
            )

        entries: list[tuple[str, str]] = []
        with open(manifest) as f:
            for line in f:
                row = json.loads(line)
                entries.append((row["motion_a"], row["prompt_a"]))

        entries.sort()
        rng = random.Random(seed)
        rng.shuffle(entries)
        self.entries = entries[:max_clips]

        print(
            f"[GeneralPromptDataset] {len(self.entries)}/{len(entries)} clips "
            f"sampled from {manifest.name} (seed={seed})"
        )

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, str]:
        filename, caption = random.choice(self.entries)
        clip_path = self.data_root / "motion_refs" / filename
        pixels = _load_clip_to_pixel_tensor(
            clip_path, self.frame_count, self.resolution, self.device,
        )
        latent = self.vae.encode_to_latent(pixels)
        return latent.to(torch.bfloat16), caption
