"""Clip loaders for MotionDirector LoRA finetune.

Two datasets:
  * SkateboardingLatentDataset — reference clips of one UCF Sports category,
    yields (latent, ucf_sports.yaml train_caption). Used as the motion-target
    source.
  * GeneralPromptDataset — random subset of generic OpenVid clips with their
    captions. Used by the prior-consistency anti-drift term in docs/02.md.
    Reuses the (prompt, clip) pairs already produced by
    scripts/prepare_openvid.py at $LL_DATA/prompts/motion_pairs_train.jsonl
    + $LL_DATA/motion_refs/.

Both VAE-encode on the fly. Latents cast to bf16 at sample-time to align with
the bf16 backbone; VAE itself runs fp32 (conv3d weight/bias dtype constraint).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from longlive.utils.wan_wrapper import WanVAEWrapper


REPO_ROOT = Path(__file__).resolve().parents[3]
UCF_PROMPTS_YAML = REPO_ROOT / "scripts" / "motion_eval" / "prompts" / "ucf_sports.yaml"


def _load_clip_to_pixel_tensor(
    path: Path,
    frame_count: int,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Load a video file → (1, 3, T, H, W) fp32 in [-1, 1] on ``device``.

    Sampling: ``frame_count`` evenly spaced indices (clips shorter than
    ``frame_count`` are loop-padded). Spatial: shorter edge resized to
    ``resolution``, then center crop to ``resolution × resolution``.

    Fp32 because the Wan VAE encoder is loaded fp32 (numerical stability;
    conv3d dtype mismatch with bf16 input + fp32 weight).
    """
    import decord  # lazy — only required at training time
    vr = decord.VideoReader(str(path), num_threads=1)
    n = len(vr)
    if n >= frame_count:
        idxs = np.linspace(0, n - 1, frame_count).round().astype(int).tolist()
    else:
        idxs = (list(range(n)) * (frame_count // n + 1))[:frame_count]
    frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3) uint8

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    _, _, H, W = x.shape
    scale = resolution / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    y0 = (new_h - resolution) // 2
    x0 = (new_w - resolution) // 2
    x = x[:, :, y0:y0 + resolution, x0:x0 + resolution]

    x = x.permute(1, 0, 2, 3).unsqueeze(0) * 2.0 - 1.0
    return x.to(device, dtype=torch.float32)


class SkateboardingLatentDataset:
    """Yields (latent, train_caption) from UCF clips of one category.

    Default category is ``Skateboarding`` because UCF Sports Action is the
    only one where MotionDirector paper releases all 7 prompts verbatim
    (Fig 8a + Fig 10) — gives paper-anchorable comparison after eval.
    """

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
        # Sort for reproducibility — single_video mode picks clips[0] which
        # must be deterministic across runs.
        self.clips.sort()

        # MotionDirector single-video setup: train on the first clip only.
        # Mirrors paper's `config_single_video.yaml` which trains on one
        # reference video. The chosen clip's path is exposed via
        # `self.train_clip_path` so downstream inversion (noise_prior) can
        # use the same reference at eval.
        if single_video:
            self.clips = self.clips[:1]
        self.train_clip_path = self.clips[0]

        # Train caption from ucf_sports.yaml (same for all clips in a category).
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
        """Returns (latent (1, F_latent, 16, H/8, W/8) bf16, caption).

        VAE runs in fp32; latent cast to bf16 here to align with the bf16
        backbone dtype for `add_noise` / forward.
        """
        clip_path = random.choice(self.clips)
        pixels = _load_clip_to_pixel_tensor(
            clip_path, self.frame_count, self.resolution, self.device,
        )
        latent = self.vae.encode_to_latent(pixels)
        return latent.to(torch.bfloat16), self.train_caption


class GeneralPromptDataset:
    """Yields (latent, caption) from a random subset of generic OpenVid clips.

    Used by the prior-consistency anti-drift term (docs/02.md §3.2). Reads
    the (prompt, motion_a) pairs already produced by
    ``scripts/prepare_openvid.py`` — no separate prep step needed.

    The subset is fixed by ``seed`` for reproducibility: with the same
    (manifest, seed, max_clips), every run trains against the same set of
    generic prompts. The default ``max_clips=50`` is sized for the
    anti-drift use case where we just need coverage of "non-skateboarding"
    behavior, not a large training set.
    """

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

        # Each row: {prompt_a, prompt_b, motion_a, motion_b, switch_frame}.
        # We only need prompt_a + motion_a (the "a" stream is the canonical
        # prompt/clip pair; "b" stream is for cross-pair experiments).
        entries: list[tuple[str, str]] = []
        with open(manifest) as f:
            for line in f:
                row = json.loads(line)
                entries.append((row["motion_a"], row["prompt_a"]))

        # Deterministic subsample: sort then seed-shuffle then take first N.
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
        """Returns (latent (1, F_latent, 16, H/8, W/8) bf16, caption)."""
        filename, caption = random.choice(self.entries)
        clip_path = self.data_root / "motion_refs" / filename
        pixels = _load_clip_to_pixel_tensor(
            clip_path, self.frame_count, self.resolution, self.device,
        )
        latent = self.vae.encode_to_latent(pixels)
        return latent.to(torch.bfloat16), caption
