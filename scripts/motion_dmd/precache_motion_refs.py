"""Pre-encode motion-DMD reference videos to VAE latents.

Loads each ref mp4 → samples 81 pixel frames at 480×832 → Wan VAE encode →
latent [F=21, C=16, H=60, W=104]. Stacks all refs and saves the cache so
training can mmap-load without paying the VAE-encode cost on every run.

The Wan VAE is a (1+4×k)-frame temporal compressor: 81 pixel frames → 21 latent
frames (1 keyframe + 5 groups of 4). Spatial 8× downsample: 480/8=60, 832/8=104.

Refs JSONL format (one object per line):
    {"video": "<filename>", "caption": "...", ...optional metadata}
Resolution order for the mp4: --refs_root / video, else $LL_DATA/motion_refs/video.

Run on lab or HPC where Wan VAE weights are available:
    python scripts/motion_dmd/precache_motion_refs.py \\
        --refs_jsonl prompts/walking_refs_v1.jsonl \\
        --output $LL_DATA/motion_dmd/walking_v1.latents.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.wan_wrapper import WanVAEWrapper  # noqa: E402

PIXEL_FRAMES = 81
PIXEL_H = 480
PIXEL_W = 832
LATENT_FRAMES = 21
LATENT_C = 16
LATENT_H = 60
LATENT_W = 104


def load_video_pixels(path: str, num_frames: int, height: int, width: int,
                      device: torch.device) -> torch.Tensor:
    """Returns [C, F, H, W] in [-1, 1] bfloat16, on `device`.

    Uses torchvision.io.read_video (already a hard dep of LongLive). decord
    is faster but optional and not installed on the HPC longlive env.
    """
    from torchvision.io import read_video
    video, _, _ = read_video(path, pts_unit="sec")  # [F_total, H, W, C] uint8
    total = video.shape[0]
    if total < num_frames:
        raise ValueError(f"Video {path} has {total} frames, need {num_frames}")
    indices = torch.linspace(0, total - 1, num_frames).long()
    frames = video[indices]                          # [F, H, W, C] uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
    frames = transforms.functional.resize(frames, [height, width], antialias=True)
    frames = frames * 2.0 - 1.0
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]
    return frames.to(device=device, dtype=torch.bfloat16)


def resolve_ref_path(ref: dict, refs_root: Path | None) -> Path:
    if "path" in ref and Path(ref["path"]).is_absolute():
        p = Path(ref["path"])
        if p.exists():
            return p
    fname = ref.get("video") or ref.get("filename") or Path(ref["path"]).name
    if refs_root is not None:
        cand = refs_root / fname
        if cand.exists():
            return cand
    ll_data = os.environ.get("LL_DATA", "")
    if ll_data:
        cand = Path(ll_data) / "motion_refs" / fname
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Cannot resolve {fname}; tried path={ref.get('path')}, "
        f"refs_root={refs_root}, $LL_DATA/motion_refs"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs_jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--refs_root", default=None,
                    help="Override directory holding the mp4s")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    refs_root = Path(args.refs_root) if args.refs_root else None

    with open(args.refs_jsonl) as f:
        refs = [json.loads(line) for line in f if line.strip()]
    print(f"[precache] {len(refs)} refs in {args.refs_jsonl}", flush=True)

    vae = WanVAEWrapper().to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    latents = []
    captions = []
    paths = []
    with torch.no_grad():
        for i, ref in enumerate(refs):
            mp4_path = resolve_ref_path(ref, refs_root)
            paths.append(str(mp4_path))
            captions.append(ref["caption"])
            print(f"[precache] [{i+1}/{len(refs)}] {mp4_path.name}", flush=True)

            pixel = load_video_pixels(
                str(mp4_path), PIXEL_FRAMES, PIXEL_H, PIXEL_W, device
            )  # [C=3, F=81, H=480, W=832]
            pixel = pixel.unsqueeze(0)  # [1, 3, 81, 480, 832]
            latent = vae.encode_to_latent(pixel)  # [1, F_lat, 16, 60, 104]
            assert latent.shape == (1, LATENT_FRAMES, LATENT_C, LATENT_H, LATENT_W), \
                f"unexpected latent shape {tuple(latent.shape)}"
            latents.append(latent.squeeze(0).to(torch.bfloat16).cpu())

    cache = torch.stack(latents, dim=0)  # [N, 21, 16, 60, 104]
    print(f"[precache] cache shape={tuple(cache.shape)} dtype={cache.dtype} "
          f"size={cache.numel() * cache.element_size() / 1e6:.1f} MB",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"latents": cache, "captions": captions, "paths": paths,
         "pixel_shape": (PIXEL_FRAMES, PIXEL_H, PIXEL_W),
         "latent_shape": (LATENT_FRAMES, LATENT_C, LATENT_H, LATENT_W)},
        str(out_path),
    )
    print(f"[precache] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
