"""50-step base Wan2.1-T2V-1.3B inference, optionally with motion LoRA loaded.

Usage:
    # Plain Wan-1.3B (no LoRA) — sanity baseline
    python scripts/motion_lora/inference.py \\
        --prompt "a person walking through a forest" \\
        --output_mp4 baseline.mp4

    # With motion LoRA
    python scripts/motion_lora/inference.py \\
        --prompt "a different person walking through a forest" \\
        --motion_lora logs/motion_lora_walking_v1/motion_lora.pt \\
        --output_mp4 with_lora.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import peft  # noqa: E402

from utils.wan_wrapper import (  # noqa: E402
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)
from scripts.motion_lora.train import (  # noqa: E402
    inject_motion_lora,
    MOTION_LORA_BLOCKS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_mp4", required=True)
    ap.add_argument("--motion_lora", default=None,
                    help="path to motion_lora.pt; omit for baseline (no LoRA)")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--num_frames", type=int, default=21,
                    help="number of LATENT frames; 21 ≈ 81 pixel frames @ 16 fps")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # ---- 1. Load base Wan2.1-T2V-1.3B (non-causal) -----------------------
    print("[inference] loading Wan2.1-T2V-1.3B base ...", flush=True)
    diffusion = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B", is_causal=False, timestep_shift=8.0
    )
    diffusion.model.requires_grad_(False)
    diffusion.to(device=device, dtype=torch.bfloat16)

    text_encoder = WanTextEncoder().to(device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    # ---- 2. Inject motion LoRA (skipped if no --motion_lora given) -------
    if args.motion_lora is not None:
        print(f"[inference] injecting motion LoRA scope ...", flush=True)
        diffusion.model = inject_motion_lora(
            diffusion.model, rank=args.lora_rank, alpha=args.lora_alpha
        )
        print(f"[inference] loading motion LoRA from {args.motion_lora}", flush=True)
        state = torch.load(args.motion_lora, map_location="cpu")
        peft.set_peft_model_state_dict(diffusion.model, state)

    diffusion.model.eval()

    # ---- 3. Encode prompt ------------------------------------------------
    with torch.no_grad():
        prompt_emb = text_encoder([args.prompt])["prompt_embeds"]  # [1, 512, 4096]

    # ---- 4. Set up 50-step rectified-flow schedule -----------------------
    sched = diffusion.scheduler
    sched.set_timesteps(args.num_steps, training=False)
    timesteps = sched.timesteps.to(device)
    sched.sigmas = sched.sigmas.to(device)
    sched.timesteps = timesteps

    # ---- 5. Sample noise + denoise loop ---------------------------------
    H_lat = args.height // 8
    W_lat = args.width // 8
    F_lat = args.num_frames
    xt = torch.randn(
        [1, F_lat, 16, H_lat, W_lat], device=device, dtype=torch.bfloat16,
    )
    cond = {"prompt_embeds": prompt_emb}

    print(f"[inference] running {args.num_steps} steps "
          f"(latent shape {tuple(xt.shape)}) ...", flush=True)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_b = t.view(1).expand(1, F_lat).to(device)  # [B, F]
            flow_pred, _ = diffusion(xt, cond, t_b)
            # FlowMatchScheduler.step takes [B*F]-flat or [B, F] timestep.
            # It internally flattens 2D → 1D, so passing t_b (= [1, F]) is fine.
            # But step's `sample` arg expects [N, C, H, W]; flatten BF.
            xt_flat = xt.reshape(F_lat, 16, H_lat, W_lat)
            flow_flat = flow_pred.reshape(F_lat, 16, H_lat, W_lat)
            xt_flat = sched.step(flow_flat, t_b, xt_flat,
                                 to_final=(i == len(timesteps) - 1))
            xt = xt_flat.reshape(1, F_lat, 16, H_lat, W_lat).to(torch.bfloat16)
            if i % 10 == 0 or i == len(timesteps) - 1:
                print(f"[inference] step {i+1}/{len(timesteps)}", flush=True)

    # ---- 6. Decode VAE ---------------------------------------------------
    print("[inference] decoding via VAE ...", flush=True)
    with torch.no_grad():
        pixels = vae.decode_to_pixel(xt)  # [1, F, C, H, W] in [-1, 1]
    pixels = (pixels.clamp(-1, 1) + 1) * 127.5
    pixels = pixels.to(torch.uint8).cpu()
    pixels = rearrange(pixels[0], "f c h w -> f h w c")

    # ---- 7. Save mp4 -----------------------------------------------------
    out_path = Path(args.output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), pixels, fps=args.fps)
    print(f"[inference] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
