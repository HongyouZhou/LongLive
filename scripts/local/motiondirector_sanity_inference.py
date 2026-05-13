"""Phase 2 sanity inference: load a teacher LoRA ckpt onto Wan-14B,
run 50-step UniPC on a couple of Skateboarding prompts to visually confirm
the ckpt is loadable + the LoRA-augmented teacher still produces coherent
video. Does NOT replace Phase 3 distillation + eval — purely an artifact
sanity check before investing the next ~5 h on DMD distillation.

Default prompts come from MotionDirector paper Fig 8a + Fig 10 Skateboarding
verbatim list:
  - "A man is skateboarding on the moon."   — subject close to train_caption
  - "A panda is skateboarding."             — cross-subject motion test

Usage (single GPU; ~3-5 min per 81-frame video on H200):
  python scripts/local/motiondirector_sanity_inference.py \\
      --lora-ckpt $LL_DATA/motiondirector_runs/skateboarding_v1/teacher_lora_final.pt \\
      --ckpt-dir $WAN_MODELS_ROOT/Wan2.1-T2V-14B \\
      --output-dir $LL_DATA/motiondirector_runs/sanity_<runid>

The sbatch wrapper at scripts/hpc/sbatch_motiondirector_sanity.sh handles
SLURM env + path resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import peft
import torch
from torchvision.io import write_video

from longlive.utils.lora_utils import configure_adapter_for_model
from wan.configs.wan_t2v_14B import t2v_14B
from wan.text2video import WanT2V


# Skateboarding verbatim 6 prompts from MotionDirector Fig 8a + Fig 10.
# Default sanity picks 2: one close to train_caption, one cross-subject.
DEFAULT_PROMPTS = [
    ("a_man_on_moon", "A man is skateboarding on the moon."),
    ("a_panda", "A panda is skateboarding."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-ckpt", required=True, help="Phase 2 teacher_lora_*.pt to load")
    p.add_argument("--ckpt-dir", required=True, help="Wan2.1-T2V-14B checkpoint directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--prompts", nargs="*", default=None,
        help="Optional list of prompt strings to override DEFAULT_PROMPTS",
    )
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--width", type=int, default=480,
                   help="Match Phase 2 training resolution (480x480 default)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=81,
                   help="Match Phase 2 frame_count (must be 4n+1)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=8.0,
                   help="Match Phase 2 config timestep_shift (8.0 for Wan-14B)")
    p.add_argument("--solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--rank", type=int, default=64, help="Must match training-time LoRA rank")
    p.add_argument("--alpha", type=int, default=64, help="Must match training-time LoRA alpha")
    return p.parse_args()


def main():
    args = parse_args()

    device_id = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_id)
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_total_gib = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
    print(f"[sanity] device: {gpu_name} ({gpu_total_gib:.1f} GiB total)", flush=True)

    if args.prompts:
        prompts = [(f"prompt_{i:02d}", s) for i, s in enumerate(args.prompts)]
    else:
        prompts = DEFAULT_PROMPTS

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[sanity] loading Wan2.1-T2V-14B from {args.ckpt_dir}", flush=True)
    pipe = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    print(
        f"[sanity] attaching LoRA (rank={args.rank}, alpha={args.alpha}) "
        f"to teacher DiT via configure_adapter_for_model('real_score') ...",
        flush=True,
    )
    adapter_cfg = {
        "type": "lora",
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": 0.0,
        "verbose": False,
    }
    pipe.model = configure_adapter_for_model(
        pipe.model,
        model_name="real_score",
        adapter_config=adapter_cfg,
        is_main_process=True,
    )

    print(f"[sanity] loading LoRA weights from {args.lora_ckpt}", flush=True)
    lora_state = torch.load(args.lora_ckpt, map_location="cpu")
    peft.set_peft_model_state_dict(pipe.model, lora_state)

    # PEFT initializes LoRA in fp32; cast to bf16 to match base Wan-14B
    # (same fix as longlive/methods/motiondirector/train.py post-PEFT).
    n_cast = 0
    for p in pipe.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    print(f"[sanity] cast {n_cast} fp32 LoRA params to bfloat16", flush=True)

    print(
        f"[sanity] model ready, generating {len(prompts)} prompt(s) × {args.seeds} seed(s)",
        flush=True,
    )
    manifest = []
    for tag, prompt in prompts:
        for s in range(args.seeds):
            seed = args.seed_base + s
            out_path = out_root / f"{tag}_seed{seed}.mp4"
            t0 = time.time()
            video = pipe.generate(
                input_prompt=prompt,
                size=(args.width, args.height),
                frame_num=args.frames,
                sample_solver=args.solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                shift=args.shift,
                seed=seed,
                offload_model=False,
            )
            v = (video.clamp(-1, 1) + 1) * 127.5
            v = v.permute(1, 2, 3, 0).to(torch.uint8).cpu()
            write_video(str(out_path), v, fps=args.fps)
            dt = time.time() - t0
            rec = {
                "tag": tag,
                "prompt": prompt,
                "seed": seed,
                "file": out_path.name,
                "walltime_s": round(dt, 2),
            }
            manifest.append(rec)
            print(f"[sanity] {tag} seed={seed} → {out_path.name} ({dt:.1f}s)", flush=True)

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sanity] done. {len(manifest)} videos in {out_root}/", flush=True)


if __name__ == "__main__":
    main()
