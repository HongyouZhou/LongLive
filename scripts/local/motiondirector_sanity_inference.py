"""Phase 2 sanity inference: load a teacher LoRA ckpt onto Wan-14B,
run 50-step UniPC on Skateboarding prompts to visually confirm the ckpt
is loadable + the LoRA-augmented teacher still produces coherent video.
Does NOT replace Phase 3 distillation + eval — purely an artifact sanity
check before investing the next ~5 h on DMD distillation.

Default prompts come from MotionDirector paper Fig 8a + Fig 10 Skateboarding
verbatim list (6 prompts):
  "A man is skateboarding on the moon."   — close to train_caption
  "An alien is skateboarding on Mars."
  "A bear is skateboarding."
  "A monkey is skateboarding."
  "A lion is skateboarding."
  "A panda is skateboarding."

Data-parallel: each torchrun rank holds a full copy of Wan-14B + LoRA and
generates its share of (prompt × seed) jobs. Same design as
scripts/local/teacher_boundary.py — no NCCL collectives, each rank passes
rank=0 to WanT2V to bypass USP sequence-parallel logic.

Usage (8-GPU on HPC; the only reliable 80 GB+ GRES allocation on Charité):
  torchrun --nproc_per_node=8 scripts/local/motiondirector_sanity_inference.py \\
      --lora-ckpt $LL_DATA/motiondirector_runs/skateboarding_v1/teacher_lora_final.pt \\
      --ckpt-dir $WAN_MODELS_ROOT/Wan2.1-T2V-14B \\
      --output-dir $LL_DATA/motiondirector_runs/sanity_<runid>

Single-GPU also works (e.g. on lab) when a 80 GB+ GPU is available:
  python scripts/local/motiondirector_sanity_inference.py [args]

The sbatch wrapper at scripts/hpc/sbatch_motiondirector_sanity.sh handles
SLURM env + torchrun launch.
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


# MotionDirector paper Fig 8a + Fig 10 Skateboarding verbatim 6 prompts.
DEFAULT_PROMPTS = [
    ("a_man_on_moon", "A man is skateboarding on the moon."),
    ("an_alien_on_mars", "An alien is skateboarding on Mars."),
    ("a_bear", "A bear is skateboarding."),
    ("a_monkey", "A monkey is skateboarding."),
    ("a_lion", "A lion is skateboarding."),
    ("a_panda", "A panda is skateboarding."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-ckpt", required=True, help="Phase 2 teacher_lora_*.pt to load")
    p.add_argument("--ckpt-dir", required=True, help="Wan2.1-T2V-14B checkpoint directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--prompts", nargs="*", default=None,
        help="Optional flat list of prompt strings, overrides DEFAULT_PROMPTS",
    )
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--width", type=int, default=480, help="Match Phase 2 training resolution")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=81, help="Match Phase 2 frame_count (4n+1)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument(
        "--shift", type=float, default=8.0,
        help="Match Phase 2 config timestep_shift (8.0 for Wan-14B)",
    )
    p.add_argument("--solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--rank", type=int, default=64, help="Must match training LoRA rank")
    p.add_argument("--alpha", type=int, default=64, help="Must match training LoRA alpha")
    return p.parse_args()


def init_distributed():
    """Each rank is fully independent (data-parallel inference, no NCCL).
    Same pattern as scripts/local/teacher_boundary.py — torchrun env vars
    used only to shard work and pick a GPU; WanT2V receives rank=0 on every
    process to bypass its USP sequence-parallel sync logic.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    torch.cuda.set_device(0)
    return 0, 0, 1


def main():
    args = parse_args()
    local_rank, rank, world_size = init_distributed()
    is_main = rank == 0

    if args.prompts:
        prompts = [(f"prompt_{i:02d}", s) for i, s in enumerate(args.prompts)]
    else:
        prompts = DEFAULT_PROMPTS

    jobs = [
        (tag, prompt, args.seed_base + s)
        for tag, prompt in prompts
        for s in range(args.seeds)
    ]
    my_jobs = jobs[rank::world_size]

    out_root = Path(args.output_dir)
    if is_main:
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_total_gib = torch.cuda.get_device_properties(local_rank).total_memory / 1024 ** 3
        print(f"[sanity] device: {gpu_name} ({gpu_total_gib:.1f} GiB total) × world_size={world_size}", flush=True)
        print(f"[sanity] total jobs={len(jobs)}, this rank={len(my_jobs)}", flush=True)
        out_root.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"[sanity] loading Wan2.1-T2V-14B from {args.ckpt_dir}", flush=True)
    pipe = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=0,                    # bypass USP collective sync — see teacher_boundary.py
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    if is_main:
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
        is_main_process=is_main,
    )

    if is_main:
        print(f"[sanity] loading LoRA weights from {args.lora_ckpt}", flush=True)
    lora_state = torch.load(args.lora_ckpt, map_location="cpu")
    peft.set_peft_model_state_dict(pipe.model, lora_state)

    # PEFT initializes LoRA in fp32; cast to bf16 to match Wan base
    # (same fix as longlive/methods/motiondirector/train.py post-PEFT).
    n_cast = 0
    for p in pipe.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if is_main:
        print(f"[sanity] cast {n_cast} fp32 LoRA params to bfloat16", flush=True)
        print(f"[sanity] model ready, beginning generation", flush=True)

    manifest_path = out_root / f"manifest_rank{rank}.jsonl"
    with open(manifest_path, "w") as manifest_f:
        for i, (tag, prompt, seed) in enumerate(my_jobs):
            out_path = out_root / f"{tag}_seed{seed}.mp4"
            if out_path.exists():
                print(f"[rank{rank}] [{i+1}/{len(my_jobs)}] skip existing {out_path.name}", flush=True)
                continue

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
            manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest_f.flush()
            print(
                f"[rank{rank}] [{i+1}/{len(my_jobs)}] {tag} seed={seed} → "
                f"{out_path.name} ({dt:.1f}s)",
                flush=True,
            )

    if is_main:
        print(f"[sanity] rank done. manifests at {out_root}/manifest_rank*.jsonl", flush=True)


if __name__ == "__main__":
    main()
