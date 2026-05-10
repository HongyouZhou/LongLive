"""Wan2.1-T2V-14B teacher boundary test: pure 50-step UniPC inference,
no LoRA, no DMD. Establishes the upper bound of what the teacher can produce
on cat-dunk / human-dunk / cat-static prompts before any motion specialization.

Outputs:
  <output_dir>/<group>/<idx:02d>_seed<seed>.mp4
  <output_dir>/manifest_rank<rank>.jsonl

DDP shards (group, idx, seed) tuples across ranks. Each rank holds a full
copy of Wan T2V 14B (no FSDP for inference; ~30 GB GPU + ~10 GB CPU T5).

Usage (single GPU smoke):
  python scripts/local/teacher_boundary.py \\
    --prompts prompts/teacher_boundary_v1.jsonl \\
    --ckpt-dir $WAN_MODELS_ROOT/Wan2.1-T2V-14B \\
    --output-dir vis/teacher_boundary_v1 \\
    --seeds 1

Usage (8-GPU full sweep, prompt x seed sharded):
  torchrun --nproc_per_node=8 scripts/local/teacher_boundary.py [args] --seeds 4
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

import torch
from torchvision.io import write_video

from wan.configs.wan_t2v_14B import t2v_14B
from wan.text2video import WanT2V


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, help="JSONL with {group, idx, prompt} per line")
    p.add_argument("--ckpt-dir", required=True, help="Wan2.1-T2V-14B checkpoint directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, default=4, help="Number of seeds per prompt")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--width", type=int, default=832, help="Video width (Wan default 480p: 832x480)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=81, help="Frame count, must be 4n+1")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--fps", type=int, default=16, help="Output mp4 fps")
    return p.parse_args()


def init_distributed():
    """Each rank is fully independent in our data-parallel design — we only
    use torchrun env vars (LOCAL_RANK / WORLD_SIZE / RANK) to shard work and
    pick a GPU. We deliberately do NOT call dist.init_process_group: Wan's
    upstream WanT2V.generate() is built for collaborative (USP / sequence
    parallel) sampling where only rank 0 VAE-decodes and returns the video;
    non-zero ranks return None. Bypass that by passing rank=0 to every
    WanT2V instance (see main()) and skipping NCCL altogether.
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

    records = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    jobs = [
        (r["group"], r["idx"], r["prompt"], args.seed_base + s)
        for r in records
        for s in range(args.seeds)
    ]
    my_jobs = jobs[rank::world_size]

    if is_main:
        print(f"[rank0] world_size={world_size}  total_jobs={len(jobs)}  jobs/rank0={len(my_jobs)}")
        print(f"[rank0] size={args.width}x{args.height}  frames={args.frames}  steps={args.steps}")
        print(f"[rank0] groups={sorted({r['group'] for r in records})}")

    out_root = Path(args.output_dir)
    for r in records:
        (out_root / r["group"]).mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"[rank0] loading Wan2.1-T2V-14B from {args.ckpt_dir}")
    # rank=0 is intentional on every process — see init_distributed() docstring.
    pipe = WanT2V(
        config=t2v_14B,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )
    if is_main:
        print(f"[rank0] model ready, beginning generation")

    manifest_path = out_root / f"manifest_rank{rank}.jsonl"
    with open(manifest_path, "w") as manifest_f:
        for i, (group, idx, prompt, seed) in enumerate(my_jobs):
            out_path = out_root / group / f"{idx:02d}_seed{seed}.mp4"
            if out_path.exists():
                print(f"[rank{rank}] [{i+1}/{len(my_jobs)}] skip existing {out_path.name}")
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
                "group": group,
                "idx": idx,
                "seed": seed,
                "prompt": prompt,
                "file": str(out_path.relative_to(out_root)),
                "walltime_s": round(dt, 2),
            }
            manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest_f.flush()
            print(f"[rank{rank}] [{i+1}/{len(my_jobs)}] {group}/{idx:02d} seed={seed} ({dt:.1f}s)")

    if is_main:
        print(f"[rank0] this rank done. manifests at {out_root}/manifest_rank*.jsonl")


if __name__ == "__main__":
    main()
