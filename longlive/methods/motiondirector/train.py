"""Phase 2: MotionDirector teacher-finetune on Wan-14B (docs/04.md).

Trains a LoRA on top of frozen Wan-14B teacher using paper recipe
(L_temporal_MSE + L_AD with alpha=sqrt(2), beta=1), in epsilon space via
B1 close-form reverse from Wan's native (flow_pred, pred_x0) outputs.

Output: teacher_lora.pt — to be loaded as `real_score` adapter for the
existing DMD trainer in Phase 3.

Distributed via FSDP + torchrun (8 GPU default, single-rank also supported).
Usage:

    torchrun --nproc_per_node=8 -m longlive.methods.motiondirector.train \\
        --config longlive/methods/motiondirector/configs/skateboarding_v1.yaml

    # 5-step smoke:
    torchrun --nproc_per_node=8 -m longlive.methods.motiondirector.train \\
        --config ... --smoke

The sbatch wrapper (scripts/hpc/sbatch_motiondirector_train.sh) handles
SLURM env + torchrun launch.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from longlive.methods.motiondirector.data import SkateboardingLatentDataset
from longlive.methods.motiondirector.losses import appearance_debias_loss
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


def _linear_warmup_schedule(optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def _save_lora_ckpt(fsdp_peft_model, out_dir: Path, tag: str, rank0: bool) -> Path | None:
    """Gather LoRA state from sharded PEFT model → save on rank 0.

    `FSDP.state_dict_type` is a collective: every rank must enter the context
    or the all-gather deadlocks. Rank-0 then extracts the LoRA sub-dict via
    PEFT's helper and writes to disk; non-rank-0 ranks return None.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_peft_model, StateDictType.FULL_STATE_DICT, save_policy):
        full = fsdp_peft_model.state_dict()
    if not rank0:
        return None
    lora_state = get_peft_model_state_dict(fsdp_peft_model, state_dict=full)
    lora_state = {k: v.detach().cpu() for k, v in lora_state.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"teacher_lora_{tag}.pt"
    torch.save(lora_state, path)
    return path


def _prune_old_ckpts(out_dir: Path, keep_last: int) -> None:
    """Keep only the keep_last most recent ckpts (excluding 'final')."""
    ckpts = sorted(
        (p for p in out_dir.glob("teacher_lora_*.pt") if "final" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    while len(ckpts) > keep_last:
        ckpts[0].unlink()
        ckpts.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--smoke", action="store_true",
        help="5-step smoke run (overrides train_steps / ckpt_interval).",
    )
    args = ap.parse_args()

    # ---------- Distributed init (torchrun-managed env) ----------
    launch_distributed_job(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    rank0 = rank == 0

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.smoke:
        cfg.train_steps = 5
        cfg.ckpt_interval = 5
        cfg.warmup_steps = 0

    if rank0:
        print("[motiondirector] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[motiondirector] device: {gpu_name} ({gpu_total_gib:.1f} GiB total) "
            f"× world_size={world_size}",
            flush=True,
        )

    # Seed: same across ranks for model init (PEFT LoRA must initialize
    # identically before FSDP wrap). Re-seeded per-rank after setup for data
    # sampling variety (different clip / noise / timestep per rank).
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    # ---------- VAE (small, per rank) ----------
    if rank0:
        print("[motiondirector] loading VAE ...", flush=True)
    vae = WanVAEWrapper()
    vae.to(device).eval()

    # ---------- Data ----------
    dataset = SkateboardingLatentDataset(
        data_root=cfg.data_root,
        vae=vae,
        frame_count=int(cfg.frame_count),
        resolution=int(cfg.resolution),
        category=str(cfg.category),
        device=device,
    )

    # ---------- Text encoder: load → encode → free ----------
    # Pre-cache null + train_caption embeddings, then drop the ~20 GB fp32
    # umt5 before loading the 28 GB Wan-14B teacher so peak GPU per rank
    # stays well clear of (text_encoder + teacher). Valid for the single-
    # caption Phase 2 first version; multi-caption training (paper Table 1,
    # 12 motions) will need to keep the encoder online instead.
    if rank0:
        print("[motiondirector] loading text encoder (cache embeddings then free) ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        null_cond = {k: v.detach().clone() for k, v in text_encoder([""]).items()}
        train_cond = {k: v.detach().clone() for k, v in text_encoder([dataset.train_caption]).items()}
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Teacher ----------
    if rank0:
        print(f"[motiondirector] loading teacher {cfg.teacher_name} ...", flush=True)
    teacher = WanDiffusionWrapper(
        model_name=cfg.teacher_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=False,
    )
    # Move base weights to GPU before PEFT wrap; gradient checkpointing flag
    # is set on the underlying nn.Module and persists across PEFT / FSDP.
    teacher.enable_gradient_checkpointing()
    teacher.model = configure_adapter_for_model(
        teacher.model,
        model_name="real_score",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )

    # ---------- FSDP wrap ----------
    # Shards the 14B teacher across ranks; LoRA params (~300 M) also sharded.
    # mixed_precision=True → bf16 compute, fp32 grad reduce + fp32 buffers.
    # `use_orig_params=True` (set inside fsdp_wrap) is required for PEFT.
    teacher.model = fsdp_wrap(
        teacher.model,
        sharding_strategy="full",
        mixed_precision=True,
        wrap_strategy="size",
    )
    teacher.model.train()  # LoRA-only training mode (base frozen by PEFT)

    # Re-seed per rank for data-sampling variety after model init is done.
    random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    # ---------- Optimizer ----------
    trainable = [p for p in teacher.model.parameters() if p.requires_grad]
    n_trainable_local = sum(p.numel() for p in trainable)
    n_trainable_global = torch.tensor(n_trainable_local, device=device)
    dist.all_reduce(n_trainable_global)
    if rank0:
        print(
            f"[motiondirector] trainable params (FSDP-sharded total across "
            f"{world_size} ranks): {int(n_trainable_global.item()):,}",
            flush=True,
        )

    optimizer = AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    lr_sched = _linear_warmup_schedule(optimizer, int(cfg.warmup_steps))

    out_dir = Path(cfg.out_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # wait for rank-0 to create dir before any rank tries to write

    sched = teacher.scheduler  # FlowMatchScheduler (already set_timesteps in wrapper init)

    # ---------- Training loop ----------
    if rank0:
        print(
            f"[motiondirector] starting training, {cfg.train_steps} steps "
            f"× effective batch {world_size} (1 clip / rank / step)",
            flush=True,
        )

    for step in range(int(cfg.train_steps)):
        t0 = time.time()

        latent, _ = dataset.sample()
        # latent: (1, F, 16, H_l, W_l) bf16 on cuda

        # Pre-cached cond_dict: null vs train_caption (single-motion Phase 2 —
        # text encoder was freed at init, no per-step forward).
        cond_dict = null_cond if random.random() < float(cfg.null_prompt_p) else train_cond

        # Add noise (uniform t across frames for non-causal teacher).
        noise = torch.randn_like(latent)
        n_frames = latent.shape[1]
        t_scalar = torch.randint(
            int(cfg.t_min), int(cfg.t_max), (1,), device=device
        )
        timestep = t_scalar.expand(1, n_frames).contiguous()  # (B=1, F)
        noisy = sched.add_noise(
            latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, latent.shape[:2])

        # Forward (B1 close-form reverse: eps_pred = flow_pred + pred_x0).
        flow_pred, pred_x0 = teacher(
            noisy,
            cond_dict,
            timestep,
        )
        eps_pred = flow_pred + pred_x0
        eps_gt = noise

        # Loss = paper L_temporal_MSE + L_AD (alpha=sqrt(2), beta=1).
        loss_mse = F.mse_loss(eps_pred, eps_gt)
        loss_ad = appearance_debias_loss(
            eps_pred, eps_gt,
            alpha=float(cfg.ad_alpha), beta=float(cfg.ad_beta),
        )
        loss = loss_mse + loss_ad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()

        dt = time.time() - t0
        if rank0 and (step % 10 == 0 or step == int(cfg.train_steps) - 1):
            cur_lr = lr_sched.get_last_lr()[0]
            print(
                f"[motiondirector] step {step:4d}/{cfg.train_steps}  "
                f"t={int(t_scalar.item()):4d}  "
                f"loss={loss.item():.4f} "
                f"(mse={loss_mse.item():.4f}, ad={loss_ad.item():.4f})  "
                f"lr={cur_lr:.2e}  dt={dt:.1f}s",
                flush=True,
            )

        if (
            int(cfg.ckpt_interval) > 0
            and (step + 1) % int(cfg.ckpt_interval) == 0
            and (step + 1) < int(cfg.train_steps)
        ):
            ckpt_path = _save_lora_ckpt(teacher.model, out_dir, str(step + 1), rank0)
            if rank0:
                _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
                print(f"[motiondirector] saved ckpt: {ckpt_path}", flush=True)

    final_path = _save_lora_ckpt(teacher.model, out_dir, "final", rank0)
    if rank0:
        print(f"[motiondirector] DONE. final ckpt: {final_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
