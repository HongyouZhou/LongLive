"""Phase 2: MotionDirector teacher-finetune on Wan-14B (docs/04.md).

Trains a LoRA on top of frozen Wan-14B teacher using paper recipe
(L_temporal_MSE + L_AD with alpha=sqrt(2), beta=1), in epsilon space via
B1 close-form reverse from Wan's native (flow_pred, pred_x0) outputs.

Output: teacher_lora.pt — to be loaded as `real_score` adapter for the
existing DMD trainer in Phase 3.

Usage:
    python -m longlive.methods.motiondirector.train \\
        --config longlive/methods/motiondirector/configs/skateboarding_v1.yaml

    # short smoke (5 steps, no mid sample):
    python -m longlive.methods.motiondirector.train \\
        --config longlive/methods/motiondirector/configs/skateboarding_v1.yaml \\
        --smoke

Single-GPU only for first version (docs/04.md §3 K1). Wan-14B + LoRA
backward at 81 frame x 480^2 needs gradient checkpointing on bf16.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from longlive.methods.motiondirector.data import SkateboardingLatentDataset
from longlive.methods.motiondirector.losses import appearance_debias_loss
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


def _save_lora_ckpt(peft_model, out_dir: Path, tag: str) -> Path:
    """Save only LoRA params (PEFT extracts them via get_peft_model_state_dict)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(peft_model)
    # Move to CPU before save to avoid CUDA tensors in the ckpt.
    state = {k: v.detach().cpu() for k, v in state.items()}
    path = out_dir / f"teacher_lora_{tag}.pt"
    torch.save(state, path)
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

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.smoke:
        cfg.train_steps = 5
        cfg.ckpt_interval = 5
        cfg.warmup_steps = 0

    # Print resolved config — per CLAUDE.md "Confirm experiment config before launch".
    print("[motiondirector] resolved config:")
    print(OmegaConf.to_yaml(cfg))

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    device = torch.device(cfg.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"[motiondirector] device: {gpu_name} ({gpu_total_gib:.1f} GiB total)", flush=True)

    # ---------- VAE (small) ----------
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
    # umt5 before loading the 28 GB Wan-14B teacher so peak GPU usage stays
    # well clear of (text_encoder + teacher) ≈ 48 GB. Valid for the
    # single-caption Phase 2 first version; multi-caption training (paper
    # Table 1, 12 motions) will need to keep the encoder online instead.
    print("[motiondirector] loading text encoder (cache embeddings then free) ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        null_cond = {k: v.detach().clone() for k, v in text_encoder([""]).items()}
        train_cond = {k: v.detach().clone() for k, v in text_encoder([dataset.train_caption]).items()}
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Teacher ----------
    print(f"[motiondirector] loading teacher {cfg.teacher_name} ...", flush=True)
    teacher = WanDiffusionWrapper(
        model_name=cfg.teacher_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=False,
    )
    teacher.to(device)
    # Enable gradient checkpointing on the base Wan model BEFORE PEFT wrap;
    # the flag persists on the underlying nn.Module after wrapping.
    teacher.enable_gradient_checkpointing()
    teacher.model = configure_adapter_for_model(
        teacher.model,
        model_name="real_score",
        adapter_config=cfg.adapter,
        is_main_process=True,
    )
    teacher.model.train()  # LoRA-only training mode (base frozen by PEFT)

    # ---------- Optimizer ----------
    trainable = [p for p in teacher.model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[motiondirector] trainable params: {n_trainable:,}", flush=True)

    optimizer = AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    lr_sched = _linear_warmup_schedule(optimizer, int(cfg.warmup_steps))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sched = teacher.scheduler  # FlowMatchScheduler (already set_timesteps in wrapper init)

    # ---------- Training loop ----------
    print(f"[motiondirector] starting training, {cfg.train_steps} steps", flush=True)

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
        if step % 10 == 0 or step == int(cfg.train_steps) - 1:
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
            ckpt_path = _save_lora_ckpt(teacher.model, out_dir, str(step + 1))
            _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
            print(f"[motiondirector] saved ckpt: {ckpt_path}", flush=True)

    final_path = _save_lora_ckpt(teacher.model, out_dir, "final")
    print(f"[motiondirector] DONE. final ckpt: {final_path}", flush=True)


if __name__ == "__main__":
    main()
