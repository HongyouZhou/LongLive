"""DRaFT-K trainer for 4-step DMD video fast adaptation.

Single-file trainer.  Setup phase mirrors longlive/methods/diffusion_ram/
train.py (FSDP + 2 PEFT adapters: `default` trainable, `anchor` zero-init
frozen = no-LoRA base).  Outer loop differs:

  for outer in range(outer_epochs):
      optimizer.zero_grad()
      # ── Reward branch (DRaFT-K backprop, expensive) ──
      generator.set_adapter("default")
      for k in range(k_rollouts_per_outer):
          noise = sample_noise(seed=outer*1009 + 31*rank + k)
          video, latent_x0 = rollout_engine.rollout_with_grad(
              noise, k_grad_steps=cfg.k_grad_steps
          )
          mf = reward_fn.score_grad(video[0])
          (-reward_coef * mf / k_rollouts_per_outer).backward()
          last_latent_x0 = latent_x0.detach()
      # ── KL anchor branch (cheap, anti-drift) ──
      if beta_kl > 0:
          for kl_step in range(n_kl_steps_per_outer):
              anchor_t = anchors[kl_step % len(anchors)]
              noise = torch.randn_like(last_latent_x0)
              x_t   = sched.add_noise(last_latent_x0, noise, anchor_t_tensor)
              # default forward (grad ON)
              v_default, _ = generator(x_t, train_cond, ts)
              # anchor forward (no_grad)
              generator.set_adapter("anchor")
              with torch.no_grad():
                  v_anchor, _ = generator(x_t, train_cond, ts)
              generator.set_adapter("default")  # restore before backward (gc safety)
              (beta_kl * F.mse_loss(v_default, v_anchor.detach()) / n_kl_steps_per_outer).backward()
      optimizer.step()

See docs/superpowers/specs/2026-05-22-draft-k-design.md for the locked
design and motivation.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import peft
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import AdamW

from longlive.methods.diffusion_draft.losses import kl_anchor_loss
from longlive.methods.motiondirector.data import SkateboardingLatentDataset
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.motion_reward import MotionFidelityRewardGrad
from longlive.utils.rl_rollout import RolloutEngine, maybe_barrier
from longlive.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


# ============================================================================
# Helpers
# ============================================================================


def _clean_fsdp_key(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


def _find_adapter_params(
    model: torch.nn.Module, adapter_tag: str
) -> list[torch.nn.Parameter]:
    needle = f".{adapter_tag}."
    return [p for n, p in model.named_parameters() if needle in n]


def _save_lora_ckpt(
    fsdp_peft_model: torch.nn.Module,
    out_dir: Path,
    tag: str,
    rank0: bool,
    adapter_name: str = "default",
) -> Path | None:
    """Gather FSDP state on rank 0 and save the named PEFT adapter.

    Saves ONLY the 'default' adapter by default — 'anchor' is zero by
    construction (PEFT B-projection zero-init) and reconstructs at load
    time, so it doesn't need to be in the checkpoint.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_peft_model, StateDictType.FULL_STATE_DICT, save_policy):
        full = fsdp_peft_model.state_dict()
    if not rank0:
        return None
    lora_state = get_peft_model_state_dict(
        fsdp_peft_model, state_dict=full, adapter_name=adapter_name
    )
    lora_state = {k: v.detach().cpu() for k, v in lora_state.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_{tag}.pt"
    torch.save(lora_state, path)
    return path


def _prune_old_ckpts(out_dir: Path, keep_last: int) -> None:
    ckpts = sorted(
        (p for p in out_dir.glob("lora_*.pt") if "final" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    while len(ckpts) > keep_last:
        ckpts[0].unlink()
        ckpts.pop(0)


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--smoke", action="store_true",
        help="2-outer × small smoke run (overrides outer_epochs / k_rollouts / n_kl).",
    )
    ap.add_argument(
        "--disable-wandb", action="store_true",
        help="Skip wandb.init — useful for local debug.",
    )
    args = ap.parse_args()

    # ---------- Distributed init ----------
    launch_distributed_job(backend="nccl")
    t_setup_start = time.time()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    rank0 = rank == 0

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.smoke:
        # Smoke override: 2 outer × K_rollouts=1 × n_kl=2 × K_grad=1.
        cfg.outer_epochs = 2
        cfg.k_rollouts_per_outer = 1
        cfg.n_kl_steps_per_outer = 2
        cfg.k_grad_steps = 1
        cfg.ckpt_interval = 2
        cfg.warmup_steps = 0

    if rank0:
        print("[diffusion_draft] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[diffusion_draft] device: {gpu_name} ({gpu_total_gib:.1f} GiB) "
            f"× world_size={world_size}",
            flush=True,
        )

    # ---------- wandb ----------
    wandb_enabled = rank0 and not args.disable_wandb
    if wandb_enabled:
        config_basename = Path(args.config).stem
        run_name = f"{config_basename}_{time.strftime('%y%m%d_%H%M')}"
        if args.smoke:
            run_name += "_smoke"
        wandb.init(
            project=getattr(cfg, "wandb_project", "longlive_diffusion_draft"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[diffusion_draft] wandb run: {wandb.run.url}", flush=True)

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    # ---------- VAE (bf16, eval) ----------
    if rank0:
        print("[diffusion_draft] loading VAE (bf16) ...", flush=True)
    vae = WanVAEWrapper()
    vae.to(device=device, dtype=torch.bfloat16).eval()

    # ---------- Dataset (ref clip + train caption) ----------
    dataset = SkateboardingLatentDataset(
        data_root=cfg.data_root,
        vae=vae,
        frame_count=int(cfg.frame_count),
        resolution=int(cfg.resolution),
        category=str(cfg.category),
        device=device,
        single_video=True,
    )
    train_caption = dataset.train_caption
    ref_clip_path = dataset.train_clip_path

    # ---------- Text encoder: load → encode → free ----------
    if rank0:
        print("[diffusion_draft] loading text encoder ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        train_cond = {k: v.detach().clone() for k, v in text_encoder([train_caption]).items()}
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Backbone + base ckpt + NVlabs baseline LoRA merge ----------
    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[diffusion_draft] building {cfg.model_name} ({arch}) ...", flush=True)
    model_kwargs = dict(
        model_name=cfg.model_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=is_causal,
    )
    if is_causal:
        model_kwargs["local_attn_size"] = int(getattr(cfg, "local_attn_size", -1))
        model_kwargs["sink_size"] = int(getattr(cfg, "sink_size", 0))
    generator = WanDiffusionWrapper(**model_kwargs)

    base_ckpt_path = os.path.expandvars(os.path.expanduser(cfg.base_ckpt))
    if rank0:
        print(f"[diffusion_draft] loading base ckpt: {base_ckpt_path}", flush=True)
    sd = torch.load(base_ckpt_path, map_location="cpu")
    if "generator" in sd:
        state = sd["generator"]
    elif "model" in sd:
        state = sd["model"]
    else:
        state = sd
    state = {_clean_fsdp_key(k): v for k, v in state.items()}
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if rank0:
        print(
            f"[diffusion_draft] base load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    del sd, state

    baseline_lora_ckpt = getattr(cfg, "baseline_lora_ckpt", None)
    if baseline_lora_ckpt:
        baseline_lora_ckpt = os.path.expandvars(os.path.expanduser(baseline_lora_ckpt))
        if rank0:
            print(
                f"[diffusion_draft] overlaying NVlabs baseline LoRA: {baseline_lora_ckpt}",
                flush=True,
            )
        generator.model = configure_adapter_for_model(
            generator.model,
            model_name="generator",
            adapter_config=cfg.baseline_adapter,
            is_main_process=rank0,
        )
        baseline_state = torch.load(baseline_lora_ckpt, map_location="cpu")
        if isinstance(baseline_state, dict) and "generator_lora" in baseline_state:
            baseline_state = baseline_state["generator_lora"]
        peft.set_peft_model_state_dict(generator.model, baseline_state)
        generator.model = generator.model.merge_and_unload()
        if rank0:
            print("[diffusion_draft] baseline LoRA merged into base weights", flush=True)
        del baseline_state

    # ---------- Attach 2 PEFT adapters: default + anchor ----------
    if rank0:
        print("[diffusion_draft] attaching adapters: default + anchor (v_ref)", flush=True)
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    peft_config_default = generator.model.peft_config["default"]
    generator.model.add_adapter("anchor", peft_config_default)
    for name, param in generator.model.named_parameters():
        if ".anchor." in name:
            param.requires_grad_(False)
    generator.model.set_adapter("default")
    generator.enable_gradient_checkpointing()

    # Cast fp32 PEFT params → bf16 (FSDP size-wrap dtype uniformity).
    n_cast = 0
    for p in generator.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if rank0:
        print(f"[diffusion_draft] cast {n_cast} fp32 params → bf16 (post-LoRA, pre-FSDP)", flush=True)

    # ---------- FSDP wrap ----------
    generator.model = fsdp_wrap(
        generator.model,
        sharding_strategy="full",
        mixed_precision=True,
        wrap_strategy="size",
    )
    generator.model.train()

    random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    # ---------- Adapter param lists (post-FSDP) ----------
    default_params = _find_adapter_params(generator.model, "default")
    anchor_params = _find_adapter_params(generator.model, "anchor")
    if rank0:
        print(
            f"[diffusion_draft] adapter param counts: "
            f"default={len(default_params)}, anchor={len(anchor_params)}",
            flush=True,
        )
        assert len(default_params) == len(anchor_params), (
            f"adapter param count mismatch: default={len(default_params)} "
            f"vs anchor={len(anchor_params)}"
        )

    # ---------- Optimizer (default adapter only) ----------
    trainable = [p for p in default_params if p.requires_grad]
    n_trainable_local = sum(p.numel() for p in trainable)
    n_trainable_global = torch.tensor(n_trainable_local, device=device)
    dist.all_reduce(n_trainable_global)
    if rank0:
        print(
            f"[diffusion_draft] trainable params (FSDP-sharded total): "
            f"{int(n_trainable_global.item()):,}",
            flush=True,
        )
    optimizer = AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    # ---------- Rollout engine ----------
    latent_b = 1
    latent_f = (int(cfg.frame_count) - 1) // 4 + 1
    latent_c = 16
    latent_h = int(cfg.resolution) // 8
    latent_w_pixel = int(cfg.resolution * 16 / 9)
    latent_w_pixel = int(getattr(cfg, "pixel_width", latent_w_pixel))
    latent_w = latent_w_pixel // 8
    latent_shape = (latent_b, latent_f, latent_c, latent_h, latent_w)

    pipeline_args = OmegaConf.create({
        "denoising_step_list": list(cfg.denoising_step_list),
        "warp_denoising_step": bool(getattr(cfg, "warp_denoising_step", True)),
        "num_frame_per_block": int(getattr(cfg, "num_frame_per_block", 3)),
        "context_noise": int(getattr(cfg, "context_noise", 0)),
        "model_kwargs": OmegaConf.create({
            "local_attn_size": int(getattr(cfg, "local_attn_size", -1)),
            "sink_size": int(getattr(cfg, "sink_size", 0)),
            "use_infinite_attention": False,
        }),
    })
    rollout_engine = RolloutEngine(
        generator=generator,
        vae=vae,
        cached_cond_dict=train_cond,
        pipeline_args=pipeline_args,
        device=device,
        latent_shape=latent_shape,
    )

    # ---------- Reward (grad-enabled, rank-0 first to avoid CoTracker hub race) ----------
    cache_root = Path(os.path.expandvars(cfg.cache_dir)) if getattr(cfg, "cache_dir", None) else None
    if rank0:
        print(f"[diffusion_draft] init reward grad (rank 0 first): ref={ref_clip_path}", flush=True)
        reward_fn = MotionFidelityRewardGrad(
            ref_path=ref_clip_path,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
        )
    dist.barrier()
    if not rank0:
        reward_fn = MotionFidelityRewardGrad(
            ref_path=ref_clip_path,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
        )
    dist.barrier()
    if rank0:
        print("[diffusion_draft] reward init complete on all ranks", flush=True)

    out_dir = Path(cfg.out_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    sched = generator.scheduler

    # ============ Outer training loop ============
    outer_epochs = int(cfg.outer_epochs)
    k_rollouts_per_outer = int(cfg.k_rollouts_per_outer)
    n_kl_steps_per_outer = int(cfg.n_kl_steps_per_outer)
    k_grad_steps = int(cfg.k_grad_steps)
    reward_coef = float(cfg.reward_coef)
    beta_kl = float(getattr(cfg, "beta_kl", 0.0))
    anchors = list(cfg.t_anchors)

    if rank0:
        print(
            f"[diffusion_draft] start: outer={outer_epochs} × K_rollouts={k_rollouts_per_outer} "
            f"× K_grad={k_grad_steps} × n_kl={n_kl_steps_per_outer} | "
            f"reward_coef={reward_coef} | beta_kl={beta_kl} | anchors={anchors}",
            flush=True,
        )

    global_step = 0
    t_train_loop_start = time.time()
    setup_time_s = t_train_loop_start - t_setup_start
    if rank0:
        print(f"[diffusion_draft] setup_time_s={setup_time_s:.1f}", flush=True)

    for outer in range(outer_epochs):
        t_outer = time.time()
        optimizer.zero_grad()

        # ── REWARD BRANCH (DRaFT-K reward-gradient backprop) ──
        generator.model.set_adapter("default")
        sum_mf, sum_reward_loss = 0.0, 0.0
        last_latent_x0 = None
        t_reward_start = time.time()
        for k in range(k_rollouts_per_outer):
            gen_seed = int(cfg.seed) + 1009 * outer + 31 * rank + k
            torch.manual_seed(gen_seed)
            noise = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
            video, latent_x0 = rollout_engine.rollout_with_grad(
                noise=noise, k_grad_steps=k_grad_steps,
            )
            # video: (B=1, F_pix, 3, H_pix, W_pix) in [0, 1], grad on if k_grad_steps > 0
            mf = reward_fn.score_grad(video[0])
            reward_loss = -reward_coef * mf / k_rollouts_per_outer
            reward_loss.backward()
            sum_mf += float(mf.detach())
            sum_reward_loss += float(reward_loss.detach())
            last_latent_x0 = latent_x0.detach()  # for KL branch (no grad needed)
        t_reward = time.time() - t_reward_start

        # ── KL ANCHOR BRANCH (cheap, anti-drift) ──
        sum_kl = 0.0
        t_kl_start = time.time()
        if beta_kl > 0.0 and last_latent_x0 is not None:
            for kl_step in range(n_kl_steps_per_outer):
                anchor_t = int(anchors[kl_step % len(anchors)])
                noise_kl = torch.randn_like(last_latent_x0)
                n_frames = last_latent_x0.shape[1]
                t_scalar = torch.tensor([anchor_t], device=device, dtype=torch.long)
                timestep = t_scalar.expand(1, n_frames).contiguous()
                x_t = sched.add_noise(
                    last_latent_x0.flatten(0, 1),
                    noise_kl.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).unflatten(0, last_latent_x0.shape[:2])

                # default forward (grad ON)
                generator.model.set_adapter("default")
                v_default, _ = generator(x_t, train_cond, timestep)

                # anchor forward (no_grad)
                generator.model.set_adapter("anchor")
                with torch.no_grad():
                    v_anchor, _ = generator(x_t, train_cond, timestep)

                # restore default BEFORE backward (gc safety)
                generator.model.set_adapter("default")

                kl = beta_kl * kl_anchor_loss(v_default, v_anchor) / n_kl_steps_per_outer
                kl.backward()
                sum_kl += float(kl.detach())
        t_kl = time.time() - t_kl_start

        optimizer.step()
        global_step += 1

        # ── LOG ──
        dt_outer = time.time() - t_outer
        avg_mf = sum_mf / max(1, k_rollouts_per_outer)
        total_reward_loss = sum_reward_loss
        avg_kl = sum_kl
        if rank0:
            print(
                f"[diffusion_draft] outer {outer:3d}/{outer_epochs}  "
                f"mf={avg_mf:.4f}  reward_loss={total_reward_loss:.4f}  "
                + (f"kl={avg_kl:.4f}  " if beta_kl > 0.0 else "")
                + f"dt={dt_outer:.1f}s (reward={t_reward:.1f}, kl={t_kl:.1f})",
                flush=True,
            )
        if wandb_enabled:
            log_dict = {
                "outer/mf": avg_mf,
                "outer/reward_loss": total_reward_loss,
                "outer/dt_total_s": dt_outer,
                "outer/dt_reward_s": t_reward,
                "outer/dt_kl_s": t_kl,
            }
            if beta_kl > 0.0:
                log_dict["outer/kl"] = avg_kl
            wandb.log(log_dict, step=global_step)

        # ── CKPT ──
        if (
            int(cfg.ckpt_interval) > 0
            and (outer + 1) % int(cfg.ckpt_interval) == 0
            and (outer + 1) < outer_epochs
        ):
            ckpt_path = _save_lora_ckpt(generator.model, out_dir, str(outer + 1), rank0)
            if rank0:
                _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
                print(f"[diffusion_draft] saved ckpt: {ckpt_path}", flush=True)

        maybe_barrier()

    train_loop_time_s = time.time() - t_train_loop_start
    final_path = _save_lora_ckpt(generator.model, out_dir, "final", rank0)
    if rank0:
        print(
            f"[diffusion_draft] DONE. setup_time_s={setup_time_s:.1f} "
            f"train_loop_time_s={train_loop_time_s:.1f}  final ckpt: {final_path}",
            flush=True,
        )
    if wandb_enabled:
        wandb.run.summary["setup_time_s"] = setup_time_s
        wandb.run.summary["train_loop_time_s"] = train_loop_time_s
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
