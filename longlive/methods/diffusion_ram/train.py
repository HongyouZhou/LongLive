"""Reinforce Adjoint Matching (RAM) trainer for the 4-step DMD base.

Single-file trainer implementing the RAM mechanism (arXiv:2605.10759, Eq. 17 +
Algorithm 1) on top of `longlive_base.pt` + NVlabs `lora.pt` merged base.
It keeps the old reward-rollout experiment's useful infrastructure, but the
method is independent from the removed DiffusionNFT trainer.

Outer loop (per epoch):

  1. set_adapter("default");  K=4 no_grad rollouts at 4-step DMD inference
     (on-policy per RAM Alg 1; NFT used "old" EMA — RAM doesn't).
  2. Score each rollout with motion_fidelity.
  3. Cross-rank all_gather over K × world_size rewards, group-normalize → r ∈ [0, 1].
  4. set_adapter("default");  for `inner_steps` gradient updates:
       a. Pick (k_idx, t_idx) with k = inner // K_noisings, t = inner % len(anchors).
       b. Forward-noise rollout_k's clean latent at anchor_t with random ε.
       c. v_default = generator(x_t, t) under "default" adapter, grad ON.
       d. set_adapter("anchor");  v_anchor = generator(x_t, t), no_grad.
       e. set_adapter("default") to restore before backward (gc safety).
       f. loss = ram_loss(v_default, v_anchor, ε, x_0_k, r_k, reward_coef).
          Optionally + beta_kl · kl_anchor_loss(...) when beta_kl > 0.
       g. backward, optimizer.step().
  5. Save LoRA ckpt every `ckpt_interval` outer epochs.

Differences from NFT:
  * 2 PEFT adapters (default + anchor); NO "old" EMA adapter.
  * 2 inner forwards per step (default + anchor); NFT had 3.
  * No `_ema_decay` / `_ema_refresh` machinery.
  * `k_idx / t_idx` cycling groups noisings under an endpoint (RAM convention).
  * Logging drops `outer/kl`, `outer/decay`, `outer/smooth_neg_over_default`;
    adds `outer/{shift_norm, target_norm, v_default_norm, v_anchor_norm, r_mean}`
    + per-anchor and per-rollout buckets.
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
import wandb
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import AdamW

from longlive.methods.diffusion_ram.losses import kl_anchor_loss, ram_loss
from longlive.data.motion_refs import make_reference_dataset
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.group_norm import group_normalize
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.motion_reward import MotionFidelityReward
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
    """Return params whose qualified name contains `.<adapter_tag>.`.

    Works on FSDP-wrapped PEFT models because FSDP only prepends
    `_fsdp_wrapped_module.` to param names; the PEFT adapter substring is
    preserved.
    """
    needle = f".{adapter_tag}."
    return [p for n, p in model.named_parameters() if needle in n]


def _save_lora_ckpt(
    fsdp_peft_model: torch.nn.Module,
    out_dir: Path,
    tag: str,
    rank0: bool,
    adapter_name: str = "default",
) -> Path | None:
    """Gather LoRA state from sharded PEFT model → save on rank 0 (one adapter).

    Saves ONLY the "default" adapter — "anchor" is zero by construction.
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
        help="2-outer-epoch × 4-inner smoke run (overrides outer_epochs / inner_steps).",
    )
    ap.add_argument(
        "--disable-wandb", action="store_true",
        help="Skip wandb.init — useful for local debug.",
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Override cfg.seed. Used by multi-seed sweeps where one yaml drives N runs.",
    )
    ap.add_argument(
        "--out-suffix", type=str, default="",
        help="Append to cfg.out_dir (e.g. '_seed3'). Keeps per-seed runs isolated.",
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
        # Smoke override: 2 outer × 4 inner (= 2 noisings × 2 endpoints, K=2).
        # The RAM startup assertion requires inner_steps == k_noisings × g_endpoints
        # AND k_rollouts == g_endpoints, so all four fields are overridden together.
        cfg.outer_epochs = 2
        cfg.inner_steps = 4
        cfg.k_rollouts = 2
        cfg.k_noisings_per_endpoint = 2
        cfg.g_endpoints_per_outer = 2
        cfg.ckpt_interval = 2
        cfg.warmup_steps = 0

    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.out_suffix:
        cfg.out_dir = f"{cfg.out_dir}{args.out_suffix}"

    # Startup invariant — RAM groups K_noisings under each rollout endpoint,
    # so the number of rollouts (K_rollouts) must equal the endpoint count.
    g_endpoints = int(getattr(cfg, "g_endpoints_per_outer", cfg.k_rollouts))
    k_noisings = int(getattr(cfg, "k_noisings_per_endpoint", cfg.inner_steps // g_endpoints))
    assert int(cfg.k_rollouts) == g_endpoints, (
        f"RAM requires k_rollouts == g_endpoints_per_outer; got "
        f"k_rollouts={cfg.k_rollouts}, g_endpoints_per_outer={g_endpoints}"
    )
    assert k_noisings * g_endpoints == int(cfg.inner_steps), (
        f"inner_steps must equal k_noisings * g_endpoints; got "
        f"inner_steps={cfg.inner_steps}, k_noisings={k_noisings}, g_endpoints={g_endpoints}"
    )

    if rank0:
        print("[diffusion_ram] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[diffusion_ram] device: {gpu_name} ({gpu_total_gib:.1f} GiB) "
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
        if args.out_suffix:
            run_name += args.out_suffix
        wandb.init(
            project=getattr(cfg, "wandb_project", "longlive_diffusion_ram"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[diffusion_ram] wandb run: {wandb.run.url}", flush=True)

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    # ---------- VAE ----------
    if rank0:
        print("[diffusion_ram] loading VAE (bf16) ...", flush=True)
    vae = WanVAEWrapper(model_name=str(cfg.model_name))
    vae.to(device=device, dtype=torch.bfloat16).eval()

    # ---------- Dataset (ref clip path + train caption only) ----------
    dataset = make_reference_dataset(cfg, vae=vae, device=device)
    train_caption = dataset.train_caption
    ref_clip_path = dataset.train_clip_path

    # ---------- Text encoder: load → encode → free ----------
    if rank0:
        print("[diffusion_ram] loading text encoder ...", flush=True)
    text_encoder = WanTextEncoder(model_name=str(cfg.model_name))
    text_encoder.to(device).eval()
    with torch.no_grad():
        train_cond = {k: v.detach().clone() for k, v in text_encoder([train_caption]).items()}
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Backbone + base ckpt + NVlabs baseline LoRA merge ----------
    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[diffusion_ram] building {cfg.model_name} ({arch}) ...", flush=True)
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
        print(f"[diffusion_ram] loading base ckpt: {base_ckpt_path}", flush=True)
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
            f"[diffusion_ram] base load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    del sd, state

    baseline_lora_ckpt = getattr(cfg, "baseline_lora_ckpt", None)
    if baseline_lora_ckpt:
        baseline_lora_ckpt = os.path.expandvars(os.path.expanduser(baseline_lora_ckpt))
        if rank0:
            print(
                f"[diffusion_ram] overlaying NVlabs baseline LoRA: {baseline_lora_ckpt}",
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
            print("[diffusion_ram] baseline LoRA merged into base weights", flush=True)
        del baseline_state

    # ---------- Attach 2 PEFT adapters (BEFORE FSDP wrap) ----------
    # RAM only needs 2: "default" (trainable) + "anchor" (zero-init, frozen LongLive base).
    # No "old" EMA adapter (NFT-specific).
    if rank0:
        print("[diffusion_ram] attaching adapters: default + anchor (frozen LongLive base)", flush=True)
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    peft_config_default = generator.model.peft_config["default"]
    generator.model.add_adapter("anchor", peft_config_default)

    # "anchor" is never trained; zero-init B-projection means LoRA delta = 0 →
    # forward through "anchor" equals the no-LoRA base.  Freeze its params.
    for name, param in generator.model.named_parameters():
        if ".anchor." in name:
            param.requires_grad_(False)

    generator.model.set_adapter("default")
    generator.enable_gradient_checkpointing()

    # Cast fp32 PEFT params to bf16 (FSDP size-based wrap requires uniform dtype).
    n_cast = 0
    for p in generator.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if rank0:
        print(f"[diffusion_ram] cast {n_cast} fp32 params → bf16 (post-LoRA, pre-FSDP)", flush=True)

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
            f"[diffusion_ram] adapter param counts: "
            f"default={len(default_params)}, anchor={len(anchor_params)}",
            flush=True,
        )
        # Sanity: when add_adapter("anchor", peft_config_default) is honored,
        # both adapters cover identical layers and the counts match.
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
            f"[diffusion_ram] trainable params (FSDP-sharded total): "
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

    # ---------- Reward (cache ref tracklets once, rank-0 first to avoid CoTracker hub race) ----------
    cache_root = Path(os.path.expandvars(cfg.cache_dir)) if getattr(cfg, "cache_dir", None) else None
    scratch_dir = Path(os.path.expandvars(cfg.scratch_dir)) / f"rank{rank}"
    if rank0:
        print(f"[diffusion_ram] init reward (rank 0 first): ref={ref_clip_path}", flush=True)
        reward_fn = MotionFidelityReward(
            ref_path=ref_clip_path,
            scratch_dir=scratch_dir,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
            fps=int(getattr(cfg, "fps", 16)),
        )
    dist.barrier()
    if not rank0:
        reward_fn = MotionFidelityReward(
            ref_path=ref_clip_path,
            scratch_dir=scratch_dir,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
            fps=int(getattr(cfg, "fps", 16)),
        )
    dist.barrier()
    if rank0:
        print("[diffusion_ram] reward init complete on all ranks", flush=True)

    out_dir = Path(cfg.out_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    sched = generator.scheduler

    # ============ Outer / inner training loop ============
    outer_epochs = int(cfg.outer_epochs)
    inner_steps = int(cfg.inner_steps)
    K = int(cfg.k_rollouts)
    adv_clip_max = float(cfg.adv_clip_max)
    anchors = list(cfg.t_anchors)
    reward_coef = float(getattr(cfg, "reward_coef", 1.0))
    beta_kl = float(getattr(cfg, "beta_kl", 0.0))
    rollout_adapter = str(getattr(cfg, "rollout_adapter", "default"))
    assert rollout_adapter in ("default", "anchor"), (
        f"rollout_adapter must be 'default' or 'anchor', got {rollout_adapter!r}"
    )

    if rank0:
        print(
            f"[diffusion_ram] start: outer={outer_epochs} × inner={inner_steps} "
            f"× K={K} (g_endpoints={g_endpoints}, k_noisings={k_noisings}) | "
            f"reward_coef={reward_coef} | beta_kl={beta_kl} | "
            f"rollout_adapter={rollout_adapter} | anchors={anchors}",
            flush=True,
        )

    global_step = 0
    t_train_loop_start = time.time()
    setup_time_s = t_train_loop_start - t_setup_start
    if rank0:
        print(f"[diffusion_ram] setup_time_s={setup_time_s:.1f}", flush=True)

    for outer in range(outer_epochs):
        t_outer = time.time()

        # ---- ROLLOUT phase ----
        # On-policy per RAM Alg 1: sample x_0 from the current trainable model.
        generator.model.set_adapter(rollout_adapter)
        rollout_seed = int(cfg.seed) + 1009 * outer + 31 * rank
        with torch.no_grad():
            rollouts = rollout_engine.rollout_k(
                k=K, dtype=torch.bfloat16, base_seed=rollout_seed,
            )
        t_rollout = time.time() - t_outer

        # ---- REWARD phase ----
        t_reward_start = time.time()
        rewards = []
        for k_idx, (video, _latent) in enumerate(rollouts):
            score = reward_fn.score(video[0], tag=f"e{outer}_k{k_idx}")
            rewards.append(score)
        t_reward = time.time() - t_reward_start

        # ---- CROSS-RANK GROUP NORM (carried over from NFT-H1) ----
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        gathered = [torch.zeros_like(rewards_t) for _ in range(world_size)]
        dist.all_gather(gathered, rewards_t)
        all_rewards = torch.cat(gathered).cpu().tolist()
        r_all, reward_diag = group_normalize(
            all_rewards, adv_clip_max=adv_clip_max,
        )
        r_tensor = r_all[rank * K : (rank + 1) * K].to(device)

        # ---- TRAINING phase ----
        generator.model.set_adapter("default")
        t_train_start = time.time()

        # Per-outer accumulators
        sum_loss = 0.0
        sum_kl = 0.0
        sum_shift_norm = 0.0
        sum_target_norm = 0.0
        sum_v_default_norm = 0.0
        sum_v_anchor_norm = 0.0
        sum_r_scalar = 0.0
        # Per-anchor / per-rollout buckets for diagnostic disaggregation
        per_anchor_shift = [0.0] * len(anchors)
        per_anchor_count = [0] * len(anchors)
        per_rollout_shift = [0.0] * K
        per_rollout_count = [0] * K

        for inner in range(inner_steps):
            # RAM-style cycling: noisings grouped under endpoint, anchor rotates inside.
            k_idx = (inner // k_noisings) % g_endpoints
            t_idx = inner % len(anchors)
            anchor_t = int(anchors[t_idx])
            _video_k, latent_k = rollouts[k_idx]
            x0_ref = latent_k.to(torch.bfloat16)

            noise = torch.randn_like(x0_ref)
            n_frames = x0_ref.shape[1]
            t_scalar = torch.tensor([anchor_t], device=device, dtype=torch.long)
            timestep = t_scalar.expand(1, n_frames).contiguous()
            noisy = sched.add_noise(
                x0_ref.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, x0_ref.shape[:2])

            # --- default forward (grad) ---
            flow_pred_default, _pred_x0_default = generator(noisy, train_cond, timestep)

            # --- anchor forward (no_grad) ---
            generator.model.set_adapter("anchor")
            with torch.no_grad():
                flow_pred_anchor, _pred_x0_anchor = generator(noisy, train_cond, timestep)

            # Restore default BEFORE backward so gradient-checkpoint recompute
            # uses the same active adapter as the original forward.
            generator.model.set_adapter("default")

            # --- loss ---
            loss_ram, diag = ram_loss(
                v_default=flow_pred_default,
                v_anchor=flow_pred_anchor,
                noise=noise,
                x0_ref=x0_ref,
                r=r_tensor[k_idx:k_idx + 1],
                reward_coef=reward_coef,
                anchor_idx=t_idx,
            )
            if beta_kl > 0.0:
                kl = kl_anchor_loss(flow_pred_default, flow_pred_anchor)
                loss = loss_ram + beta_kl * kl
            else:
                kl = torch.zeros((), device=device, dtype=loss_ram.dtype)
                loss = loss_ram

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.detach())
            sum_kl += float(kl.detach())
            sum_shift_norm += float(diag["ram/shift_norm"])
            sum_target_norm += float(diag["ram/target_norm"])
            sum_v_default_norm += float(diag["ram/v_default_norm"])
            sum_v_anchor_norm += float(diag["ram/v_anchor_norm"])
            sum_r_scalar += float(diag["ram/r_scalar"])
            per_anchor_shift[t_idx] += float(diag["ram/shift_norm"])
            per_anchor_count[t_idx] += 1
            per_rollout_shift[k_idx] += float(diag["ram/shift_norm"])
            per_rollout_count[k_idx] += 1
            global_step += 1

        t_train = time.time() - t_train_start

        # ---- LOG ----
        dt_outer = time.time() - t_outer
        n = max(1, inner_steps)
        avg_loss = sum_loss / n
        avg_kl = sum_kl / n
        avg_shift = sum_shift_norm / n
        avg_target = sum_target_norm / n
        avg_v_default = sum_v_default_norm / n
        avg_v_anchor = sum_v_anchor_norm / n
        avg_r = sum_r_scalar / n
        if rank0:
            print(
                f"[diffusion_ram] outer {outer:3d}/{outer_epochs}  "
                f"rewards mean={reward_diag['reward/raw_mean']:.3f} "
                f"std={reward_diag['reward/raw_std']:.3f}  "
                f"loss={avg_loss:.4f}  r_mean={avg_r:.3f}  "
                f"shift_norm={avg_shift:.3f}  "
                f"v_def={avg_v_default:.3f}  v_anc={avg_v_anchor:.3f}  "
                + (f"kl={avg_kl:.4f}  " if beta_kl > 0.0 else "")
                + f"dt={dt_outer:.1f}s (rollout={t_rollout:.1f}, "
                f"reward={t_reward:.1f}, train={t_train:.1f})",
                flush=True,
            )
        if wandb_enabled:
            log_dict = {
                "outer/loss": avg_loss,
                "outer/shift_norm": avg_shift,
                "outer/target_norm": avg_target,
                "outer/v_default_norm": avg_v_default,
                "outer/v_anchor_norm": avg_v_anchor,
                "outer/r_mean": avg_r,
                "outer/dt_total_s": dt_outer,
                "outer/dt_rollout_s": t_rollout,
                "outer/dt_reward_s": t_reward,
                "outer/dt_train_s": t_train,
                **{f"reward/{k.split('/')[-1]}": v for k, v in reward_diag.items()},
            }
            if beta_kl > 0.0:
                log_dict["outer/kl"] = avg_kl
            # Per-anchor / per-rollout shift norms
            for t_i, t_v in enumerate(anchors):
                if per_anchor_count[t_i]:
                    log_dict[f"ram/shift_norm_t{t_i}"] = (
                        per_anchor_shift[t_i] / per_anchor_count[t_i]
                    )
            for k_i in range(K):
                if per_rollout_count[k_i]:
                    log_dict[f"ram/shift_norm_k{k_i}"] = (
                        per_rollout_shift[k_i] / per_rollout_count[k_i]
                    )
            wandb.log(log_dict, step=global_step)

        if (
            int(cfg.ckpt_interval) > 0
            and (outer + 1) % int(cfg.ckpt_interval) == 0
            and (outer + 1) < outer_epochs
        ):
            ckpt_path = _save_lora_ckpt(generator.model, out_dir, str(outer + 1), rank0)
            if rank0:
                _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
                print(f"[diffusion_ram] saved ckpt: {ckpt_path}", flush=True)

        maybe_barrier()

    train_loop_time_s = time.time() - t_train_loop_start
    final_path = _save_lora_ckpt(generator.model, out_dir, "final", rank0)
    if rank0:
        print(
            f"[diffusion_ram] DONE. setup_time_s={setup_time_s:.1f} "
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
