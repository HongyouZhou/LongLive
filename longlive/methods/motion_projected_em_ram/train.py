"""Motion-Projected EM-RAM trainer for the 4-step DMD base.

Single-file trainer implementing an empirical EM / mirror-descent variant of
RAM on top of `longlive_base.pt` + NVlabs `lora.pt` merged base.  The outer
loop computes a reward-tilted endpoint distribution under a KL budget; the
inner loop distills selected endpoints through RAM's residual target.

Outer loop (per epoch):

  1. set_adapter("default");  K=4 no_grad rollouts at 4-step DMD inference
     (on-policy per RAM Alg 1; NFT used "old" EMA; RAM doesn't).
  2. Score each rollout with tracklet direction plus speed-ratio consistency.
  3. Cross-rank all_gather over K x world_size rewards, then E-step:
     q_i proportional to exp(A_i / eta), with KL(q || uniform) controlled by config.
  4. set_adapter("default");  for `inner_steps` gradient updates:
       a. Pick (k_idx, t_idx) with k = inner // K_noisings, t = inner % len(anchors).
       b. Forward-noise rollout_k's clean latent at anchor_t with random eps.
       c. v_default = generator(x_t, t) under "default" adapter, grad ON.
       d. set_adapter("anchor");  v_anchor = generator(x_t, t), no_grad.
       e. set_adapter("default") to restore before backward (gc safety).
       f. loss = motion_projected_em_ram_loss(v_default, v_anchor, eps, x_0_k, alpha_k, reward_coef).
          Optionally + beta_kl * kl_anchor_loss(...) when beta_kl > 0.
       g. backward, optimizer.step().
  5. Save LoRA ckpt every `ckpt_interval` outer epochs.

Differences from NFT:
  * 2 PEFT adapters (default + anchor); NO "old" EMA adapter.
  * 2 inner forwards per step (default + anchor); NFT had 3.
  * No `_ema_decay` / `_ema_refresh` machinery.
  * `k_idx / t_idx` cycling groups noisings under an endpoint (RAM convention).
  * Logging reports raw-vs-projected shift norms, EM selection stats, and
    reward component means.
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

from longlive.methods.motion_projected_em_ram.losses import (
    em_tilt_alpha_and_weights,
    feature_consistency_gates,
    feature_consistency_weights,
    kl_anchor_loss,
    mode_cover_velocity_loss,
    motion_projected_em_ram_loss,
    residual_bucket_time_weights,
    reward_weighted_velocity_loss,
    score_consistency_weights,
    time_local_reward_weighted_velocity_loss,
)
from longlive.methods.motion_projected_em_ram.reward import MotionProjectedEMReward
from longlive.data.motion_refs import make_reference_dataset
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
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
    """Gather LoRA state from sharded PEFT model and save on rank 0 (one adapter).

    Saves ONLY the "default" adapter; "anchor" is zero by construction.
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


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    seen = set()
    out = []
    for prompt in prompts:
        p = str(prompt).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _cover_step_schedule(inner_steps: int, cover_steps: int) -> list[bool]:
    cover_steps = max(0, min(int(cover_steps), int(inner_steps)))
    if cover_steps == 0:
        return [False] * int(inner_steps)
    schedule = []
    for idx in range(int(inner_steps)):
        prev = (idx * cover_steps) // int(inner_steps)
        cur = ((idx + 1) * cover_steps) // int(inner_steps)
        schedule.append(cur > prev)
    return schedule


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--smoke", action="store_true",
        help="2-outer-epoch x 4-inner smoke run (overrides outer_epochs / inner_steps).",
    )
    ap.add_argument(
        "--disable-wandb", action="store_true",
        help="Skip wandb.init; useful for local debug.",
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
        # Smoke override: 2 outer x 4 inner (= 2 noisings x 2 endpoints, K=2).
        # The startup assertion requires inner_steps == k_noisings x g_endpoints
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

    subspace_mode = str(getattr(cfg, "subspace_mode", "coarse_motion"))
    reference_motion_scope = str(getattr(cfg, "reference_motion_scope", "frame"))
    reference_motion_positive = bool(getattr(cfg, "reference_motion_positive", False))
    reference_motion_mix = float(getattr(cfg, "reference_motion_mix", 1.0))
    reference_motion_temporal_center = bool(
        getattr(cfg, "reference_motion_temporal_center", False)
    )
    lambda_reference_orthogonal = float(getattr(cfg, "lambda_reference_orthogonal", 0.0))
    reference_subspace_modes = ("reference_motion", "hybrid_reference_motion")
    assert subspace_mode in ("coarse_motion", *reference_subspace_modes), (
        f"subspace_mode must be 'coarse_motion', 'reference_motion', or "
        f"'hybrid_reference_motion', got {subspace_mode!r}"
    )
    assert reference_motion_scope in ("frame", "global"), (
        f"reference_motion_scope must be 'frame' or 'global', got {reference_motion_scope!r}"
    )

    # Startup invariant: group K_noisings under each rollout endpoint,
    # so the number of rollouts (K_rollouts) must equal the endpoint count.
    g_endpoints = int(getattr(cfg, "g_endpoints_per_outer", cfg.k_rollouts))
    k_noisings = int(getattr(cfg, "k_noisings_per_endpoint", cfg.inner_steps // g_endpoints))
    assert int(cfg.k_rollouts) == g_endpoints, (
        f"Motion-Projected EM-RAM requires k_rollouts == g_endpoints_per_outer; got "
        f"k_rollouts={cfg.k_rollouts}, g_endpoints_per_outer={g_endpoints}"
    )
    assert k_noisings * g_endpoints == int(cfg.inner_steps), (
        f"inner_steps must equal k_noisings * g_endpoints; got "
        f"inner_steps={cfg.inner_steps}, k_noisings={k_noisings}, g_endpoints={g_endpoints}"
    )

    if rank0:
        print("[motion_projected_em_ram] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[motion_projected_em_ram] device: {gpu_name} ({gpu_total_gib:.1f} GiB) "
            f"x world_size={world_size}",
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
            project=getattr(cfg, "wandb_project", "longlive_motion_projected_em_ram"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[motion_projected_em_ram] wandb run: {wandb.run.url}", flush=True)

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    # ---------- VAE ----------
    if rank0:
        print("[motion_projected_em_ram] loading VAE (bf16) ...", flush=True)
    vae = WanVAEWrapper()
    vae.to(device=device, dtype=torch.bfloat16).eval()

    # ---------- Dataset (ref clip path + train caption only) ----------
    dataset = make_reference_dataset(cfg, vae=vae, device=device)
    train_caption = dataset.train_caption
    ref_clip_path = dataset.train_clip_path
    mstep_objective = str(getattr(cfg, "mstep_objective", "alpha_shift"))
    reference_latent = None
    if subspace_mode in reference_subspace_modes:
        if rank0:
            print("[motion_projected_em_ram] encoding reference latent for subspace projector ...", flush=True)
        reference_latent, _ = dataset.sample()
        reference_latent = reference_latent.detach().to(device=device, dtype=torch.bfloat16)
        if rank0:
            print(
                f"[motion_projected_em_ram] reference latent shape={tuple(reference_latent.shape)}",
                flush=True,
            )

    # ---------- Text encoder: load, encode, free ----------
    if rank0:
        print("[motion_projected_em_ram] loading text encoder ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        train_cond = {k: v.detach().clone() for k, v in text_encoder([train_caption]).items()}
        cover_prompt_source = str(getattr(cfg, "cover_prompt_source", "train_caption"))
        raw_cover_prompts = list(getattr(cfg, "cover_prompts", []))
        if cover_prompt_source == "train_caption":
            cover_prompts = [train_caption]
        elif cover_prompt_source == "train_and_eval_prompts":
            cover_prompts = [train_caption, *raw_cover_prompts]
        elif cover_prompt_source == "config":
            cover_prompts = raw_cover_prompts
        else:
            raise ValueError(
                "cover_prompt_source must be 'train_caption', "
                f"'train_and_eval_prompts', or 'config', got {cover_prompt_source!r}"
            )
        cover_prompts = _dedupe_prompts(cover_prompts)
        if mstep_objective == "two_stream_on_policy_cover" and not cover_prompts:
            raise ValueError("two_stream_on_policy_cover requires at least one cover prompt")
        cover_cond_list = []
        if mstep_objective == "two_stream_on_policy_cover":
            cover_cond_batch = {
                k: v.detach().clone() for k, v in text_encoder(cover_prompts).items()
            }
            cover_cond_list = [
                {k: v[i:i + 1].detach().clone() for k, v in cover_cond_batch.items()}
                for i in range(len(cover_prompts))
            ]
    del text_encoder
    torch.cuda.empty_cache()
    if rank0 and mstep_objective == "two_stream_on_policy_cover":
        print(
            f"[motion_projected_em_ram] cover prompts: {len(cover_prompts)} "
            f"(source={cover_prompt_source})",
            flush=True,
        )

    # ---------- Backbone + base ckpt + NVlabs baseline LoRA merge ----------
    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[motion_projected_em_ram] building {cfg.model_name} ({arch}) ...", flush=True)
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
        print(f"[motion_projected_em_ram] loading base ckpt: {base_ckpt_path}", flush=True)
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
            f"[motion_projected_em_ram] base load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    del sd, state

    baseline_lora_ckpt = getattr(cfg, "baseline_lora_ckpt", None)
    if baseline_lora_ckpt:
        baseline_lora_ckpt = os.path.expandvars(os.path.expanduser(baseline_lora_ckpt))
        if rank0:
            print(
                f"[motion_projected_em_ram] overlaying NVlabs baseline LoRA: {baseline_lora_ckpt}",
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
            print("[motion_projected_em_ram] baseline LoRA merged into base weights", flush=True)
        del baseline_state

    # ---------- Attach 2 PEFT adapters (BEFORE FSDP wrap) ----------
    # Motion-Projected EM-RAM only needs 2: "default" (trainable) + "anchor" (zero-init, frozen LongLive base).
    # No "old" EMA adapter (NFT-specific).
    if rank0:
        print("[motion_projected_em_ram] attaching adapters: default + anchor (frozen LongLive base)", flush=True)
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    peft_config_default = generator.model.peft_config["default"]
    generator.model.add_adapter("anchor", peft_config_default)

    # "anchor" is never trained; zero-init B-projection means LoRA delta = 0,
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
        print(f"[motion_projected_em_ram] cast {n_cast} fp32 params -> bf16 (post-LoRA, pre-FSDP)", flush=True)

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
            f"[motion_projected_em_ram] adapter param counts: "
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
            f"[motion_projected_em_ram] trainable params (FSDP-sharded total): "
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
    reward_motion_mode = str(getattr(cfg, "reward_motion_mode", "tracklet_scalar"))
    reward_kwargs = dict(
        ref_path=ref_clip_path,
        scratch_dir=scratch_dir,
        device=device,
        cache_dir=cache_root,
        n_frames=int(getattr(cfg, "reward_n_frames", 16)),
        grid_size=int(getattr(cfg, "reward_grid_size", 30)),
        fps=int(getattr(cfg, "fps", 16)),
        speed_lower=float(getattr(cfg, "reward_speed_lower", 0.5)),
        speed_upper=float(getattr(cfg, "reward_speed_upper", 2.0)),
        speed_penalty_coef=float(getattr(cfg, "reward_speed_penalty_coef", 0.25)),
        motion_mode=reward_motion_mode,
        bucket_count=int(getattr(cfg, "reward_bucket_count", 3)),
        moving_percentile=float(getattr(cfg, "reward_moving_percentile", 60.0)),
        min_moving_tracks=int(getattr(cfg, "reward_min_moving_tracks", 4)),
        moving_speed_floor=float(getattr(cfg, "reward_moving_speed_floor", 1e-5)),
    )
    if rank0:
        print(
            f"[motion_projected_em_ram] init reward (rank 0 first): "
            f"mode={reward_motion_mode} ref={ref_clip_path}",
            flush=True,
        )
        reward_fn = MotionProjectedEMReward(**reward_kwargs)
    dist.barrier()
    if not rank0:
        reward_fn = MotionProjectedEMReward(**reward_kwargs)
    dist.barrier()
    if rank0:
        print("[motion_projected_em_ram] reward init complete on all ranks", flush=True)

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
    em_target_kl = float(getattr(cfg, "em_target_kl", 0.10))
    em_eta_cfg = getattr(cfg, "em_eta", None)
    em_eta = None if em_eta_cfg is None else float(em_eta_cfg)
    em_weight_clip = float(getattr(cfg, "em_weight_clip", 4.0))
    em_alpha_mode = str(getattr(cfg, "em_alpha_mode", "positive_excess"))
    em_alpha_max = float(getattr(cfg, "em_alpha_max", 1.0))
    em_std_floor = float(getattr(cfg, "em_std_floor", 1e-4))
    lambda_motion = float(getattr(cfg, "lambda_motion", 1.0))
    lambda_static = float(getattr(cfg, "lambda_static", 0.05))
    motion_pool = int(getattr(cfg, "motion_pool", 2))
    motion_temporal_center = bool(getattr(cfg, "motion_temporal_center", True))
    shift_coef = float(getattr(cfg, "shift_coef", 0.25))
    loss_weight_mode = str(getattr(cfg, "loss_weight_mode", "em_weight"))
    loss_weight_clip_min = float(getattr(cfg, "loss_weight_clip_min", 0.0))
    loss_weight_clip_max = float(getattr(cfg, "loss_weight_clip_max", 2.0))
    anchor_beta = float(getattr(cfg, "anchor_beta", 0.1))
    local_anchor_beta = float(getattr(cfg, "local_anchor_beta", anchor_beta))
    cover_step_ratio = float(getattr(cfg, "cover_step_ratio", 0.25))
    cover_rollouts_per_outer = int(getattr(cfg, "cover_rollouts_per_outer", 1))
    cover_loss_weight = float(getattr(cfg, "cover_loss_weight", 1.0))
    time_weight_mode = str(getattr(cfg, "time_weight_mode", "none"))
    time_weight_temperature = float(getattr(cfg, "time_weight_temperature", 0.25))
    time_weight_min = float(getattr(cfg, "time_weight_min", 0.0))
    time_weight_max = float(getattr(cfg, "time_weight_max", 1.0))
    time_weight_normalize_mean = bool(getattr(cfg, "time_weight_normalize_mean", False))
    rollout_adapter = str(getattr(cfg, "rollout_adapter", "default"))
    reward_mode = str(getattr(cfg, "reward_mode", "absolute"))
    reward_relative_margin = float(getattr(cfg, "reward_relative_margin", 0.0))
    reward_relative_gate = bool(getattr(cfg, "reward_relative_gate", True))
    feature_selector_mode = str(getattr(cfg, "feature_selector_mode", "none"))
    feature_selector_direction_min = float(
        getattr(cfg, "feature_selector_direction_min", 0.0)
    )
    feature_selector_speed_penalty_max_cfg = getattr(
        cfg, "feature_selector_speed_penalty_max", None
    )
    feature_selector_speed_penalty_max = (
        None
        if feature_selector_speed_penalty_max_cfg is None
        else float(feature_selector_speed_penalty_max_cfg)
    )
    feature_selector_speed_ratio_min_cfg = getattr(
        cfg, "feature_selector_speed_ratio_min", None
    )
    feature_selector_speed_ratio_min = (
        None
        if feature_selector_speed_ratio_min_cfg is None
        else float(feature_selector_speed_ratio_min_cfg)
    )
    feature_selector_speed_ratio_max_cfg = getattr(
        cfg, "feature_selector_speed_ratio_max", None
    )
    feature_selector_speed_ratio_max = (
        None
        if feature_selector_speed_ratio_max_cfg is None
        else float(feature_selector_speed_ratio_max_cfg)
    )
    feature_selector_fallback_topk = int(getattr(cfg, "feature_selector_fallback_topk", 0))
    feature_selector_fallback_speed_penalty_coef = float(
        getattr(
            cfg,
            "feature_selector_fallback_speed_penalty_coef",
            getattr(cfg, "reward_speed_penalty_coef", 0.25),
        )
    )
    feature_weight_direction_center = float(
        getattr(cfg, "feature_weight_direction_center", 0.0)
    )
    feature_weight_direction_temperature = float(
        getattr(cfg, "feature_weight_direction_temperature", 0.25)
    )
    feature_weight_speed_penalty_coef = float(
        getattr(cfg, "feature_weight_speed_penalty_coef", getattr(cfg, "reward_speed_penalty_coef", 0.25))
    )
    feature_weight_min = float(getattr(cfg, "feature_weight_min", 0.25))
    feature_weight_max = float(getattr(cfg, "feature_weight_max", 1.5))
    feature_weight_normalize_mean = bool(getattr(cfg, "feature_weight_normalize_mean", True))
    feature_score_center = float(getattr(cfg, "feature_score_center", 0.0))
    feature_score_temperature = float(getattr(cfg, "feature_score_temperature", 0.25))
    feature_score_min = float(getattr(cfg, "feature_score_min", 0.0))
    feature_score_max = float(getattr(cfg, "feature_score_max", 1.0))
    feature_score_normalize_mean = bool(getattr(cfg, "feature_score_normalize_mean", False))
    assert rollout_adapter in ("default", "anchor"), (
        f"rollout_adapter must be 'default' or 'anchor', got {rollout_adapter!r}"
    )
    assert reward_mode in ("absolute", "baseline_relative"), (
        f"reward_mode must be 'absolute' or 'baseline_relative', got {reward_mode!r}"
    )
    assert feature_selector_mode in ("none", "component_gate", "component_weight", "score_weight"), (
        f"feature_selector_mode must be 'none', 'component_gate', "
        f"'component_weight', or 'score_weight', got {feature_selector_mode!r}"
    )
    assert mstep_objective in ("alpha_shift", "reward_weighted_velocity", "two_stream_on_policy_cover"), (
        f"mstep_objective must be 'alpha_shift', 'reward_weighted_velocity', "
        f"or 'two_stream_on_policy_cover', "
        f"got {mstep_objective!r}"
    )
    assert loss_weight_mode == "em_weight", (
        f"loss_weight_mode currently supports only 'em_weight', got {loss_weight_mode!r}"
    )
    assert loss_weight_clip_min <= loss_weight_clip_max, (
        f"loss_weight_clip_min must be <= loss_weight_clip_max, got "
        f"{loss_weight_clip_min} > {loss_weight_clip_max}"
    )
    if mstep_objective == "reward_weighted_velocity":
        assert beta_kl == 0.0, (
            "reward_weighted_velocity uses anchor_beta as its explicit "
            "LongLive anchor; set beta_kl=0.0 to avoid mixing objectives"
        )
    if mstep_objective == "two_stream_on_policy_cover":
        assert beta_kl == 0.0, (
            "two_stream_on_policy_cover has explicit local_anchor_beta and "
            "cover_loss_weight; set beta_kl=0.0 to avoid mixing objectives"
        )
        assert reward_motion_mode == "residual_bucket", (
            "two_stream_on_policy_cover currently requires "
            "reward_motion_mode='residual_bucket' for time-local weights"
        )
        assert time_weight_mode == "residual_bucket_direction", (
            "two_stream_on_policy_cover currently supports only "
            "time_weight_mode='residual_bucket_direction'"
        )
        assert 0.0 <= cover_step_ratio <= 1.0, (
            f"cover_step_ratio must be in [0, 1], got {cover_step_ratio}"
        )
        assert cover_rollouts_per_outer > 0, (
            f"cover_rollouts_per_outer must be positive, got {cover_rollouts_per_outer}"
        )
    if reward_mode == "baseline_relative":
        assert rollout_adapter == "default", (
            "baseline_relative reward compares the trainable default adapter against "
            "the frozen anchor adapter, so rollout_adapter must be 'default'"
        )

    if rank0:
        print(
            f"[motion_projected_em_ram] start: outer={outer_epochs} x inner={inner_steps} "
            f"x K={K} (g_endpoints={g_endpoints}, k_noisings={k_noisings}) | "
            f"reward_coef={reward_coef} | beta_kl={beta_kl} | "
            f"mstep_objective={mstep_objective} | shift_coef={shift_coef} | "
            f"loss_weight_mode={loss_weight_mode} | "
            f"loss_weight_clip={loss_weight_clip_min}/{loss_weight_clip_max} | "
            f"anchor_beta={anchor_beta} | "
            f"local_anchor_beta={local_anchor_beta} | "
            f"cover_step_ratio={cover_step_ratio} | "
            f"cover_rollouts={cover_rollouts_per_outer} | "
            f"cover_loss_weight={cover_loss_weight} | "
            f"time_weight_mode={time_weight_mode} | "
            f"time_weight_temp={time_weight_temperature} | "
            f"time_weight_minmax={time_weight_min}/{time_weight_max} | "
            f"em_target_kl={em_target_kl} | em_eta={em_eta} | "
            f"em_alpha_mode={em_alpha_mode} | em_alpha_max={em_alpha_max} | "
            f"em_std_floor={em_std_floor} | lambda_motion={lambda_motion} | "
            f"lambda_static={lambda_static} | motion_pool={motion_pool} | "
            f"lambda_reference_orthogonal={lambda_reference_orthogonal} | "
            f"motion_temporal_center={motion_temporal_center} | "
            f"subspace_mode={subspace_mode} | reference_scope={reference_motion_scope} | "
            f"reference_positive={reference_motion_positive} | "
            f"reference_mix={reference_motion_mix} | "
            f"reference_temporal_center={reference_motion_temporal_center} | "
            f"rollout_adapter={rollout_adapter} | reward_mode={reward_mode} | "
            f"reward_motion_mode={reward_motion_mode} | "
            f"relative_margin={reward_relative_margin} | "
            f"relative_gate={reward_relative_gate} | "
            f"feature_selector={feature_selector_mode} | "
            f"selector_dir_min={feature_selector_direction_min} | "
            f"selector_speed_penalty_max={feature_selector_speed_penalty_max} | "
            f"selector_speed_ratio_min={feature_selector_speed_ratio_min} | "
            f"selector_speed_ratio_max={feature_selector_speed_ratio_max} | "
            f"selector_fallback_topk={feature_selector_fallback_topk} | "
            f"selector_fallback_speed_penalty_coef={feature_selector_fallback_speed_penalty_coef} | "
            f"feature_weight_center={feature_weight_direction_center} | "
            f"feature_weight_temp={feature_weight_direction_temperature} | "
            f"feature_weight_speed_coef={feature_weight_speed_penalty_coef} | "
            f"feature_weight_minmax={feature_weight_min}/{feature_weight_max} | "
            f"feature_weight_normalize={feature_weight_normalize_mean} | "
            f"feature_score_center={feature_score_center} | "
            f"feature_score_temp={feature_score_temperature} | "
            f"feature_score_minmax={feature_score_min}/{feature_score_max} | "
            f"feature_score_normalize={feature_score_normalize_mean} | "
            f"anchors={anchors}",
            flush=True,
        )

    global_step = 0
    t_train_loop_start = time.time()
    setup_time_s = t_train_loop_start - t_setup_start
    if rank0:
        print(f"[motion_projected_em_ram] setup_time_s={setup_time_s:.1f}", flush=True)

    for outer in range(outer_epochs):
        t_outer = time.time()

        # ---- ROLLOUT phase ----
        # On-policy per RAM Alg 1: sample x_0 from the current trainable model.
        generator.model.set_adapter(rollout_adapter)
        rollout_seed = int(cfg.seed) + 1009 * outer + 31 * rank
        with torch.no_grad():
            if reward_mode == "baseline_relative":
                structured_rollouts = rollout_engine.rollout_k_structured(
                    k=K, dtype=torch.bfloat16, base_seed=rollout_seed,
                )
                rollouts = [
                    (out.video, out.latent_x0, out.noise.detach().clone())
                    for out in structured_rollouts
                ]
            else:
                rollouts = [
                    (video, latent, None)
                    for video, latent in rollout_engine.rollout_k(
                        k=K, dtype=torch.bfloat16, base_seed=rollout_seed,
                    )
                ]
        t_rollout = time.time() - t_outer

        # ---- REWARD phase ----
        t_reward_start = time.time()
        rewards = []
        reward_components = []
        relative_components = []
        for k_idx, (video, _latent, _noise) in enumerate(rollouts):
            score = reward_fn.score(video[0], tag=f"e{outer}_k{k_idx}_student")
            student_components = dict(getattr(reward_fn, "last_components", {}))
            if reward_mode == "baseline_relative":
                noise = rollouts[k_idx][2]
                assert noise is not None, "baseline_relative reward requires rollout noise"
                generator.model.set_adapter("anchor")
                with torch.no_grad():
                    anchor_out = rollout_engine.rollout(noise.detach().clone())
                    rollout_engine.pipeline.kv_cache1 = None
                    rollout_engine.pipeline.crossattn_cache = None
                generator.model.set_adapter(rollout_adapter)
                baseline_score = reward_fn.score(
                    anchor_out.video[0], tag=f"e{outer}_k{k_idx}_baseline"
                )
                baseline_components = dict(getattr(reward_fn, "last_components", {}))
                rel_score = float(score) - float(baseline_score) - reward_relative_margin
                rewards.append(rel_score)
                reward_components.append(student_components)
                relative_components.append({
                    "student_score": float(score),
                    "baseline_score": float(baseline_score),
                    "relative_score": float(rel_score),
                    "accepted": float(rel_score > 0.0),
                    "student_direction": float(student_components.get("direction", 0.0)),
                    "baseline_direction": float(baseline_components.get("direction", 0.0)),
                    "student_speed_penalty": float(student_components.get("speed_penalty", 0.0)),
                    "baseline_speed_penalty": float(baseline_components.get("speed_penalty", 0.0)),
                    "student_speed_ratio": float(student_components.get("speed_ratio", 0.0)),
                    "baseline_speed_ratio": float(baseline_components.get("speed_ratio", 0.0)),
                })
                del anchor_out
            else:
                rewards.append(score)
                reward_components.append(student_components)
        t_reward = time.time() - t_reward_start
        reward_component_diag = {}
        if reward_components:
            component_keys = sorted({key for c in reward_components for key in c})
            for key in component_keys:
                vals = [
                    float(c[key])
                    for c in reward_components
                    if key in c and isinstance(c[key], (int, float))
                ]
                if vals:
                    reward_component_diag[f"reward/{key}"] = sum(vals) / len(vals)
        if relative_components:
            for key in (
                "student_score",
                "baseline_score",
                "relative_score",
                "accepted",
                "student_direction",
                "baseline_direction",
                "student_speed_penalty",
                "baseline_speed_penalty",
                "student_speed_ratio",
                "baseline_speed_ratio",
            ):
                vals = [float(c[key]) for c in relative_components if key in c]
                if vals:
                    reward_component_diag[f"reward/{key}"] = sum(vals) / len(vals)
            if rank0:
                accepted = sum(1 for c in relative_components if c["accepted"] > 0.0)
                print(
                    f"[motion_projected_em_ram] relative reward outer {outer:3d}: "
                    f"student={reward_component_diag.get('reward/student_score', 0.0):.3f} "
                    f"baseline={reward_component_diag.get('reward/baseline_score', 0.0):.3f} "
                    f"rel={reward_component_diag.get('reward/relative_score', 0.0):.3f} "
                    f"accepted={accepted}/{len(relative_components)} "
                    f"margin={reward_relative_margin:.3f}",
                    flush=True,
                )

        # ---- E-step: cross-rank empirical reward tilt ----
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        gathered = [torch.zeros_like(rewards_t) for _ in range(world_size)]
        dist.all_gather(gathered, rewards_t)
        all_rewards = torch.cat(gathered)
        alpha_all, loss_weight_all, em_diag = em_tilt_alpha_and_weights(
            all_rewards,
            target_kl=em_target_kl,
            eta=em_eta,
            adv_clip_max=adv_clip_max,
            weight_clip=em_weight_clip,
            alpha_mode=em_alpha_mode,
            alpha_max=em_alpha_max,
            std_floor=em_std_floor,
            device=device,
        )
        alpha_tensor = alpha_all[rank * K : (rank + 1) * K].to(device)
        loss_weight_tensor = loss_weight_all[rank * K : (rank + 1) * K].to(device)
        if reward_mode == "baseline_relative" and reward_relative_gate:
            accepted_t = torch.tensor(
                [1.0 if float(r) > 0.0 else 0.0 for r in rewards],
                device=device,
                dtype=alpha_tensor.dtype,
            )
            alpha_tensor = alpha_tensor * accepted_t
            loss_weight_tensor = loss_weight_tensor * accepted_t
        feature_selector_diag = {}
        if feature_selector_mode == "component_gate":
            feature_gates, feature_selector_diag = feature_consistency_gates(
                reward_components,
                direction_min=feature_selector_direction_min,
                speed_penalty_max=feature_selector_speed_penalty_max,
                speed_ratio_min=feature_selector_speed_ratio_min,
                speed_ratio_max=feature_selector_speed_ratio_max,
                fallback_topk=feature_selector_fallback_topk,
                fallback_speed_penalty_coef=feature_selector_fallback_speed_penalty_coef,
                device=device,
            )
            alpha_tensor = alpha_tensor * feature_gates.to(dtype=alpha_tensor.dtype)
            loss_weight_tensor = loss_weight_tensor * feature_gates.to(dtype=loss_weight_tensor.dtype)
        elif feature_selector_mode == "component_weight":
            feature_weights, feature_selector_diag = feature_consistency_weights(
                reward_components,
                direction_center=feature_weight_direction_center,
                direction_temperature=feature_weight_direction_temperature,
                speed_penalty_coef=feature_weight_speed_penalty_coef,
                min_weight=feature_weight_min,
                max_weight=feature_weight_max,
                normalize_mean=feature_weight_normalize_mean,
                device=device,
            )
            alpha_tensor = alpha_tensor * feature_weights.to(dtype=alpha_tensor.dtype)
            loss_weight_tensor = loss_weight_tensor * feature_weights.to(dtype=loss_weight_tensor.dtype)
        elif feature_selector_mode == "score_weight":
            feature_weights, feature_selector_diag = score_consistency_weights(
                rewards,
                score_center=feature_score_center,
                score_temperature=feature_score_temperature,
                min_weight=feature_score_min,
                max_weight=feature_score_max,
                normalize_mean=feature_score_normalize_mean,
                device=device,
            )
            alpha_tensor = alpha_tensor * feature_weights.to(dtype=alpha_tensor.dtype)
            loss_weight_tensor = loss_weight_tensor * feature_weights.to(dtype=loss_weight_tensor.dtype)
        loss_weight_tensor = loss_weight_tensor.clamp(
            float(loss_weight_clip_min),
            float(loss_weight_clip_max),
        )
        time_weight_tensor = None
        time_weight_diag = {}
        if mstep_objective == "two_stream_on_policy_cover":
            time_weight_tensor, time_weight_diag = residual_bucket_time_weights(
                reward_components,
                latent_frames=latent_f,
                bucket_count=int(getattr(cfg, "reward_bucket_count", 3)),
                direction_temperature=time_weight_temperature,
                min_weight=time_weight_min,
                max_weight=time_weight_max,
                normalize_mean=time_weight_normalize_mean,
                device=device,
            )
        if mstep_objective == "reward_weighted_velocity":
            em_diag["em/local_loss_weight_min"] = float(loss_weight_tensor.min())
            em_diag["em/local_loss_weight_max"] = float(loss_weight_tensor.max())
            em_diag["em/local_loss_weight_mean"] = float(loss_weight_tensor.mean())
        elif mstep_objective == "two_stream_on_policy_cover":
            em_diag["em/local_loss_weight_min"] = float(loss_weight_tensor.min())
            em_diag["em/local_loss_weight_max"] = float(loss_weight_tensor.max())
            em_diag["em/local_loss_weight_mean"] = float(loss_weight_tensor.mean())
            for key, value in time_weight_diag.items():
                em_diag[key] = value

        # ---- COVER phase ----
        # Teacher/anchor rollouts provide an explicit mode-covering stream.
        cover_rollouts = []
        t_cover = 0.0
        if mstep_objective == "two_stream_on_policy_cover":
            t_cover_start = time.time()
            generator.model.set_adapter("anchor")
            cover_gen = torch.Generator(device=device)
            cover_base_seed = int(cfg.seed) + 7919 * outer + 37 * rank
            for cover_idx in range(cover_rollouts_per_outer):
                prompt_idx = (outer * cover_rollouts_per_outer + cover_idx) % len(cover_cond_list)
                cover_cond = cover_cond_list[prompt_idx]
                rollout_engine.set_cached_cond_dict(cover_cond)
                cover_gen.manual_seed(cover_base_seed + cover_idx)
                cover_noise = torch.randn(
                    latent_shape,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=cover_gen,
                )
                cover_out = rollout_engine.rollout(cover_noise)
                rollout_engine.pipeline.kv_cache1 = None
                rollout_engine.pipeline.crossattn_cache = None
                cover_rollouts.append({
                    "latent": cover_out.latent_x0.detach(),
                    "cond": cover_cond,
                    "prompt_idx": prompt_idx,
                })
                del cover_out
            rollout_engine.set_cached_cond_dict(train_cond)
            generator.model.set_adapter("default")
            t_cover = time.time() - t_cover_start

        # ---- TRAINING phase ----
        generator.model.set_adapter("default")
        t_train_start = time.time()

        # Per-outer accumulators
        sum_loss = 0.0
        sum_kl = 0.0
        sum_motion_loss = 0.0
        sum_static_loss = 0.0
        sum_raw_shift_norm = 0.0
        sum_motion_shift_norm = 0.0
        sum_target_norm = 0.0
        sum_v_default_norm = 0.0
        sum_v_anchor_norm = 0.0
        sum_alpha = 0.0
        sum_alpha_eff = 0.0
        optional_diag_keys = (
            "mpem/coarse_shift_norm",
            "mpem/reference_basis_norm",
            "mpem/reference_aligned_shift_norm",
            "mpem/reference_projection_residual_norm",
            "mpem/reference_alignment_cos",
            "mpem/reference_coef_mean",
            "mpem/reference_coef_abs_mean",
            "mpem/reference_mix",
            "mpem/reference_orthogonal_loss",
            "mpem/projected_shift_norm",
            "mpem/unweighted_motion_loss",
            "mpem/anchor_loss",
            "mpem/loss_weight",
            "mpem/loss_weight_min",
            "mpem/loss_weight_max",
            "mpem/shift_coef",
            "mpem/anchor_beta",
            "mpem/time_weight_mean",
            "mpem/time_weight_min",
            "mpem/time_weight_max",
            "mpem/active_frame_rate",
            "mpem/cover_loss",
            "mpem/cover_loss_weight",
            "mpem/cover_stream",
        )
        sum_optional_diag = {key: 0.0 for key in optional_diag_keys}
        count_optional_diag = {key: 0 for key in optional_diag_keys}
        # Per-anchor / per-rollout buckets for diagnostic disaggregation
        per_anchor_shift = [0.0] * len(anchors)
        per_anchor_count = [0] * len(anchors)
        per_rollout_shift = [0.0] * K
        per_rollout_count = [0] * K
        cover_steps = (
            int(round(inner_steps * cover_step_ratio))
            if mstep_objective == "two_stream_on_policy_cover"
            else 0
        )
        cover_schedule = _cover_step_schedule(inner_steps, cover_steps)
        ref_inner = 0
        cover_inner = 0

        for inner in range(inner_steps):
            use_cover = (
                mstep_objective == "two_stream_on_policy_cover"
                and cover_schedule[inner]
            )
            if use_cover:
                cover_item = cover_rollouts[cover_inner % len(cover_rollouts)]
                k_idx = -1
                t_idx = cover_inner % len(anchors)
                x0_ref = cover_item["latent"].to(torch.bfloat16)
                active_cond = cover_item["cond"]
                cover_inner += 1
            else:
                # Existing MP-EM-RAM keeps noisings grouped under each endpoint.
                # The two-stream objective distributes fewer reference updates
                # across all sampled endpoints instead of dropping the tail K.
                ref_step = ref_inner if mstep_objective == "two_stream_on_policy_cover" else inner
                if mstep_objective == "two_stream_on_policy_cover":
                    k_idx = ref_step % K
                    t_idx = ref_step % len(anchors)
                else:
                    k_idx = (ref_step // k_noisings) % g_endpoints
                    t_idx = ref_step % len(anchors)
                _video_k, latent_k, _noise_k = rollouts[k_idx]
                x0_ref = latent_k.to(torch.bfloat16)
                active_cond = train_cond
                ref_inner += 1
            anchor_t = int(anchors[t_idx])

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
            flow_pred_default, _pred_x0_default = generator(noisy, active_cond, timestep)

            # --- anchor forward (no_grad) ---
            generator.model.set_adapter("anchor")
            with torch.no_grad():
                flow_pred_anchor, _pred_x0_anchor = generator(noisy, active_cond, timestep)

            # Restore default BEFORE backward so gradient-checkpoint recompute
            # uses the same active adapter as the original forward.
            generator.model.set_adapter("default")

            # --- loss ---
            if use_cover:
                loss_motion_projected_em_ram, diag = mode_cover_velocity_loss(
                    v_default=flow_pred_default,
                    v_anchor=flow_pred_anchor,
                    cover_loss_weight=cover_loss_weight,
                    lambda_static=lambda_static,
                    anchor_idx=t_idx,
                )
                kl = torch.zeros((), device=device, dtype=loss_motion_projected_em_ram.dtype)
                loss = loss_motion_projected_em_ram
            elif mstep_objective == "alpha_shift":
                loss_motion_projected_em_ram, diag = motion_projected_em_ram_loss(
                    v_default=flow_pred_default,
                    v_anchor=flow_pred_anchor,
                    noise=noise,
                    x0_ref=x0_ref,
                    alpha=alpha_tensor[k_idx:k_idx + 1],
                    x0_reference=reference_latent,
                    reward_coef=reward_coef,
                    lambda_motion=lambda_motion,
                    lambda_static=lambda_static,
                    subspace_mode=subspace_mode,
                    motion_pool=motion_pool,
                    motion_temporal_center=motion_temporal_center,
                    reference_motion_scope=reference_motion_scope,
                    reference_motion_positive=reference_motion_positive,
                    reference_motion_mix=reference_motion_mix,
                    reference_motion_temporal_center=reference_motion_temporal_center,
                    lambda_reference_orthogonal=lambda_reference_orthogonal,
                    anchor_idx=t_idx,
                )
                if beta_kl > 0.0:
                    kl = kl_anchor_loss(flow_pred_default, flow_pred_anchor)
                    loss = loss_motion_projected_em_ram + beta_kl * kl
                else:
                    kl = torch.zeros((), device=device, dtype=loss_motion_projected_em_ram.dtype)
                    loss = loss_motion_projected_em_ram
            elif mstep_objective == "reward_weighted_velocity":
                loss_motion_projected_em_ram, diag = reward_weighted_velocity_loss(
                    v_default=flow_pred_default,
                    v_anchor=flow_pred_anchor,
                    noise=noise,
                    x0_ref=x0_ref,
                    loss_weight=loss_weight_tensor[k_idx:k_idx + 1],
                    x0_reference=reference_latent,
                    shift_coef=shift_coef,
                    anchor_beta=anchor_beta,
                    lambda_motion=lambda_motion,
                    lambda_static=lambda_static,
                    subspace_mode=subspace_mode,
                    motion_pool=motion_pool,
                    motion_temporal_center=motion_temporal_center,
                    reference_motion_scope=reference_motion_scope,
                    reference_motion_positive=reference_motion_positive,
                    reference_motion_mix=reference_motion_mix,
                    reference_motion_temporal_center=reference_motion_temporal_center,
                    lambda_reference_orthogonal=lambda_reference_orthogonal,
                    anchor_idx=t_idx,
                )
                kl = torch.zeros((), device=device, dtype=loss_motion_projected_em_ram.dtype)
                loss = loss_motion_projected_em_ram
            elif mstep_objective == "two_stream_on_policy_cover":
                assert time_weight_tensor is not None
                loss_motion_projected_em_ram, diag = time_local_reward_weighted_velocity_loss(
                    v_default=flow_pred_default,
                    v_anchor=flow_pred_anchor,
                    noise=noise,
                    x0_ref=x0_ref,
                    loss_weight=loss_weight_tensor[k_idx:k_idx + 1],
                    time_weight=time_weight_tensor[k_idx:k_idx + 1],
                    x0_reference=reference_latent,
                    shift_coef=shift_coef,
                    local_anchor_beta=local_anchor_beta,
                    lambda_motion=lambda_motion,
                    lambda_static=lambda_static,
                    subspace_mode=subspace_mode,
                    motion_pool=motion_pool,
                    motion_temporal_center=motion_temporal_center,
                    reference_motion_scope=reference_motion_scope,
                    reference_motion_positive=reference_motion_positive,
                    reference_motion_mix=reference_motion_mix,
                    reference_motion_temporal_center=reference_motion_temporal_center,
                    lambda_reference_orthogonal=lambda_reference_orthogonal,
                    anchor_idx=t_idx,
                )
                kl = torch.zeros((), device=device, dtype=loss_motion_projected_em_ram.dtype)
                loss = loss_motion_projected_em_ram
            else:
                raise AssertionError(f"unhandled mstep_objective={mstep_objective!r}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.detach())
            sum_kl += float(kl.detach())
            sum_motion_loss += float(diag["mpem/motion_loss"])
            sum_static_loss += float(diag["mpem/static_loss"])
            sum_raw_shift_norm += float(diag["mpem/raw_shift_norm"])
            sum_motion_shift_norm += float(diag["mpem/motion_shift_norm"])
            sum_target_norm += float(diag["mpem/target_norm"])
            sum_v_default_norm += float(diag["mpem/v_default_norm"])
            sum_v_anchor_norm += float(diag["mpem/v_anchor_norm"])
            sum_alpha += float(diag["mpem/alpha"])
            sum_alpha_eff += float(diag["mpem/alpha_eff"])
            per_anchor_shift[t_idx] += float(diag["mpem/motion_shift_norm"])
            per_anchor_count[t_idx] += 1
            if k_idx >= 0:
                per_rollout_shift[k_idx] += float(diag["mpem/motion_shift_norm"])
                per_rollout_count[k_idx] += 1
            for key in optional_diag_keys:
                if key in diag:
                    sum_optional_diag[key] += float(diag[key])
                    count_optional_diag[key] += 1
            global_step += 1

        t_train = time.time() - t_train_start

        # ---- LOG ----
        dt_outer = time.time() - t_outer
        n = max(1, inner_steps)
        avg_loss = sum_loss / n
        avg_kl = sum_kl / n
        avg_motion_loss = sum_motion_loss / n
        avg_static_loss = sum_static_loss / n
        avg_raw_shift = sum_raw_shift_norm / n
        avg_motion_shift = sum_motion_shift_norm / n
        avg_target = sum_target_norm / n
        avg_v_default = sum_v_default_norm / n
        avg_v_anchor = sum_v_anchor_norm / n
        avg_alpha = sum_alpha / n
        avg_alpha_eff = sum_alpha_eff / n
        avg_optional_diag = {
            key: sum_optional_diag[key] / max(1, count_optional_diag[key])
            for key in optional_diag_keys
            if count_optional_diag[key]
        }
        if rank0:
            ref_diag = ""
            residual_reward_diag = ""
            update_label = (
                "weight"
                if mstep_objective in ("reward_weighted_velocity", "two_stream_on_policy_cover")
                else "alpha"
            )
            if "mpem/reference_alignment_cos" in avg_optional_diag:
                ref_diag = (
                    f"ref_cos={avg_optional_diag['mpem/reference_alignment_cos']:.3f} "
                    f"ref_coef={avg_optional_diag['mpem/reference_coef_mean']:.3f}/"
                    f"{avg_optional_diag['mpem/reference_coef_abs_mean']:.3f} "
                )
            if reward_motion_mode == "residual_bucket":
                bucket_diag = " ".join(
                    f"b{idx}={reward_component_diag.get(f'reward/bucket{idx}_direction', 0.0):.3f}"
                    for idx in range(int(getattr(cfg, "reward_bucket_count", 3)))
                )
                residual_reward_diag = (
                    f"resid={reward_component_diag.get('reward/residual_score', 0.0):.3f} "
                    f"global={reward_component_diag.get('reward/global_direction', 0.0):.3f} "
                    f"{bucket_diag}  "
                )
            print(
                f"[motion_projected_em_ram] outer {outer:3d}/{outer_epochs}  "
                f"rewards mean={em_diag['reward/raw_mean']:.3f} "
                f"std={em_diag['reward/raw_std']:.3f}  "
                f"dir={reward_component_diag.get('reward/direction', 0.0):.3f} "
                f"speed={reward_component_diag.get('reward/speed_ratio', 0.0):.3f} "
                f"sp_pen={reward_component_diag.get('reward/speed_penalty', 0.0):.3f} "
                + residual_reward_diag
                + f"loss={avg_loss:.4f}  motion={avg_motion_loss:.4f} "
                f"static={avg_static_loss:.4f}  "
                f"em_kl={em_diag['em/kl']:.3f}  "
                f"ess={em_diag['em/ess']:.1f}  "
                f"{update_label}={avg_alpha:.3f}/{avg_alpha_eff:.3f}  "
                + (
                    f"sel={feature_selector_diag.get('feature_selector/accepted', 0.0):.0f}/{K} "
                    f"sel_dir={feature_selector_diag.get('feature_selector/direction_mean', 0.0):.3f} "
                    f"sel_sp={feature_selector_diag.get('feature_selector/speed_penalty_mean', 0.0):.3f}  "
                    if feature_selector_mode == "component_gate"
                    else ""
                )
                + (
                    f"w={feature_selector_diag.get('feature_selector/weight_mean', 0.0):.3f} "
                    f"w_rng={feature_selector_diag.get('feature_selector/weight_min', 0.0):.3f}/"
                    f"{feature_selector_diag.get('feature_selector/weight_max', 0.0):.3f} "
                    f"w_dir={feature_selector_diag.get('feature_selector/direction_mean', 0.0):.3f} "
                    f"w_sp={feature_selector_diag.get('feature_selector/speed_penalty_mean', 0.0):.3f}  "
                    if feature_selector_mode == "component_weight"
                    else ""
                )
                + (
                    f"w={feature_selector_diag.get('feature_selector/weight_mean', 0.0):.3f} "
                    f"w_rng={feature_selector_diag.get('feature_selector/weight_min', 0.0):.3f}/"
                    f"{feature_selector_diag.get('feature_selector/weight_max', 0.0):.3f} "
                    f"w_score={feature_selector_diag.get('feature_selector/score_mean', 0.0):.3f}  "
                    if feature_selector_mode == "score_weight"
                    else ""
                )
                + f"shift={avg_raw_shift:.3f}->{avg_motion_shift:.3f}  "
                + ref_diag
                + (
                    f"ref_resid={avg_optional_diag['mpem/reference_projection_residual_norm']:.3f}  "
                    if "mpem/reference_projection_residual_norm" in avg_optional_diag
                    else ""
                )
                + (
                    f"ref_orth={avg_optional_diag['mpem/reference_orthogonal_loss']:.4f}  "
                    if "mpem/reference_orthogonal_loss" in avg_optional_diag
                    and lambda_reference_orthogonal > 0.0
                    else ""
                )
                + (
                    f"anchor={avg_optional_diag['mpem/anchor_loss']:.4f}  "
                    if "mpem/anchor_loss" in avg_optional_diag
                    else ""
                )
                + (
                    f"tw={avg_optional_diag['mpem/time_weight_mean']:.3f} "
                    f"cover={avg_optional_diag.get('mpem/cover_stream', 0.0):.2f} "
                    f"cover_loss={avg_optional_diag.get('mpem/cover_loss', 0.0):.4f}  "
                    if mstep_objective == "two_stream_on_policy_cover"
                    else ""
                )
                + f"v_def={avg_v_default:.3f}  v_anc={avg_v_anchor:.3f}  "
                + (f"kl={avg_kl:.4f}  " if beta_kl > 0.0 else "")
                + f"dt={dt_outer:.1f}s (rollout={t_rollout:.1f}, "
                f"reward={t_reward:.1f}, cover={t_cover:.1f}, train={t_train:.1f})",
                flush=True,
            )
        if wandb_enabled:
            log_dict = {
                "outer/loss": avg_loss,
                "outer/motion_loss": avg_motion_loss,
                "outer/static_loss": avg_static_loss,
                "outer/raw_shift_norm": avg_raw_shift,
                "outer/motion_shift_norm": avg_motion_shift,
                "outer/target_norm": avg_target,
                "outer/v_default_norm": avg_v_default,
                "outer/v_anchor_norm": avg_v_anchor,
                "outer/alpha_mean": avg_alpha,
                "outer/alpha_eff_mean": avg_alpha_eff,
                "outer/dt_total_s": dt_outer,
                "outer/dt_rollout_s": t_rollout,
                "outer/dt_reward_s": t_reward,
                "outer/dt_cover_s": t_cover,
                "outer/dt_train_s": t_train,
                **{f"em/{k.split('/')[-1]}": v for k, v in em_diag.items() if k.startswith("em/")},
                **{f"reward/{k.split('/')[-1]}": v for k, v in em_diag.items() if k.startswith("reward/")},
                **{f"reward/{k.split('/')[-1]}": v for k, v in reward_component_diag.items()},
            }
            log_dict.update(feature_selector_diag)
            if beta_kl > 0.0:
                log_dict["outer/kl"] = avg_kl
            log_dict.update(avg_optional_diag)
            # Per-anchor / per-rollout shift norms
            for t_i, t_v in enumerate(anchors):
                if per_anchor_count[t_i]:
                    log_dict[f"mpem/motion_shift_norm_t{t_i}"] = (
                        per_anchor_shift[t_i] / per_anchor_count[t_i]
                    )
            for k_i in range(K):
                if per_rollout_count[k_i]:
                    log_dict[f"mpem/motion_shift_norm_k{k_i}"] = (
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
                print(f"[motion_projected_em_ram] saved ckpt: {ckpt_path}", flush=True)

        maybe_barrier()

    train_loop_time_s = time.time() - t_train_loop_start
    final_path = _save_lora_ckpt(generator.model, out_dir, "final", rank0)
    if rank0:
        print(
            f"[motion_projected_em_ram] DONE. setup_time_s={setup_time_s:.1f} "
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
