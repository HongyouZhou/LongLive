"""MotionDirector LoRA finetune on the few-step LongLive model.

Frozen starting point = NVlabs LongLive complete inference product
(longlive_base.pt + lora.pt merged), matching configs/longlive_inference.yaml.
On top of that we attach a new LoRA trained with MotionDirector loss
(L_temporal_MSE + L_AD, alpha=sqrt(2), beta=1) to push motion back toward
mode-covering — see docs/00.md §1 candidate idea 1.

Distributed via FSDP + torchrun (8 GPU default, single-rank also supported).
Usage:

    torchrun --nproc_per_node=8 -m longlive.methods.motiondirector.train \\
        --config longlive/methods/motiondirector/configs/skateboarding_fewstep.yaml

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
import wandb
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from longlive.methods.motiondirector.data import GeneralPromptDataset, SkateboardingLatentDataset
from longlive.methods.motiondirector.losses import (
    appearance_debias_loss,
    prior_consistency_loss,
    trajectory_cosine_loss,
)
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


def _clean_fsdp_key(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


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
    path = out_dir / f"lora_{tag}.pt"
    torch.save(lora_state, path)
    return path


def _prune_old_ckpts(out_dir: Path, keep_last: int) -> None:
    """Keep only the keep_last most recent ckpts (excluding 'final')."""
    ckpts = sorted(
        (p for p in out_dir.glob("lora_*.pt") if "final" not in p.name),
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
    ap.add_argument(
        "--disable-wandb", action="store_true",
        help="Skip wandb.init — useful for local debug.",
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

    # ---------- wandb (rank 0 only) ----------
    wandb_enabled = rank0 and not args.disable_wandb
    if wandb_enabled:
        config_basename = Path(args.config).stem
        run_name = f"{config_basename}_{time.strftime('%y%m%d_%H%M')}"
        if args.smoke:
            run_name += "_smoke"
        wandb.init(
            project=getattr(cfg, "wandb_project", "longlive_motiondirector"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[motiondirector] wandb run: {wandb.run.url}", flush=True)

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
        single_video=bool(getattr(cfg, "single_video", False)),
    )

    # General-prompt dataset (anti-drift anchor in trajectory_cosine mode only).
    # docs/02.md §5.2 — reuses scripts/prepare_openvid.py output, no extra prep.
    # Guarded by lambda_anchor>0 so an ablation with lambda_anchor=0 skips the
    # disable_adapter / second-forward path entirely (Risk 3: that path tripped
    # torch.utils.checkpoint mismatch under FSDP + PEFT in run 8939766).
    loss_space = str(getattr(cfg, "loss_space", "eps"))
    lambda_anchor_init = float(getattr(cfg, "lambda_anchor", 1.0))
    general_dataset = None
    if loss_space == "trajectory_cosine" and lambda_anchor_init > 0.0:
        general_dataset = GeneralPromptDataset(
            data_root=cfg.data_root,
            vae=vae,
            frame_count=int(cfg.frame_count),
            resolution=int(cfg.resolution),
            device=device,
            manifest_rel=str(getattr(cfg, "general_manifest_rel", "prompts/motion_pairs_train.jsonl")),
            max_clips=int(getattr(cfg, "general_max_clips", 50)),
            seed=int(getattr(cfg, "general_seed", 0)),
        )

    # ---------- Text encoder: load → encode → free ----------
    # Pre-cache null + train_caption embeddings, then drop the ~20 GB fp32
    # umt5 before loading the backbone so peak GPU per rank stays well clear
    # of (text_encoder + backbone). Valid for single-caption training only;
    # multi-caption training (paper Table 1, 12 motions) needs the encoder
    # to stay online instead.
    #
    # In trajectory_cosine mode we ALSO pre-encode the general subset's
    # captions here so the encoder can still be freed afterwards. Memory cost:
    # ~50 cond_dicts × ~1 MB each = trivial.
    if rank0:
        print("[motiondirector] loading text encoder (cache embeddings then free) ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        null_cond = {k: v.detach().clone() for k, v in text_encoder([""]).items()}
        train_cond = {k: v.detach().clone() for k, v in text_encoder([dataset.train_caption]).items()}

        # caption (str) → cond_dict mapping; deduplicates if any captions
        # happen to repeat across entries (they may in OpenVid). Sample-time
        # lookup is then a single dict access.
        general_caption_to_cond: dict[str, dict] | None = None
        if general_dataset is not None:
            general_caption_to_cond = {}
            for _, caption in general_dataset.entries:
                if caption in general_caption_to_cond:
                    continue
                cond = text_encoder([caption])
                general_caption_to_cond[caption] = {
                    k: v.detach().clone() for k, v in cond.items()
                }
            if rank0:
                print(
                    f"[motiondirector] pre-encoded {len(general_caption_to_cond)} "
                    f"unique captions from {len(general_dataset.entries)} entries",
                    flush=True,
                )
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Build wrapper + load NVlabs frozen starting point ----------
    # Mirrors configs/longlive_inference.yaml verbatim so the frozen starting
    # point equals NVlabs's complete few-step inference product:
    #   1. WanDiffusionWrapper with model_kwargs (local_attn_size, sink_size,
    #      timestep_shift) matching upstream
    #   2. load_state_dict from longlive_base.pt at the wrapper level —
    #      state-dict keys carry `model.` prefix (wrapper.model = CausalWanModel)
    #   3. attach NVlabs lora.pt as a transient PEFT adapter, load its
    #      weights, merge into base via merge_and_unload()
    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[motiondirector] building {cfg.model_name} ({arch}) ...", flush=True)
    model_kwargs = dict(
        model_name=cfg.model_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=is_causal,
    )
    if is_causal:
        model_kwargs["local_attn_size"] = int(getattr(cfg, "local_attn_size", -1))
        model_kwargs["sink_size"] = int(getattr(cfg, "sink_size", 0))
    generator = WanDiffusionWrapper(**model_kwargs)

    # Step 2: load longlive_base.pt → wrapper level (state-dict keys carry
    # `model.` prefix because the wrapper holds self.model = CausalWanModel).
    base_ckpt_path = os.path.expandvars(os.path.expanduser(cfg.base_ckpt))
    if rank0:
        print(f"[motiondirector] loading base ckpt: {base_ckpt_path}", flush=True)
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
            f"[motiondirector] base ckpt load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if missing and len(missing) <= 10:
            print(f"  missing keys: {missing}", flush=True)
        if unexpected and len(unexpected) <= 10:
            print(f"  unexpected keys: {unexpected}", flush=True)
    del sd, state

    # Step 3: overlay NVlabs lora.pt — the LoRA-side of NVlabs's released
    # LongLive few-step product. Attach as a transient PEFT adapter, load
    # weights, merge into base. After merge_and_unload() the wrapper holds
    # a plain CausalWanModel again, ready for our own LoRA.
    baseline_lora_ckpt = getattr(cfg, "baseline_lora_ckpt", None)
    if baseline_lora_ckpt:
        baseline_lora_ckpt = os.path.expandvars(os.path.expanduser(baseline_lora_ckpt))
        if rank0:
            print(
                f"[motiondirector] overlaying NVlabs baseline LoRA: {baseline_lora_ckpt}",
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
        set_peft_model_state_dict(generator.model, baseline_state)
        generator.model = generator.model.merge_and_unload()
        if rank0:
            print("[motiondirector] baseline LoRA merged into base weights", flush=True)
        del baseline_state

    # Attach the LoRA we will train (MotionDirector loss).
    # `model_name='generator'` targets CausalWanAttentionBlock to match the
    # backbone we just loaded (longlive_base.pt = CausalWanModel). Calling
    # with a non-matching dispatch key produces zero LoRA layers attached
    # (silent — PEFT accepts empty target_modules) and the training loop
    # runs as a no-op.
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    # Enable gradient checkpointing AFTER LoRA attach — upstream
    # trainer/distillation.py order. PEFT re-wraps modules; setting gc on
    # the pre-wrap model risks losing the flag on wrapped layers.
    generator.enable_gradient_checkpointing()

    # PEFT creates LoRA adapters in float32; FSDP's size-based auto-wrap groups
    # them with the bfloat16 base params and fails "uniform dtype" validation.
    # Cast every fp32 param down to bfloat16 to match base — same fix as
    # longlive/trainer/distillation.py.
    n_cast = 0
    for p in generator.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if rank0:
        print(f"[motiondirector] cast {n_cast} fp32 params to bfloat16 (post-LoRA, pre-FSDP)", flush=True)

    # ---------- FSDP wrap ----------
    # Shards the backbone across ranks; LoRA params also sharded.
    # mixed_precision=True → bf16 compute, fp32 grad reduce + fp32 buffers.
    # `use_orig_params=True` (set inside fsdp_wrap) is required for PEFT.
    generator.model = fsdp_wrap(
        generator.model,
        sharding_strategy="full",
        mixed_precision=True,
        wrap_strategy="size",
    )
    generator.model.train()  # LoRA-only training mode (base frozen by PEFT)

    # Re-seed per rank for data-sampling variety after model init is done.
    random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    # ---------- Optimizer ----------
    trainable = [p for p in generator.model.parameters() if p.requires_grad]
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

    sched = generator.scheduler  # FlowMatchScheduler (already set_timesteps in wrapper init)

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

        # Pre-cached cond_dict: null vs train_caption (single-caption training;
        # text encoder was freed at init, no per-step forward).
        cond_dict = null_cond if random.random() < float(cfg.null_prompt_p) else train_cond

        # Add noise (uniform t across frames — single scalar per step).
        # t_sampling_mode controls where t comes from:
        #   "uniform" (default):   t ~ Uniform[t_min, t_max]
        #   "anchor_only":         t sampled from cfg.t_anchors (e.g. the
        #                          4 few-step denoising indices [1000, 750,
        #                          500, 250]) — keeps training on the
        #                          few-step model's well-defined timesteps only.
        noise = torch.randn_like(latent)
        n_frames = latent.shape[1]
        t_mode = str(getattr(cfg, "t_sampling_mode", "uniform"))
        if t_mode == "anchor_only":
            anchors = list(getattr(cfg, "t_anchors", [1000, 750, 500, 250]))
            t_scalar = torch.tensor(
                [anchors[torch.randint(0, len(anchors), (1,)).item()]],
                device=device, dtype=torch.long,
            )
        else:
            t_scalar = torch.randint(
                int(cfg.t_min), int(cfg.t_max), (1,), device=device
            )
        timestep = t_scalar.expand(1, n_frames).contiguous()  # (B=1, F)
        noisy = sched.add_noise(
            latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, latent.shape[:2])

        # Forward — wrapper returns flow velocity + derived x0.
        #   flow_pred = velocity prediction (ε − x0 in flow matching)
        #   pred_x0   = x_t − σ_t · flow_pred  (deterministic algebra)
        flow_pred, pred_x0 = generator(
            noisy,
            cond_dict,
            timestep,
        )

        # Loss space selection:
        #   "eps" (paper recipe): supervises (flow_pred + pred_x0) toward ε_gt.
        #     Algebraically = (1−σ_t)² · MSE(flow_pred, ε−x0). Weight peaks at
        #     low t (≈ pixel detail regime), near zero at high t (motion regime).
        #     Mismatched for DMD few-step student: weight at t=1000 ≈ 0.
        #   "x0": supervises pred_x0 toward clean latent x0_gt directly.
        #     Algebraically = σ_t² · MSE(flow_pred, ε−x0). Weight peaks at
        #     high t (motion / coarse-structure regime), matches what DMD
        #     distillation actually trained the student to output (x0 at the
        #     4 few-step anchors).
        #   "trajectory_cosine": no L_MSE on eps/x0 at all. Supervision is
        #     inter-frame delta cosine on pred_x0 vs reference latent
        #     (L_motion) plus prior-consistency MSE between LoRA-on and
        #     LoRA-off student on a general (non-reference) clip (L_anchor).
        #     See docs/02.md.
        loss_space = str(getattr(cfg, "loss_space", "eps"))
        if loss_space == "trajectory_cosine":
            # ---- L_motion (reference batch) ----
            loss_motion = trajectory_cosine_loss(pred_x0, latent)

            lambda_motion = float(getattr(cfg, "lambda_motion", 1.0))
            lambda_anchor = float(getattr(cfg, "lambda_anchor", 1.0))

            # ---- L_anchor (general batch) — gated by lambda_anchor > 0 ----
            # The LoRA-off forward inside `disable_adapter()` interacts badly
            # with FSDP + torch.utils.checkpoint (mismatched saved-tensor count
            # on recompute, see run 8939766 traceback). Until that's resolved
            # via a second-base-model copy (option C in conversation), we skip
            # the entire anti-drift path when lambda_anchor == 0.
            if lambda_anchor > 0.0 and general_dataset is not None:
                z_gen, caption_g = general_dataset.sample()
                cond_gen = general_caption_to_cond[caption_g]

                noise_g = torch.randn_like(z_gen)
                n_frames_g = z_gen.shape[1]
                if t_mode == "anchor_only":
                    anchors_g = list(getattr(cfg, "t_anchors", [1000, 750, 500, 250]))
                    t_scalar_g = torch.tensor(
                        [anchors_g[torch.randint(0, len(anchors_g), (1,)).item()]],
                        device=device, dtype=torch.long,
                    )
                else:
                    t_scalar_g = torch.randint(
                        int(cfg.t_min), int(cfg.t_max), (1,), device=device,
                    )
                timestep_g = t_scalar_g.expand(1, n_frames_g).contiguous()
                noisy_g = sched.add_noise(
                    z_gen.flatten(0, 1),
                    noise_g.flatten(0, 1),
                    timestep_g.flatten(0, 1),
                ).unflatten(0, z_gen.shape[:2])

                _, pred_x0_lora = generator(noisy_g, cond_gen, timestep_g)
                with generator.model.disable_adapter():
                    with torch.no_grad():
                        _, pred_x0_base = generator(noisy_g, cond_gen, timestep_g)

                loss_anchor = prior_consistency_loss(pred_x0_lora, pred_x0_base)
            else:
                loss_anchor = torch.zeros((), device=device, dtype=loss_motion.dtype)

            loss = lambda_motion * loss_motion + lambda_anchor * loss_anchor

            # Logging slots — reuse mse / ad keys so existing wandb
            # bucketing infra still produces meaningful series; the
            # semantics differ but the dashboard columns line up.
            loss_mse = loss_motion
            loss_ad = loss_anchor
        elif loss_space in ("eps", "x0"):
            if loss_space == "x0":
                target_pred, target_gt = pred_x0, latent
            else:
                target_pred, target_gt = flow_pred + pred_x0, noise

            # Loss = L_MSE + ad_weight * L_AD (alpha=sqrt(2), beta=1 by paper).
            # ad_weight defaults to 1.0 (paper recipe); 0.0 ablates L_AD.
            loss_mse = F.mse_loss(target_pred, target_gt)
            loss_ad = appearance_debias_loss(
                target_pred, target_gt,
                alpha=float(cfg.ad_alpha), beta=float(cfg.ad_beta),
            )
            ad_weight = float(getattr(cfg, "ad_weight", 1.0))
            loss = loss_mse + ad_weight * loss_ad
        else:
            raise ValueError(f"unknown loss_space: {loss_space}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()

        dt = time.time() - t0
        cur_lr = lr_sched.get_last_lr()[0]
        t_val = int(t_scalar.item())
        if rank0 and (step % 10 == 0 or step == int(cfg.train_steps) - 1):
            print(
                f"[motiondirector] step {step:4d}/{cfg.train_steps}  "
                f"t={t_val:4d}  "
                f"loss={loss.item():.4f} "
                f"(mse={loss_mse.item():.4f}, ad={loss_ad.item():.4f})  "
                f"lr={cur_lr:.2e}  dt={dt:.1f}s",
                flush=True,
            )
        if wandb_enabled:
            # Bucket loss by t — high t (≥600) has near-zero loss because
            # the B1 close-form identity `eps = flow_pred + pred_x0` collapses
            # to eps ≈ eps_gt as sigma → 1; low/mid t carry the actual
            # learning signal. Bucketed series let the dashboard show real
            # trend instead of t-variance noise.
            log_dict = {
                "loss/total": loss.item(),
                "loss/mse": loss_mse.item(),
                "loss/ad": loss_ad.item(),
                "lr": cur_lr,
                "step_t": t_val,
                "step_dt_s": dt,
            }
            if t_val < 200:
                log_dict["loss/mse_low_t"] = loss_mse.item()
                log_dict["loss/ad_low_t"] = loss_ad.item()
            elif t_val < 600:
                log_dict["loss/mse_mid_t"] = loss_mse.item()
                log_dict["loss/ad_mid_t"] = loss_ad.item()
            else:
                log_dict["loss/mse_high_t"] = loss_mse.item()
                log_dict["loss/ad_high_t"] = loss_ad.item()
            wandb.log(log_dict, step=step)

        if (
            int(cfg.ckpt_interval) > 0
            and (step + 1) % int(cfg.ckpt_interval) == 0
            and (step + 1) < int(cfg.train_steps)
        ):
            ckpt_path = _save_lora_ckpt(generator.model, out_dir, str(step + 1), rank0)
            if rank0:
                _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
                print(f"[motiondirector] saved ckpt: {ckpt_path}", flush=True)

    final_path = _save_lora_ckpt(generator.model, out_dir, "final", rank0)
    if rank0:
        print(f"[motiondirector] DONE. final ckpt: {final_path}", flush=True)
    if wandb_enabled:
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
