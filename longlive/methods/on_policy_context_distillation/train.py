"""On-policy context distillation trainer for the 4-step DMD base.

Outer loop:

  1. Roll out the current trainable student adapter with the production
     4-step causal sampler.
  2. Re-noise the student's generated endpoint latents at DMD anchor timesteps.
  3. Match the trainable student velocity to a frozen context teacher velocity
     on those on-policy states.
  4. Optionally add a small no-context base anchor.

This intentionally does not reuse the removed DiffusionNFT beta-interpolation
or EMA self-mirror machinery.
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
from torch.optim import AdamW

from longlive.data.motion_refs import SkateboardingLatentDataset
from longlive.methods.on_policy_context_distillation.losses import (
    anchor_velocity_loss,
    context_velocity_distillation_loss,
)
from longlive.methods.on_policy_context_distillation.teacher import (
    freeze_adapter_params,
    load_teacher_adapter,
    optional_path,
)
from longlive.utils.checkpoints import (
    cast_fp32_params_to_bf16,
    clean_fsdp_key,
    find_adapter_params,
    merge_lora_into_transformer,
    prune_old_lora_checkpoints,
    save_lora_checkpoint,
)
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.rl_rollout import RolloutEngine, maybe_barrier
from longlive.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


def _load_base_generator(generator: WanDiffusionWrapper, ckpt_path: str, rank0: bool):
    ckpt_path = os.path.expandvars(os.path.expanduser(str(ckpt_path)))
    if rank0:
        print(f"[on_policy_context_distillation] loading base ckpt: {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "generator" in sd:
        state = sd["generator"]
    elif "model" in sd:
        state = sd["model"]
    else:
        state = sd
    state = {clean_fsdp_key(k): v for k, v in state.items()}
    missing, unexpected = generator.load_state_dict(state, strict=False)
    del sd, state
    return missing, unexpected


def _encode_prompt(
    text_encoder: WanTextEncoder,
    prompt: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {k: v.detach().clone() for k, v in text_encoder([prompt]).items()}


def _latent_shape_from_cfg(cfg) -> tuple[int, int, int, int, int]:
    latent_b = 1
    latent_f = (int(cfg.frame_count) - 1) // 4 + 1
    latent_c = 16
    latent_h = int(cfg.resolution) // 8
    latent_w_pixel = int(cfg.resolution * 16 / 9)
    latent_w_pixel = int(getattr(cfg, "pixel_width", latent_w_pixel))
    latent_w = latent_w_pixel // 8
    return latent_b, latent_f, latent_c, latent_h, latent_w


def _pipeline_args_from_cfg(cfg):
    return OmegaConf.create({
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


def _apply_smoke_overrides(cfg) -> None:
    cfg.outer_epochs = 2
    cfg.inner_steps = 4
    cfg.k_rollouts = 2
    cfg.k_noisings_per_endpoint = 2
    cfg.g_endpoints_per_outer = 2
    cfg.ckpt_interval = 1
    cfg.warmup_steps = 0
    cfg.allow_base_teacher = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

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
        _apply_smoke_overrides(cfg)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.out_suffix:
        cfg.out_dir = f"{cfg.out_dir}{args.out_suffix}"

    g_endpoints = int(getattr(cfg, "g_endpoints_per_outer", cfg.k_rollouts))
    k_noisings = int(getattr(cfg, "k_noisings_per_endpoint", cfg.inner_steps // g_endpoints))
    assert int(cfg.k_rollouts) == g_endpoints, (
        "on-policy context distillation requires "
        f"k_rollouts == g_endpoints_per_outer; got {cfg.k_rollouts} and {g_endpoints}"
    )
    assert k_noisings * g_endpoints == int(cfg.inner_steps), (
        f"inner_steps must equal k_noisings * g_endpoints; got "
        f"inner_steps={cfg.inner_steps}, k_noisings={k_noisings}, g_endpoints={g_endpoints}"
    )

    if rank0:
        print("[on_policy_context_distillation] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[on_policy_context_distillation] device: {gpu_name} "
            f"({gpu_total_gib:.1f} GiB) x world_size={world_size}",
            flush=True,
        )

    wandb_enabled = rank0 and not args.disable_wandb
    if wandb_enabled:
        run_name = f"{Path(args.config).stem}_{time.strftime('%y%m%d_%H%M')}"
        if args.smoke:
            run_name += "_smoke"
        if args.out_suffix:
            run_name += args.out_suffix
        wandb.init(
            project=getattr(cfg, "wandb_project", "longlive_on_policy_context_distillation"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[on_policy_context_distillation] wandb run: {wandb.run.url}", flush=True)

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    if rank0:
        print("[on_policy_context_distillation] loading VAE (bf16) ...", flush=True)
    vae = WanVAEWrapper(model_name=str(cfg.model_name))
    vae.to(device=device, dtype=torch.bfloat16).eval()

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

    teacher_caption = str(getattr(cfg, "teacher_caption", "") or "").strip()
    if not teacher_caption:
        teacher_caption = train_caption

    if rank0:
        print("[on_policy_context_distillation] loading text encoder ...", flush=True)
    text_encoder = WanTextEncoder(model_name=str(cfg.model_name))
    text_encoder.to(device).eval()
    train_cond = _encode_prompt(text_encoder, train_caption)
    teacher_cond = train_cond if teacher_caption == train_caption else _encode_prompt(text_encoder, teacher_caption)
    del text_encoder
    torch.cuda.empty_cache()

    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[on_policy_context_distillation] building {cfg.model_name} ({arch}) ...", flush=True)
    model_kwargs = dict(
        model_name=cfg.model_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=is_causal,
    )
    if is_causal:
        model_kwargs["local_attn_size"] = int(getattr(cfg, "local_attn_size", -1))
        model_kwargs["sink_size"] = int(getattr(cfg, "sink_size", 0))
    generator = WanDiffusionWrapper(**model_kwargs)

    missing, unexpected = _load_base_generator(generator, cfg.base_ckpt, rank0)
    if rank0:
        print(
            "[on_policy_context_distillation] base load: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    baseline_lora_ckpt = optional_path(getattr(cfg, "baseline_lora_ckpt", None))
    if baseline_lora_ckpt:
        if rank0:
            print(
                "[on_policy_context_distillation] merging baseline LoRA: "
                f"{baseline_lora_ckpt}",
                flush=True,
            )
        generator.model = merge_lora_into_transformer(
            generator.model,
            model_name="generator",
            adapter_config=cfg.baseline_adapter,
            lora_ckpt=baseline_lora_ckpt,
            is_main_process=rank0,
        )

    if rank0:
        print(
            "[on_policy_context_distillation] attaching adapters: "
            "default(student) + teacher(frozen) + optional anchor",
            flush=True,
        )
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    peft_config_default = generator.model.peft_config["default"]
    generator.model.add_adapter("teacher", peft_config_default)

    beta_anchor = float(getattr(cfg, "beta_anchor", 0.0))
    use_anchor = beta_anchor > 0.0
    if use_anchor:
        generator.model.add_adapter("anchor", peft_config_default)

    teacher_lora_ckpt = optional_path(getattr(cfg, "teacher_lora_ckpt", None))
    if teacher_lora_ckpt:
        if rank0:
            print(
                "[on_policy_context_distillation] loading teacher adapter: "
                f"{teacher_lora_ckpt}",
                flush=True,
            )
        load_teacher_adapter(
            generator.model,
            teacher_lora_ckpt,
            adapter_name="teacher",
            state_key=str(getattr(cfg, "teacher_lora_key", "generator_lora")),
        )
    elif not bool(getattr(cfg, "allow_base_teacher", False)):
        raise ValueError(
            "teacher_lora_ckpt is required. Set allow_base_teacher=true only "
            "for smoke tests or explicit no-context ablations."
        )
    elif rank0:
        print(
            "[on_policy_context_distillation][warn] using base teacher; "
            "this is a smoke/ablation mode, not context distillation.",
            flush=True,
        )

    n_teacher_frozen = freeze_adapter_params(generator.model, "teacher")
    n_anchor_frozen = freeze_adapter_params(generator.model, "anchor") if use_anchor else 0
    if rank0:
        print(
            "[on_policy_context_distillation] frozen adapter params: "
            f"teacher={n_teacher_frozen:,}, anchor={n_anchor_frozen:,}",
            flush=True,
        )

    generator.model.set_adapter("default")
    generator.enable_gradient_checkpointing()
    n_cast = cast_fp32_params_to_bf16(generator.model)
    if rank0:
        print(
            f"[on_policy_context_distillation] cast {n_cast} fp32 params to bf16",
            flush=True,
        )

    generator.model = fsdp_wrap(
        generator.model,
        sharding_strategy="full",
        mixed_precision=True,
        wrap_strategy="size",
    )
    generator.model.train()

    random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    student_params = find_adapter_params(generator.model, "default")
    teacher_params = find_adapter_params(generator.model, "teacher")
    anchor_params = find_adapter_params(generator.model, "anchor") if use_anchor else []
    if rank0:
        print(
            "[on_policy_context_distillation] adapter param counts: "
            f"default={len(student_params)}, teacher={len(teacher_params)}, "
            f"anchor={len(anchor_params)}",
            flush=True,
        )

    trainable = [p for p in student_params if p.requires_grad]
    n_trainable_local = sum(p.numel() for p in trainable)
    n_trainable_global = torch.tensor(n_trainable_local, device=device)
    dist.all_reduce(n_trainable_global)
    if rank0:
        print(
            "[on_policy_context_distillation] trainable params "
            f"(FSDP-sharded total): {int(n_trainable_global.item()):,}",
            flush=True,
        )

    optimizer = AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    rollout_engine = RolloutEngine(
        generator=generator,
        vae=vae,
        cached_cond_dict=train_cond,
        pipeline_args=_pipeline_args_from_cfg(cfg),
        device=device,
        latent_shape=_latent_shape_from_cfg(cfg),
    )

    out_dir = Path(cfg.out_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    sched = generator.scheduler
    outer_epochs = int(cfg.outer_epochs)
    inner_steps = int(cfg.inner_steps)
    k_rollouts = int(cfg.k_rollouts)
    anchors = list(cfg.t_anchors)
    distill_weight = float(getattr(cfg, "distill_weight", 1.0))
    save_trajectory = bool(getattr(cfg, "save_rollout_trajectory", False))
    trajectory_device = str(getattr(cfg, "trajectory_device", "cpu"))

    if rank0:
        print(
            "[on_policy_context_distillation] start: "
            f"outer={outer_epochs} x inner={inner_steps} x K={k_rollouts} "
            f"(g_endpoints={g_endpoints}, k_noisings={k_noisings}) | "
            f"distill_weight={distill_weight} | beta_anchor={beta_anchor} | "
            f"anchors={anchors}",
            flush=True,
        )

    global_step = 0
    t_train_loop_start = time.time()
    setup_time_s = t_train_loop_start - t_setup_start
    if rank0:
        print(f"[on_policy_context_distillation] setup_time_s={setup_time_s:.1f}", flush=True)

    for outer in range(outer_epochs):
        t_outer = time.time()

        generator.model.set_adapter("default")
        rollout_seed = int(cfg.seed) + 1009 * outer + 31 * rank
        rollouts = rollout_engine.rollout_k_structured(
            k=k_rollouts,
            dtype=torch.bfloat16,
            base_seed=rollout_seed,
            return_trajectory=save_trajectory,
            trajectory_device=trajectory_device,
        )
        t_rollout = time.time() - t_outer

        generator.model.set_adapter("default")
        t_train_start = time.time()

        sum_loss = 0.0
        sum_distill = 0.0
        sum_anchor = 0.0
        sum_delta = 0.0
        sum_cos = 0.0
        sum_student_norm = 0.0
        sum_teacher_norm = 0.0
        per_anchor_delta = [0.0] * len(anchors)
        per_anchor_count = [0] * len(anchors)
        per_rollout_delta = [0.0] * k_rollouts
        per_rollout_count = [0] * k_rollouts

        for inner in range(inner_steps):
            k_idx = (inner // k_noisings) % g_endpoints
            t_idx = inner % len(anchors)
            anchor_t = int(anchors[t_idx])
            x0_ref = rollouts[k_idx].latent_x0.to(device=device, dtype=torch.bfloat16)

            noise = torch.randn_like(x0_ref)
            n_frames = x0_ref.shape[1]
            t_scalar = torch.tensor([anchor_t], device=device, dtype=torch.long)
            timestep = t_scalar.expand(1, n_frames).contiguous()
            noisy = sched.add_noise(
                x0_ref.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, x0_ref.shape[:2])

            flow_student, _pred_x0_student = generator(noisy, train_cond, timestep)

            generator.model.set_adapter("teacher")
            with torch.no_grad():
                flow_teacher, _pred_x0_teacher = generator(noisy, teacher_cond, timestep)

            if use_anchor:
                generator.model.set_adapter("anchor")
                with torch.no_grad():
                    flow_anchor, _pred_x0_anchor = generator(noisy, train_cond, timestep)
            else:
                flow_anchor = None

            generator.model.set_adapter("default")

            loss_distill, diag = context_velocity_distillation_loss(
                flow_student,
                flow_teacher,
            )
            loss = distill_weight * loss_distill
            if use_anchor:
                anchor_loss = anchor_velocity_loss(flow_student, flow_anchor)
                loss = loss + beta_anchor * anchor_loss
            else:
                anchor_loss = torch.zeros((), device=device, dtype=loss.dtype)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.detach())
            sum_distill += float(loss_distill.detach())
            sum_anchor += float(anchor_loss.detach())
            sum_delta += float(diag["distill/delta_norm"])
            sum_cos += float(diag["distill/cosine"])
            sum_student_norm += float(diag["distill/student_norm"])
            sum_teacher_norm += float(diag["distill/teacher_norm"])
            per_anchor_delta[t_idx] += float(diag["distill/delta_norm"])
            per_anchor_count[t_idx] += 1
            per_rollout_delta[k_idx] += float(diag["distill/delta_norm"])
            per_rollout_count[k_idx] += 1
            global_step += 1

        t_train = time.time() - t_train_start
        dt_outer = time.time() - t_outer
        n = max(1, inner_steps)
        avg_loss = sum_loss / n
        avg_distill = sum_distill / n
        avg_anchor = sum_anchor / n
        avg_delta = sum_delta / n
        avg_cos = sum_cos / n
        avg_student_norm = sum_student_norm / n
        avg_teacher_norm = sum_teacher_norm / n

        if rank0:
            print(
                f"[on_policy_context_distillation] outer {outer:3d}/{outer_epochs}  "
                f"loss={avg_loss:.4f}  distill={avg_distill:.4f}  "
                + (f"anchor={avg_anchor:.4f}  " if use_anchor else "")
                + f"delta={avg_delta:.3f}  cos={avg_cos:.3f}  "
                f"v_s={avg_student_norm:.3f}  v_t={avg_teacher_norm:.3f}  "
                f"dt={dt_outer:.1f}s (rollout={t_rollout:.1f}, train={t_train:.1f})",
                flush=True,
            )

        if wandb_enabled:
            log_dict = {
                "outer/loss": avg_loss,
                "outer/distill_loss": avg_distill,
                "outer/anchor_loss": avg_anchor,
                "outer/delta_norm": avg_delta,
                "outer/cosine": avg_cos,
                "outer/student_norm": avg_student_norm,
                "outer/teacher_norm": avg_teacher_norm,
                "outer/dt_total_s": dt_outer,
                "outer/dt_rollout_s": t_rollout,
                "outer/dt_train_s": t_train,
            }
            for t_i, _t_v in enumerate(anchors):
                if per_anchor_count[t_i]:
                    log_dict[f"distill/delta_norm_t{t_i}"] = (
                        per_anchor_delta[t_i] / per_anchor_count[t_i]
                    )
            for k_i in range(k_rollouts):
                if per_rollout_count[k_i]:
                    log_dict[f"distill/delta_norm_k{k_i}"] = (
                        per_rollout_delta[k_i] / per_rollout_count[k_i]
                    )
            wandb.log(log_dict, step=global_step)

        if (
            int(cfg.ckpt_interval) > 0
            and (outer + 1) % int(cfg.ckpt_interval) == 0
            and (outer + 1) < outer_epochs
        ):
            ckpt_path = save_lora_checkpoint(
                generator.model,
                out_dir,
                str(outer + 1),
                rank0=rank0,
            )
            if rank0:
                prune_old_lora_checkpoints(out_dir, int(cfg.ckpt_keep_last))
                print(f"[on_policy_context_distillation] saved ckpt: {ckpt_path}", flush=True)

        maybe_barrier()

    train_loop_time_s = time.time() - t_train_loop_start
    final_path = save_lora_checkpoint(generator.model, out_dir, "final", rank0=rank0)
    if rank0:
        print(
            "[on_policy_context_distillation] DONE. "
            f"setup_time_s={setup_time_s:.1f} train_loop_time_s={train_loop_time_s:.1f} "
            f"final ckpt: {final_path}",
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
