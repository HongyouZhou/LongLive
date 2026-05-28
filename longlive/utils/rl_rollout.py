"""K-rollout sampler shared by on-policy finetune methods on the 4-step DMD base.

Each outer epoch needs `K` rollouts of the same caption with whichever PEFT
adapter the trainer activates. Adapter selection is the trainer's
responsibility, not this module's. We reuse
`longlive.pipeline.causal_inference.CausalInferencePipeline` so the rollout goes
through the exact same 4-step DMD sampler as inference / eval — there is no
separate "training-time sampler" that could drift from production.

Two integration details worth noting:

  (1) `CausalInferencePipeline` internally calls `self.text_encoder(text_prompts)`
      every time. We pre-encode our single training caption once at trainer
      startup (text encoder loaded → encoded → freed, see motiondirector
      pattern) and pass a tiny stub that returns the cached dict regardless of
      the `text_prompts` argument. This lets us run rollouts without keeping
      the ~20 GB umt5 fp32 model resident.

  (2) The pipeline's `inference()` allocates its own KV / cross-attention
      caches inside the call.  With our FSDP-wrapped backbone the model
      weights are sharded across ranks but the caches are per-rank tensors
      — no extra coordination required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from longlive.pipeline.causal_inference import CausalInferencePipeline


@dataclass(frozen=True)
class RolloutOutput:
    """Structured rollout result for future method APIs."""

    video: torch.Tensor
    latent_x0: torch.Tensor
    noise: torch.Tensor | None = None
    trajectory: list[dict] | None = None

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video, self.latent_x0


class _CachedTextEncoder:
    """Stub that ignores text_prompts and returns the pre-encoded cond dict.

    Used to bypass the text-encoder forward inside `CausalInferencePipeline.
    inference()` for the case where we know we'll always rollout the same
    caption(s).  The cond dict shape must match `WanTextEncoder.__call__`:
    `{"prompt_embeds": (B, L, D)}`.
    """

    def __init__(self, cond_dict: dict, batch_size: int = 1):
        # Detach + clone so we don't accidentally keep autograd graph alive.
        self._cond_dict_template = {
            k: v.detach() for k, v in cond_dict.items()
        }
        self._batch_size = batch_size

    @property
    def device(self):
        # CausalInferencePipeline doesn't use this, but defining it matches
        # the WanTextEncoder interface in case any path inspects it.
        any_value = next(iter(self._cond_dict_template.values()))
        return any_value.device

    def __call__(self, text_prompts) -> dict:
        # We assume callers pass `text_prompts=[caption]` of length B.  If they
        # pass more, broadcast the cached embedding across the batch.
        n = len(text_prompts) if hasattr(text_prompts, "__len__") else 1
        if n == 1:
            return {k: v.clone() for k, v in self._cond_dict_template.items()}
        return {
            k: v.expand(n, *v.shape[1:]).contiguous().clone()
            for k, v in self._cond_dict_template.items()
        }


class RolloutEngine:
    """K-rollout sampler that wraps `CausalInferencePipeline`.

    The trainer constructs one of these and calls `rollout()` K times per
    outer epoch. Adapter switching is the trainer's responsibility; this class
    is unaware of which adapter is active.
    """

    def __init__(
        self,
        generator,      # WanDiffusionWrapper, FSDP-wrapped .model, PEFT-equipped
        vae,            # WanVAEWrapper (frozen)
        cached_cond_dict: dict,
        pipeline_args,  # OmegaConf with denoising_step_list, model_kwargs, etc.
        device: torch.device,
        latent_shape: tuple[int, int, int, int, int],  # (B, F_lat, C, H_lat, W_lat)
    ):
        self.device = device
        self.latent_shape = tuple(latent_shape)

        # Build the inference pipeline with our FSDP-wrapped generator + the
        # cached cond stub.  We pass our VAE so the pipeline doesn't try to
        # load a second copy.
        self._stub_text_encoder = _CachedTextEncoder(
            cached_cond_dict, batch_size=self.latent_shape[0]
        )
        self.pipeline = CausalInferencePipeline(
            args=pipeline_args,
            device=device,
            generator=generator,
            text_encoder=self._stub_text_encoder,
            vae=vae,
        )

        # Cached for log messages — the placeholder caption string never
        # reaches the encoder, but pipeline.inference signature wants a list.
        self._placeholder_prompts = ["<cached>"] * self.latent_shape[0]

    def rollout(
        self,
        noise: torch.Tensor,
        *,
        return_trajectory: bool = False,
        trajectory_device: str | torch.device | None = None,
    ) -> RolloutOutput:
        """Run one full 4-step DMD rollout with a structured result.

        Args:
            noise: (B, F_lat, C, H_lat, W_lat) Gaussian noise tensor in the
                latent shape.  Caller pre-generates K of these with distinct
                random seeds.

        Returns:
            (video_pixel, latent_x0) where:
              * video_pixel: (B, F_pix, 3, H_pix, W_pix) in [0, 1] (pipeline
                already applies `(v * 0.5 + 0.5).clamp(0, 1)`).
              * latent_x0:   (B, F_lat, C, H_lat, W_lat) clean latent at the
                end of denoising.
              * trajectory:  optional list of denoising states visited by the
                sampler. Each item stores a block-local noisy latent and its
                timestep; this is intended for on-policy distillation methods.
        """
        with torch.no_grad():
            inference_kwargs = {
                "noise": noise,
                "text_prompts": self._placeholder_prompts,
                "return_latents": True,
            }
            if return_trajectory:
                inference_kwargs["return_trajectory"] = True
                inference_kwargs["trajectory_device"] = trajectory_device
            result = self.pipeline.inference(**inference_kwargs)
        if return_trajectory:
            video, latent, trajectory = result
        else:
            video, latent = result
            trajectory = None
        return RolloutOutput(
            video=video,
            latent_x0=latent,
            noise=noise,
            trajectory=trajectory,
        )

    @torch.no_grad()
    def rollout_one(self, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible tuple API for existing trainers."""
        return self.rollout(noise).as_tuple()

    def rollout_with_grad(self, noise: torch.Tensor, k_grad_steps: int | None = None) -> RolloutOutput:
        """Reserved API for DRaFT-style methods that backprop through rollout.

        The current inference pipeline writes denoised blocks into a preallocated
        latent buffer, which is fine for sampling but not a reliable autograd
        boundary.  Keep this explicit so future methods do not accidentally
        assume they are training through the rollout when gradients are absent.
        """
        raise NotImplementedError(
            "rollout_with_grad requires a grad-preserving inference path; "
            "use rollout() for no-grad sampling."
        )

    @torch.no_grad()
    def rollout_k(
        self,
        k: int,
        dtype: torch.dtype,
        base_seed: Optional[int] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate K rollouts sequentially.

        We use distinct manual seeds when `base_seed` is given so that the
        same outer epoch produces reproducible rollouts across debugging
        runs.  Pass `None` for stochastic-by-default behavior (uses the
        current torch state).

        Returns list of (video_pixel, latent_x0).
        """
        results = []
        gen = torch.Generator(device=self.device) if base_seed is not None else None
        for i in range(k):
            if gen is not None:
                gen.manual_seed(int(base_seed) + i)
                noise = torch.randn(
                    self.latent_shape, device=self.device, dtype=dtype, generator=gen,
                )
            else:
                noise = torch.randn(self.latent_shape, device=self.device, dtype=dtype)
            results.append(self.rollout_one(noise))
            # Free the pipeline's per-call KV caches between rollouts.  The
            # pipeline re-initialises them inside inference(), so this is
            # belt-and-braces; we add it because K rollouts back-to-back
            # would otherwise hold K × cache memory if anything leaked.
            self.pipeline.kv_cache1 = None
            self.pipeline.crossattn_cache = None
        return results

    @torch.no_grad()
    def rollout_k_structured(
        self,
        k: int,
        dtype: torch.dtype,
        base_seed: Optional[int] = None,
        *,
        return_trajectory: bool = False,
        trajectory_device: str | torch.device | None = None,
    ) -> list[RolloutOutput]:
        """Generate K rollouts and return structured outputs."""
        results: list[RolloutOutput] = []
        gen = torch.Generator(device=self.device) if base_seed is not None else None
        for i in range(k):
            if gen is not None:
                gen.manual_seed(int(base_seed) + i)
                noise = torch.randn(
                    self.latent_shape, device=self.device, dtype=dtype, generator=gen,
                )
            else:
                noise = torch.randn(self.latent_shape, device=self.device, dtype=dtype)
            results.append(
                self.rollout(
                    noise,
                    return_trajectory=return_trajectory,
                    trajectory_device=trajectory_device,
                )
            )
            self.pipeline.kv_cache1 = None
            self.pipeline.crossattn_cache = None
        return results


def maybe_barrier() -> None:
    """No-op when single-process, dist.barrier() when distributed.

    Used between rollout phase and training phase so all ranks stay in
    lockstep across the adapter switch.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
