# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os

from longlive.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from longlive.specs.wan import get_config_value, get_wan_model_spec_from_config

from longlive.utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from longlive.utils.debug_option import DEBUG
import torch.distributed as dist

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.model_kwargs = getattr(args, "model_kwargs", {})
        self.generator = WanDiffusionWrapper(
            **self.model_kwargs, is_causal=True) if generator is None else generator
        self.model_spec = getattr(self.generator, "model_spec", None)
        if self.model_spec is None:
            self.model_spec = get_wan_model_spec_from_config(self.model_kwargs)
        self.text_encoder = WanTextEncoder(model_name=self.model_spec.name) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(model_name=self.model_spec.name) if vae is None else vae
        self.cache_spec = self.model_spec.cache

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = self.cache_spec.transformer_blocks
        self.frame_seq_length = self.cache_spec.frame_seq_length

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = get_config_value(self.model_kwargs, "local_attn_size", -1)

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        return_trajectory: bool = False,
        trajectory_device: str | torch.device | None = None,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        kv_cache_size, kv_policy = self._resolve_kv_cache_size(num_output_frames)
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        trajectory = [] if return_trajectory else None
        trajectory_device = torch.device(trajectory_device) if trajectory_device is not None else noise.device

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if trajectory is not None:
                    trajectory.append({
                        "noisy_latent": noisy_input.detach().to(trajectory_device),
                        "timestep": timestep.detach().to(trajectory_device),
                        "block_start_frame": current_start_frame,
                        "num_frames": current_num_frames,
                        "denoise_step_index": index,
                    })

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
            # Step 2.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents and return_trajectory:
            return video, output.to(noise.device), trajectory
        elif return_latents:
            return video, output.to(noise.device)
        elif return_trajectory:
            return video, trajectory
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            kv_cache_size = self.cache_spec.default_kv_tokens(self.local_attn_size)

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros(self.cache_spec.kv_shape(batch_size, kv_cache_size), dtype=dtype, device=device),
                "v": torch.zeros(self.cache_spec.kv_shape(batch_size, kv_cache_size), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros(self.cache_spec.crossattn_shape(batch_size), dtype=dtype, device=device),
                "v": torch.zeros(self.cache_spec.crossattn_shape(batch_size), dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _resolve_kv_cache_size(self, num_output_frames: int) -> tuple[int, str]:
        local_attn_cfg = get_config_value(self.model_kwargs, "local_attn_size", -1)
        if int(local_attn_cfg) != -1:
            return (
                self.cache_spec.kv_tokens_for_frames(num_output_frames, int(local_attn_cfg)),
                f"int->local, size={local_attn_cfg}",
            )
        return (
            self.cache_spec.kv_tokens_for_frames(num_output_frames, -1),
            "global (-1)",
        )

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model spec's global default.
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if int(local_attn_size_value) == -1:
            target_size = self.cache_spec.attention_tokens(-1)
            policy = "global"
        else:
            target_size = self.cache_spec.attention_tokens(int(local_attn_size_value))
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
