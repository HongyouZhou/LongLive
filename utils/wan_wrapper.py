# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import json
import os
import types
from typing import List, Optional
import torch
from torch import nn
import torch.distributed as dist

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from utils.nfs_serial import nfs_serial
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_model_infinity import CausalWanModel as CausalWanModelInfinity


# Root containing the pretrained Wan2.1 checkpoints
# (Wan2.1-T2V-{14B,1.3B}/ subdirectories live underneath).
# Defaults to repo-relative `wan_models` to match the historical layout;
# override with WAN_MODELS_ROOT for HPC/cluster runs where weights live
# under a separate $PROJECT_DATA tree.
WAN_MODELS_ROOT = os.environ.get("WAN_MODELS_ROOT", "wan_models")


def _load_wan_with_meta(model_cls, path, **extra_kwargs):
    """Load a Wan diffusion model, using meta init on non-rank-0 ranks.

    When loading a large teacher (e.g. 14B) across multiple ranks, the default
    from_pretrained materialises the full fp32 weights on every rank's CPU —
    on 2 ranks this peaks at ~2× model_size × 4 bytes. For a 14B model on a
    125GB-RAM box that OOMs before FSDP ever gets to shard.

    Fix: only rank 0 loads the weights. Other ranks build an architecture-only
    copy on meta device (0 bytes), then materialise empty CUDA tensors for
    FSDP to broadcast into via `sync_module_states=True`.
    """
    # Use torchrun's RANK/WORLD_SIZE env vars — these are set by the launcher
    # before any Python code runs, so this works correctly even when called
    # before dist.init_process_group(). Relying on dist.is_initialized() here
    # was a bug: if model construction happens before init_process_group, every
    # rank fell through to "full load" and 8-way concurrent NFS reads of the
    # 14B safetensors thrashed the shared FS into a hang.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1 or rank == 0:
        # Load weights directly in bf16 to halve both CPU and GPU footprint
        # during FSDP wrap. Cast any stray fp32 buffers (RoPE freqs etc.) so
        # FSDP's size-based auto-wrap sees uniform dtype.
        with nfs_serial():
            model = model_cls.from_pretrained(
                path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, **extra_kwargs
            )
        for buf_name, buf in model.named_buffers():
            if buf.dtype == torch.float32:
                buf.data = buf.data.to(torch.bfloat16)
        return model

    # Non-rank-0: construct empty model (no weight loading), then move to CUDA.
    from accelerate import init_empty_weights

    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(extra_kwargs)

    # Build on bf16 default so `to_empty()` halves GPU allocation vs fp32
    # (14B × 4 bytes = 56GB → 14B × 2 bytes = 28GB). FSDP's mixed_precision
    # already runs compute in bf16 so this matches runtime.
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights():
            model = model_cls(**cfg)
    finally:
        torch.set_default_dtype(original_dtype)
    if hasattr(model, "register_to_config"):
        model.register_to_config(**cfg)
    # Convert meta tensors to empty CUDA tensors; FSDP will broadcast real
    # values from rank 0 during wrap (sync_module_states=True).
    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to_empty(device=device)
    return model

class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        with nfs_serial():
            self.text_encoder.load_state_dict(
                torch.load(f"{WAN_MODELS_ROOT}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                           map_location='cpu', weights_only=False)
            )
        
        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=f"{WAN_MODELS_ROOT}/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        with nfs_serial():
            self.model = _video_vae(
                pretrained_path=f"{WAN_MODELS_ROOT}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                z_dim=16,
            ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 120) -> torch.Tensor:
        """
        Decode latent frames to pixel space.
        
        Args:
            latent: Latent tensor with shape [batch_size, num_frames, num_channels, height, width]
            use_cache: Whether to use cached decoding (for streaming)
            chunk_size: Number of latent frames to decode at once (default 240 to avoid OOM)
        
        Returns:
            Decoded video tensor with shape [batch_size, num_frames, num_channels, height, width]
        """
        # latent shape: [batch_size, num_frames, num_channels, height, width]
        # zs shape after permute: [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
            else:
                decoded_chunks = []
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]  # [C, chunk_frames, H, W]
                    self.model.clear_cache()
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())
                    
                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                self.model.clear_cache()
            output.append(decoded)
        
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_infinite_attention=False
    ):
        super().__init__()

        # Non-rank-0 ranks use meta-device init to avoid duplicate 14B-in-CPU
        # materialisation (see _load_wan_with_meta). FSDP's sync_module_states
        # broadcasts real weights from rank 0 during wrap. NOTE: intentionally
        # don't pass torch_dtype=bf16 — Wan keeps RoPE/sincos buffers in fp32
        # and FSDP's size-based auto-wrap refuses to flatten mixed-dtype
        # groups; FSDP's MixedPrecision handles runtime bf16 casting.
        path = f"{WAN_MODELS_ROOT}/{model_name}/"
        if is_causal:
            model_cls = CausalWanModelInfinity if use_infinite_attention else CausalWanModel
            self.model = _load_wan_with_meta(
                model_cls, path, local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = _load_wan_with_meta(WanModel, path)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        sink_recache_after_switch=False
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                sink_recache_after_switch=sink_recache_after_switch
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    sink_recache_after_switch=sink_recache_after_switch
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        sink_recache_after_switch=sink_recache_after_switch
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        sink_recache_after_switch=sink_recache_after_switch
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
