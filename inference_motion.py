"""Motion-conditioned inference for LongLive fine-tuned with motion_encoder.

Usage:
    python inference_motion.py --config_path configs/longlive_inference_motion.yaml \
        --prompt "a cat running" --motion_ref assets/ref.mp4 --output out.mp4

The config reuses fields from the training motion config, plus inference-only:
    lora_ckpt, motion_encoder_ckpt, output_folder, num_output_frames, num_samples
"""
import argparse
import os
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from model.motion_encoder import MotionEncoder
from utils.dataset import _load_motion_video
from utils.misc import set_seed


def load_motion_encoder(config, vae, device, dtype):
    motion_cfg = config.motion_encoder
    me = MotionEncoder(
        vae_wrapper=vae,
        dim=getattr(motion_cfg, "dim", 4096),
        num_layers=getattr(motion_cfg, "num_layers", 4),
        num_heads=getattr(motion_cfg, "num_heads", 16),
        tokens_per_frame=getattr(motion_cfg, "tokens_per_frame", 64),
        max_tokens=getattr(motion_cfg, "max_tokens", 512),
    ).to(device=device, dtype=dtype)
    ckpt_path = getattr(config, "motion_encoder_ckpt", None)
    if ckpt_path is None:
        lora_ckpt = getattr(config, "lora_ckpt", None)
        if lora_ckpt:
            blob = torch.load(lora_ckpt, map_location="cpu")
            if isinstance(blob, dict) and "motion_encoder" in blob:
                me.load_state_dict(blob["motion_encoder"], strict=True)
                return me
        raise FileNotFoundError(
            "No motion_encoder weights found. Set motion_encoder_ckpt or "
            "provide a lora_ckpt that bundles 'motion_encoder'."
        )
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["motion_encoder"] if isinstance(blob, dict) and "motion_encoder" in blob else blob
    me.load_state_dict(sd, strict=True)
    return me


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--motion_ref", required=True,
                        help="Path to reference video (mp4/webm/etc.)")
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.distributed = False
    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    pipeline = CausalInferencePipeline(config, device=device)

    if config.generator_ckpt:
        blob = torch.load(config.generator_ckpt, map_location="cpu")
        gen_sd = blob["generator"] if "generator" in blob else blob.get("model", blob)
        pipeline.generator.load_state_dict(gen_sd, strict=True)

    from utils.lora_utils import configure_lora_for_model
    import peft
    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )
        lora_ckpt = getattr(config, "lora_ckpt", None)
        if lora_ckpt:
            lora_blob = torch.load(lora_ckpt, map_location="cpu")
            lora_sd = lora_blob["generator_lora"] if "generator_lora" in lora_blob else lora_blob
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_sd)

    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    motion_encoder = load_motion_encoder(config, pipeline.vae, device, dtype)
    motion_encoder.eval()

    # ---- Motion encoding ----
    ref_len = getattr(config.motion_encoder, "ref_length", 16)
    ref_size = tuple(getattr(config.motion_encoder, "image_size", (480, 832)))
    motion_video = _load_motion_video(args.motion_ref, ref_len, ref_size).unsqueeze(0)
    motion_video = motion_video.to(device=device, dtype=dtype)
    with torch.no_grad():
        motion_tokens = motion_encoder(motion_video)
    print(f"[motion] tokens shape: {tuple(motion_tokens.shape)}")

    # ---- Run inference ----
    prompts = [args.prompt] * args.num_samples
    sampled_noise = torch.randn(
        [args.num_samples, args.num_frames, 16, 60, 104],
        device=device, dtype=dtype,
    )
    with torch.no_grad():
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            motion_tokens=motion_tokens,
            return_latents=True,
        )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    video = (video.clamp(0, 1) * 255).to(torch.uint8)
    for i in range(video.shape[0]):
        out_path = args.output if args.num_samples == 1 else args.output.replace(
            ".mp4", f"_{i}.mp4")
        write_video(out_path, video[i].permute(0, 2, 3, 1).cpu(), fps=16)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
