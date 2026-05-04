"""Persistent single-GPU eval worker for VBench short generation.

Design
------
One process per GPU (set ``CUDA_VISIBLE_DEVICES`` before launch). After
boot it emits ``{"status": "ready"}`` on stdout and then reads one
line-delimited JSON request at a time from stdin, emitting one response
per request. Process exits on EOF.

Request schema::

    {
      "prompt": "...",
      "seed": 0,
      "output_mp4": "/abs/path/to/out.mp4"
    }

Response on success::

    {"ok": true, "output_mp4": "...", "runtime_s": 5.2}

Response on failure::

    {"ok": false, "error": "...", "traceback": "...", "request": "..."}

Model + LoRA load happens once at boot; all requests in a dispatch run
share the same checkpoint (VBench evaluates a single ckpt at a time).

Mirrors ``scripts/local/inference.py``'s ckpt + LoRA loading sequence so the generated
videos are byte-identical to what NVlabs's stock inference would produce
for the same prompt + seed + ckpt.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Dispatcher reads line-delimited JSON from our stdout. Pipelines and peft
# helpers are chatty; redirect every stray print to stderr while keeping
# stdout exclusive for our response protocol.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


@contextlib.contextmanager
def _stdout_for_response():
    prev, sys.stdout = sys.stdout, _REAL_STDOUT
    try:
        yield
    finally:
        sys.stdout = prev


from longlive.pipeline import CausalInferencePipeline  # noqa: E402
from longlive.utils.misc import set_seed  # noqa: E402
from longlive.utils.memory import DynamicSwapInstaller  # noqa: E402
from longlive.utils.lora_utils import configure_lora_for_model  # noqa: E402
import peft  # noqa: E402


def _log(msg: str) -> None:
    """Diagnostics go to stderr so stdout stays clean for JSON responses."""
    print(f"[worker pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


class EvalWorker:
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.config.distributed = False
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        _log("building CausalInferencePipeline (generator + text_encoder + VAE)...")
        t0 = time.time()
        torch.set_grad_enabled(False)
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)

        # Load base generator weights (same logic as scripts/local/inference.py:73-95).
        if self.config.generator_ckpt:
            _log(f"loading base generator from {self.config.generator_ckpt}")
            sd = torch.load(self.config.generator_ckpt, map_location="cpu")
            if "generator" in sd or "generator_ema" in sd:
                gen_sd = sd["generator_ema" if self.config.use_ema else "generator"]
            elif "model" in sd:
                gen_sd = sd["model"]
            else:
                raise ValueError(
                    f"generator state dict not found in {self.config.generator_ckpt}; "
                    f"keys={list(sd.keys())}")
            if self.config.use_ema:
                clean = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in gen_sd.items()}
                missing, unexpected = self.pipeline.generator.load_state_dict(clean, strict=False)
                if missing:
                    _log(f"missing keys (head 8): {missing[:8]}")
                if unexpected:
                    _log(f"unexpected keys (head 8): {unexpected[:8]}")
            else:
                self.pipeline.generator.load_state_dict(gen_sd)

        # Apply LoRA + load LoRA weights (mirrors scripts/local/inference.py:97-131).
        self.pipeline.is_lora_enabled = False
        if getattr(self.config, "adapter", None):
            _log("wrapping generator with LoRA adapter")
            self.pipeline.generator.model = configure_lora_for_model(
                self.pipeline.generator.model,
                model_name="generator",
                lora_config=self.config.adapter,
                is_main_process=True,
            )
            lora_path = getattr(self.config, "lora_ckpt", None)
            if lora_path:
                _log(f"loading LoRA weights from {lora_path}")
                blob = torch.load(lora_path, map_location="cpu")
                if isinstance(blob, dict) and "generator_lora" in blob:
                    peft.set_peft_model_state_dict(
                        self.pipeline.generator.model, blob["generator_lora"])
                else:
                    peft.set_peft_model_state_dict(
                        self.pipeline.generator.model, blob)
            self.pipeline.is_lora_enabled = True

        # Move to device + dtype (matches scripts/local/inference.py:134-139).
        self.pipeline = self.pipeline.to(dtype=self.dtype)
        # Always low-memory on eval pool; 8 workers × ~5GB gen + critic budget.
        DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

        _log(f"boot complete in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def handle(self, req: dict) -> dict:
        t0 = time.time()
        prompt = req["prompt"]
        seed = int(req.get("seed", 0))
        out_path = req["output_mp4"]
        set_seed(seed)

        # Latent shape mirrors scripts/local/inference.py:194-196 — fixed at the model's
        # canonical [B, T_lat, 16, 60, 104] (480×832 pixel after VAE decode).
        T_lat = int(self.config.num_output_frames)
        noise = torch.randn(
            [1, T_lat, 16, 60, 104], device=self.device, dtype=self.dtype)

        video, _ = self.pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
            return_latents=True,
            low_memory=True,
            profile=False,
        )
        # video is [B, T, C, H, W] in [0,1]. write_video wants [T, H, W, C] uint8.
        frames = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu())
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        write_video(out_path, frames[0].to(torch.uint8), fps=16)
        self.pipeline.vae.model.clear_cache()

        return {
            "ok": True,
            "output_mp4": out_path,
            "runtime_s": round(time.time() - t0, 2),
        }

    def _emit(self, obj: dict) -> None:
        with _stdout_for_response():
            print(json.dumps(obj, ensure_ascii=False), flush=True)

    def loop(self) -> None:
        self._emit({"status": "ready"})
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
                resp = self.handle(req)
            except Exception as e:
                resp = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "request": raw,
                }
            self._emit(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    args = ap.parse_args()
    worker = EvalWorker(args.config_path)
    worker.loop()


if __name__ == "__main__":
    main()
