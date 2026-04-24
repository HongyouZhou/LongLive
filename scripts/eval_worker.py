"""Persistent single-GPU eval worker for Motion-Recache.

Design
------
One process per A40 (set ``CUDA_VISIBLE_DEVICES`` before launch). After
boot it emits ``{"status": "ready"}`` on stdout. It then reads one
line-delimited JSON request at a time from stdin, and emits one
response per request. The process exits on EOF.

Request schema::

    {
      "ckpt_path": "/abs/path/to/checkpoint_model_XXXXXX/model.pt",
      "prompt": "...",
      "motion_ref": "/abs/path/to/motion.mp4",
      "seed": 42,
      "output_mp4": "/abs/path/to/out.mp4",
      "num_frames": 81        # optional, default 81
    }

Response schema on success::

    {
      "ok": true,
      "output_mp4": "...",
      "ckpt_step": 2400,
      "metrics": {"gen_flow_mag": ..., "ref_flow_mag": ..., "flow_abs_diff": ..., "flow_rel_diff": ...},
      "runtime_s": 28.5
    }

Response on failure::

    {"ok": false, "error": "description", "request": {...}}

Model load happens once; LoRA + motion_encoder weights hot-reload when
``ckpt_path`` changes (Wan base generator stays in memory).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from typing import Optional

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
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

from pipeline import CausalInferencePipeline  # noqa: E402
from model.motion_encoder import MotionEncoder  # noqa: E402
from utils.dataset import _load_motion_video  # noqa: E402
from utils.misc import set_seed  # noqa: E402
from utils.lora_utils import configure_lora_for_model  # noqa: E402
from utils.optical_flow import compute_flow_l2_vs_reference  # noqa: E402
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
        self._current_ckpt_path: Optional[str] = None
        self._current_ckpt_step: Optional[int] = None
        self._build_models()

    def _build_models(self) -> None:
        _log("building pipeline (generator + text_encoder + VAE)...")
        t0 = time.time()
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)

        if self.config.generator_ckpt:
            _log(f"loading base generator weights from {self.config.generator_ckpt}")
            blob = torch.load(self.config.generator_ckpt, map_location="cpu")
            gen_sd = blob["generator"] if "generator" in blob else blob.get("model", blob)
            self.pipeline.generator.load_state_dict(gen_sd, strict=True)

        if getattr(self.config, "adapter", None):
            _log("wrapping generator with LoRA adapter (weights loaded per-request)")
            self.pipeline.generator.model = configure_lora_for_model(
                self.pipeline.generator.model,
                model_name="generator",
                lora_config=self.config.adapter,
                is_main_process=True,
            )

        self.pipeline = self.pipeline.to(dtype=self.dtype)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

        _log("building motion_encoder (weights loaded per-request)")
        motion_cfg = self.config.motion_encoder
        self.motion_encoder = MotionEncoder(
            vae_wrapper=self.pipeline.vae,
            dim=getattr(motion_cfg, "dim", 4096),
            num_layers=getattr(motion_cfg, "num_layers", 4),
            num_heads=getattr(motion_cfg, "num_heads", 16),
            tokens_per_frame=getattr(motion_cfg, "tokens_per_frame", 64),
            max_tokens=getattr(motion_cfg, "max_tokens", 512),
        ).to(device=self.device, dtype=self.dtype)
        self.motion_encoder.eval()
        self.ref_len = getattr(motion_cfg, "ref_length", 16)
        self.ref_size = tuple(getattr(motion_cfg, "image_size", (480, 832)))

        _log(f"boot complete in {time.time()-t0:.1f}s")

    def _reload_ckpt(self, ckpt_path: str) -> None:
        if ckpt_path == self._current_ckpt_path:
            return
        _log(f"hot-reloading LoRA + motion_encoder from {ckpt_path}")
        t0 = time.time()
        blob = torch.load(ckpt_path, map_location="cpu")
        if "generator_lora" not in blob:
            raise ValueError(
                f"checkpoint {ckpt_path} missing 'generator_lora' key "
                f"(found {list(blob.keys())})")
        peft.set_peft_model_state_dict(
            self.pipeline.generator.model, blob["generator_lora"])

        if "motion_encoder" not in blob:
            raise ValueError(
                f"checkpoint {ckpt_path} missing 'motion_encoder' key "
                f"(found {list(blob.keys())})")
        self.motion_encoder.load_state_dict(blob["motion_encoder"], strict=True)
        self.motion_encoder.to(device=self.device, dtype=self.dtype)
        self.motion_encoder.eval()

        self._current_ckpt_path = ckpt_path
        self._current_ckpt_step = int(blob.get("step", -1))
        _log(f"reload done in {time.time()-t0:.1f}s (step={self._current_ckpt_step})")

    @torch.no_grad()
    def _encode_motion(self, motion_ref_path: str) -> torch.Tensor:
        video = _load_motion_video(motion_ref_path, self.ref_len, self.ref_size)
        video = video.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return self.motion_encoder(video)

    @torch.no_grad()
    def handle(self, req: dict) -> dict:
        t0 = time.time()
        self._reload_ckpt(req["ckpt_path"])

        seed = int(req.get("seed", 0))
        set_seed(seed)

        motion_tokens = self._encode_motion(req["motion_ref"])

        num_frames = int(req.get("num_frames", 81))
        noise = torch.randn(
            [1, num_frames, 16, 60, 104],
            device=self.device, dtype=self.dtype,
        )
        video, _ = self.pipeline.inference(
            noise=noise,
            text_prompts=[req["prompt"]],
            motion_tokens=motion_tokens,
            return_latents=True,
        )
        video = (video.clamp(0, 1) * 255).to(torch.uint8)  # [B, T, 3, H, W]
        out_path = req["output_mp4"]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        gen_chw = video[0].permute(0, 2, 3, 1).cpu()  # [T, H, W, 3]
        write_video(out_path, gen_chw, fps=16)

        # Optical flow metric (Farneback on CPU, uses the uint8 frames)
        ref_frames = _load_motion_video(
            req["motion_ref"], self.ref_len, self.ref_size)  # [-1,1] float
        ref_uint8 = ((ref_frames.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        ref_thwc = ref_uint8.permute(0, 2, 3, 1)
        metrics = compute_flow_l2_vs_reference(gen_chw, ref_thwc)

        return {
            "ok": True,
            "output_mp4": out_path,
            "ckpt_step": self._current_ckpt_step,
            "metrics": metrics,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    worker = EvalWorker(args.config_path)
    worker.loop()


if __name__ == "__main__":
    main()
