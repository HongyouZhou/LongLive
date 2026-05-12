"""Three CLIP-based metrics aligned to LOVEU-TGVE 2023 official ``run_eval.py``.

MotionDirector (arXiv:2310.08465) Tables 1 & 2 use exactly this metric trio
(see paper §4.2 + A.2). LOVEU-TGVE's reference implementation lives at:
    https://github.com/showlab/loveu-tgve-2023/blob/main/scripts/run_eval.py

Metrics (all return raw CLIP logits / scaled scores — NOT cosine similarities;
this matches paper Table reporting convention):

1. ``clip_score_text(frames, prompt)`` — "Appearance Diversity" / text alignment.
   CLIP ViT-L/14 image-text logits per frame, mean over frames.

2. ``clip_score_frame(frames)`` — "Temporal Consistency".
   CLIP ViT-L/14 image features per frame, off-diagonal mean of pairwise
   cosine similarity matrix.

3. ``pick_score(frames, prompt)`` — PickScore (Kirstain et al. 2023).
   PickScore_v1 (CLIP-H/14 fine-tune) image-text score per frame,
   mean over frames. No softmax (we report absolute scores, not pairwise
   A/B preferences).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

CLIP_L14 = "openai/clip-vit-large-patch14"
PICK_MODEL = "yuvalkirstain/PickScore_v1"
PICK_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class CLIPMetrics:
    """Lazy-loaded CLIP + PickScore models for the three LOVEU-style metrics."""

    def __init__(self, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self._clip = None
        self._clip_proc = None
        self._pick = None
        self._pick_proc = None

    # ------ lazy loaders ------
    def _ensure_clip(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPModel, AutoProcessor
        self._clip_proc = AutoProcessor.from_pretrained(CLIP_L14)
        self._clip = CLIPModel.from_pretrained(CLIP_L14).to(self.device, dtype=self.dtype).eval()

    def _ensure_pick(self) -> None:
        if self._pick is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self._pick_proc = AutoProcessor.from_pretrained(PICK_PROCESSOR)
        self._pick = AutoModel.from_pretrained(PICK_MODEL).to(self.device, dtype=self.dtype).eval()

    # ------ metrics ------
    @torch.no_grad()
    def clip_score_text(self, frames: list, prompt: str) -> float:
        """LOVEU's ``clip_score_text``: mean per-frame CLIP image-text logit."""
        self._ensure_clip()
        inputs = self._clip_proc(text=[prompt], images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Cast image pixel values to model dtype; keep input_ids as int.
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        logits = self._clip(**inputs).logits_per_image.detach().cpu().float().numpy()
        return float(logits.mean())

    @torch.no_grad()
    def clip_score_frame(self, frames: list) -> float:
        """LOVEU's ``clip_score_frame``: off-diagonal mean of pairwise frame cosine sim."""
        self._ensure_clip()
        inputs = self._clip_proc(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        feats = self._clip.get_image_features(**inputs).detach().cpu().float().numpy()
        # Pairwise cosine = (X / ||X||) @ (X / ||X||).T after L2-normalising rows.
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        unit = feats / norms
        sim = unit @ unit.T
        np.fill_diagonal(sim, 0.0)
        n = len(frames)
        if n < 2:
            return 0.0
        return float(sim.sum() / (n * (n - 1)))

    @torch.no_grad()
    def pick_score(self, frames: list, prompt: str) -> float:
        """LOVEU's ``pick_score``: mean per-frame PickScore (image-text similarity)."""
        self._ensure_pick()
        img_in = self._pick_proc(
            images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt",
        )
        txt_in = self._pick_proc(
            text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt",
        )
        img_in = {k: v.to(self.device) for k, v in img_in.items()}
        txt_in = {k: v.to(self.device) for k, v in txt_in.items()}
        if "pixel_values" in img_in:
            img_in["pixel_values"] = img_in["pixel_values"].to(self.dtype)

        ie = self._pick.get_image_features(**img_in)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = self._pick.get_text_features(**txt_in)
        te = te / te.norm(dim=-1, keepdim=True)
        scale = self._pick.logit_scale.exp()
        s = (scale * (te @ ie.T)[0]).detach().cpu().float().numpy()
        return float(s.mean())

    def score_video(self, frames_pil: list, prompt: Optional[str]) -> dict:
        """Compute all three metrics on one video's frames (PIL list).

        Returns ``{app_div, temp_consist, pick_score}``. If ``prompt`` is None,
        only ``temp_consist`` is computed (others require text).
        """
        out: dict[str, float] = {
            "temp_consist": self.clip_score_frame(frames_pil),
        }
        if prompt is not None:
            out["app_div"] = self.clip_score_text(frames_pil, prompt)
            out["pick_score"] = self.pick_score(frames_pil, prompt)
        return out


def _smoke():
    """Standalone smoke test: ``python -m scripts.motion_eval.metrics.clip_metrics``."""
    import argparse
    from .video_io import read_video_frames, frames_to_pil

    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()

    frames = read_video_frames(args.mp4)
    pil = frames_to_pil(frames)
    metrics = CLIPMetrics()
    out = metrics.score_video(pil, args.prompt)
    for k, v in out.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    _smoke()
