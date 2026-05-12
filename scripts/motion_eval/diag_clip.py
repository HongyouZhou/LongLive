"""Diagnostic: compare LOVEU's official run_eval.py CLIP-Text path against
``CLIPMetrics.clip_score_text``.

LOVEU-TGVE 2023 ``scripts/run_eval.py`` computes:

    outputs = model(**inputs)
    score   = outputs.logits_per_image.mean()

We compute the equivalent via ``get_image_features`` / ``get_text_features``
+ manual logit construction (so that the same code path can be reused for
PickScore, which has no ``logits_per_image`` convenience attribute).

The two paths should be numerically identical up to fp32 rounding. If
``clip_score_text`` falls behind paper-reported magnitudes (cosines ~0.01
instead of 0.20+), this script tells you whether to blame the metric
implementation or the upstream model output.

Usage::

    python scripts/motion_eval/diag_clip.py \\
        --mp4 /path/to/video.mp4 \\
        --prompt "the text prompt"

Suggested probes (post baseline_v1 — pick one high- and one low- app_div):

    # high app_div in smoke (4.30)
    --mp4 $LL_DATA/motion_eval_runs/baseline_v1_8893601/videos/012f3ca5.mp4 \\
    --prompt "Tap Nixon shooting a basketball while a woman helps him, high-contrast style."

    # low app_div in smoke (-6.15)
    --mp4 $LL_DATA/motion_eval_runs/baseline_v1_8893601/videos/001b64a6.mp4 \\
    --prompt "A ship sails on the sea during sunset, 2D vector art."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, CLIPModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.clip_metrics import CLIP_L14, CLIPMetrics  # noqa: E402
from metrics.video_io import frames_to_pil, read_video_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="Subsample every Nth frame (default 1 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    frames = read_video_frames(args.mp4, stride=args.stride)
    pil = frames_to_pil(frames)
    print(f"[diag] frames loaded   : {len(pil)} (stride={args.stride}) from {args.mp4}")
    print(f"[diag] prompt          : {args.prompt!r}")
    print()

    # -------- Official LOVEU path: model(**inputs).logits_per_image --------
    proc = AutoProcessor.from_pretrained(CLIP_L14)
    model = CLIPModel.from_pretrained(CLIP_L14).to(args.device, dtype=torch.float32).eval()

    inputs = proc(
        text=[args.prompt], images=pil, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    official_logits = out.logits_per_image  # (N_frames, 1)
    official_mean = official_logits.mean().item()

    print("== Official LOVEU path  (model(**inputs).logits_per_image) ==")
    print(f"   shape               : {tuple(official_logits.shape)}")
    print(f"   mean                : {official_mean:.6f}")
    print(f"   min / max           : {official_logits.min().item():.4f} / "
          f"{official_logits.max().item():.4f}")
    print()

    # -------- Manual reconstruction (clip_score_text body) --------
    m = CLIPMetrics(device=args.device, dtype=torch.float32)
    m._ensure_clip()

    img_in = m._clip_proc(images=pil, return_tensors="pt")
    txt_in = m._clip_proc(
        text=[args.prompt], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    img_in = {k: v.to(args.device) for k, v in img_in.items()}
    txt_in = {k: v.to(args.device) for k, v in txt_in.items()}

    with torch.no_grad():
        ie = m._embed_image(m._clip, **img_in)
        te = m._embed_text(m._clip, **txt_in)
        ie_norm = ie / (ie.norm(dim=-1, keepdim=True) + 1e-12)
        te_norm = te / (te.norm(dim=-1, keepdim=True) + 1e-12)
        scale = m._clip.logit_scale.exp()
        cos = ie_norm @ te_norm.T  # (N_frames, 1)
        manual_logits = scale * cos

    print("== Manual reconstruction (CLIPMetrics._embed_* + scale * (ie @ te.T)) ==")
    print(f"   image embed shape   : {tuple(ie.shape)}")
    print(f"   text  embed shape   : {tuple(te.shape)}")
    print(f"   logit_scale.exp()   : {scale.item():.4f}")
    print(f"   cos mean / min / max: {cos.mean().item():.4f} / "
          f"{cos.min().item():.4f} / {cos.max().item():.4f}")
    print(f"   logit mean          : {manual_logits.mean().item():.6f}")
    print(f"   logit min / max     : {manual_logits.min().item():.4f} / "
          f"{manual_logits.max().item():.4f}")
    print()

    # -------- High-level API call (what run_eval.py actually invokes) --------
    api_score = m.clip_score_text(pil, args.prompt)
    print(f"== CLIPMetrics.clip_score_text(pil, prompt)  →  {api_score:.6f}")
    print()

    # -------- Verdict --------
    delta = abs(official_mean - api_score)
    print(f"[delta] official - api  = {delta:.6f}")
    if delta < 1e-3:
        print("[verdict] IDENTICAL → clip_score_text is faithful to LOVEU's official path.")
        print("          Low app_div values reflect genuine LongLive output↔prompt alignment.")
    else:
        print("[verdict] DIVERGENT → bug in _embed_image / _embed_text path.")
        print("          Investigate clip_metrics.py compat shims.")


if __name__ == "__main__":
    main()
