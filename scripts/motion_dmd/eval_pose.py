"""Evaluate motion-LoRA outputs by pose similarity + CLIP text alignment.

Pose backend: mediapipe (lightweight, pip install mediapipe). Outputs 33 body
landmarks per frame; we use a subset of 17 stable body keypoints for distance.
A DWPose backend can be plugged in later via --backend dwpose.

Distance metric: per-frame Procrustes-aligned L2 over 17 keypoints, then DTW
along the temporal axis to handle small speed differences. Lower = better.

CLIP text alignment: per-frame openai/clip-vit-base-patch32 image-text cosine
similarity, averaged across frames. Higher = better.

Output: one JSON per line, e.g.
  {"reference": "ref.mp4", "generated": "gen.mp4", "prompt": "...",
   "pose_dist": 0.234, "clip_text": 0.272, "n_frames_paired": 81}

Usage:
  python scripts/motion_lora/eval_pose.py \\
      --reference $LL_DATA/motion_refs/celebv_X.mp4 \\
      --generated_dir logs/motion_lora_walking_v1/inference/ \\
      --prompts_jsonl logs/motion_lora_walking_v1/inference/prompts.jsonl \\
      --output logs/motion_lora_walking_v1/inference/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pose extraction (mediapipe backend)
# ---------------------------------------------------------------------------

# 17 keypoint indices from MediaPipe Pose's 33-point output that match the
# COCO body skeleton (excluding face landmarks 1-10 except nose, plus hand/foot
# extensions). Using a stable subset keeps the L2 robust to small mediapipe
# detection jitter on face landmarks that aren't motion-relevant.
COCO_FROM_MEDIAPIPE = [
    0,   # nose
    11, 12,  # shoulders L/R
    13, 14,  # elbows L/R
    15, 16,  # wrists L/R
    23, 24,  # hips L/R
    25, 26,  # knees L/R
    27, 28,  # ankles L/R
    29, 30,  # heels L/R
    31, 32,  # foot indices L/R
]
N_KEYPOINTS = len(COCO_FROM_MEDIAPIPE)  # 17


def extract_pose_sequence(video_path: str) -> np.ndarray:
    """Return [T, 17, 2] array of normalized (x, y) keypoints in [0, 1].
    Frames where pose detection fails get NaN.
    """
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    out = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)
        if res.pose_landmarks is None:
            out.append(np.full((N_KEYPOINTS, 2), np.nan, dtype=np.float32))
            continue
        lm = res.pose_landmarks.landmark
        kp = np.array(
            [[lm[i].x, lm[i].y] for i in COCO_FROM_MEDIAPIPE],
            dtype=np.float32,
        )
        out.append(kp)
    cap.release()
    pose.close()
    return np.stack(out, axis=0) if out else np.zeros((0, N_KEYPOINTS, 2),
                                                       dtype=np.float32)


# ---------------------------------------------------------------------------
# Procrustes alignment (similarity transform: rotation + scale + translation)
# ---------------------------------------------------------------------------

def procrustes_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src [N, 2] to dst [N, 2] by similarity transform; return aligned src.
    Uses Kabsch + isotropic scale. NaN-aware: only fits on rows present in both.
    """
    mask = ~(np.isnan(src).any(axis=1) | np.isnan(dst).any(axis=1))
    if mask.sum() < 3:
        return src.copy()
    s = src[mask]
    d = dst[mask]
    s_mean, d_mean = s.mean(axis=0), d.mean(axis=0)
    s0 = s - s_mean
    d0 = d - d_mean
    norm_s = np.linalg.norm(s0)
    norm_d = np.linalg.norm(d0)
    if norm_s < 1e-8:
        return src.copy()
    s_norm = s0 / norm_s
    d_norm = d0 / norm_d
    U, _, Vt = np.linalg.svd(d_norm.T @ s_norm)
    R = U @ Vt
    scale = norm_d / norm_s
    aligned = (src - s_mean) @ R.T * scale + d_mean
    return aligned


def per_frame_pose_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Procrustes-align each frame of seq_a to seq_b and return per-frame L2.
    Both seq_a and seq_b are [T, N, 2]; lengths must match (caller's responsibility).
    Returns [T] L2 distances; frames where alignment failed → NaN.
    """
    T = min(len(seq_a), len(seq_b))
    out = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        if np.isnan(seq_a[t]).any() or np.isnan(seq_b[t]).any():
            continue
        aligned = procrustes_align(seq_a[t], seq_b[t])
        out[t] = float(np.linalg.norm(aligned - seq_b[t]))
    return out


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Dynamic-time-warping distance over per-frame Procrustes-aligned L2.
    Cost(i, j) = || procrustes(seq_a[i]) - seq_b[j] ||.
    Returns normalized total cost (path-length normalized).
    """
    Ta, Tb = len(seq_a), len(seq_b)
    if Ta == 0 or Tb == 0:
        return float("nan")

    # Pre-compute cost matrix [Ta, Tb]
    cost = np.full((Ta, Tb), np.inf, dtype=np.float32)
    for i in range(Ta):
        for j in range(Tb):
            if np.isnan(seq_a[i]).any() or np.isnan(seq_b[j]).any():
                continue
            aligned = procrustes_align(seq_a[i], seq_b[j])
            cost[i, j] = float(np.linalg.norm(aligned - seq_b[j]))

    if not np.isfinite(cost).any():
        return float("nan")
    median_cost = np.nanmedian(cost[np.isfinite(cost)])
    cost[~np.isfinite(cost)] = median_cost  # fill NaN frames with median

    # DTW
    D = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            D[i, j] = cost[i - 1, j - 1] + min(
                D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
            )
    return float(D[Ta, Tb] / (Ta + Tb))


# ---------------------------------------------------------------------------
# CLIP text alignment
# ---------------------------------------------------------------------------

def clip_text_alignment(video_path: str, prompt: str,
                         model=None, processor=None) -> tuple[float, int]:
    """Per-frame CLIP image-text cosine, averaged. Returns (mean_sim, n_frames)."""
    import torch
    import cv2
    if model is None or processor is None:
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            model = model.cuda()

    device = next(model.parameters()).device

    cap = cv2.VideoCapture(video_path)
    sims = []
    # Sample every 4th frame to save compute (still ~5 fps at 16 fps source)
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % 4 == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(
                text=[prompt], images=frame_rgb, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
            sim = (img_emb @ txt_emb.T).squeeze().item()
            sims.append(sim)
        frame_idx += 1
    cap.release()
    if not sims:
        return float("nan"), 0
    return float(np.mean(sims)), len(sims)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True,
                    help="path to reference mp4 (the one motion-LoRA was trained on)")
    ap.add_argument("--generated_dir", required=True,
                    help="dir containing generated mp4s (e.g. .../inference/)")
    ap.add_argument("--prompts_jsonl", default=None,
                    help="optional: jsonl mapping mp4 basename -> prompt for CLIP-text alignment. "
                         "If omitted, only pose distance is computed")
    ap.add_argument("--output", required=True,
                    help="output jsonl with per-(ref, gen) scores")
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe"],
                    help="pose backend; only mediapipe supported in v1")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    gen_dir = Path(args.generated_dir)
    out_path = Path(args.output)
    if not ref_path.exists():
        sys.exit(f"reference not found: {ref_path}")
    if not gen_dir.exists():
        sys.exit(f"generated_dir not found: {gen_dir}")

    print(f"[eval] extracting pose from reference {ref_path.name} ...", flush=True)
    ref_pose = extract_pose_sequence(str(ref_path))
    print(f"[eval] reference pose seq shape: {ref_pose.shape}", flush=True)

    prompt_map: dict[str, str] = {}
    if args.prompts_jsonl:
        with open(args.prompts_jsonl) as f:
            for line in f:
                row = json.loads(line)
                prompt_map[row["video"]] = row["prompt"]
        print(f"[eval] loaded {len(prompt_map)} prompt mappings", flush=True)

    # Lazy-load CLIP only if we'll use it
    clip_model = clip_proc = None
    if prompt_map:
        from transformers import CLIPModel, CLIPProcessor
        import torch
        model_name = "openai/clip-vit-base-patch32"
        clip_proc = CLIPProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_path, "w")

    gen_files = sorted(p for p in gen_dir.iterdir() if p.suffix == ".mp4")
    print(f"[eval] {len(gen_files)} generated mp4s in {gen_dir}", flush=True)

    for gen in gen_files:
        print(f"[eval] {gen.name} ...", flush=True)
        gen_pose = extract_pose_sequence(str(gen))
        if gen_pose.shape[0] == 0:
            print(f"  failed to read frames; skipping", flush=True)
            continue
        d = dtw_distance(ref_pose, gen_pose)

        prompt = prompt_map.get(gen.name)
        clip_score, n_clip = (float("nan"), 0)
        if prompt and clip_model is not None:
            clip_score, n_clip = clip_text_alignment(
                str(gen), prompt, model=clip_model, processor=clip_proc
            )

        row = {
            "reference": ref_path.name,
            "generated": gen.name,
            "prompt": prompt,
            "pose_dist": d,
            "clip_text": clip_score,
            "n_frames_paired": int(min(len(ref_pose), len(gen_pose))),
            "n_clip_frames": n_clip,
        }
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()
        print(f"  pose_dist={d:.4f}  clip_text={clip_score:.4f}", flush=True)

    f_out.close()
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
