"""
Read-only diagnostic: estimate per-video head yaw range / angular velocity over
the cross-pair training motion_refs, to bound how much head amplitude the data
permits. No training touched.

Pipeline:
    motion_refs/*.mp4 -> sample N frames -> MediaPipe FaceLandmarker (tasks API)
    -> facial_transformation_matrix -> yaw/pitch/roll (deg) -> per-video stats
    -> dataset histogram.

Outputs JSON {video: {yaw_range, yaw_p95_step, ...}} and prints summary.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def rot_to_euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Returns (pitch_x, yaw_y, roll_z) in degrees from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def head_pose_series(video_path: Path, max_frames: int, landmarker) -> list[tuple[float, float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    out = []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0])  # 4x4
                R = M[:3, :3]
                pitch, yaw, roll = rot_to_euler_yxz(R)
                out.append((yaw, pitch, roll))
        f_idx += 1
    cap.release()
    return out


def per_video_stats(series: list[tuple[float, float, float]]) -> dict:
    if len(series) < 2:
        return {"n": len(series)}
    arr = np.array(series, dtype=np.float64)
    yaw = arr[:, 0]
    pitch = arr[:, 1]
    yaw_diff = np.abs(np.diff(yaw))
    return {
        "n": int(len(series)),
        "yaw_min": float(yaw.min()),
        "yaw_max": float(yaw.max()),
        "yaw_range": float(yaw.max() - yaw.min()),
        "yaw_std": float(yaw.std()),
        "yaw_step_p50": float(np.percentile(yaw_diff, 50)),
        "yaw_step_p95": float(np.percentile(yaw_diff, 95)),
        "yaw_step_max": float(yaw_diff.max()),
        "pitch_range": float(pitch.max() - pitch.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/hongyou/dev/data/wm/prompts/motion_pairs_cross_train.jsonl")
    ap.add_argument("--root", default="/home/hongyou/dev/data/wm/motion_refs")
    ap.add_argument("--model", default="scripts/_models/face_landmarker.task")
    ap.add_argument("--out", default="logs/diag_head_pose_cross_train.json")
    ap.add_argument("--max-frames-per-video", type=int, default=24,
                    help="Sampled frames per video (16 ref @ training)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos")
    args = ap.parse_args()

    refs = []
    seen = set()
    with open(args.jsonl) as f:
        for line in f:
            entry = json.loads(line)
            for key in ("motion_a", "motion_b"):
                v = entry.get(key)
                if v and v not in seen:
                    seen.add(v)
                    refs.append(v)
    if args.limit > 0:
        refs = refs[: args.limit]
    print(f"[diag] {len(refs)} unique motion_ref videos", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    base = mp_python.BaseOptions(model_asset_path=args.model)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )

    results = {}
    n_no_face = 0
    with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
        for i, name in enumerate(refs):
            p = Path(args.root) / name
            if not p.exists():
                continue
            try:
                series = head_pose_series(p, args.max_frames_per_video, landmarker)
            except Exception as e:
                print(f"[diag] err {name}: {e}", file=sys.stderr)
                series = []
            if len(series) < 2:
                n_no_face += 1
                results[name] = {"n": len(series)}
                continue
            results[name] = per_video_stats(series)
            if (i + 1) % 50 == 0:
                print(f"[diag] {i+1}/{len(refs)} processed", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    yaw_ranges = np.array([v["yaw_range"] for v in results.values() if "yaw_range" in v])
    yaw_p95s = np.array([v["yaw_step_p95"] for v in results.values() if "yaw_step_p95" in v])
    yaw_maxs = np.array([v["yaw_step_max"] for v in results.values() if "yaw_step_max" in v])
    pitch_ranges = np.array([v["pitch_range"] for v in results.values() if "pitch_range" in v])
    print()
    print(f"videos with face landmarks: {len(yaw_ranges)} / {len(refs)}  (no-face: {n_no_face})")
    print(f"  yaw_range          (deg, per-video peak-to-peak)")
    print(f"    p10/p50/p90    : {np.percentile(yaw_ranges,10):.1f} / {np.percentile(yaw_ranges,50):.1f} / {np.percentile(yaw_ranges,90):.1f}")
    print(f"    p95/max        : {np.percentile(yaw_ranges,95):.1f} / {yaw_ranges.max():.1f}")
    print(f"  yaw step P95     (deg, between adjacent sampled frames)")
    print(f"    p50/p95/max    : {np.percentile(yaw_p95s,50):.2f} / {np.percentile(yaw_p95s,95):.2f} / {yaw_p95s.max():.2f}")
    print(f"  yaw step max     (deg)")
    print(f"    p50/p95/max    : {np.percentile(yaw_maxs,50):.2f} / {np.percentile(yaw_maxs,95):.2f} / {yaw_maxs.max():.2f}")
    print(f"  pitch_range  p50/p95: {np.percentile(pitch_ranges,50):.1f} / {np.percentile(pitch_ranges,95):.1f}")
    print()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
