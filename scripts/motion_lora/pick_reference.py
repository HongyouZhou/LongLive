"""Pick candidate reference videos from OpenVid-1M for motion-LoRA training.

Filters the OpenVid-1M.csv by caption keyword(s) + score thresholds + camera
motion, intersects with the actual mp4 files present in `motion_refs/`, and
emits top-K candidates ranked by aesthetic × motion score.

Output: one JSON object per line, e.g.
  {"video": "celebv_X.mp4", "caption": "...", "motion": 4.21, "aesthetic": 5.8,
   "camera": "static", "seconds": 4.7, "fps": 29.97, "path": "/full/path"}

Usage:
  python scripts/motion_lora/pick_reference.py \\
      --keyword walking --keyword person \\
      --camera_motion static \\
      --motion_min 2.5 --motion_max 7.0 --aesthetic_min 5.0 \\
      --seconds_min 3.0 --seconds_max 8.0 \\
      --refs_dir $LL_DATA/motion_refs \\
      --csv     $LL_DATA/meta/data/train/OpenVid-1M.csv \\
      --top_k 20 \\
      --output candidates_walking.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="path to OpenVid-1M.csv")
    ap.add_argument("--refs_dir", required=True,
                    help="path to motion_refs/ (mp4 files actually present)")
    ap.add_argument("--keyword", action="append", default=[],
                    help="caption substring(s); ALL must appear (case-insensitive)")
    ap.add_argument("--keyword_any", action="append", default=[],
                    help="any of these substrings must appear")
    ap.add_argument("--exclude", action="append", default=[],
                    help="caption substring(s) to EXCLUDE (case-insensitive)")
    ap.add_argument("--camera_motion", default=None,
                    help="exact match for camera motion column "
                         "(static/pan_left/pan_right/zoom_in/zoom_out/tilt_up/...)")
    ap.add_argument("--motion_min", type=float, default=0.0)
    ap.add_argument("--motion_max", type=float, default=1e9)
    ap.add_argument("--aesthetic_min", type=float, default=0.0)
    ap.add_argument("--aesthetic_max", type=float, default=1e9)
    ap.add_argument("--seconds_min", type=float, default=0.0)
    ap.add_argument("--seconds_max", type=float, default=1e9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--output", default=None,
                    help="output JSONL path; if omitted, prints to stdout")
    args = ap.parse_args()

    refs_dir = Path(args.refs_dir).resolve()
    if not refs_dir.exists():
        raise FileNotFoundError(refs_dir)
    present = set(p.name for p in refs_dir.iterdir() if p.suffix == ".mp4")
    print(f"[pick] refs_dir has {len(present)} mp4 files", flush=True)

    keywords = [k.lower() for k in args.keyword]
    keywords_any = [k.lower() for k in args.keyword_any]
    excludes = [k.lower() for k in args.exclude]

    candidates = []
    n_total = n_kept = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            video = row["video"]
            if video not in present:
                continue
            caption = row["caption"]
            cap_lower = caption.lower()

            # Keyword filters
            if keywords and not all(k in cap_lower for k in keywords):
                continue
            if keywords_any and not any(k in cap_lower for k in keywords_any):
                continue
            if excludes and any(k in cap_lower for k in excludes):
                continue

            # Numeric filters
            try:
                motion = float(row["motion score"])
                aesthetic = float(row["aesthetic score"])
                seconds = float(row["seconds"])
                fps = float(row["fps"])
            except (KeyError, ValueError):
                continue
            if not (args.motion_min <= motion <= args.motion_max):
                continue
            if not (args.aesthetic_min <= aesthetic <= args.aesthetic_max):
                continue
            if not (args.seconds_min <= seconds <= args.seconds_max):
                continue

            camera = row.get("camera motion", "")
            if args.camera_motion is not None and camera != args.camera_motion:
                continue

            candidates.append({
                "video": video,
                "caption": caption,
                "motion": motion,
                "aesthetic": aesthetic,
                "camera": camera,
                "seconds": seconds,
                "fps": fps,
                "path": str(refs_dir / video),
            })
            n_kept += 1

    print(f"[pick] scanned {n_total} CSV rows, {n_kept} match all filters",
          flush=True)

    # Rank: prefer mid-range motion (best for learning generic dynamics) +
    # high aesthetic. Score = aesthetic - |motion - 4.0| * 0.5.
    candidates.sort(
        key=lambda r: r["aesthetic"] - abs(r["motion"] - 4.0) * 0.5,
        reverse=True,
    )
    candidates = candidates[: args.top_k]

    if args.output:
        with open(args.output, "w") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[pick] wrote {len(candidates)} candidates to {args.output}",
              flush=True)
    print(f"[pick] top {min(5, len(candidates))} preview:", flush=True)
    for i, c in enumerate(candidates[:5]):
        cap_short = c["caption"][:120].replace("\n", " ")
        print(f"  [{i}] motion={c['motion']:.2f} aes={c['aesthetic']:.2f} "
              f"cam={c['camera']:10s} {c['seconds']:.1f}s  {c['video']}",
              flush=True)
        print(f"      {cap_short}", flush=True)


if __name__ == "__main__":
    main()
