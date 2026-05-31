"""Run structured reference-motion diagnostics on generated videos."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from longlive.methods.motion_projected_em_ram.motion_features import (
    build_reference_motion_descriptor,
    descriptor_pair_metrics,
    descriptor_to_npz,
)
from scripts.motion_eval.metrics.motion_fidelity import CoTrackerWrapper
from scripts.motion_eval.metrics.video_io import read_video_frames


def _frame_size(video_path: Path) -> tuple[int, int]:
    frames = read_video_frames(video_path, max_frames=1)
    return int(frames.shape[1]), int(frames.shape[2])


def _descriptor_for_video(
    tracker: CoTrackerWrapper,
    video_path: Path,
    *,
    bucket_count: int,
    moving_percentile: float,
    min_moving_tracks: int,
    moving_speed_floor: float,
):
    height, width = _frame_size(video_path)
    tracks, visibility = tracker.get_tracklets(video_path)
    return build_reference_motion_descriptor(
        tracks,
        visibility,
        frame_height=height,
        frame_width=width,
        bucket_count=bucket_count,
        moving_percentile=moving_percentile,
        min_moving_tracks=min_moving_tracks,
        moving_speed_floor=moving_speed_floor,
    )


def _collect_gen_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.gen]
    for gen_dir in args.gen_dir:
        paths.extend(sorted(Path(gen_dir).glob("*.mp4")))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("no generated videos found")
    return unique


def run(args: argparse.Namespace) -> None:
    ref_path = Path(args.ref)
    gen_paths = _collect_gen_paths(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = CoTrackerWrapper(
        device=args.device,
        cache_dir=args.cache_dir,
        n_frames=args.n_frames,
        grid_size=args.grid_size,
    )
    ref_desc = _descriptor_for_video(
        tracker,
        ref_path,
        bucket_count=args.bucket_count,
        moving_percentile=args.moving_percentile,
        min_moving_tracks=args.min_moving_tracks,
        moving_speed_floor=args.moving_speed_floor,
    )
    descriptor_to_npz(str(out_dir / "reference_descriptor.npz"), ref_desc)

    rows = []
    for idx, gen_path in enumerate(gen_paths):
        gen_desc = _descriptor_for_video(
            tracker,
            gen_path,
            bucket_count=args.bucket_count,
            moving_percentile=args.moving_percentile,
            min_moving_tracks=args.min_moving_tracks,
            moving_speed_floor=args.moving_speed_floor,
        )
        descriptor_to_npz(str(out_dir / f"gen_{idx:03d}_descriptor.npz"), gen_desc)
        metrics = descriptor_pair_metrics(
            gen_desc,
            ref_desc,
            speed_lower=args.speed_lower,
            speed_upper=args.speed_upper,
            speed_penalty_coef=args.speed_penalty_coef,
        )
        rows.append({
            "video": str(gen_path),
            "video_name": gen_path.name,
            **metrics,
        })

    csv_path = out_dir / "reference_motion_diagnostics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    preferred = ["video", "video_name", "residual_score", "residual_direction",
                 "residual_speed_ratio", "residual_speed_penalty", "global_direction"]
    fieldnames = preferred + [k for k in fieldnames if k not in preferred]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {}
    numeric_keys = [k for k in fieldnames if k not in ("video", "video_name")]
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            means[key] = sum(vals) / len(vals)
    summary = {
        "ref": str(ref_path),
        "n": len(rows),
        "reference_summary": ref_desc.summary,
        "mean": means,
    }
    with open(out_dir / "reference_motion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[reference_motion] wrote {csv_path}")
    print(json.dumps(summary["mean"], indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--gen", action="append", default=[])
    p.add_argument("--gen_dir", action="append", default=[])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--grid_size", type=int, default=30)
    p.add_argument("--bucket_count", type=int, default=3)
    p.add_argument("--moving_percentile", type=float, default=60.0)
    p.add_argument("--min_moving_tracks", type=int, default=4)
    p.add_argument("--moving_speed_floor", type=float, default=1e-5)
    p.add_argument("--speed_lower", type=float, default=0.5)
    p.add_argument("--speed_upper", type=float, default=2.0)
    p.add_argument("--speed_penalty_coef", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
