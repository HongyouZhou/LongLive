"""Compute amplitude metrics for an alpha-sweep set of generated videos.

Reads every .mp4 under <sweep_dir>/alpha_*/, runs CoTracker3 (reusing the
wrapper from scripts.motion_eval.metrics.motion_fidelity), computes per-video
tracklet-based amplitude (mean velocity, total path length, max displacement
in pixels), and writes a CSV summarising alpha vs amplitude.

Output CSV columns:
    alpha, prompt_tag, seed, video, mean_v_px, total_path_px, max_displ_px
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# CoTrackerWrapper lives in motion_eval/metrics; that dir is not a package,
# so add it to sys.path explicitly.
METRICS_DIR = REPO_ROOT / "scripts" / "motion_eval" / "metrics"
sys.path.insert(0, str(METRICS_DIR))
from motion_fidelity import CoTrackerWrapper  # noqa: E402


def amplitude(tracks: np.ndarray, vis: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean_velocity_px, total_path_px, max_displ_px) per Yatim-style tracks."""
    T, N, _ = tracks.shape
    v = tracks[1:] - tracks[:-1]
    pair_vis = vis[1:] & vis[:-1]
    mag = np.linalg.norm(v, axis=2)
    visible_mag = mag[pair_vis]
    mean_v = float(visible_mag.mean()) if visible_mag.size else 0.0

    track_lens, max_displ = [], []
    for n in range(N):
        if vis[0, n]:
            ms = mag[:, n][pair_vis[:, n]]
            if ms.size > 0:
                track_lens.append(float(ms.sum()))
            visible_idx = np.where(vis[:, n])[0]
            if visible_idx.size > 1:
                d = np.linalg.norm(tracks[visible_idx, n] - tracks[0, n], axis=1)
                max_displ.append(float(d.max()))
    total_path = float(np.mean(track_lens)) if track_lens else 0.0
    max_d = float(np.mean(max_displ)) if max_displ else 0.0
    return mean_v, total_path, max_d


def parse_video_name(p: Path) -> tuple[str, int]:
    """`<tag>_seed<N>.mp4` → ('tag', N). Falls back to (stem, 0)."""
    m = re.match(r"(?P<tag>.+?)_seed(?P<seed>\d+)$", p.stem)
    if m:
        return m.group("tag"), int(m.group("seed"))
    return p.stem, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="Root containing alpha_*/ subdirs of .mp4")
    ap.add_argument("--output", required=True, help="Path to write the result CSV")
    ap.add_argument("--cache-dir", default=None, help="CoTracker tracklet cache dir (defaults to sweep-dir/cache)")
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (sweep_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    alpha_dirs = sorted(sweep_dir.glob("alpha_*/"), key=lambda p: -int(p.name.split("_")[1]))
    if not alpha_dirs:
        print(f"[amp] no alpha_*/ subdirs under {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    wrapper = CoTrackerWrapper(
        device=args.device, cache_dir=cache_dir,
        n_frames=args.n_frames, grid_size=args.grid_size,
    )

    rows = []
    for ad in alpha_dirs:
        alpha = int(ad.name.split("_")[1])
        videos = sorted(ad.glob("*.mp4"))
        print(f"[amp] alpha={alpha}: {len(videos)} videos", flush=True)
        for v in videos:
            tracks, vis = wrapper.get_tracklets(v)
            mv, tp, md = amplitude(tracks, vis)
            tag, seed = parse_video_name(v)
            rows.append({
                "alpha": alpha, "prompt_tag": tag, "seed": seed,
                "video": v.name, "mean_v_px": mv,
                "total_path_px": tp, "max_displ_px": md,
            })
            print(f"  alpha={alpha:>3} {tag:<24} seed={seed}: "
                  f"mean_v={mv:.3f}  path={tp:.2f}  max_d={md:.2f}", flush=True)

    # Write CSV
    fieldnames = ["alpha", "prompt_tag", "seed", "video", "mean_v_px", "total_path_px", "max_displ_px"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print per-alpha summary
    print()
    print(f"{'alpha':>5}  {'n':>3}  {'mean_v_px':>10}  {'path_px':>9}  {'max_d_px':>9}")
    print("-" * 50)
    for ad in alpha_dirs:
        alpha = int(ad.name.split("_")[1])
        sub = [r for r in rows if r["alpha"] == alpha]
        if not sub: continue
        mv = np.mean([r["mean_v_px"] for r in sub])
        tp = np.mean([r["total_path_px"] for r in sub])
        md = np.mean([r["max_displ_px"] for r in sub])
        print(f"{alpha:>5}  {len(sub):>3}  {mv:>10.3f}  {tp:>9.3f}  {md:>9.3f}")

    print()
    print(f"[amp] wrote {out}")


if __name__ == "__main__":
    main()
