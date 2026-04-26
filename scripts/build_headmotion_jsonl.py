"""
Build cross-pair JSONL from the head-motion master JSON produced by
``scripts/mine_openvid_headmotion.py``.

Pure offline filter: reads master JSON + OpenVid CSV, picks videos passing the
yaw threshold (and that have actually been saved to motion_refs/), buckets by
yaw tier, cross-pairs within bucket so prompt-from-X aligns with motion-from-Y
of similar yaw amplitude, and writes train/val JSONL with disjoint video sets.

Re-run any time with a different ``--yaw_threshold`` to resize the pool with no
network cost. Output schema is identical to the existing
``motion_pairs_cross_train.jsonl`` so ``utils/dataset.py:MotionSwitchDataset``
loads it transparently.
"""

import argparse
import csv
import json
import random
from pathlib import Path


def load_captions(csv_path: Path) -> dict:
    captions = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions[row["video"]] = row["caption"]
    return captions


def yaw_bucket(yaw_range: float, threshold: float) -> str:
    if yaw_range < 30.0:
        return "low"
    if yaw_range < 50.0:
        return "mid"
    return "high"


def cross_pair_within(videos: list[str], captions: dict, rng: random.Random) -> list[dict]:
    """Circular-shift cross-pair: prompt from X, motion from Y = next(X) within bucket.
    Guarantees X != Y (assuming len(videos) >= 2)."""
    pairs = []
    if len(videos) < 2:
        for v in videos:
            pairs.append({
                "prompt_a": captions[v],
                "prompt_b": captions[v],
                "motion_a": v,
                "motion_b": v,
                "switch_frame": -1,
            })
        return pairs
    shuffled = rng.sample(videos, len(videos))
    rotated = shuffled[1:] + shuffled[:1]
    for X, Y in zip(shuffled, rotated):
        assert X != Y
        pairs.append({
            "prompt_a": captions[X],
            "prompt_b": captions[X],
            "motion_a": Y,
            "motion_b": Y,
            "switch_frame": -1,
        })
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master_json",
                    default="logs/diag_head_pose_all_parts.json",
                    help="Master yaw stats JSON from mine_openvid_headmotion.py")
    ap.add_argument("--meta_csv",
                    default="/home/hongyou/dev/data/wm/meta/data/train/OpenVid-1M.csv")
    ap.add_argument("--output_dir",
                    default="/home/hongyou/dev/data/wm/prompts")
    ap.add_argument("--yaw_threshold", type=float, default=20.0,
                    help="Minimum yaw_range (deg) to include")
    ap.add_argument("--max_pool", type=int, default=0,
                    help="Cap final pool size; 0 = no cap. When set, keep top-N by yaw_range")
    ap.add_argument("--motion_refs_dir",
                    default="/home/hongyou/dev/data/wm/motion_refs",
                    help="Verify mp4 actually exists on disk; "
                         "supersedes the `saved` flag in master JSON entries")
    ap.add_argument("--skip_existence_check", action="store_true",
                    help="Trust master JSON's saved flag, do not stat each mp4 file")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_suffix", default="_cross_headmotion")
    ap.add_argument("--also_existing_diag", nargs="*", default=[],
                    help="Extra diag JSONs to merge (e.g. logs/diag_head_pose_cross_{train,val}.json)")
    args = ap.parse_args()

    # ---- Load master JSON, optionally merge older diag JSONs.
    master_path = Path(args.master_json)
    if master_path.exists():
        with open(master_path) as f:
            master = json.load(f)
    else:
        master = {}
        print(f"[warn] master JSON {master_path} not found; using only --also_existing_diag inputs")
    for extra in args.also_existing_diag:
        with open(extra) as f:
            master.update(json.load(f))
    print(f"[info] {len(master)} entries in merged master")

    # ---- Filter by yaw_threshold; verify on-disk presence.
    motion_refs_dir = Path(args.motion_refs_dir)
    if not args.skip_existence_check:
        on_disk = {p.name for p in motion_refs_dir.iterdir() if p.suffix == ".mp4"}
        print(f"[info] motion_refs/ on disk: {len(on_disk)} mp4")
    else:
        on_disk = None  # signal: trust master's `saved` flag
    candidates = []
    n_below_thresh = n_missing = n_err = 0
    for name, meta in master.items():
        if meta.get("error"):
            n_err += 1
            continue
        yr = meta.get("yaw_range")
        if yr is None or yr < args.yaw_threshold:
            n_below_thresh += 1
            continue
        if on_disk is not None:
            if name not in on_disk:
                n_missing += 1
                continue
        elif not meta.get("saved", False):
            n_missing += 1
            continue
        candidates.append((name, yr))
    print(f"[info] filter: yaw<thresh={n_below_thresh}, missing_mp4={n_missing}, errors={n_err}")
    candidates.sort(key=lambda x: -x[1])  # by yaw_range desc
    if args.max_pool > 0 and len(candidates) > args.max_pool:
        candidates = candidates[:args.max_pool]
    n = len(candidates)
    print(f"[info] yaw_range >= {args.yaw_threshold}: {n} candidates")
    if n < 4:
        raise SystemExit("too few candidates to build train/val")

    # ---- Distribution sanity (which buckets dominate).
    bucket_count = {"low": 0, "mid": 0, "high": 0}
    for _, yr in candidates:
        bucket_count[yaw_bucket(yr, args.yaw_threshold)] += 1
    print(f"[info] bucket distribution: {bucket_count}")

    # ---- Load captions.
    captions = load_captions(Path(args.meta_csv))
    print(f"[info] loaded {len(captions)} captions from CSV")
    candidates = [(n_, yr) for n_, yr in candidates if n_ in captions]
    if len(candidates) != n:
        print(f"[warn] {n - len(candidates)} videos dropped (no caption in CSV)")
    n = len(candidates)

    # ---- Split videos disjointly into train/val (90/10), seeded.
    rng = random.Random(args.seed)
    cand_names = [name for name, _ in candidates]
    rng.shuffle(cand_names)
    n_val = max(1, int(n * args.val_frac))
    val_set = set(cand_names[:n_val])
    train_set = set(cand_names[n_val:])
    print(f"[info] split: train={len(train_set)}  val={len(val_set)}")

    # ---- Re-bucket per split (yaw_range lookup).
    yr_by_name = {n_: yr for n_, yr in candidates}

    def bucket_split(names: set) -> dict:
        buckets = {"low": [], "mid": [], "high": []}
        for n_ in names:
            buckets[yaw_bucket(yr_by_name[n_], args.yaw_threshold)].append(n_)
        return buckets

    train_buckets = bucket_split(train_set)
    val_buckets = bucket_split(val_set)
    print(f"[info] train buckets: { {k: len(v) for k, v in train_buckets.items()} }")
    print(f"[info] val   buckets: { {k: len(v) for k, v in val_buckets.items()} }")

    # ---- Cross-pair within each bucket.
    train_pairs = []
    for bk, vs in train_buckets.items():
        train_pairs.extend(cross_pair_within(vs, captions, rng))
    rng.shuffle(train_pairs)
    val_pairs = []
    for bk, vs in val_buckets.items():
        val_pairs.extend(cross_pair_within(vs, captions, rng))
    rng.shuffle(val_pairs)

    # ---- Sanity asserts.
    train_video_set = set()
    val_video_set = set()
    for p in train_pairs:
        train_video_set.add(p["motion_a"])
        train_video_set.add(p["motion_b"])
    for p in val_pairs:
        val_video_set.add(p["motion_a"])
        val_video_set.add(p["motion_b"])
    overlap = train_video_set & val_video_set
    assert not overlap, f"train/val overlap: {sorted(overlap)[:5]}"
    for p in train_pairs + val_pairs:
        assert p["motion_a"] == p["motion_b"], "schema invariant"
    print(f"[info] train_pairs={len(train_pairs)}  val_pairs={len(val_pairs)}  disjoint OK")

    # ---- Write JSONL.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"motion_pairs{args.output_suffix}_train.jsonl"
    val_path = out_dir / f"motion_pairs{args.output_suffix}_val.jsonl"
    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[done] wrote {train_path} ({len(train_pairs)} rows)")
    print(f"[done] wrote {val_path} ({len(val_pairs)} rows)")


if __name__ == "__main__":
    main()
