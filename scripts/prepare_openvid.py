"""Download one OpenVid-1M zip part, extract high-motion clips, build motion-pair JSONL.

Usage:
    python scripts/prepare_openvid.py \
        --part 0 \
        --num_keep 1000 \
        --data_root /home/hongyou/dev/data/wm

Output layout under ``data_root``:
    meta/data/train/OpenVid-1M.csv    (already downloaded separately)
    openvid_raw/OpenVid_part{N}.zip   (temporary; deleted after extraction unless --keep_zip)
    motion_refs/*.mp4                 (num_keep selected videos)
    prompts/motion_pairs_train.jsonl  (90% of selected)
    prompts/motion_pairs_val.jsonl    (10%)
"""
import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import csv
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0,
                    help="OpenVid part index to download")
    ap.add_argument("--num_keep", type=int, default=1000,
                    help="Number of clips to keep after filtering")
    ap.add_argument("--data_root", type=str, default="/home/hongyou/dev/data/wm")
    ap.add_argument("--min_motion", type=float, default=3.0,
                    help="Minimum motion score (OpenVid metric) to keep")
    ap.add_argument("--min_aesthetic", type=float, default=4.5)
    ap.add_argument("--min_seconds", type=float, default=2.0)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--keep_zip", action="store_true",
                    help="Do not delete the downloaded zip after extraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross_pair", action="store_true",
                    help="Build cross-paired JSONL (prompt from X, motion from Y != X) "
                         "stratified by motion_score bucket. Keeps self-pair JSONL untouched.")
    ap.add_argument("--output_suffix", type=str, default="",
                    help="Suffix for JSONL output filename (e.g. '_cross'). "
                         "Empty -> motion_pairs_{train,val}.jsonl (overwrite). "
                         "'_cross' -> motion_pairs_cross_{train,val}.jsonl.")
    ap.add_argument("--skip_download", action="store_true",
                    help="Skip the HF download step even if zip is missing.")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip the zip extraction step; assume refs_dir already populated. "
                         "When set, `selected` is derived from existing refs_dir contents.")
    args = ap.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    raw_dir = data_root / "openvid_raw"
    refs_dir = data_root / "motion_refs"
    prompts_dir = data_root / "prompts"
    meta_csv = data_root / "meta/data/train/OpenVid-1M.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    assert meta_csv.exists(), f"Missing {meta_csv}. Download it first."

    # ---- 1. Load captions
    print(f"Loading caption CSV {meta_csv} ...")
    caption_by_video: dict[str, str] = {}
    score_by_video: dict[str, float] = {}
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                motion_score = float(row["motion score"])
                aesthetic = float(row["aesthetic score"])
                seconds = float(row["seconds"])
            except (ValueError, KeyError):
                continue
            if (motion_score >= args.min_motion
                    and aesthetic >= args.min_aesthetic
                    and args.min_seconds <= seconds <= args.max_seconds):
                caption_by_video[row["video"]] = row["caption"]
                score_by_video[row["video"]] = motion_score
    print(f"  filtered captions: {len(caption_by_video)}")

    # ---- 2. Download zip if missing
    zip_name = f"OpenVid_part{args.part}.zip"
    zip_path = raw_dir / zip_name
    if not zip_path.exists() and not args.skip_download:
        print(f"Downloading {zip_name} ...")
        downloaded = hf_hub_download(
            repo_id="nkp37/OpenVid-1M",
            filename=zip_name,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"  at {downloaded}")
    elif zip_path.exists():
        print(f"Zip already present at {zip_path}")
    else:
        print(f"Zip missing but --skip_download set; relying on refs_dir contents.")

    # ---- 3. List zip contents (or refs_dir), intersect with filtered captions
    if args.skip_extract:
        print("Skipping extract; reusing refs_dir contents as selected pool ...")
        existing = sorted(
            os.path.basename(p) for p in os.listdir(refs_dir)
            if p.endswith(".mp4")
        )
        selected = [n for n in existing if n in caption_by_video]
        print(f"  refs_dir has {len(existing)} mp4s, matched-in-caption: {len(selected)}")
    else:
        print("Reading zip index ...")
        with zipfile.ZipFile(zip_path) as zf:
            all_names = [n for n in zf.namelist() if n.endswith(".mp4")]
        print(f"  {len(all_names)} mp4 files in zip")

        eligible = [n for n in all_names if os.path.basename(n) in caption_by_video]
        print(f"  intersect with filtered captions: {len(eligible)}")

        random.shuffle(eligible)
        selected = eligible[: args.num_keep]
        print(f"  selecting {len(selected)}")

        # ---- 4. Extract only selected
        print("Extracting selected clips ...")
        with zipfile.ZipFile(zip_path) as zf:
            for i, name in enumerate(selected):
                dst = refs_dir / os.path.basename(name)
                if dst.exists():
                    continue
                with zf.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                if (i + 1) % 100 == 0:
                    print(f"  extracted {i + 1}/{len(selected)}")
        print("Done extracting.")
        # Normalize `selected` to basenames for downstream pairing logic
        selected = [os.path.basename(n) for n in selected]

    # ---- 5. Build pair JSONL
    print("Building motion-pair JSONL ...")
    if args.cross_pair:
        # Stratify by motion_score bucket, then circular-shift within bucket
        # so prompt is from video X, motion_ref from Y = next(X) within bucket.
        buckets: dict[str, list[str]] = {"low": [], "mid": [], "high": []}
        for v in selected:
            s = float(score_by_video[v])
            if s < 4.0:
                buckets["low"].append(v)
            elif s < 6.0:
                buckets["mid"].append(v)
            else:
                buckets["high"].append(v)
        for k, vs in buckets.items():
            print(f"  bucket {k}: {len(vs)} clips")

        pairs = []
        for _, videos in buckets.items():
            if len(videos) < 2:
                # Too few to cross-pair; fall back to self-pair for this bucket
                for v in videos:
                    pairs.append({
                        "prompt_a": caption_by_video[v], "prompt_b": caption_by_video[v],
                        "motion_a": v, "motion_b": v, "switch_frame": -1,
                    })
                continue
            shuffled = random.sample(videos, len(videos))
            rotated = shuffled[1:] + shuffled[:1]   # guarantees Y != X
            for X, Y in zip(shuffled, rotated):
                assert X != Y, f"cross-pair collision: {X}"
                pairs.append({
                    "prompt_a": caption_by_video[X],
                    "prompt_b": caption_by_video[X],
                    "motion_a": Y,
                    "motion_b": Y,
                    "switch_frame": -1,
                })
    else:
        # Self-paired: motion_a == motion_b == prompt source video
        pairs = []
        for name in selected:
            base = os.path.basename(name) if not args.skip_extract else name
            caption = caption_by_video[base]
            pairs.append({
                "prompt_a": caption,
                "prompt_b": caption,
                "motion_a": base,
                "motion_b": base,
                "switch_frame": -1,
            })

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    suffix = args.output_suffix
    train_jsonl = prompts_dir / f"motion_pairs{suffix}_train.jsonl"
    val_jsonl = prompts_dir / f"motion_pairs{suffix}_val.jsonl"
    with open(train_jsonl, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_jsonl, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {train_jsonl}  ({len(train_pairs)} rows)")
    print(f"  wrote {val_jsonl}    ({len(val_pairs)} rows)")

    # ---- 6. Cleanup
    if not args.keep_zip and zip_path.exists() and not args.skip_extract:
        print(f"Removing zip {zip_path} to save disk ...")
        zip_path.unlink()
    print("All done.")


if __name__ == "__main__":
    main()
