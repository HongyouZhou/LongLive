"""
Retry errored / skipped clips from a previous mining sweep.

Two recovery sources:

1. **Errored clips** in master JSON (entries with non-empty "error" field).
   These are typically signal.alarm timeouts or ffmpeg decode failures
   that left the clip with no yaw stats. We re-screen them with
   conservative settings (small worker pool, longer alarm) to recover
   ones that were transient failures.

2. **Skipped parts** recorded in mining_progress.json (per_part_stats
   entries with negative ``errors`` field, e.g. the manual skip of
   part 43 due to OOM cascade caused by the cv2.setLogLevel bug).
   These parts were never screened; we mine them fresh.

For each part with retry candidates we re-download the zip, extract
just the needed clips, screen them, then update master JSON and copy
passing mp4s to motion_refs/. Idempotent — safe to re-run.

Usage:
    python scripts/retry_errors.py                       # default conservative
    python scripts/retry_errors.py --workers 4 --alarm_sec 30
    python scripts/retry_errors.py --include_skipped_parts 43 99
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Reuse the full pipeline from mine_openvid_headmotion.
sys.path.insert(0, str(Path(__file__).parent))
from mine_openvid_headmotion import (
    DEFAULT_FACE_KEYWORDS,
    DEFAULT_MODEL_PATH,
    _worker_init,
    _worker_screen,
    download_part,
    ensure_face_landmarker,
    log,
    write_json_atomic,
)


def load_caption_csv(csv_path: Path, face_kws: list[str],
                     min_motion: float, min_aesthetic: float,
                     min_seconds: float, max_seconds: float) -> dict:
    """Same pre-filter as the main script (kept local to avoid coupling)."""
    import csv
    captions = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ms = float(row["motion score"])
                ae = float(row["aesthetic score"])
                sec = float(row["seconds"])
            except (ValueError, KeyError):
                continue
            if not (ms >= min_motion and ae >= min_aesthetic
                    and min_seconds <= sec <= max_seconds):
                continue
            cap_lc = row["caption"].lower()
            if not any(k in cap_lc for k in face_kws):
                continue
            captions[row["video"]] = row["caption"]
    return captions


def collect_retry_candidates(master_data: dict,
                             progress_data: dict,
                             include_skipped_parts: list[int]) -> dict[int, set]:
    """Returns {part_idx: set(clip_name_to_retry)}.

    ``set()`` (empty) means the entire part is needed (skipped part).
    Non-empty set lists specific errored clips.
    """
    retry_by_part: dict[int, set] = {}

    # 1. Errored individual clips.
    for name, meta in master_data.items():
        if meta.get("error") and "part" in meta:
            p = int(meta["part"])
            retry_by_part.setdefault(p, set()).add(name)

    # 2. Skipped parts (errors == -1 marker we wrote, or explicit list).
    skipped = set()
    per_part = progress_data.get("per_part_stats", {})
    for k, v in per_part.items():
        if v.get("errors", 0) == -1:
            skipped.add(int(k))
    skipped.update(include_skipped_parts)

    for p in skipped:
        # Empty set = full part retry, supersedes any individual entries.
        retry_by_part[p] = set()

    return retry_by_part


def process_part_retry(part_idx: int,
                       wanted: set,
                       caption_by_video: dict,
                       motion_refs_dir: Path,
                       motion_refs_seen: set,
                       tmp_dir: Path,
                       master_json_path: Path,
                       progress_path: Path,
                       save_threshold: float,
                       pool,
                       master_data: dict,
                       progress_data: dict,
                       raw_dir: Path) -> dict:
    t_part = time.time()
    log(f"-- retry part {part_idx} (target: {len(wanted) or 'ALL face_kw'} clips) --")

    # Download zip (idempotent: skip if on disk).
    zip_path = download_part(part_idx, raw_dir)

    # List zip mp4s.
    with zipfile.ZipFile(zip_path) as zf:
        names_all = [n for n in zf.namelist() if n.endswith(".mp4")]

    if wanted:
        # Targeted retry: extract specified clips that exist in zip.
        candidates = [n for n in names_all
                      if os.path.basename(n) in wanted]
    else:
        # Full part: same logic as main script (face_kw + dedup).
        candidates = [n for n in names_all
                      if os.path.basename(n) in caption_by_video
                      and os.path.basename(n) not in motion_refs_seen]
    log(f"  zip mp4 total = {len(names_all)},  to screen = {len(candidates)}")

    if not candidates:
        log(f"  no candidates for retry; skipping part {part_idx}")
        return {"requested": len(wanted), "screened": 0, "saved": 0, "wall_seconds": int(time.time() - t_part)}

    part_tmp = tmp_dir / f"retry_part{part_idx}"
    part_tmp.mkdir(parents=True, exist_ok=True)

    log(f"  extracting to {part_tmp}")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        for n in candidates:
            dst = part_tmp / os.path.basename(n)
            with zf.open(n) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
    log(f"  extracted in {(time.time() - t0) / 60:.1f} min")

    # Screen.
    log(f"  screening (conservative pool)")
    t0 = time.time()
    screen_args = [(os.path.basename(n), str(part_tmp / os.path.basename(n)))
                   for n in candidates]
    n_pass = 0
    n_err = 0
    for i, (name, stats) in enumerate(pool.imap_unordered(_worker_screen, screen_args, chunksize=1)):
        if "error" in stats:
            n_err += 1
            master_data[name] = {"part": part_idx, **stats, "saved": False}
        elif stats.get("n", 0) < 2:
            master_data[name] = {"part": part_idx, **stats, "saved": False}
        else:
            yr = stats["yaw_range"]
            saved = yr >= save_threshold
            master_data[name] = {"part": part_idx, **stats, "saved": saved}
            if saved:
                src_path = part_tmp / name
                dst_path = motion_refs_dir / name
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    motion_refs_seen.add(name)
                n_pass += 1
        if (i + 1) % 200 == 0:
            log(f"    screened {i + 1}/{len(screen_args)}  pass={n_pass}  err={n_err}")
    log(f"  done part {part_idx}: screened {len(screen_args)} in "
        f"{(time.time() - t0) / 60:.1f} min, pass={n_pass}, err={n_err}")

    # Persist.
    write_json_atomic(master_json_path, master_data)
    progress_data.setdefault("retried_parts", {})[str(part_idx)] = {
        "requested": len(wanted),
        "screened": len(candidates),
        "saved": n_pass,
        "errors": n_err,
        "wall_seconds": int(time.time() - t_part),
    }
    write_json_atomic(progress_path, progress_data)

    # Cleanup.
    try:
        shutil.rmtree(part_tmp)
    except Exception as e:
        log(f"  tmp cleanup err: {e}")
    try:
        zip_path.unlink()
    except Exception as e:
        log(f"  zip cleanup err: {e}")

    return progress_data["retried_parts"][str(part_idx)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default="/home/hongyou/dev/data/wm")
    ap.add_argument("--save_threshold", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="Pool size. Smaller (default 4) for stability "
                         "since we are explicitly retrying flaky clips.")
    ap.add_argument("--alarm_sec", type=int, default=30,
                    help="Per-video signal.alarm (deg). Longer than the "
                         "main script's 15s to recover transient slowdowns.")
    ap.add_argument("--face_keywords", default=DEFAULT_FACE_KEYWORDS)
    ap.add_argument("--master_json", default="logs/diag_head_pose_all_parts.json")
    ap.add_argument("--progress_json", default="logs/mining_progress.json")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--tmp_dir", default="/home/hongyou/longlive_work/openvid_screen")
    ap.add_argument("--min_motion", type=float, default=3.0)
    ap.add_argument("--min_aesthetic", type=float, default=4.5)
    ap.add_argument("--min_seconds", type=float, default=2.0)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--include_skipped_parts", type=int, nargs="*", default=[],
                    help="Extra part indices to fully retry (in addition to "
                         "any auto-detected from progress.json).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print plan and exit without doing any work.")
    args = ap.parse_args()

    # Patch the worker's alarm to user-supplied length. _screen_video reads
    # MINE_ALARM_SEC; setting it here propagates to forkserver workers.
    os.environ["MINE_ALARM_SEC"] = str(args.alarm_sec)

    data_root = Path(args.data_root)
    if data_root.parent.is_symlink():
        sys.exit(f"REFUSING: {data_root.parent} is a symlink — write would leak through sshfs")

    motion_refs_dir = data_root / "motion_refs"
    raw_dir = data_root / "openvid_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = data_root / "meta/data/train/OpenVid-1M.csv"

    master_json = Path(args.master_json)
    progress_json = Path(args.progress_json)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not master_json.exists():
        sys.exit(f"missing master JSON {master_json}")
    if not progress_json.exists():
        sys.exit(f"missing progress JSON {progress_json}")

    # Load.
    with open(master_json) as f:
        master_data = json.load(f)
    with open(progress_json) as f:
        progress_data = json.load(f)
    log(f"master: {len(master_data)} entries; progress: "
        f"{len(progress_data.get('processed_parts', []))} parts processed")

    # Identify retries.
    retry_by_part = collect_retry_candidates(
        master_data, progress_data, args.include_skipped_parts)
    if not retry_by_part:
        log("nothing to retry. exiting.")
        return
    n_targeted = sum(len(v) for v in retry_by_part.values() if v)
    n_full = sum(1 for v in retry_by_part.values() if not v)
    log(f"plan: {len(retry_by_part)} parts to retry "
        f"({n_targeted} targeted clips + {n_full} full-part redos)")
    log(f"  parts: {sorted(retry_by_part.keys())}")

    if args.dry_run:
        log("dry_run: exiting without changes")
        return

    # Cap.
    motion_refs_seen = set(p.name for p in motion_refs_dir.iterdir() if p.suffix == ".mp4")
    log(f"motion_refs/ already has {len(motion_refs_seen)} mp4s")

    ensure_face_landmarker(args.model_path)

    face_kws = args.face_keywords.lower().split()
    caption_by_video = load_caption_csv(
        meta_csv, face_kws,
        args.min_motion, args.min_aesthetic, args.min_seconds, args.max_seconds,
    )
    log(f"CSV face_kw filter: {len(caption_by_video)} captions")

    mp.set_start_method("forkserver", force=True)
    pool = mp.Pool(processes=args.workers,
                   initializer=_worker_init,
                   initargs=(args.model_path,),
                   maxtasksperchild=10)

    t_all = time.time()
    n_done = 0
    for part_idx in sorted(retry_by_part.keys()):
        wanted = retry_by_part[part_idx]
        try:
            stats = process_part_retry(
                part_idx, wanted, caption_by_video,
                motion_refs_dir, motion_refs_seen, tmp_dir,
                master_json, progress_json,
                args.save_threshold,
                pool, master_data, progress_data, raw_dir,
            )
            n_done += 1
            log(f"  part {part_idx} retry stats: {stats}")
        except Exception as e:
            log(f"  part {part_idx} FAILED: {type(e).__name__}: {e}")
            progress_data.setdefault("retry_failed_parts", []).append(part_idx)
            write_json_atomic(progress_json, progress_data)

    pool.close()
    pool.join()

    log(f"DONE retry. {n_done}/{len(retry_by_part)} parts in "
        f"{(time.time() - t_all) / 3600:.2f} h. "
        f"motion_refs total: {len(motion_refs_seen)}")


if __name__ == "__main__":
    main()
