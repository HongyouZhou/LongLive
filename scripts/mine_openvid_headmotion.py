"""
Mine OpenVid-1M for head-motion-rich clips.

Pipeline per zip part (rolling, peak disk ~50GB working set):
    HF download → CSV face_kw pre-filter → extract candidates → multiprocess
    MediaPipe yaw detection → save mp4 with yaw_range >= save_threshold to
    motion_refs/, append all candidates' stats to master JSON → delete zip+tmp.

Resumable via logs/mining_progress.json.

Design notes:
- Runs on lab (32 cores, 125GB RAM, 3.3TB local nvme, HF reachable).
- Cross-part overlap: download thread + screen Pool work concurrently.
- Each Pool worker holds its own FaceLandmarker (TFLite is not fork-safe).
"""

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_FACE_KEYWORDS = "face person man woman boy girl child people head"
HF_REPO_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"
DEFAULT_MODEL_PATH = "scripts/_models/face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_face_landmarker(model_path: str) -> None:
    p = Path(model_path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    log(f"face_landmarker model missing at {p}; downloading...")
    r = requests.get(FACE_LANDMARKER_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(p, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    log(f"  saved {p.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Worker-side: each Pool process owns a FaceLandmarker.
# ---------------------------------------------------------------------------

_landmarker = None  # per-process global


def _worker_init(model_path: str) -> None:
    global _landmarker
    import mediapipe as mp_pkg  # noqa: F401  (loaded into worker)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)


def _rot_to_euler_yxz(R) -> tuple[float, float, float]:
    import math
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


def _screen_video(path: str, max_frames: int = 24) -> dict:
    """Returns {} on failure (no detectable face / cannot decode)."""
    import cv2
    import mediapipe as mp_pkg
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return {}
    step = max(1, total // max_frames)
    series = []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_pkg.Image(image_format=mp_pkg.ImageFormat.SRGB, data=rgb)
            res = _landmarker.detect(mp_image)
            if res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0])
                pitch, yaw, roll = _rot_to_euler_yxz(M[:3, :3])
                series.append((yaw, pitch, roll))
        f_idx += 1
    cap.release()

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


def _worker_screen(args_tuple: tuple) -> tuple[str, dict]:
    name, path = args_tuple
    try:
        return name, _screen_video(path)
    except Exception as e:
        return name, {"error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Main-side: download + extraction + orchestration.
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[mine {ts}] {msg}", flush=True)


def load_face_kw_csv(csv_path: Path, face_kws: list[str],
                     min_motion: float, min_aesthetic: float,
                     min_seconds: float, max_seconds: float) -> tuple[dict, dict]:
    """Returns (caption_by_video, motion_score_by_video) for all CSV rows
    that pass the existing motion/aesthetic/seconds filter AND have at least
    one face_kw in their caption (lowercased)."""
    log(f"loading CSV {csv_path}")
    caption = {}
    score = {}
    n_total = 0
    n_face = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
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
            n_face += 1
            caption[row["video"]] = row["caption"]
            score[row["video"]] = ms
    log(f"  CSV total={n_total}, after filter+face_kw={n_face}")
    return caption, score


def download_part(part_idx: int, raw_dir: Path, retries: int = 3) -> Path:
    """Download one zip part via huggingface_hub. Returns local path."""
    from huggingface_hub import hf_hub_download
    name = f"OpenVid_part{part_idx}.zip"
    dst = raw_dir / name
    if dst.exists():
        log(f"  part{part_idx} zip already on disk ({dst.stat().st_size / 1e9:.1f} GB), skip download")
        return dst
    last_err = None
    for attempt in range(retries):
        try:
            log(f"  part{part_idx} downloading (attempt {attempt + 1}/{retries})...")
            t0 = time.time()
            path = hf_hub_download(
                repo_id="nkp37/OpenVid-1M",
                filename=name,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            dt = time.time() - t0
            sz = os.path.getsize(path) / 1e9
            log(f"  part{part_idx} downloaded {sz:.1f} GB in {dt / 60:.1f} min ({sz * 1024 / dt:.0f} MB/s)")
            return Path(path)
        except Exception as e:
            last_err = e
            log(f"  part{part_idx} download error: {type(e).__name__}: {e}")
            time.sleep(5)
    raise RuntimeError(f"part{part_idx} download failed after {retries} attempts: {last_err}")


def process_part(part_idx: int,
                 zip_path: Path,
                 caption_by_video: dict,
                 motion_refs_dir: Path,
                 tmp_dir: Path,
                 master_json_path: Path,
                 progress_path: Path,
                 save_threshold: float,
                 max_per_part: int,
                 pool,
                 master_data: dict,
                 progress_data: dict) -> dict:
    """Process one part: extract face candidates, screen, save passing, delete tmp.
    Returns per-part stats dict."""
    t_part = time.time()

    # 1. List zip contents, intersect with face-keyword video set.
    with zipfile.ZipFile(zip_path) as zf:
        names_all = [n for n in zf.namelist() if n.endswith(".mp4")]
    candidates = [n for n in names_all
                  if os.path.basename(n) in caption_by_video
                  and os.path.basename(n) not in motion_refs_seen]
    if max_per_part and len(candidates) > max_per_part:
        candidates = candidates[:max_per_part]
    log(f"  part{part_idx}: zip has {len(names_all)} mp4, face_kw match = {len(candidates)}")

    if not candidates:
        return {"face_candidates": 0, "passed_yaw": 0, "errors": 0,
                "wall_seconds": time.time() - t_part}

    # 2. Extract candidates to tmp_dir.
    part_tmp = tmp_dir / f"part{part_idx}"
    part_tmp.mkdir(parents=True, exist_ok=True)
    log(f"  part{part_idx}: extracting {len(candidates)} candidates to {part_tmp}")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        for n in candidates:
            dst = part_tmp / os.path.basename(n)
            with zf.open(n) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
    log(f"  part{part_idx}: extracted in {(time.time() - t0) / 60:.1f} min")

    # 3. Multiprocess yaw screening.
    log(f"  part{part_idx}: screening with mediapipe...")
    t0 = time.time()
    screen_args = [(os.path.basename(n), str(part_tmp / os.path.basename(n)))
                   for n in candidates]
    n_pass = 0
    n_err = 0
    for i, (name, stats) in enumerate(pool.imap_unordered(_worker_screen, screen_args, chunksize=8)):
        if "error" in stats:
            n_err += 1
            master_data[name] = {"part": part_idx, **stats, "saved": False}
        elif stats.get("n", 0) < 2:
            master_data[name] = {"part": part_idx, **stats, "saved": False}
        else:
            yaw_range = stats["yaw_range"]
            saved = yaw_range >= save_threshold
            master_data[name] = {"part": part_idx, **stats, "saved": saved}
            if saved:
                src_path = part_tmp / name
                dst_path = motion_refs_dir / name
                shutil.move(str(src_path), str(dst_path))
                motion_refs_seen.add(name)
                n_pass += 1
        if (i + 1) % 500 == 0:
            log(f"    screened {i + 1}/{len(screen_args)}  pass={n_pass}  err={n_err}")
    log(f"  part{part_idx}: screened {len(screen_args)} in {(time.time() - t0) / 60:.1f} min,"
        f" pass={n_pass} ({100 * n_pass / max(1, len(screen_args)):.1f}%), err={n_err}")

    # 4. Persist master + progress, atomic rename.
    write_json_atomic(master_json_path, master_data)
    progress_data.setdefault("processed_parts", []).append(part_idx)
    progress_data["last_updated_at"] = datetime.now().isoformat()
    progress_data.setdefault("per_part_stats", {})[str(part_idx)] = {
        "face_candidates": len(candidates),
        "passed_yaw": n_pass,
        "errors": n_err,
        "wall_seconds": int(time.time() - t_part),
    }
    write_json_atomic(progress_path, progress_data)

    # 5. Cleanup: delete zip + tmp dir.
    try:
        shutil.rmtree(part_tmp)
    except Exception as e:
        log(f"  part{part_idx}: tmp cleanup err: {e}")
    try:
        zip_path.unlink()
    except Exception as e:
        log(f"  part{part_idx}: zip cleanup err: {e}")

    return progress_data["per_part_stats"][str(part_idx)]


def write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


# Module-level set populated lazily so workers don't accidentally pick it up
# via fork (we use spawn). Updated by main process only.
motion_refs_seen: set = set()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default="/home/hongyou/dev/data/wm")
    ap.add_argument("--save_threshold", type=float, default=15.0,
                    help="yaw_range (deg) for permanent save to motion_refs/")
    ap.add_argument("--max_per_part", type=int, default=0, help="0 = no cap")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--face_keywords", default=DEFAULT_FACE_KEYWORDS,
                    help="space-separated keyword list")
    ap.add_argument("--start_part", type=int, default=0)
    ap.add_argument("--end_part", type=int, default=170)
    ap.add_argument("--master_json", default="logs/diag_head_pose_all_parts.json")
    ap.add_argument("--progress_json", default="logs/mining_progress.json")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--tmp_dir", default="/home/hongyou/longlive_work/openvid_screen")
    ap.add_argument("--min_motion", type=float, default=3.0)
    ap.add_argument("--min_aesthetic", type=float, default=4.5)
    ap.add_argument("--min_seconds", type=float, default=2.0)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--restart", action="store_true",
                    help="ignore existing progress JSON and restart from start_part")
    args = ap.parse_args()

    # ---- Safety: data_root must NOT be a symlink (could write through sshfs to arp).
    data_root = Path(args.data_root)
    parent = data_root.parent  # /home/hongyou/dev/data
    if parent.is_symlink():
        sys.exit(f"REFUSING TO RUN: {parent} is a symlink "
                 f"({os.readlink(parent)}). Mining must write to a real local dir.")
    if data_root.is_symlink():
        sys.exit(f"REFUSING TO RUN: {data_root} is a symlink. Aborting.")

    motion_refs_dir = data_root / "motion_refs"
    motion_refs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_root / "openvid_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = data_root / "meta/data/train/OpenVid-1M.csv"
    if not meta_csv.exists():
        sys.exit(f"missing CSV {meta_csv}")
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    master_json = Path(args.master_json)
    progress_json = Path(args.progress_json)
    master_json.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load existing master + progress.
    master_data = {}
    if master_json.exists() and not args.restart:
        with open(master_json) as f:
            master_data = json.load(f)
        log(f"loaded existing master JSON ({len(master_data)} entries)")
    progress_data = {}
    processed_parts = set()
    if progress_json.exists() and not args.restart:
        with open(progress_json) as f:
            progress_data = json.load(f)
        processed_parts = set(progress_data.get("processed_parts", []))
        log(f"resume mode: skip {len(processed_parts)} already-processed parts")
    if "started_at" not in progress_data:
        progress_data["started_at"] = datetime.now().isoformat()

    # ---- Snapshot motion_refs_seen for dedup (existing 1000 mp4s).
    global motion_refs_seen
    motion_refs_seen = set(p.name for p in motion_refs_dir.iterdir() if p.suffix == ".mp4")
    log(f"motion_refs/ already has {len(motion_refs_seen)} mp4s (will be skipped on collision)")

    # ---- Ensure mediapipe model file is on disk (auto-download if missing).
    ensure_face_landmarker(args.model_path)

    # ---- Load CSV face_kw map.
    face_kws = args.face_keywords.lower().split()
    caption_by_video, motion_score_by_video = load_face_kw_csv(
        meta_csv, face_kws,
        args.min_motion, args.min_aesthetic, args.min_seconds, args.max_seconds,
    )

    # ---- Build parts plan.
    parts_plan = [p for p in range(args.start_part, args.end_part) if p not in processed_parts]
    if not parts_plan:
        log("nothing to do (all parts processed). Use --restart to redo.")
        return
    log(f"plan: {len(parts_plan)} parts ({parts_plan[0]}..{parts_plan[-1]})  workers={args.workers}")

    # ---- Spawn-mode pool (TFLite/cv2 not fork-safe with already-imported state).
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=args.workers,
                   initializer=_worker_init,
                   initargs=(args.model_path,))

    # ---- Download thread (one part ahead).
    download_q: queue.Queue = queue.Queue(maxsize=1)
    download_err: list = []

    def _downloader():
        try:
            for p in parts_plan:
                zp = download_part(p, raw_dir)
                download_q.put((p, zp))
            download_q.put(None)
        except Exception as e:
            download_err.append(e)
            download_q.put(None)

    threading.Thread(target=_downloader, daemon=True).start()

    # ---- Main loop: pull next ready zip, screen, save, delete.
    t_all = time.time()
    n_done = 0
    while True:
        item = download_q.get()
        if item is None:
            break
        part_idx, zp = item
        log(f"-- part {part_idx} --")
        try:
            stats = process_part(part_idx, zp, caption_by_video,
                                 motion_refs_dir, tmp_dir,
                                 master_json, progress_json,
                                 args.save_threshold, args.max_per_part,
                                 pool, master_data, progress_data)
            n_done += 1
            log(f"  part{part_idx} stats: {stats}")
            log(f"  cumulative: {n_done}/{len(parts_plan)} parts done,"
                f" master_entries={len(master_data)},"
                f" motion_refs={len(motion_refs_seen)},"
                f" elapsed={int((time.time() - t_all) / 60)} min")
        except Exception as e:
            log(f"  part{part_idx} FAILED: {type(e).__name__}: {e}")
            # Don't kill loop — record failure and move on.
            progress_data.setdefault("failed_parts", []).append(part_idx)
            write_json_atomic(progress_json, progress_data)

    pool.close()
    pool.join()

    if download_err:
        log(f"download thread had errors: {download_err}")

    log(f"DONE. {n_done} parts processed in {(time.time() - t_all) / 3600:.2f} h."
        f" master={len(master_data)}, motion_refs={len(motion_refs_seen)}")


if __name__ == "__main__":
    main()
