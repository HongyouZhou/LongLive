"""
Fetch only the head-motion-mined clips from OpenVid-1M directly on HPC.

Replaces the previous "rsync 360 GB motion_refs/ from lab" approach. Uses
the master JSON (which records `part` for every screened clip) to figure
out which OpenVid zip parts contain the clips referenced by the training
JSONL. For each such part: download zip from HF, extract only the wanted
clips, save to motion_refs/, delete zip. Roll part-by-part to keep peak
disk small (one zip ~30-50 GB).

Inputs:
    master_all.json   merged master from scripts/merge_master_jsons.py
    motion_pairs_*.jsonl  emitted by scripts/build_headmotion_jsonl.py;
                          we collect motion_a / motion_b clip names

Outputs:
    motion_refs/<clip>.mp4   only the clips referenced by the JSONLs
"""

import argparse
import json
import os
import shutil
import socket
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Avoid silent socket hangs (same fix as mine_openvid_headmotion).
socket.setdefaulttimeout(120)

from huggingface_hub import hf_hub_download


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[fetch {ts}] {msg}", flush=True)


def collect_wanted(jsonls: list[str]) -> set[str]:
    wanted: set[str] = set()
    for path in jsonls:
        if not Path(path).exists():
            log(f"  warn: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                for k in ("motion_a", "motion_b"):
                    v = e.get(k)
                    if v:
                        wanted.add(v)
    return wanted


def fetch_part(part_idx: int, clips: list[str], raw_dir: Path,
               output_dir: Path, retries: int = 3) -> tuple[int, int]:
    """Download zip, extract `clips`, delete zip. Returns (n_extracted, n_skipped)."""
    name = f"OpenVid_part{part_idx}.zip"
    zip_path = raw_dir / name

    last_err = None
    for attempt in range(retries):
        try:
            log(f"  part{part_idx}: downloading (attempt {attempt + 1}/{retries})")
            t0 = time.time()
            local = hf_hub_download(
                repo_id="nkp37/OpenVid-1M",
                filename=name,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            zip_path = Path(local)
            sz = zip_path.stat().st_size / 1e9
            dt = time.time() - t0
            log(f"  part{part_idx}: downloaded {sz:.1f} GB in {dt / 60:.1f} min ({sz * 1024 / dt:.0f} MB/s)")
            break
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            if "404" in err_text or "entry not found" in err_text:
                log(f"  part{part_idx}: 404 on HF -- skip")
                return 0, len(clips)
            log(f"  part{part_idx}: download err {type(e).__name__}: {e}")
            time.sleep(5)
    else:
        log(f"  part{part_idx}: failed after {retries} attempts: {last_err}")
        return 0, len(clips)

    # Extract wanted clips.
    n_extracted = 0
    n_missing_in_zip = 0
    wanted_set = set(clips)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names_in_zip = {os.path.basename(n): n for n in zf.namelist()
                        if n.endswith(".mp4")}
        for clip in clips:
            entry = names_in_zip.get(clip)
            if entry is None:
                n_missing_in_zip += 1
                continue
            dst = output_dir / clip
            with zf.open(entry) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
            n_extracted += 1
    log(f"  part{part_idx}: extracted {n_extracted}/{len(clips)} in "
        f"{(time.time() - t0):.0f}s (missing in zip: {n_missing_in_zip})")

    # Cleanup.
    try:
        zip_path.unlink()
    except Exception as e:
        log(f"  part{part_idx}: zip cleanup err: {e}")

    return n_extracted, n_missing_in_zip


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master_json", required=True,
                    help="merged master from merge_master_jsons.py "
                         "(provides clip -> part lookup)")
    ap.add_argument("--jsonls", nargs="+", required=True,
                    help="motion_pairs_*.jsonl files; "
                         "motion_a/motion_b fields are extracted")
    ap.add_argument("--output_dir", required=True,
                    help="motion_refs/ destination directory")
    ap.add_argument("--raw_dir", required=True,
                    help="tmp dir for zip downloads (rolling, "
                         "~50 GB peak per part)")
    ap.add_argument("--start_part", type=int, default=0)
    ap.add_argument("--end_part", type=int, default=183)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load master.
    log(f"loading master {args.master_json}")
    with open(args.master_json) as f:
        master = json.load(f)
    log(f"  {len(master)} entries in master")

    # ---- Collect wanted clip names from JSONLs.
    wanted = collect_wanted(args.jsonls)
    log(f"  {len(wanted)} unique clips referenced by JSONLs")

    # ---- Group by part.
    by_part: dict[int, list[str]] = defaultdict(list)
    n_no_part = 0
    for name in wanted:
        meta = master.get(name)
        if not meta or "part" not in meta:
            n_no_part += 1
            continue
        by_part[int(meta["part"])].append(name)
    parts = sorted(p for p in by_part if args.start_part <= p < args.end_part)
    log(f"  in {len(parts)} parts (range [{args.start_part}, {args.end_part}))"
        f"; clips with no part lookup: {n_no_part}")

    # ---- Skip clips already on disk.
    on_disk = {p.name for p in output_dir.iterdir() if p.suffix == ".mp4"}
    log(f"  motion_refs/ already has {len(on_disk)} mp4 (will skip dedup)")

    # ---- For each part: dedup wanted by on_disk, fetch.
    total_extracted = 0
    total_skipped = 0
    t_all = time.time()
    for i, part_idx in enumerate(parts):
        wanted_in_part = [c for c in by_part[part_idx] if c not in on_disk]
        if not wanted_in_part:
            log(f"part {part_idx} ({i + 1}/{len(parts)}): all "
                f"{len(by_part[part_idx])} on disk, skip")
            continue
        log(f"part {part_idx} ({i + 1}/{len(parts)}): "
            f"{len(wanted_in_part)} clips to fetch")
        n_ext, n_skip = fetch_part(part_idx, wanted_in_part, raw_dir, output_dir)
        on_disk.update(wanted_in_part[:n_ext])  # rough; assume first N succeeded
        total_extracted += n_ext
        total_skipped += n_skip
        log(f"  cumulative: {total_extracted} extracted, "
            f"{total_skipped} skipped, "
            f"{int((time.time() - t_all) / 60)} min elapsed")

    log(f"DONE. {total_extracted} clips extracted in "
        f"{(time.time() - t_all) / 3600:.2f} h. "
        f"motion_refs/ now has {len(on_disk)} mp4.")


if __name__ == "__main__":
    main()
