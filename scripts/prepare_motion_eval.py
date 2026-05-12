"""Stage UCF Sports Action + LOVEU-TGVE 2023 reference videos for motion eval.

Usage::

    python scripts/prepare_motion_eval.py \\
        --data_root "$LL_DATA" \\
        --datasets ucf,loveu

Output layout under ``data_root``::

    ucf_sports/
      _raw/                              # original zip + unzipped tree (deleted unless --keep_zip)
      videos/<category>/<clip_id>.mp4    # rule-based filtered, ffmpeg-normalized
      manifest.csv                       # category, clip_id, path, h, w, n_frames
    loveu_tgve/
      _raw/                              # original zip (deleted unless --keep_zip)
      videos/<video_id>.mp4
      prompts.csv                        # LOVEU's official LOVEU-TGVE-2023_Dataset.csv, verbatim

UCF filter rules (rule-based, reproducible — paper's 95-clip filter is
unreleased, so we replicate the spirit rather than the literal list):
    - drop clips with ``height < min_h`` or ``width < min_w`` (default 720x480)
    - drop clips with fewer than ``min_frames`` frames (default 16)

Golf-Swing category handling: UCF Sports ships three view sub-directories
(Golf-Swing-Front / -Back / -Side). MotionDirector's public repo collapses
them to a single ``playing_golf`` LoRA. We follow that release convention
and collapse the three subdirs into a single ``Golf-Swing`` category.
Clip IDs get a ``front_`` / ``back_`` / ``side_`` prefix to stay unique.

Cross-machine staging: pass ``--remote_host hongyou@lab`` (or set
``LL_REMOTE_HOST``) to rsync the already-prepared output from a peer
(canonically lab) instead of downloading fresh. In remote mode we skip
download + extract + filter entirely and pull the peer's
``ucf_sports/`` and ``loveu_tgve/`` directories wholesale. This mirrors
``scripts/hpc/fetch_data.sh``'s ``LL_REMOTE_HOST`` mode.

HPC users: prefer the wrapper at ``scripts/hpc/fetch_motion_eval.sh``,
which sets up the env + paths before calling this script.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional


UCF_URL = "https://www.crcv.ucf.edu/data/ucf_sports_actions.zip"
LOVEU_GDRIVE_ID = "1D7ZVm66IwlKhS6UINoDgFiFJp_mLIQ0W"

UCF_CATEGORIES = [
    "Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse",
    "Run-Side", "Skateboarding", "Swing-Bench", "Swing-SideAngle", "Walk-Front",
]


def _remote_mode(remote_host: Optional[str]) -> bool:
    return bool(remote_host)


# Canonical lab data root. The lab user is "hongyou" and $LL_DATA on lab
# resolves to ~/dev/data/wm = /home/hongyou/dev/data/wm. Matches the
# hardcoded path in scripts/hpc/fetch_data.sh, deliberately consistent.
LAB_DATA_ROOT = "/home/hongyou/dev/data/wm"


def _rsync(remote_host: str, src: str, dst: str) -> None:
    print(f"[motion_eval] rsync {remote_host}:{src} -> {dst}")
    Path(dst).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-aP", "--inplace", f"{remote_host}:{src}/", f"{dst}/"],
        check=True,
    )


def _ucf_download(raw_dir: Path) -> Path:
    """Local-mode UCF download (curl). Remote mode is handled in main()
    by rsyncing the already-prepared ucf_sports/ directory wholesale."""
    zip_path = raw_dir / "ucf_sports_actions.zip"
    if zip_path.exists():
        print(f"[ucf] zip already at {zip_path}")
        return zip_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ucf] downloading {UCF_URL} -> {zip_path} (~1.66 GiB)")

    # Some clusters (Charite HPC observed 2026-05-12) have stale system
    # CA bundles or TLS-intercepting proxies that fail to verify the
    # crcv.ucf.edu cert. Try in order:
    #   1. system CA  (curl default) — works on most machines
    #   2. certifi's bundle           — fresh CAs shipped with Python
    #   3. --insecure                 — last resort for known public URL
    attempts = [
        ("system CA", []),
    ]
    try:
        import certifi  # type: ignore
        attempts.append(("certifi CA", ["--cacert", certifi.where()]))
    except ImportError:
        pass
    attempts.append(("insecure (TLS verify skipped)", ["--insecure"]))

    last_err: Optional[subprocess.CalledProcessError] = None
    for label, extra in attempts:
        if last_err is not None:
            print(f"[ucf] retry with {label}")
        try:
            subprocess.run(
                ["curl", "-L", "--fail", *extra, "-o", str(zip_path), UCF_URL],
                check=True,
            )
            return zip_path
        except subprocess.CalledProcessError as e:
            # curl exit 60 = SSL cert problem. Retry. Other errors = fail fast.
            if e.returncode != 60:
                raise
            last_err = e
            if zip_path.exists():
                zip_path.unlink()
    # All attempts failed.
    raise RuntimeError(
        f"curl failed for {UCF_URL} after {len(attempts)} attempts; "
        f"last exit {last_err.returncode if last_err else '?'}. "
        "Check HPC network / proxy."
    )


def _ucf_extract(zip_path: Path, raw_dir: Path) -> Path:
    """Unzip into raw_dir/ucf_action_sports/<Category>/<clip>/..."""
    marker = raw_dir / ".extracted"
    if marker.exists():
        print(f"[ucf] already extracted (marker {marker})")
    else:
        print(f"[ucf] extracting {zip_path} -> {raw_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.touch()

    # Locate the top-level directory. Common name is "ucf_action_sports" but
    # be defensive — pick the only subdirectory if there's exactly one.
    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(candidates) == 0:
        raise RuntimeError(f"No directory found in {raw_dir} after extraction")
    if len(candidates) == 1:
        return candidates[0]
    for name in ("ucf_action_sports", "ucf_sports_actions"):
        for c in candidates:
            if c.name == name:
                return c
    raise RuntimeError(
        f"Multiple dirs in {raw_dir} after extract, none named expected: {[c.name for c in candidates]}"
    )


def _probe_video(path: Path):
    """Return (height, width, n_frames) for a video, or None if unreadable."""
    try:
        # Use decord if available (already in longlive env); fall back to ffprobe.
        try:
            from decord import VideoReader  # type: ignore

            vr = VideoReader(str(path))
            n = len(vr)
            if n == 0:
                return None
            frame = vr[0].asnumpy()
            h, w = frame.shape[:2]
            return int(h), int(w), int(n)
        except ImportError:
            pass
        # ffprobe fallback
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split("\n")
        if len(out) < 3:
            return None
        w, h, n = int(out[0]), int(out[1]), int(out[2])
        return h, w, n
    except Exception as e:  # noqa: BLE001
        print(f"[ucf] probe failed for {path.name}: {e}", file=sys.stderr)
        return None


def _normalize_ucf_category(raw_category: str, clip_id: str) -> tuple[str, str]:
    """Collapse Golf-Swing-{Front,Back,Side} -> Golf-Swing with view-prefixed clip ID."""
    if raw_category.startswith("Golf-Swing-"):
        view = raw_category.split("-")[-1].lower()
        return "Golf-Swing", f"{view}_{clip_id}"
    return raw_category, clip_id


def _ffmpeg_normalize(src: Path, dst: Path) -> bool:
    """Remux src to dst.mp4. Try stream-copy first; fall back to libx264 re-encode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Try remux (instant for already-H.264 streams).
    p = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "copy", "-an", str(dst)],
        capture_output=True,
    )
    if p.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    # Fall back: re-encode.
    if dst.exists():
        dst.unlink()
    p = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(dst)],
        capture_output=True,
    )
    if p.returncode != 0:
        print(f"[ucf] ffmpeg failed for {src}: {p.stderr.decode(errors='replace')}",
              file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return False
    return True


def _ucf_filter_and_layout(
    raw_root: Path, out_videos: Path, manifest_csv: Path,
    min_h: int, min_w: int, min_frames: int,
) -> None:
    """Walk raw_root, apply size+length filter, write normalized mp4s + manifest.

    Handles both common UCF Sports archive layouts by walking each top-level
    category recursively (the categories — Diving, Golf-Swing-{Front,Back,Side},
    etc. — are always at depth-1 from raw_root). For each category we collect
    every .avi/.mp4/.mov file at any depth and infer clip_id from the closest
    parent directory above the file (or the file stem if the file sits flat
    at the category root).
    """
    if manifest_csv.exists():
        print(f"[ucf] manifest already present at {manifest_csv}, skipping filter")
        return

    out_videos.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_categories = set()

    top_entries = sorted(raw_root.iterdir())
    n_dirs = sum(1 for e in top_entries if e.is_dir())
    print(f"[ucf] walking {raw_root} ({len(top_entries)} top-level entries, "
          f"{n_dirs} dirs)")
    for e in top_entries[:20]:
        kind = "dir" if e.is_dir() else "file"
        print(f"[ucf]   {kind}: {e.name}")

    VID_EXTS = (".avi", ".mp4", ".mov")
    for cat_dir in top_entries:
        if not cat_dir.is_dir():
            continue
        raw_category = cat_dir.name
        video_files: list[Path] = []
        for root, _dirs, files in os.walk(cat_dir):
            for fname in files:
                if fname.lower().endswith(VID_EXTS):
                    video_files.append(Path(root) / fname)
        if not video_files:
            print(f"[ucf]   {raw_category}: 0 video files (skipping)")
            continue
        print(f"[ucf]   {raw_category}: {len(video_files)} candidate video file(s)")

        for src in sorted(video_files):
            # clip_id = closest non-trivial parent dir name, else file stem.
            # Layouts seen in the wild:
            #   <cat>/<clip_num>/<file>.avi      -> clip_id = <clip_num>
            #   <cat>/<file>.avi                 -> clip_id = file stem
            if src.parent == cat_dir:
                raw_clip_id = src.stem
            else:
                raw_clip_id = src.parent.name

            category, clip_id = _normalize_ucf_category(raw_category, raw_clip_id)
            seen_categories.add(category)

            probe = _probe_video(src)
            if probe is None:
                continue
            h, w, n = probe
            if h < min_h or w < min_w or n < min_frames:
                print(f"[ucf]   drop {raw_category}/{raw_clip_id} ({w}x{h}, {n} frames)")
                continue

            dst = out_videos / category / f"{clip_id}.mp4"
            if dst.exists():
                # Same logical clip seen via a sibling .avi/.mov (rare); skip.
                continue
            if not _ffmpeg_normalize(src, dst):
                continue
            rows.append({
                "category": category,
                "clip_id": clip_id,
                "path": str(dst.relative_to(out_videos.parent)),
                "height": h,
                "width": w,
                "n_frames": n,
            })

    rows.sort(key=lambda r: (r["category"], r["clip_id"]))

    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "clip_id", "path", "height", "width", "n_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ucf] wrote {manifest_csv} with {len(rows)} clips "
          f"across {len(seen_categories)} categories")

    if len(rows) == 0:
        # Don't silently succeed — caller's cleanup would delete _raw/ and
        # destroy our ability to debug. Raise so the exception path keeps
        # _raw/ around for inspection.
        manifest_csv.unlink(missing_ok=True)  # don't leave a 0-row stub
        raise RuntimeError(
            f"UCF filter produced 0 clips from {raw_root}. Check the "
            "[ucf] debug log above (top-level dirs, per-category video "
            "counts) and the actual zip layout. _raw/ has been preserved "
            f"at {raw_root.parent} — `ls -R {raw_root.parent}/_raw | head -50` "
            "to inspect."
        )


def _loveu_download(raw_dir: Path) -> Path:
    """Local-mode LOVEU download (gdown). Remote mode is handled in main()."""
    zip_path = raw_dir / "loveu-tgve-2023.zip"
    if zip_path.exists():
        print(f"[loveu] zip already at {zip_path}")
        return zip_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[loveu] downloading via gdown -> {zip_path}")
    try:
        import gdown  # type: ignore
    except ImportError:
        raise RuntimeError(
            "gdown not installed. `pip install gdown` in the longlive env, "
            "or set LL_REMOTE_HOST to rsync from a peer."
        )
    gdown.download(id=LOVEU_GDRIVE_ID, output=str(zip_path), quiet=False)
    if not zip_path.exists():
        raise RuntimeError(f"gdown finished but {zip_path} not found")
    return zip_path


def _loveu_extract(zip_path: Path, raw_dir: Path, out_videos: Path, out_prompts: Path) -> None:
    marker = raw_dir / ".extracted"
    if marker.exists():
        print(f"[loveu] already extracted (marker {marker})")
    else:
        print(f"[loveu] extracting {zip_path} -> {raw_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.touch()

    # Find the dataset CSV (LOVEU-TGVE-2023_Dataset.csv) and the videos dir.
    csv_src = None
    video_sources: list[Path] = []
    for root, _dirs, files in os.walk(raw_dir):
        for name in files:
            p = Path(root) / name
            if name == "LOVEU-TGVE-2023_Dataset.csv":
                csv_src = p
            elif name.lower().endswith((".mp4", ".mov", ".avi")):
                video_sources.append(p)

    if csv_src is None:
        raise RuntimeError(f"LOVEU-TGVE-2023_Dataset.csv not found under {raw_dir}")
    if not out_prompts.exists():
        shutil.copy2(csv_src, out_prompts)
        print(f"[loveu] copied prompts to {out_prompts}")

    out_videos.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in video_sources:
        dst = out_videos / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"[loveu] {len(video_sources)} videos total, {copied} newly copied to {out_videos}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=os.environ.get("LL_DATA"),
                    help="Data root (default reads $LL_DATA env var)")
    ap.add_argument("--datasets", type=str, default="ucf,loveu",
                    help="Comma-separated subset of {ucf, loveu}")
    ap.add_argument("--min_h", type=int, default=720)
    ap.add_argument("--min_w", type=int, default=480)
    ap.add_argument("--min_frames", type=int, default=16)
    ap.add_argument("--keep_zip", action="store_true",
                    help="Keep downloaded zip and unzipped _raw/ after layout")
    ap.add_argument("--remote_host", type=str, default=os.environ.get("LL_REMOTE_HOST"),
                    help="Rsync from this peer instead of fresh download "
                         "(default reads $LL_REMOTE_HOST)")
    args = ap.parse_args()

    if not args.data_root:
        raise SystemExit("--data_root unset and $LL_DATA not in env")

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    requested = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}
    valid = {"ucf", "loveu"}
    if not requested.issubset(valid):
        raise SystemExit(f"Unknown datasets {requested - valid}, valid={valid}")

    remote = args.remote_host

    if "ucf" in requested:
        ucf_root = data_root / "ucf_sports"
        if _remote_mode(remote):
            # Pull the already-prepared output from a peer (canonically lab).
            # Skips download + extract + filter — those ran once on the peer.
            _rsync(remote, f"{LAB_DATA_ROOT}/ucf_sports", str(ucf_root))
            mf = ucf_root / "manifest.csv"
            if not mf.exists():
                raise RuntimeError(
                    f"After rsync, expected {mf}. Peer may not have run "
                    "prepare_motion_eval.py yet."
                )
            print(f"[ucf] synced from {remote}: manifest.csv present at {mf}")
        else:
            ucf_raw = ucf_root / "_raw"
            ucf_videos = ucf_root / "videos"
            ucf_manifest = ucf_root / "manifest.csv"

            zip_path = _ucf_download(ucf_raw)
            tree_root = _ucf_extract(zip_path, ucf_raw)
            _ucf_filter_and_layout(
                tree_root, ucf_videos, ucf_manifest,
                min_h=args.min_h, min_w=args.min_w, min_frames=args.min_frames,
            )
            if not args.keep_zip:
                print(f"[ucf] removing {ucf_raw} (--keep_zip not set)")
                shutil.rmtree(ucf_raw, ignore_errors=True)

    if "loveu" in requested:
        loveu_root = data_root / "loveu_tgve"
        if _remote_mode(remote):
            _rsync(remote, f"{LAB_DATA_ROOT}/loveu_tgve", str(loveu_root))
            pf = loveu_root / "prompts.csv"
            if not pf.exists():
                raise RuntimeError(
                    f"After rsync, expected {pf}. Peer may not have run "
                    "prepare_motion_eval.py yet."
                )
            print(f"[loveu] synced from {remote}: prompts.csv present at {pf}")
        else:
            loveu_raw = loveu_root / "_raw"
            loveu_videos = loveu_root / "videos"
            loveu_prompts = loveu_root / "prompts.csv"

            zip_path = _loveu_download(loveu_raw)
            _loveu_extract(zip_path, loveu_raw, loveu_videos, loveu_prompts)
            if not args.keep_zip:
                print(f"[loveu] removing {loveu_raw} (--keep_zip not set)")
                shutil.rmtree(loveu_raw, ignore_errors=True)

    print("[motion_eval] all done.")


if __name__ == "__main__":
    main()
