"""Post-hoc motion-customization eval driver.

Reads:
  --prompts_manifest <path>   # JSONL from scripts/motion_eval/build_manifest.py
                              # (prompt_id, dataset, key, prompt, ref_videos, ...)
  --gen_dir <path>            # contains manifest.json (prompt_id -> mp4) + videos/
  --ref_root <path>           # $LL_DATA — ref_videos in prompt manifest are relative
  --output <path>             # CSV: per-prompt scores

For each prompt_id present in both manifests:
  - Load gen mp4 + ref mp4(s)
  - Compute 4 metrics (3 CLIP-based + Yatim motion fidelity)
  - Append row to CSV

Resume: skip prompt_ids already in --output CSV.

End-of-run summary: print per-dataset means.

This driver is method-agnostic: it does not import longlive.*. Any framework
that produces (prompt_id -> mp4 path) compatible with the prompt manifest
schema can feed into this eval.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Support both `python scripts/motion_eval/run_eval.py ...` and
# `python -m scripts.motion_eval.run_eval ...`. The bare-filename invocation
# needs the script's directory on sys.path so the `metrics.*` package resolves.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.clip_metrics import CLIPMetrics  # noqa: E402
from metrics.motion_fidelity import MotionFidelity  # noqa: E402
from metrics.video_io import frames_to_pil, read_video_frames  # noqa: E402


CSV_COLUMNS = [
    "prompt_id", "dataset", "key", "prompt", "gen_path",
    "app_div", "temp_consist", "pick_score", "motion_fidelity",
    "ok", "error",
]


def _load_prompts_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["prompt_id"]] = r
    return rows


def _load_gen_manifest(gen_dir: Path) -> dict[str, str]:
    path = gen_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did scripts/motion_eval/eval_dispatch.py run on this gen_dir?"
        )
    with open(path) as f:
        return json.load(f)


def _read_existing_csv(path: Path) -> set[str]:
    """Return prompt_ids already scored (for resume)."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ok") == "True" and row.get("prompt_id"):
                done.add(row["prompt_id"])
    return done


def _open_csv_for_append(path: Path) -> "csv.DictWriter":
    new = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if new:
        writer.writeheader()
        fh.flush()
    return writer, fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_manifest", required=True,
                    help="JSONL from scripts/motion_eval/build_manifest.py")
    ap.add_argument("--gen_dir", required=True,
                    help="Generation output dir (contains manifest.json + videos/)")
    ap.add_argument("--ref_root", required=True,
                    help="Root for resolving ref_videos relative paths (typically $LL_DATA)")
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--motion_fidelity_cache",
                    default=None,
                    help="Directory for cached CoTracker3 tracklets. "
                         "Defaults to <gen_dir>/cache/tracklets")
    ap.add_argument("--n_frames_mf", type=int, default=16,
                    help="Frames sampled per video for motion fidelity")
    ap.add_argument("--grid_size_mf", type=int, default=30,
                    help="CoTracker3 query grid size (motion fidelity)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_motion_fidelity", action="store_true",
                    help="Drop the Yatim metric (e.g., for cotracker-less envs)")
    # wandb logging (independent eval run, project=longlive-motion-eval).
    # Aggregates 8 scalars + a per-prompt wandb.Table. Disable via --no_wandb
    # or WANDB_MODE=disabled. wandb.init failure is non-fatal — scores.csv
    # remains the ground-truth artifact.
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "longlive-motion-eval"),
                    help="wandb project name (env WANDB_PROJECT, default longlive-motion-eval)")
    ap.add_argument("--wandb_run_name", default=None,
                    help="wandb run name (default = output dir basename)")
    ap.add_argument("--ckpt_tag", default=None,
                    help="Ckpt path / identifier to record in wandb config")
    ap.add_argument("--no_wandb", action="store_true",
                    help="Skip wandb logging entirely")
    args = ap.parse_args()

    prompts_path = Path(args.prompts_manifest)
    gen_dir = Path(args.gen_dir)
    ref_root = Path(args.ref_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts_manifest(prompts_path)
    gen_map = _load_gen_manifest(gen_dir)
    print(f"[eval] {len(prompts)} prompts in manifest, {len(gen_map)} entries in gen manifest")

    common = sorted(set(prompts) & set(gen_map))
    if args.limit is not None:
        common = common[: args.limit]
    print(f"[eval] {len(common)} prompts to score")

    done = _read_existing_csv(output)
    pending = [pid for pid in common if pid not in done]
    if done:
        print(f"[eval] resuming: {len(done)} already in CSV, {len(pending)} pending")

    if not pending:
        print("[eval] nothing to score; computing aggregates and exiting")
    else:
        clip = CLIPMetrics(device=args.device)
        mf = None
        if not args.skip_motion_fidelity:
            cache = (Path(args.motion_fidelity_cache)
                     if args.motion_fidelity_cache
                     else gen_dir / "cache" / "tracklets")
            mf = MotionFidelity(
                device=args.device, cache_dir=cache,
                n_frames=args.n_frames_mf, grid_size=args.grid_size_mf,
            )
            print(f"[eval] motion fidelity cache: {cache}")

        writer, fh = _open_csv_for_append(output)
        try:
            for i, pid in enumerate(pending, 1):
                t0 = time.time()
                row = prompts[pid]
                gen_rel = gen_map[pid]
                gen_path = gen_dir / gen_rel
                ref_paths = [ref_root / p for p in row["ref_videos"]]

                result = {
                    "prompt_id": pid,
                    "dataset": row["dataset"],
                    "key": json.dumps(row["key"], ensure_ascii=False, sort_keys=True),
                    "prompt": row["prompt"],
                    "gen_path": str(gen_path),
                    "app_div": "",
                    "temp_consist": "",
                    "pick_score": "",
                    "motion_fidelity": "",
                    "ok": "False",
                    "error": "",
                }

                try:
                    if not gen_path.exists():
                        raise FileNotFoundError(f"{gen_path} missing")
                    frames = read_video_frames(gen_path)
                    pil = frames_to_pil(frames)
                    clip_out = clip.score_video(pil, row["prompt"])
                    result["app_div"] = f"{clip_out['app_div']:.6f}"
                    result["temp_consist"] = f"{clip_out['temp_consist']:.6f}"
                    result["pick_score"] = f"{clip_out['pick_score']:.6f}"

                    if mf is not None:
                        score = mf.score(gen_path, ref_paths)
                        result["motion_fidelity"] = f"{score:.6f}"

                    result["ok"] = "True"
                except Exception as e:  # noqa: BLE001
                    result["error"] = f"{type(e).__name__}: {e}"[:300]
                    print(f"[eval] FAIL {pid}: {result['error']}", file=sys.stderr)

                writer.writerow(result)
                fh.flush()
                dt = time.time() - t0
                if i % 10 == 0 or i == len(pending):
                    print(f"[eval] {i}/{len(pending)}  last={dt:.1f}s  pid={pid}",
                          flush=True)
        finally:
            fh.close()

    # ---- Aggregates ----
    print()
    print("[eval] aggregates by dataset:")
    sums: dict[str, dict[str, list[float]]] = {}
    with open(output) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("ok") != "True":
                continue
            ds = r.get("dataset", "unknown")
            d = sums.setdefault(ds, {
                "app_div": [], "temp_consist": [], "pick_score": [], "motion_fidelity": [],
            })
            for k in d:
                v = r.get(k, "")
                if v != "":
                    try:
                        d[k].append(float(v))
                    except ValueError:
                        pass

    for ds in sorted(sums):
        parts = [f"{ds}: n={len(sums[ds]['app_div'])}"]
        for k in ("app_div", "temp_consist", "pick_score", "motion_fidelity"):
            vals = sums[ds][k]
            if vals:
                parts.append(f"{k}={sum(vals)/len(vals):.4f}")
            else:
                parts.append(f"{k}=NA")
        print("  " + "  ".join(parts))
    print()
    print(f"[eval] wrote {output}")

    _maybe_log_to_wandb(args, sums, output, gen_dir, ref_root, common)


def _maybe_log_to_wandb(args, sums, output, gen_dir, ref_root, common):
    """Log aggregates + per-prompt table to wandb. Non-fatal on failure.

    Scalars land in Charts as ``<dataset>/<metric>_mean``. The full per-prompt
    table is logged as ``per_prompt`` so outliers can be sorted/filtered in the
    wandb UI without grepping the CSV.
    """
    if args.no_wandb or os.environ.get("WANDB_MODE", "") == "disabled":
        return
    try:
        import wandb
    except ImportError:
        print("[eval] wandb not installed; skipping wandb logging", file=sys.stderr)
        return

    run_name = args.wandb_run_name or output.parent.name
    config = {
        "ckpt": args.ckpt_tag or "",
        "run_dir": str(gen_dir),
        "ref_root": str(ref_root),
        "n_prompts_scored": len(common),
        "skip_motion_fidelity": args.skip_motion_fidelity,
        "limit": args.limit,
        "n_frames_mf": args.n_frames_mf,
        "grid_size_mf": args.grid_size_mf,
    }

    run = None
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            job_type="motion_eval",
            reinit=True,
        )

        scalars: dict = {}
        for ds in sorted(sums):
            scalars[f"{ds}/n"] = len(sums[ds]["app_div"])
            for k in ("app_div", "temp_consist", "pick_score", "motion_fidelity"):
                vals = sums[ds][k]
                if vals:
                    scalars[f"{ds}/{k}_mean"] = sum(vals) / len(vals)
        wandb.log(scalars)

        table_cols = [c for c in CSV_COLUMNS if c != "gen_path"]
        table = wandb.Table(columns=table_cols)
        with open(output) as f:
            reader = csv.DictReader(f)
            for r in reader:
                table.add_data(*[r.get(c, "") for c in table_cols])
        wandb.log({"per_prompt": table})

        url = getattr(run, "url", None)
        if url:
            print(f"[eval] wandb run: {url}")
    except Exception as e:  # noqa: BLE001
        print(f"[eval] wandb logging failed: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        if run is not None:
            try:
                wandb.finish()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
