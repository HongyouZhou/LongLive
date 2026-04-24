"""8-GPU eval dispatcher for Motion-Recache.

Boots N persistent ``eval_worker.py`` subprocesses (one per GPU), feeds
each a stream of JSON requests from ``eval_set.jsonl``, and aggregates
responses into ``summary.json`` plus optional wandb video/metric logging.

The dispatcher itself can be re-run against the same workers by setting
``--reuse`` once the workers exist; for simplicity the first cut spawns
workers every call. Cold boot per worker ~20s so 8 parallel ~20s total.

Example
-------
    python scripts/eval_dispatch.py \
        --config configs/longlive_inference_motion.yaml \
        --ckpt /tmp/ckpt_test/model.pt \
        --eval_set prompts/eval_set.jsonl \
        --output_dir /tmp/eval_out \
        --num_gpus 8 \
        --limit 8

Outputs
-------
``<output_dir>/<motion_stem>_s<seed>.mp4``   per request
``<output_dir>/summary.json``                aggregated metrics
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_REF_ROOT = Path("/home/hongyou/dev/data/wm/motion_refs")


def _resolve_motion_ref(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = MOTION_REF_ROOT / rel_or_abs
    if candidate.exists():
        return str(candidate)
    return str(p)


def _load_eval_set(path: Path, limit: Optional[int]) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


class Worker:
    """Thin wrapper around a persistent eval_worker subprocess."""

    def __init__(self, gpu_id: int, config_path: str, python_bin: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        script = REPO_ROOT / "scripts" / "eval_worker.py"
        cmd = [python_bin, str(script), "--config_path", config_path]
        self.gpu_id = gpu_id
        self.proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_pump = threading.Thread(
            target=self._pump_stderr, daemon=True)
        self._stderr_pump.start()
        self._await_ready()

    def _pump_stderr(self) -> None:
        for line in self.proc.stderr:
            sys.stderr.write(f"[gpu{self.gpu_id}] {line}")
            sys.stderr.flush()

    def _await_ready(self) -> None:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"[gpu{self.gpu_id}] worker exited before ready signal")
        msg = json.loads(line)
        if msg.get("status") != "ready":
            raise RuntimeError(
                f"[gpu{self.gpu_id}] unexpected ready payload: {msg}")
        print(f"[dispatch] gpu{self.gpu_id} ready", flush=True)

    def send(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"[gpu{self.gpu_id}] worker died while waiting for response")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def _preload_ramdisk(eval_set: list[dict], ramdisk: Path) -> dict[str, str]:
    """Copy each distinct motion_ref to /dev/shm so all 8 workers hit RAM."""
    ramdisk.mkdir(exist_ok=True)
    mapping = {}
    for row in eval_set:
        src = _resolve_motion_ref(row["motion_ref"])
        dst = ramdisk / Path(src).name
        if not dst.exists():
            shutil.copy2(src, dst)
        mapping[row["motion_ref"]] = str(dst)
    print(f"[dispatch] preloaded {len(mapping)} motion_refs to {ramdisk}",
          flush=True)
    return mapping


def _build_request(
    row: dict,
    *,
    ckpt_path: str,
    output_dir: Path,
    motion_ref_abs: str,
    num_frames: int,
) -> dict:
    stem = Path(row["motion_ref"]).stem
    seed = int(row.get("seed", 0))
    out_mp4 = output_dir / f"{stem}_s{seed}.mp4"
    return {
        "ckpt_path": ckpt_path,
        "prompt": row["prompt"],
        "motion_ref": motion_ref_abs,
        "seed": seed,
        "output_mp4": str(out_mp4),
        "num_frames": num_frames,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_set = _load_eval_set(Path(args.eval_set), args.limit)
    print(f"[dispatch] {len(eval_set)} evaluation tasks", flush=True)

    if args.preload_ramdisk:
        ramdisk_map = _preload_ramdisk(eval_set, Path(args.ramdisk_path))
    else:
        ramdisk_map = {r["motion_ref"]: _resolve_motion_ref(r["motion_ref"])
                       for r in eval_set}

    # Build request queue
    q: "queue.Queue[Optional[tuple[int, dict]]]" = queue.Queue()
    for idx, row in enumerate(eval_set):
        req = _build_request(
            row,
            ckpt_path=os.path.abspath(args.ckpt),
            output_dir=output_dir,
            motion_ref_abs=ramdisk_map[row["motion_ref"]],
            num_frames=args.num_frames,
        )
        q.put((idx, req))

    # Resolve GPU id list: --gpu_ids takes precedence over --num_gpus.
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(args.num_gpus))
    gpu_ids = gpu_ids[: len(eval_set)]
    num_gpus = len(gpu_ids)

    # Boot workers
    workers: list[Worker] = []
    for gpu_id in gpu_ids:
        workers.append(Worker(gpu_id, args.config, args.python_bin))

    results: list[Optional[dict]] = [None] * len(eval_set)

    def worker_loop(w: Worker) -> None:
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            idx, req = item
            t0 = time.time()
            try:
                resp = w.send(req)
            except Exception as e:
                resp = {
                    "ok": False, "error": f"dispatcher: {e}", "request": req}
            resp["_idx"] = idx
            resp["_gpu"] = w.gpu_id
            resp["_wall_s"] = round(time.time() - t0, 2)
            results[idx] = resp
            print(
                f"[dispatch] idx={idx:03d} gpu={w.gpu_id} "
                f"ok={resp.get('ok', False)} wall={resp['_wall_s']}s",
                flush=True,
            )

    threads = [threading.Thread(target=worker_loop, args=(w,), daemon=True)
               for w in workers]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_total = time.time() - t_start

    for w in workers:
        w.close()

    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "num_tasks": len(eval_set),
        "num_gpus": num_gpus,
        "total_wall_s": round(t_total, 2),
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"[dispatch] done in {t_total:.1f}s  "
        f"summary={output_dir/'summary.json'}",
        flush=True,
    )

    if args.wandb_run and args.wandb_run.lower() not in ("", "null", "none"):
        _log_to_wandb(args, summary)


def _log_to_wandb(args: argparse.Namespace, summary: dict) -> None:
    import wandb
    step = None
    for r in summary["results"]:
        if r and r.get("ok") and r.get("ckpt_step") is not None:
            step = int(r["ckpt_step"])
            break
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run,
        resume="allow",
    )
    logged_videos = 0
    total_rel_diff = []
    for r in summary["results"]:
        if not r or not r.get("ok"):
            continue
        total_rel_diff.append(r["metrics"]["flow_rel_diff"])
        if logged_videos < args.wandb_max_videos:
            tag = Path(r["output_mp4"]).stem
            wandb.log(
                {f"eval/vid_{tag}": wandb.Video(r["output_mp4"],
                                                fps=16, format="mp4")},
                step=step,
            )
            logged_videos += 1
    if total_rel_diff:
        import numpy as np
        wandb.log(
            {"eval/flow_rel_diff_mean": float(np.mean(total_rel_diff)),
             "eval/flow_rel_diff_median": float(np.median(total_rel_diff))},
            step=step,
        )
    run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/longlive_inference_motion.yaml")
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint .pt containing generator_lora + motion_encoder")
    p.add_argument("--eval_set", default="prompts/eval_set.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_gpus", type=int, default=8,
                   help="Number of GPUs starting from id 0. Ignored if --gpu_ids is set.")
    p.add_argument("--gpu_ids", type=str, default="",
                   help="Comma-separated physical GPU ids to use (e.g. '3' or '0,3,5'). "
                        "Overrides --num_gpus.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of eval set rows (e.g. 8 for smoke)")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--preload_ramdisk", action="store_true")
    p.add_argument("--ramdisk_path", default="/dev/shm/longlive_motion_refs")
    p.add_argument(
        "--python_bin",
        default=str(Path.home() / "miniforge3/envs/longlive-blackwell/bin/python"),
    )
    p.add_argument("--wandb_run", default="",
                   help='Name of wandb run (empty / "null" disables wandb)')
    p.add_argument("--wandb_project", default="longlive-eval")
    p.add_argument("--wandb_entity", default="hongyou")
    p.add_argument("--wandb_max_videos", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
