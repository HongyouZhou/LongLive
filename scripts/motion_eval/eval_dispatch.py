"""Multi-GPU eval dispatcher for motion-customization eval generation.

Boots N persistent ``scripts/vbench/eval_worker.py`` subprocesses (one per
GPU; reused verbatim — the worker is method-agnostic) and feeds each a
stream of JSON requests built from a motion-eval manifest. Output mp4
filename = ``{prompt_id}.mp4``. After generation, writes a
``manifest.json`` mapping prompt_id -> mp4 path that ``run_eval.py``
consumes (the "option C" handoff contract — eval doesn't glob filenames).

Inputs
------
--config <path>             # configs/motion_eval_inference.yaml
--ckpt <path>               # LoRA checkpoint to evaluate
--base_ckpt <path>          # Optional. Override config.generator_ckpt
--manifest <path>           # JSONL from scripts/motion_eval/build_manifest.py
--output_dir <dir>          # videos go to <output_dir>/videos/<prompt_id>.mp4
--gpu_ids 0,1,2,3,4,5,6,7
--limit N                   # cap to first N rows (smoke runs)

Outputs
-------
<output_dir>/videos/<prompt_id>.mp4   # one mp4 per prompt
<output_dir>/manifest.json            # {prompt_id: relative-mp4-path}, eval input
<output_dir>/dispatch_log.jsonl       # per-request audit log
<output_dir>/config.snapshot.yaml     # frozen rendered config
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "vbench" / "eval_worker.py"


def _load_manifest(path: Path, limit: Optional[int]) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def _render_config(args: argparse.Namespace, output_dir: Path) -> Path:
    cfg = OmegaConf.load(args.config)
    if args.base_ckpt:
        cfg.generator_ckpt = args.base_ckpt
    cfg.lora_ckpt = args.ckpt
    cfg.output_folder = str(output_dir / "videos")
    snap = output_dir / "config.snapshot.yaml"
    OmegaConf.save(cfg, snap)
    return snap


class Worker:
    """Persistent eval_worker subprocess on one GPU."""

    def __init__(self, gpu_id: int, config_path: str, python_bin: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [python_bin, str(WORKER_SCRIPT), "--config_path", config_path]
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
        self._stderr_pump = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_pump.start()
        self._await_ready()

    def _pump_stderr(self) -> None:
        for line in self.proc.stderr:
            sys.stderr.write(f"[gpu{self.gpu_id}] {line}")
            sys.stderr.flush()

    def _await_ready(self) -> None:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"[gpu{self.gpu_id}] worker exited before ready signal")
        msg = json.loads(line)
        if msg.get("status") != "ready":
            raise RuntimeError(f"[gpu{self.gpu_id}] unexpected ready payload: {msg}")
        print(f"[dispatch] gpu{self.gpu_id} ready", flush=True)

    def send(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"[gpu{self.gpu_id}] worker died while waiting for response")
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


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.manifest), args.limit)
    print(f"[dispatch] {len(manifest)} prompts in manifest", flush=True)

    def _video_path(row: dict) -> Path:
        return videos_dir / f"{row['prompt_id']}.mp4"

    pending = [r for r in manifest if not _video_path(r).exists()]
    skipped = len(manifest) - len(pending)
    if skipped:
        print(f"[dispatch] skipping {skipped} prompts already done", flush=True)
    print(f"[dispatch] {len(pending)} videos to generate", flush=True)

    snapshot = _render_config(args, output_dir)
    print(f"[dispatch] rendered config snapshot to {snapshot}", flush=True)

    # Always emit the eval manifest (prompt_id -> mp4 path), even if pending
    # is empty — eval phase needs it to find every video, including ones
    # that were already on disk from a prior partial run.
    eval_manifest = {
        r["prompt_id"]: str(_video_path(r).relative_to(output_dir))
        for r in manifest
    }
    eval_manifest_path = output_dir / "manifest.json"
    with open(eval_manifest_path, "w") as f:
        json.dump(eval_manifest, f, indent=2, ensure_ascii=False)
    print(f"[dispatch] wrote {eval_manifest_path}", flush=True)

    if not pending:
        print("[dispatch] nothing to generate; exiting", flush=True)
        return

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        sys.exit("[dispatch] --gpu_ids is required (e.g. '0,1,2,3,4,5,6,7')")
    gpu_ids = gpu_ids[: len(pending)]
    print(f"[dispatch] using GPUs: {gpu_ids}", flush=True)

    # Build request queue. 1 sample per prompt (no num_samples knob for
    # motion eval — Yatim metric is deterministic on a single sample;
    # variance estimation across samples isn't the point here).
    q: "queue.Queue[Optional[tuple[int, dict]]]" = queue.Queue()
    for idx, row in enumerate(pending):
        q.put((idx, {
            "prompt": row["prompt"],
            "seed": int(args.seed),
            "output_mp4": str(_video_path(row)),
        }))

    # Boot workers in parallel.
    workers: list[Worker] = []
    boot_lock = threading.Lock()

    def boot(gpu_id: int) -> None:
        w = Worker(gpu_id, str(snapshot), args.python_bin)
        with boot_lock:
            workers.append(w)

    boot_threads = [threading.Thread(target=boot, args=(g,), daemon=False)
                    for g in gpu_ids]
    for t in boot_threads:
        t.start()
    for t in boot_threads:
        t.join()
    if len(workers) != len(gpu_ids):
        sys.exit(f"[dispatch] only {len(workers)}/{len(gpu_ids)} workers booted")

    log_path = output_dir / "dispatch_log.jsonl"
    log_lock = threading.Lock()

    def append_log(entry: dict) -> None:
        with log_lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    done_count = [0]
    fail_count = [0]
    progress_lock = threading.Lock()

    def worker_loop(w: Worker) -> None:
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            _idx, req = item
            t0 = time.time()
            try:
                resp = w.send(req)
            except Exception as e:
                resp = {"ok": False, "error": f"dispatcher: {e}"}
            wall = round(time.time() - t0, 2)
            entry = {
                "mp4": Path(req["output_mp4"]).name,
                "gpu": w.gpu_id,
                "wall_s": wall,
                "ok": bool(resp.get("ok")),
            }
            if not resp.get("ok"):
                entry["error"] = resp.get("error", "")[:300]
            append_log(entry)
            with progress_lock:
                done_count[0] += 1
                if not resp.get("ok"):
                    fail_count[0] += 1
                if done_count[0] % 20 == 0 or done_count[0] == len(pending):
                    print(
                        f"[dispatch] {done_count[0]}/{len(pending)}  "
                        f"fails={fail_count[0]}",
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

    print(f"[dispatch] generated {done_count[0]} videos in {t_total:.1f}s "
          f"({fail_count[0]} failures)", flush=True)
    if fail_count[0]:
        sys.exit(f"[dispatch] {fail_count[0]} prompts failed; see {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/motion_eval_inference.yaml")
    p.add_argument("--ckpt", required=True,
                   help="LoRA checkpoint to evaluate (.pt)")
    p.add_argument("--base_ckpt", default="",
                   help="Base generator .pt (defaults to config.generator_ckpt)")
    p.add_argument("--manifest", required=True,
                   help="manifest.jsonl from scripts/motion_eval/build_manifest.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", required=True,
                   help="Comma-separated physical GPU ids")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--python_bin",
        default=str(Path.home() / "miniforge3/envs/longlive/bin/python"),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
