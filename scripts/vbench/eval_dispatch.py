"""Multi-GPU eval dispatcher for VBench short generation.

Boots N persistent ``eval_worker.py`` subprocesses (one per GPU), feeds
each a stream of JSON requests built from a manifest of VBench prompts,
and writes the resulting mp4s to ``<output_dir>/videos/<sha8>.mp4``.

Cold boot is ~20s per worker (parallel), then each prompt streams through
the queue. Re-running on the same output_dir skips prompts whose mp4
already exists, so interrupts are essentially free to recover from.

Inputs
------
--config <path>             # YAML; we override generator_ckpt + lora_ckpt + output_folder
--ckpt <path>               # checkpoint to evaluate. NVlabs paper format = lora.pt;
                            # our training output = checkpoint_model_NNNNNN/model.pt
                            #     (single file with generator_lora + critic_lora keys).
--base_ckpt <path>          # base generator weights (rarely changes; defaults to
                            # config.generator_ckpt). Override only when you want to
                            # evaluate a different base.
--manifest <path>           # JSONL from build_vbench_prompts.py
--output_dir <dir>          # videos go to <output_dir>/videos/<sha8>.mp4
--gpu_ids 0,1,2,3,4,5,6,7   # comma-separated physical GPU ids
--limit N                   # cap to first N rows (for smoke runs)

Outputs
-------
<output_dir>/videos/<sha8>.mp4    # one mp4 per prompt (sha8 from manifest)
<output_dir>/dispatch_log.jsonl   # one row per request: {sha8, ok, gpu, wall_s, ...}
<output_dir>/config.snapshot.yaml # frozen copy of the rendered config used
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


def _sanitize(prompt: str) -> str:
    """Make a prompt safe to use as a filename component.

    VBench's standard mode globs `{prompt}-*.mp4` in videos_path and
    matches against literal prompt text. Linux only truly forbids `/`
    and NUL in filenames, so we keep punctuation intact and only
    replace those plus collapse whitespace runs.
    """
    return prompt.replace("/", "_").replace("\x00", "").strip()


def _render_config(args: argparse.Namespace, output_dir: Path) -> Path:
    """Materialize a config snapshot with run-specific overrides applied.

    Worker subprocesses load this snapshot rather than the original yaml so
    that ckpt paths are unambiguous and frozen for the duration of the run.
    """
    cfg = OmegaConf.load(args.config)
    if args.base_ckpt:
        cfg.generator_ckpt = args.base_ckpt
    cfg.lora_ckpt = args.ckpt
    # output_folder is unused in worker (we write per-request paths) but
    # keep the field consistent so downstream tooling that reads the snapshot
    # sees the run's directory.
    cfg.output_folder = str(output_dir / "videos")

    snap = output_dir / "config.snapshot.yaml"
    OmegaConf.save(cfg, snap)
    return snap


class Worker:
    """Thin wrapper around a persistent eval_worker subprocess on one GPU."""

    def __init__(self, gpu_id: int, config_path: str, python_bin: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        script = REPO_ROOT / "scripts" / "vbench" / "eval_worker.py"
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


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.manifest), args.limit)
    n_samples = int(args.num_samples)
    print(f"[dispatch] {len(manifest)} prompts × {n_samples} samples in manifest",
          flush=True)

    # Filename = "<prompt>-<sample_idx>.mp4". VBench's standard mode globs
    # `<prompt>-*.mp4` and expects multiple samples per prompt for variance
    # estimation; the paper's protocol is 5 samples per prompt.
    def _video_path(row: dict, sample_idx: int) -> Path:
        return videos_dir / f"{_sanitize(row['prompt'])}-{sample_idx}.mp4"

    # Build full task list (one entry per (prompt, sample_idx)) then drop
    # those whose mp4 already exists for idempotent resume.
    all_tasks: list[tuple[dict, int]] = [
        (r, s) for r in manifest for s in range(n_samples)
    ]
    pending = [t for t in all_tasks if not _video_path(*t).exists()]
    skipped = len(all_tasks) - len(pending)
    if skipped:
        print(f"[dispatch] skipping {skipped} (prompt, sample) pairs already done",
              flush=True)
    print(f"[dispatch] {len(pending)} videos to generate", flush=True)
    if not pending:
        print("[dispatch] nothing to do; exiting", flush=True)
        return

    snapshot = _render_config(args, output_dir)
    print(f"[dispatch] rendered config snapshot to {snapshot}", flush=True)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        sys.exit("[dispatch] --gpu_ids is required (e.g. '0,1,2,3,4,5,6,7')")
    gpu_ids = gpu_ids[: len(pending)]
    print(f"[dispatch] using GPUs: {gpu_ids}", flush=True)

    # Optional: feed LLM-augmented prompts to the generator while keeping
    # raw VBench prompts as filenames. VBench's paper protocol for Wan-based
    # models uses VBench's official augmented prompts (Qwen2.5-3B, seed 42)
    # — matches `Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt`. Raw
    # prompts are too short (~10 words) and hurt aesthetic / semantic dims.
    aug_map: dict[str, str] = {}
    if args.augmented_prompts and args.raw_prompts:
        with open(args.raw_prompts) as fr, open(args.augmented_prompts) as fa:
            raw_lines = [ln.rstrip("\n") for ln in fr]
            aug_lines = [ln.rstrip("\n") for ln in fa]
        if len(raw_lines) != len(aug_lines):
            sys.exit(f"[dispatch] raw / augmented prompt files length mismatch "
                     f"({len(raw_lines)} vs {len(aug_lines)})")
        aug_map = dict(zip(raw_lines, aug_lines))
        n_matched = sum(1 for r in manifest if r["prompt"] in aug_map)
        print(f"[dispatch] augmented prompts loaded: {len(aug_map)} pairs, "
              f"{n_matched}/{len(manifest)} manifest prompts have an aug version",
              flush=True)
        if n_matched < len(manifest):
            unmatched = [r["prompt"] for r in manifest if r["prompt"] not in aug_map]
            print(f"[dispatch] WARNING: first 3 unmatched: {unmatched[:3]}",
                  flush=True)

    # Build request queue.
    q: "queue.Queue[Optional[tuple[int, dict]]]" = queue.Queue()
    for idx, (row, sample_idx) in enumerate(pending):
        # Generator sees aug prompt (paper protocol); filename uses raw prompt
        # (VBench standard mode globs `<raw>-*.mp4`).
        gen_prompt = aug_map.get(row["prompt"], row["prompt"])
        q.put((idx, {
            "prompt": gen_prompt,
            # Seed varies per sample so the 5 videos for one prompt aren't
            # identical noise (defeats VBench's variance estimation).
            "seed": int(args.seed) + sample_idx,
            "output_mp4": str(_video_path(row, sample_idx)),
        }))

    # Boot workers in parallel (each is ~20s cold; serial would block 8×20s).
    workers: list[Worker] = []
    boot_threads = []
    boot_lock = threading.Lock()

    def boot(gpu_id: int) -> None:
        w = Worker(gpu_id, str(snapshot), args.python_bin)
        with boot_lock:
            workers.append(w)

    for gpu_id in gpu_ids:
        t = threading.Thread(target=boot, args=(gpu_id,), daemon=False)
        t.start()
        boot_threads.append(t)
    for t in boot_threads:
        t.join()
    if len(workers) != len(gpu_ids):
        sys.exit(f"[dispatch] only {len(workers)}/{len(gpu_ids)} workers booted")

    # Per-prompt log; appended atomically.
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
            idx, req = item
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
    p.add_argument("--config", default="configs/vbench_short.yaml")
    p.add_argument("--ckpt", required=True,
                   help="LoRA checkpoint to evaluate (.pt). For NVlabs "
                        "paper baseline this is longlive_models/models/lora.pt; "
                        "for our training output it is "
                        "logs/.../checkpoint_model_NNNNNN/model.pt.")
    p.add_argument("--base_ckpt", default="",
                   help="Base generator .pt (defaults to config.generator_ckpt)")
    p.add_argument("--manifest", required=True,
                   help="manifest.jsonl from build_vbench_prompts.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_ids", required=True,
                   help="Comma-separated physical GPU ids (e.g. '0,1,2,3,4,5,6,7')")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=1,
                   help="Samples per prompt (VBench protocol = 5 for paper-comparable scores)")
    p.add_argument("--augmented_prompts", default="",
                   help="Optional. Path to VBench's LLM-augmented prompt file "
                        "(line-aligned with --raw_prompts). When provided, "
                        "the generator sees the augmented version while the "
                        "mp4 filename keeps the raw VBench prompt for "
                        "scorer compatibility. For Wan-based models, use "
                        "Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt "
                        "from the VBench repo (paper protocol).")
    p.add_argument("--raw_prompts", default="",
                   help="Required iff --augmented_prompts is set. Line-aligned "
                        "raw prompts (VBench's all_dimension.txt).")
    p.add_argument(
        "--python_bin",
        default=str(Path.home() / "miniforge3/envs/longlive/bin/python"),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
