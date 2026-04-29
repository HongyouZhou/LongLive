"""Parse VBench's prompt-metadata JSON into the artifacts our dispatcher
and scorer consume.

Inputs
------
--vbench_info <path>        # VBench_full_info.json from the VBench repo
                            # (https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json)
--output_dir <dir>          # where to write the three artifacts below

Outputs (under <output_dir>/)
-----------------------------
prompts.txt                 # 1 prompt per line, sha8-sorted (deterministic ordering)
prompt_map.json             # {"<sha8>.mp4": "<prompt text>"}, consumed by VBench scorer
manifest.jsonl              # {"sha8": "...", "prompt": "...", "dimensions": [...]} per row,
                            # consumed by eval_dispatch.py to know which dim a prompt covers

Each prompt becomes ONE video. Many VBench dimensions share prompts, so we
generate per-prompt (not per-prompt-per-dim) and the scorer figures out
which dims a video covers via prompt-text matching against full_info.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbench_info", required=True,
                    help="Path to VBench_full_info.json (from VBench repo)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap to first N prompts (for smoke / debug runs)")
    args = ap.parse_args()

    info_path = Path(args.vbench_info)
    if not info_path.exists():
        sys.exit(f"[build] VBench info not found at {info_path}")
    with open(info_path) as f:
        info = json.load(f)

    # Dedup by prompt text; aggregate the dim list. VBench_full_info has one
    # row per (prompt, dim) tuple in some versions and per-prompt in others —
    # be robust to both.
    by_prompt: dict[str, list[str]] = {}
    for row in info:
        prompt = row.get("prompt_en") or row.get("prompt")
        if not prompt:
            continue
        dims = row.get("dimension") or []
        if isinstance(dims, str):
            dims = [dims]
        bucket = by_prompt.setdefault(prompt, [])
        for d in dims:
            if d not in bucket:
                bucket.append(d)

    # sha8 ordering (deterministic across machines, no Python set quirks)
    rows = sorted(
        ({"sha8": sha8(p), "prompt": p, "dimensions": d}
         for p, d in by_prompt.items()),
        key=lambda r: r["sha8"],
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "prompts.txt", "w") as f:
        for r in rows:
            f.write(r["prompt"] + "\n")

    prompt_map = {f"{r['sha8']}.mp4": r["prompt"] for r in rows}
    with open(out_dir / "prompt_map.json", "w") as f:
        json.dump(prompt_map, f, indent=2, ensure_ascii=False)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build] {len(rows)} unique prompts → {out_dir}", flush=True)
    dims = sorted({d for r in rows for d in r["dimensions"]})
    print(f"[build] {len(dims)} dimensions: {dims}", flush=True)


if __name__ == "__main__":
    main()
