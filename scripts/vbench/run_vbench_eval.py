"""Run VBench scoring on a directory of generated mp4s.

This is a thin wrapper around VBench's Python API. It runs *all* 16
dimensions in one Python process so detectron2 / CLIP / DOVER weights
load only once (running each dim as a separate subprocess wastes ~10
minutes per call on model loading alone).

Inputs
------
--videos_dir <path>         # directory of mp4s; each filename matches a key
                            # in prompt_map.json
--prompt_map <path>         # {"<sha8>.mp4": "<prompt>"} from build_vbench_prompts.py
--vbench_info <path>        # VBench_full_info.json (provides per-prompt dim list)
--output_dir <path>         # vbench writes per-dim JSONs here
--dimensions <csv>          # optional, default = "all" (run all 16 dims)
--device cuda

Outputs (under --output_dir/)
-----------------------------
results_<dimension>_eval_results.json   # one per dim, vbench native format
results_<dimension>_full_info.json      # one per dim, prompt-level scores

Notes
-----
This script must run inside the dedicated `vbench` mamba env (where
detectron2 + vbench are installed); it is incompatible with the
`longlive` env's torch 2.8 + cu128.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# All 16 standard VBench dimensions. We run them all by default to match
# the paper's Total / Quality / Semantic aggregation; subset via --dimensions
# for debug/smoke runs.
ALL_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--vbench_info", required=True,
                    help="VBench_full_info.json path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dimensions", default="all",
                    help="comma-separated dim list, or 'all' for all 16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dimensions == "all":
        dims = ALL_DIMENSIONS
    else:
        dims = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        unknown = set(dims) - set(ALL_DIMENSIONS)
        if unknown:
            sys.exit(f"[score] unknown VBench dimensions: {sorted(unknown)}")

    # Filter out dims with zero matching videos in videos_dir.
    # VBench's standard mode crashes with ZeroDivisionError when a dim
    # iterates an empty video list (background_consistency, etc.).
    # This matters mostly for --limit smoke runs; a full 946-prompt run
    # has every dim covered.
    present_prompts = set()
    for fname in os.listdir(videos_dir):
        m = re.match(r"^(.+)-\d+\.mp4$", fname)
        if m:
            present_prompts.add(m.group(1))
    if not present_prompts:
        sys.exit(f"[score] no <prompt>-N.mp4 files in {videos_dir}")

    with open(args.vbench_info) as f:
        full_info = json.load(f)

    def _norm(p: str) -> str:
        # Mirror eval_dispatch._sanitize: only `/` is replaced.
        return p.replace("/", "_").strip()

    dim_to_prompts: dict[str, set[str]] = {d: set() for d in dims}
    for row in full_info:
        prompt = row.get("prompt_en") or row.get("prompt") or ""
        prompt = _norm(prompt)
        row_dims = row.get("dimension") or []
        if isinstance(row_dims, str):
            row_dims = [row_dims]
        for d in row_dims:
            if d in dim_to_prompts:
                dim_to_prompts[d].add(prompt)

    covered_dims = [d for d in dims
                    if dim_to_prompts[d] & present_prompts]
    skipped = [d for d in dims if d not in covered_dims]
    if skipped:
        print(f"[score] skipping {len(skipped)} dims with no video coverage "
              f"(--limit smoke?): {skipped}", flush=True)
    dims = covered_dims
    if not dims:
        sys.exit("[score] no VBench dim has ≥1 matching video; nothing to score")

    # Lazy import so we can read --help without the heavy dep installed.
    try:
        from vbench import VBench  # type: ignore
        import torch
    except ImportError as e:
        sys.exit(f"[score] vbench not installed in this env: {e}")

    # VBench's multiple_objects dim calls `device.type` directly — passing
    # a string fails with AttributeError. Coerce here to torch.device.
    device = torch.device(args.device)

    print(f"[score] videos_dir = {videos_dir}", flush=True)
    print(f"[score] dims       = {dims}", flush=True)
    print(f"[score] output     = {output_dir}", flush=True)

    bench = VBench(device, args.vbench_info, str(output_dir))
    t0 = time.time()
    # Standard (default) mode: VBench reads VBench_full_info.json and globs
    # `<prompt>-*.mp4` in videos_path. Filenames must be literal prompt
    # text + dash + index — see eval_dispatch.py's _video_path helper.
    # custom_input mode is intentionally NOT used: it disables 6/16 dims
    # (object_class, spatial_relationship, scene, color, appearance_style,
    # multiple_objects) that need ground-truth metadata from full_info.json.
    bench.evaluate(
        videos_path=str(videos_dir),
        name="longlive_eval",
        dimension_list=dims,
    )
    print(f"[score] vbench evaluate done in {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
