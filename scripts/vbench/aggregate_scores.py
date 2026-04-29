"""Aggregate VBench per-dim results into a single Total / Quality / Semantic
triple matching the paper's reporting format (i.e. the leaderboard).

Reads ``<run_dir>/vbench_out/<name>_eval_results.json`` produced by
run_vbench_eval.py and writes ``<run_dir>/summary.json`` plus one
human-readable line to stdout.

Constants below mirror VBench's `scripts/constant.py` (commit on master at
2026-04). The leaderboard formula has two ingredients many naive aggregators
miss:

1. **Min-Max normalization per dim** — each dim's raw score is mapped
   ``(raw - Min) / (Max - Min)`` BEFORE being averaged. This rescales dims
   like ``temporal_style`` (raw max ≈ 0.36) onto the same 0..1 range as
   ``subject_consistency`` (raw max = 1.0). Without it, low-cap dims
   pull Semantic down by 10+ points.
2. **Total = (4·Quality + 1·Semantic) / 5** — Quality is weighted 4× heavier
   than Semantic. Verified: ``0.8·86.97 + 0.2·76.47 = 84.87`` matches the
   LongLive paper Table 1 exactly.

If VBench updates its constants, copy them here verbatim from
``Vchitect/VBench/scripts/constant.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Min/Max are leaderboard-derived; they shift each dim onto a shared [0, 1].
# Using underscored keys to match the JSON output of run_vbench_eval.py;
# upstream constant.py uses spaces.
NORMALIZE_DIC = {
    "subject_consistency":    {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal_flickering":    {"Min": 0.6293, "Max": 1.0},
    "motion_smoothness":      {"Min": 0.706,  "Max": 0.9975},
    "dynamic_degree":         {"Min": 0.0,    "Max": 1.0},
    "aesthetic_quality":      {"Min": 0.0,    "Max": 1.0},
    "imaging_quality":        {"Min": 0.0,    "Max": 1.0},
    "object_class":           {"Min": 0.0,    "Max": 1.0},
    "multiple_objects":       {"Min": 0.0,    "Max": 1.0},
    "human_action":           {"Min": 0.0,    "Max": 1.0},
    "color":                  {"Min": 0.0,    "Max": 1.0},
    "spatial_relationship":   {"Min": 0.0,    "Max": 1.0},
    "scene":                  {"Min": 0.0,    "Max": 0.8222},
    "appearance_style":       {"Min": 0.0009, "Max": 0.2855},
    "temporal_style":         {"Min": 0.0,    "Max": 0.364},
    "overall_consistency":    {"Min": 0.0,    "Max": 0.364},
}

# dynamic_degree counted at 0.5× because it's a binary motion-presence signal,
# not a continuous quality metric (per VBench paper).
DIM_WEIGHT = {
    "subject_consistency":    1.0,
    "background_consistency": 1.0,
    "temporal_flickering":    1.0,
    "motion_smoothness":      1.0,
    "aesthetic_quality":      1.0,
    "imaging_quality":        1.0,
    "dynamic_degree":         0.5,
    "object_class":           1.0,
    "multiple_objects":       1.0,
    "human_action":           1.0,
    "color":                  1.0,
    "spatial_relationship":   1.0,
    "scene":                  1.0,
    "appearance_style":       1.0,
    "temporal_style":         1.0,
    "overall_consistency":    1.0,
}

QUALITY_LIST = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "aesthetic_quality", "imaging_quality",
    "dynamic_degree",
]
SEMANTIC_LIST = [
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "appearance_style", "temporal_style",
    "overall_consistency",
]

# Total = (QUALITY_WEIGHT·Q + SEMANTIC_WEIGHT·S) / (QUALITY_WEIGHT+SEMANTIC_WEIGHT)
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def _load_eval_results(vbench_out: Path) -> dict:
    """VBench writes ONE <name>_eval_results.json containing
    ``{dim: [score, [per_prompt_scores]]}`` for ALL dims at end of run.
    Pick most recent if multiple runs share the dir."""
    candidates = list(vbench_out.glob("*_eval_results.json"))
    if not candidates:
        return {}
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(chosen) as f:
        return json.load(f)


def _raw_score(eval_results: dict, dim: str) -> float | None:
    """Returns raw (un-normalized) dim score in [0, 1], or None if missing."""
    val = eval_results.get(dim)
    if val is None:
        return None
    if isinstance(val, list) and val:
        val = val[0]
    return float(val)


def _normalized(raw: float, dim: str) -> float:
    """Apply Min/Max → clamp to [0, 1] (a model can exceed leaderboard max)."""
    bounds = NORMALIZE_DIC[dim]
    n = (raw - bounds["Min"]) / (bounds["Max"] - bounds["Min"])
    return max(0.0, min(1.0, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="vbench_runs/<run-id>")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    vbench_out = run_dir / "vbench_out"
    if not vbench_out.exists():
        sys.exit(f"[agg] {vbench_out} not found")

    eval_results = _load_eval_results(vbench_out)

    per_dim_raw: dict[str, float | None] = {}
    per_dim_norm: dict[str, float | None] = {}
    for d in QUALITY_LIST + SEMANTIC_LIST:
        raw = _raw_score(eval_results, d)
        per_dim_raw[d] = raw
        per_dim_norm[d] = None if raw is None else _normalized(raw, d)

    def weighted(dims: list[str]) -> float | None:
        present = [(d, DIM_WEIGHT[d]) for d in dims
                   if per_dim_norm.get(d) is not None]
        if not present:
            return None
        wsum = sum(w for _, w in present)
        return sum(per_dim_norm[d] * w for d, w in present) / wsum  # type: ignore[operator]

    quality = weighted(QUALITY_LIST)
    semantic = weighted(SEMANTIC_LIST)
    if quality is not None and semantic is not None:
        total = (QUALITY_WEIGHT * quality + SEMANTIC_WEIGHT * semantic) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT)
    else:
        total = None

    # Convert to 0–100 for the human-facing summary.
    quality_pct  = None if quality  is None else quality  * 100.0
    semantic_pct = None if semantic is None else semantic * 100.0
    total_pct    = None if total    is None else total    * 100.0

    # Try to enrich with ckpt info from the snapshot.
    ckpt = None
    snap = run_dir / "config.snapshot.yaml"
    if snap.exists():
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(snap)
            ckpt = {
                "generator_ckpt": str(cfg.get("generator_ckpt", "")),
                "lora_ckpt": str(cfg.get("lora_ckpt", "")),
                "num_output_frames": int(cfg.get("num_output_frames", -1)),
            }
        except Exception as e:
            ckpt = {"error": f"could not parse snapshot: {e}"}

    # per_dim in summary reports the *normalized* score × 100, since that is
    # what the paper Table 1 cell-level numbers refer to. raw is also stored
    # for debugging / re-aggregation against future VBench releases.
    summary = {
        "run_dir": str(run_dir),
        "total":    None if total_pct    is None else round(total_pct, 3),
        "quality":  None if quality_pct  is None else round(quality_pct, 3),
        "semantic": None if semantic_pct is None else round(semantic_pct, 3),
        "per_dim":  {d: (None if v is None else round(v * 100.0, 3))
                     for d, v in per_dim_norm.items()},
        "per_dim_raw": {d: (None if v is None else round(v, 6))
                        for d, v in per_dim_raw.items()},
        "ckpt": ckpt,
    }
    out_path = run_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"
    print(f"[agg] Total={fmt(total_pct)}  Quality={fmt(quality_pct)}  "
          f"Semantic={fmt(semantic_pct)}  →  {out_path}", flush=True)
    missing = [d for d, v in per_dim_norm.items() if v is None]
    if missing:
        print(f"[agg] WARNING: {len(missing)} dims missing: {missing}",
              flush=True)


if __name__ == "__main__":
    main()
