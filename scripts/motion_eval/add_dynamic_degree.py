"""Post-hoc utility: add VBench Dynamic Degree columns to an existing scores.csv.

Reads the CSV, walks every row's gen_path, computes Dynamic Degree (RAFT
top-5% flow magnitude, VBench algorithm), writes a new CSV with two extra
columns appended:

    dynamic_score   continuous, mean of per-pair top-5% flow magnitude
    is_dynamic      "True"/"False" per VBench resolution-scaled threshold

Idempotent: rows where ``dynamic_score`` is already populated are skipped.
Use this to retro-fit the metric onto runs that finished before the metric
existed in scripts/motion_eval/run_eval.py.

Usage:
    python scripts/motion_eval/add_dynamic_degree.py \\
        --input  $LL_DATA/motion_eval_runs/baseline_v1_fixed_8894159/scores.csv \\
        --output $LL_DATA/motion_eval_runs/baseline_v1_fixed_8894159/scores.csv

(Reading and writing the same path is supported via tmp+rename.)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.dynamic_degree import DynamicDegree  # noqa: E402


EXTRA_COLS = ("dynamic_score", "is_dynamic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Existing scores.csv path")
    ap.add_argument("--output", required=True,
                    help="Output CSV (may equal --input — atomic rename used)")
    ap.add_argument("--cache_dir", default=None,
                    help="RAFT flow cache (default <input_dir>/cache/dynamic_degree)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only compute the first N missing rows (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute all rows, even those with dynamic_score already set "
                         "(use after fixing a metric bug to invalidate stale values)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else inp.parent / "cache" / "dynamic_degree"

    with open(inp, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Append the new columns at the schema level if not already present
    for col in EXTRA_COLS:
        if col not in fieldnames:
            fieldnames.append(col)

    # Identify rows still needing DD
    pending_idx = []
    for i, r in enumerate(rows):
        if r.get("ok") != "True":
            continue
        if not args.force and r.get("dynamic_score", "") != "":
            continue
        if not r.get("gen_path"):
            continue
        pending_idx.append(i)

    if args.limit is not None:
        pending_idx = pending_idx[: args.limit]

    print(f"[add_dd] {len(rows)} rows total, {len(pending_idx)} pending DD")
    if not pending_idx:
        print("[add_dd] nothing to do")
        _write(out, fieldnames, rows)
        return

    dd = DynamicDegree(device=args.device, cache_dir=cache_dir)
    print(f"[add_dd] cache: {cache_dir}")

    for n, i in enumerate(pending_idx, 1):
        r = rows[i]
        gp = r["gen_path"]
        t0 = time.time()
        try:
            res = dd.score(gp)
            r["dynamic_score"] = f"{res['dynamic_score']:.6f}"
            r["is_dynamic"] = str(res["is_dynamic"])
            ok = True
        except Exception as e:  # noqa: BLE001
            r["dynamic_score"] = ""
            r["is_dynamic"] = ""
            r.setdefault("error", "")
            if not r["error"]:
                r["error"] = f"DD: {type(e).__name__}: {e}"[:300]
            ok = False
            print(f"[add_dd] FAIL {r.get('prompt_id', '?')}: {type(e).__name__}: {e}",
                  file=sys.stderr)
        dt = time.time() - t0
        if n % 10 == 0 or n == len(pending_idx):
            print(f"[add_dd] {n}/{len(pending_idx)}  last={dt:.1f}s  "
                  f"pid={r.get('prompt_id', '?')}  ok={ok}", flush=True)
        # Flush every 25 rows to disk for crash safety
        if n % 25 == 0:
            _write(out, fieldnames, rows)

    _write(out, fieldnames, rows)
    print(f"[add_dd] wrote {out}")


def _write(out_path: Path, fieldnames: list[str], rows: list[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Make sure every row has all fields; missing → ""
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
