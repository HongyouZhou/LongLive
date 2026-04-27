"""
Merge per-stream master JSONs into a single master_all.json.

After dual-stream mining the head-motion master metadata is split across:
    logs/diag_head_pose_all_parts.json   # single-stream phase (parts 0..121)
    logs/master_A.json                   # stream A (parts 122..145)
    logs/master_B.json                   # stream B (parts 146..182)

Each entry is keyed by clip name. There is no overlap by design (different
parts contain different clips), but if a clip appears in more than one
master, prefer the entry with ``saved == True`` and the larger ``yaw_range``.

Usage:
    python scripts/merge_master_jsons.py
    python scripts/merge_master_jsons.py --out logs/master_all.json
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", default=[
        "logs/diag_head_pose_all_parts.json",
        "logs/master_A.json",
        "logs/master_B.json",
    ])
    ap.add_argument("--out", default="logs/master_all.json")
    args = ap.parse_args()

    merged: dict = {}
    n_per_input = {}
    n_replaced = 0

    for src in args.inputs:
        p = Path(src)
        if not p.exists():
            print(f"[warn] {src} not found, skipping")
            continue
        with open(p) as f:
            d = json.load(f)
        n_per_input[src] = len(d)
        for name, meta in d.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = meta
                continue
            # Both have entries; prefer saved=True, then larger yaw_range.
            ex_saved = bool(existing.get("saved"))
            new_saved = bool(meta.get("saved"))
            if new_saved and not ex_saved:
                merged[name] = meta
                n_replaced += 1
                continue
            if ex_saved and not new_saved:
                continue
            ex_yr = existing.get("yaw_range", -1)
            new_yr = meta.get("yaw_range", -1)
            if new_yr is not None and ex_yr is not None and new_yr > ex_yr:
                merged[name] = meta
                n_replaced += 1

    print(f"[info] inputs:")
    for k, v in n_per_input.items():
        print(f"  {k}: {v} entries")
    print(f"[info] merged: {len(merged)} entries  (replaced on collision: {n_replaced})")

    saved = sum(1 for v in merged.values() if v.get("saved"))
    errors = sum(1 for v in merged.values() if v.get("error"))
    yaw_known = sum(1 for v in merged.values() if "yaw_range" in v)
    print(f"[info] saved=True: {saved}    yaw_known: {yaw_known}    error: {errors}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1, sort_keys=False)
    tmp.rename(out_path)
    print(f"[done] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
