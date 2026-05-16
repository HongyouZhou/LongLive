"""One-off diagnostic: inspect a .pt checkpoint's top-level structure.

Use to verify a state-dict layout before writing load_state_dict code that
depends on the exact key naming (e.g. "generator" vs "model" vs raw sd).

Run on a compute node (frontend forbidden for heavy torch.load — see
memory `hpc-frontend-no-compute`):

    srun -p compute -c 8 --mem=64G -t 00:05:00 \\
        python3 scripts/local/inspect_ckpt.py <path>
"""
from __future__ import annotations

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".pt checkpoint to inspect")
    ap.add_argument("--n_inner", type=int, default=5,
                    help="Sample N inner keys per top-level dict (default 5)")
    args = ap.parse_args()

    sd = torch.load(args.path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        print(f"top-level type: {type(sd).__name__}")
        return

    print(f"top-level keys ({len(sd)}): {list(sd.keys())}")
    for k in list(sd.keys()):
        v = sd[k]
        if isinstance(v, dict):
            inner = list(v.keys())
            print(f"  {k}: dict with {len(inner)} entries; first {args.n_inner}:")
            for ik in inner[: args.n_inner]:
                iv = v[ik]
                if hasattr(iv, "shape"):
                    print(f"    {ik}: shape={tuple(iv.shape)}, dtype={iv.dtype}")
                else:
                    print(f"    {ik}: type={type(iv).__name__}")
        elif hasattr(v, "shape"):
            print(f"  {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")


if __name__ == "__main__":
    main()
