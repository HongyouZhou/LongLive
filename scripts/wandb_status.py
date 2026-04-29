#!/usr/bin/env python
"""Quick wandb status pull for the most recent LongLive run.

Usage:
    python scripts/wandb_status.py              # latest running/finished run
    python scripts/wandb_status.py <run_id>     # specific run
    python scripts/wandb_status.py --tail 30    # last 30 generator rows
"""
import argparse
import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", default=None)
    ap.add_argument("--project", default="hongyou/longlive")
    ap.add_argument("--tail", type=int, default=20)
    args = ap.parse_args()

    api = wandb.Api()
    if args.run_id:
        run = api.run(f"{args.project}/{args.run_id}")
    else:
        run = next(iter(api.runs(args.project, per_page=1, order="-created_at")))

    print(f"run={run.id}  state={run.state}  latest_step={run.summary.get('_step')}")
    cfg_uni = (run.config.get("unidad") or {}).get("dual_domain_dmd") or {}
    print(f"config: guidance_scale={run.config.get('guidance_scale')}  "
          f"unidad_score_weight={cfg_uni.get('unidad_score_weight')}  "
          f"score_lr={cfg_uni.get('score_lr')}\n")

    hist = list(run.scan_history(page_size=500))
    gen = [r for r in hist if r.get("unidad_dmd_ratio") is not None]

    print(f"{'step':>5}  {'uni_dmd':>8}  {'ratio':>8}  {'turn':>8}  "
          f"{'gen_loss':>9}  {'critic':>8}  {'iter_t':>7}")
    for r in gen[-args.tail:]:
        print(f"{r['_step']:>5}  "
              f"{r.get('unidad_score_dmd_loss', 0):>8.4f}  "
              f"{r['unidad_dmd_ratio']:>8.4f}  "
              f"{r.get('unidad_score_turn_loss', 0):>8.4f}  "
              f"{r.get('generator_loss', 0):>9.4f}  "
              f"{r.get('critic_loss', 0):>8.4f}  "
              f"{r.get('per iteration time', 0):>7.1f}")

    if len(gen) >= 5:
        ratios = [r["unidad_dmd_ratio"] for r in gen]
        n_spike = sum(1 for x in ratios if x > 0.5)
        print(f"\nratio summary: n={len(ratios)}  median={sorted(ratios)[len(ratios)//2]:.3f}  "
              f"max={max(ratios):.3f}  spikes>0.5={n_spike} ({100*n_spike/len(ratios):.0f}%)")


if __name__ == "__main__":
    main()
