#!/bin/bash
#SBATCH --job-name=add_dynamic_degree
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Post-hoc: add VBench Dynamic Degree to existing scores.csv files.
# Runs scripts/motion_eval/add_dynamic_degree.py against the two completed
# eval runs (baseline_v1_fixed + motiondirector_v1) in parallel on two GPUs
# from the 8-GPU allocation (the rest are idle — 8-GPU GRES is more
# available on Charité than singletons, and the bigger memory budget helps
# RAFT-Large at high res).
#
# Idempotent: if scores.csv already has dynamic_score populated for a row,
# it's skipped. Re-submit freely to fill in any gaps.

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"

source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/local/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../scripts/local/train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo" >&2
    exit 1
fi
cd "$PROJECT_DIR"

: "${PROJECT_DATA:?PROJECT_DATA not set}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export HF_HOME="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"

RUNS=(
    "baseline_v1_fixed_8894159"
    "motiondirector_v1_8910377"
)
# Sanity check that every run dir + scores.csv exists before launching
for r in "${RUNS[@]}"; do
    p="$LL_DATA/motion_eval_runs/$r/scores.csv"
    if [ ! -f "$p" ]; then
        echo "[SLURM][error] missing scores.csv: $p" >&2
        exit 1
    fi
    echo "[SLURM] target: $p"
done

# Two parallel runs, one per GPU. 8-GPU allocation; only GPUs 0 and 1 used.
i=0
pids=()
for r in "${RUNS[@]}"; do
    p="$LL_DATA/motion_eval_runs/$r/scores.csv"
    echo "[SLURM] launching add_dynamic_degree on $r (GPU $i)"
    CUDA_VISIBLE_DEVICES=$i python scripts/motion_eval/add_dynamic_degree.py \
        --input  "$p" \
        --output "$p" \
        --device cuda \
        ${LL_DD_FORCE:+--force} \
        > "logs/add_dynamic_degree-${SLURM_JOB_ID}-${r}.out" 2>&1 &
    pids+=($!)
    i=$((i+1))
done

# Wait for both parallel jobs; surface failure if any
fail=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        echo "[SLURM][error] pid $pid failed" >&2
        fail=1
    fi
done
echo "[SLURM] all add_dynamic_degree workers finished"
exit $fail
