#!/bin/bash
#SBATCH --job-name=motiondirector_train
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4:1
#SBATCH --cpus-per-task=8
# Wan-14B teacher load peaks ~50 GB CPU (28 GB bf16 weights + ~20 GB fp32 umt5
# text encoder), then drops once .to(device) runs. 256 GB gives comfortable
# headroom and avoids cgroup memory-pressure NFS stalls (sbatch_train.sh §11
# postmortem).
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# GPU sizing — Wan-14B + umt5 fp32 + LoRA + activations needs ~70 GB GPU peak.
# 40 GB A100s (DGX s-sc-dgx01/02 + other A100-40GB nodes) cannot fit it;
# even with --exclude they may still bind via plain `--gres=gpu:1`. So pin the
# GRES type to a known 80 GB variant. Override on the sbatch CLI when needed:
#   sbatch --gres=gpu:nvidia_h200:1        scripts/hpc/sbatch_motiondirector_train.sh   # 141 GB Hopper, fastest
#   sbatch --gres=gpu:nvidia_h100_80gb:1   scripts/hpc/sbatch_motiondirector_train.sh   # only on s-sc-pgpu08
# Default (this header) = nvidia_a100-sxm4:1 — 80 GB A100, least contested.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Phase 2 (docs/04.md) — MotionDirector teacher-finetune on Wan-14B.
# Trains a LoRA on top of frozen Wan-14B teacher (single-GPU, K1) using
# paper L_temporal_MSE + L_AD recipe in epsilon space via B1 close-form
# reverse. Output ckpt is consumed by Phase 3 DMD as `real_score` adapter.
#
# Usage (always via submit.sh wrapper — captures $JID):
#
#   source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
#   # Override config:
#   LL_MD_CONFIG=longlive/methods/motiondirector/configs/skateboarding_v1.yaml \
#     source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
#   # 5-step smoke instead of full 500:
#   LL_MD_SMOKE=1 source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
# Pin GPU type via the sbatch CLI when queue allows (Wan-14B fits comfortably
# on H200 80 GB; tight on A100 80 GB):
#   sbatch --gres=gpu:nvidia_h200:1        scripts/hpc/sbatch_motiondirector_train.sh
#   sbatch --gres=gpu:nvidia_a100-sxm4:1   scripts/hpc/sbatch_motiondirector_train.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPU:    ${SLURM_GPUS_ON_NODE:-1}"
if [ -r /sys/fs/cgroup/memory.max ]; then
    echo "[SLURM] cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
fi

##############################
# Activate mamba environment
##############################
source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

##############################
# Working directory
##############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/local/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../scripts/local/train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo. Set LL_REPO or sbatch from repo root." >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Data + cache paths (mirror sbatch_train.sh "Data source — explicit, no symlinks")
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
# WANDB_API_KEY + HF_TOKEN come from ~/.bashrc (not used by Phase 2 v1, but exported for consistency)
export WANDB_DIR="$PROJECT_DIR/wandb"

echo "[SLURM] Data root:       $LL_DATA"
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"
echo "[SLURM] HF_HOME:          $HF_HOME"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/motiondirector_runs"

##############################
# Run
##############################
: "${LL_MD_CONFIG:=longlive/methods/motiondirector/configs/skateboarding_v1.yaml}"
echo "[SLURM] config:  $LL_MD_CONFIG"

EXTRA_ARGS=()
if [ -n "${LL_MD_SMOKE:-}" ]; then
    EXTRA_ARGS+=(--smoke)
    echo "[SLURM] SMOKE mode — 5 steps only"
fi

# Allocator fragmentation tolerance for the large per-step activation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

echo "[SLURM] Starting python ..."
python -m longlive.methods.motiondirector.train \
    --config "$LL_MD_CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished."
