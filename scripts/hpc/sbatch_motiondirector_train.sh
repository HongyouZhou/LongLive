#!/bin/bash
#SBATCH --job-name=motiondirector_train
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 8 ranks × Wan-1.3B ≈ 8 × 5 GB CPU peak — 1.3B is small, but keep 900 GB to
# match sbatch_train.sh policy of headroom against cgroup memory-pressure
# NFS stalls (see sbatch_train.sh §11 postmortem).
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# 40 GB A100s — overkill exclude for 1.3B but kept for consistency with
# sbatch_train.sh; remove via sbatch CLI if you want them.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# MotionDirector LoRA finetune on the few-step LongLive 1.3B model.
# See docs/00.md §1.1 (research anchor, route 1) and docs/01.md (two-LoRA
# layering + load flow).
#
# Default = 8 GPU on a single pgpu node (any type, scheduler-friendlier than
# single-GPU jobs which sit in queue while full-node jobs go through).
#
# Usage (always via submit.sh wrapper — captures $JID):
#
#   source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
#   # Override config:
#   LL_MD_CONFIG=longlive/methods/motiondirector/configs/skateboarding_fewstep.yaml \
#     source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
#   # 5-step smoke instead of full 500:
#   LL_MD_SMOKE=1 source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
# Pin GPU type via sbatch CLI when queue allows:
#   sbatch --gres=gpu:nvidia_h200:8        scripts/hpc/sbatch_motiondirector_train.sh
#   sbatch --gres=gpu:nvidia_a100-sxm4:8   scripts/hpc/sbatch_motiondirector_train.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
# Print the actual GPU name + memory so we don't have to back-derive from
# OOM error capacity numbers. Charité's GRES strings have been confirmed
# ambiguous (asked for h200 and still got 40 GB A100).
echo "[SLURM] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
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
export WANDB_DIR="$PROJECT_DIR/wandb"

echo "[SLURM] Data root:       $LL_DATA"
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/motiondirector_runs"

##############################
# Distributed env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TORCH_NCCL_BLOCKING_WAIT=1

##############################
# Run
##############################
: "${LL_MD_CONFIG:=longlive/methods/motiondirector/configs/skateboarding_fewstep.yaml}"
echo "[SLURM] config:  $LL_MD_CONFIG"

EXTRA_ARGS=()
if [ -n "${LL_MD_SMOKE:-}" ]; then
    EXTRA_ARGS+=(--smoke)
    echo "[SLURM] SMOKE mode — 5 steps only"
fi

echo "[SLURM] Launching torchrun on $GPUS_PER_NODE GPU(s), master=$MASTER_ADDR:$MASTER_PORT"
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    -m longlive.methods.motiondirector.train \
    --config "$LL_MD_CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished."
