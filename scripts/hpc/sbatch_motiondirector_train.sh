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
#   # Disable the post-training eval phase (motion_eval + vbench on the
#   # `lora_final.pt` that just landed):
#   LL_MD_EVAL=0 source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
#
#   # Smoke eval (4 prompts per dataset):
#   LL_MD_EVAL_LIMIT=4 source scripts/hpc/submit.sh sbatch_motiondirector_train.sh
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

echo "[SLURM] Training phase finished."

##############################
# Post-training eval phase — same allocation, no extra queueing.
#
# Runs motion_eval (Yatim CoTracker3 + LOVEU CLIP/PickScore) and VBench on
# the `lora_final.pt` that just landed in cfg.out_dir.
#
# Best-effort: motion_eval failure does not block vbench. Smoke mode
# (LL_MD_SMOKE=1) skips eval since 5-step ckpt isn't meaningful to measure.
##############################
if [ -n "${LL_MD_SMOKE:-}" ]; then
    echo "[SLURM] Smoke mode — skipping post-training eval."
    echo "[SLURM] Job finished."
    exit 0
fi

if [ "${LL_MD_EVAL:-1}" = "0" ]; then
    echo "[SLURM] LL_MD_EVAL=0 — skipping post-training eval."
    echo "[SLURM] Job finished."
    exit 0
fi

# Resolve the OmegaConf out_dir (handles ${oc.env:LL_DATA} interpolation).
CKPT_DIR=$(python -c "
import sys, os
from omegaconf import OmegaConf
cfg = OmegaConf.load('$LL_MD_CONFIG')
print(OmegaConf.to_container(cfg, resolve=True)['out_dir'])
")
CKPT="$CKPT_DIR/lora_final.pt"

if [ ! -f "$CKPT" ]; then
    echo "[SLURM][error] post-train eval: lora_final.pt missing at $CKPT" >&2
    echo "[SLURM] Job finished (training OK, eval skipped)."
    exit 0
fi

# Run-id prefix = config basename (drops .yaml). Makes motion_eval_runs/ and
# vbench_runs/ trivially attributable back to the training config.
CONFIG_BASENAME=$(basename "$LL_MD_CONFIG" .yaml)
EVAL_PREFIX="${LL_MD_RUN_PREFIX:-$CONFIG_BASENAME}"

echo "[SLURM] Post-train eval"
echo "[SLURM]   ckpt   = $CKPT"
echo "[SLURM]   prefix = $EVAL_PREFIX"

: "${LL_MD_VBENCH_CONFIG:=configs/vbench_short.yaml}"
: "${LL_MD_EVAL_DATASETS:=ucf,loveu}"
: "${LL_MD_EVAL_GPUS:=0,1,2,3,4,5,6,7}"

EVAL_LIMIT_ARGS=()
if [ -n "${LL_MD_EVAL_LIMIT:-}" ]; then
    EVAL_LIMIT_ARGS+=(--limit "$LL_MD_EVAL_LIMIT")
fi

# VBench paths (mirror sbatch_vbench.sh — run_vbench.sh requires these).
: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"
: "${VBENCH_INFO:=$VBENCH_REPO_DIR/vbench/VBench_full_info.json}"
export VBENCH_REPO_DIR VBENCH_INFO
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"
mkdir -p "$LL_DATA/motion_eval_runs" "$LL_DATA/vbench_runs" "$TORCH_HOME"

# Best-effort: don't let motion_eval errors abort vbench.
set +e

echo "[SLURM] === motion_eval ==="
RUN_ID_MOTION="${EVAL_PREFIX}_${SLURM_JOB_ID}"
bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
    "$CKPT" "configs/motion_eval_inference.yaml" "$RUN_ID_MOTION" \
    --gpus "$LL_MD_EVAL_GPUS" \
    --datasets "$LL_MD_EVAL_DATASETS" \
    "${EVAL_LIMIT_ARGS[@]}"
RC_MOTION=$?
echo "[SLURM] motion_eval exit=$RC_MOTION"

echo "[SLURM] === vbench ==="
RUN_ID_VBENCH="${EVAL_PREFIX}_${SLURM_JOB_ID}"
if [ ! -f "$VBENCH_INFO" ]; then
    echo "[SLURM][warn] VBench_full_info.json missing at $VBENCH_INFO — skipping vbench." >&2
    RC_VBENCH=99
else
    bash "$PROJECT_DIR/scripts/vbench/run_vbench.sh" \
        "$CKPT" "$LL_MD_VBENCH_CONFIG" "$RUN_ID_VBENCH" \
        --gpus "$LL_MD_EVAL_GPUS" \
        "${EVAL_LIMIT_ARGS[@]}"
    RC_VBENCH=$?
fi
echo "[SLURM] vbench exit=$RC_VBENCH"

set -e

echo "[SLURM] Job finished (train=OK, motion_eval=$RC_MOTION, vbench=$RC_VBENCH)."
