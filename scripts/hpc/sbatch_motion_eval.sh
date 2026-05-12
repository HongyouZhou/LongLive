#!/bin/bash
#SBATCH --job-name=motion_eval
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
# 8 workers each load Wan-1.3B + T5-XXL + LoRA → ~18 GB CPU each at boot.
# Scoring phase loads CLIP-L/14 + PickScore (CLIP-H/14) + CoTracker3 — no
# detectron2, so peak is lower than VBench. 400 G is a comfortable budget.
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# 40GB DGX A100s can't fit Wan-1.3B + CLIP-H + CoTracker3 + worker text encoders
# even with our reduced footprint.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Usage:
#   sbatch scripts/hpc/sbatch_motion_eval.sh <ckpt> [<run_id_prefix>]
#
# <ckpt> can be:
#   - absolute path
#   - relative to $LL_DATA (e.g. "longlive_models/models/lora.pt")
#   - relative to repo root (e.g. "logs/.../model.pt")
#
# Examples:
#   # Uni-DAD baseline on full LOVEU + UCF eval (paper-comparable)
#   sbatch scripts/hpc/sbatch_motion_eval.sh \
#       longlive_models/models/lora.pt baseline
#
#   # Smoke test: 8 prompts on the released ckpt
#   LL_MOTION_EVAL_LIMIT=8 sbatch scripts/hpc/sbatch_motion_eval.sh \
#       longlive_models/models/lora.pt smoke
#
#   # Only LOVEU (skip UCF reconstruction)
#   LL_MOTION_EVAL_DATASETS=loveu sbatch scripts/hpc/sbatch_motion_eval.sh \
#       longlive_models/models/lora.pt loveu_only
#
# Env-var overrides:
#   LL_MOTION_EVAL_LIMIT=N             cap to first N prompts (smoke runs)
#   LL_MOTION_EVAL_CONFIG=<path>       default configs/motion_eval_inference.yaml
#   LL_MOTION_EVAL_GPUS=0,1,...        default uses all 8 from SLURM allocation
#   LL_MOTION_EVAL_DATASETS=ucf,loveu  subset

set -e

if [ "$#" -lt 1 ]; then
    echo "[SLURM][error] usage: sbatch $0 <ckpt> [<run_id_prefix>]" >&2
    exit 1
fi

CKPT_ARG="$1"
RUN_ID_PREFIX="${2:-motion_eval}"

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"

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
# Data + cache paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"   # CoTracker3 ckpt lands here
export WANDB_DIR="$PROJECT_DIR/wandb"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/motion_eval_runs" "$TORCH_HOME"

##############################
# Resolve ckpt path: absolute > $LL_DATA-relative > $PROJECT_DIR-relative
##############################
case "$CKPT_ARG" in
    /*) CKPT="$CKPT_ARG" ;;
     *) if   [ -f "$LL_DATA/$CKPT_ARG"     ]; then CKPT="$LL_DATA/$CKPT_ARG"
        elif [ -f "$PROJECT_DIR/$CKPT_ARG" ]; then CKPT="$PROJECT_DIR/$CKPT_ARG"
        else CKPT="$CKPT_ARG"
        fi ;;
esac
if [ ! -f "$CKPT" ]; then
    echo "[SLURM][error] ckpt not found: $CKPT_ARG" >&2
    echo "  tried: $CKPT" >&2
    echo "  also:  $LL_DATA/$CKPT_ARG" >&2
    echo "  also:  $PROJECT_DIR/$CKPT_ARG" >&2
    exit 1
fi

##############################
# Compose run-id and forward to run_motion_eval.sh
##############################
RUN_ID="${RUN_ID_PREFIX}_${SLURM_JOB_ID}"
: "${LL_MOTION_EVAL_CONFIG:=configs/motion_eval_inference.yaml}"
: "${LL_MOTION_EVAL_GPUS:=0,1,2,3,4,5,6,7}"
: "${LL_MOTION_EVAL_DATASETS:=ucf,loveu}"

EXTRA_ARGS=()
if [ -n "${LL_MOTION_EVAL_LIMIT:-}" ]; then
    EXTRA_ARGS+=(--limit "$LL_MOTION_EVAL_LIMIT")
fi
EXTRA_ARGS+=(--gpus "$LL_MOTION_EVAL_GPUS")
EXTRA_ARGS+=(--datasets "$LL_MOTION_EVAL_DATASETS")

echo "[SLURM] ckpt      = $CKPT"
echo "[SLURM] run_id    = $RUN_ID"
echo "[SLURM] config    = $LL_MOTION_EVAL_CONFIG"
echo "[SLURM] gpus      = $LL_MOTION_EVAL_GPUS"
echo "[SLURM] datasets  = $LL_MOTION_EVAL_DATASETS"
[ -n "${LL_MOTION_EVAL_LIMIT:-}" ] && echo "[SLURM] limit     = $LL_MOTION_EVAL_LIMIT"

bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
    "$CKPT" "$LL_MOTION_EVAL_CONFIG" "$RUN_ID" "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished."
