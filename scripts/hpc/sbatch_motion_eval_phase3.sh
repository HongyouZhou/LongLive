#!/bin/bash
#SBATCH --job-name=motiondirector_eval
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Phase 3 motion eval: auto-discover the latest Phase 3 student LoRA ckpt
# and run the standard Phase 1 eval pipeline (UCF + LOVEU, 4 metrics).
# Designed to be submitted with `--dependency=afterok:<train_jid>` from
# scripts/hpc/submit_phase3.sh so it kicks off automatically once Phase 3
# training completes successfully.
#
# Usage (chained, via submit_phase3.sh — preferred):
#   source scripts/hpc/submit_phase3.sh
#
# Manual (when training is already done):
#   source scripts/hpc/submit.sh sbatch_motion_eval_phase3.sh
#
# Env-var overrides:
#   LL_PHASE3_LOGDIR=<path>       skip auto-discovery, use this logdir
#   LL_PHASE3_RUN_ID=<id>         eval run_id prefix (default motiondirector_v1)
#   LL_MOTION_EVAL_LIMIT=N        cap to first N prompts (smoke runs)
#   LL_MOTION_EVAL_CONFIG=<path>  default configs/motion_eval_inference.yaml
#   LL_MOTION_EVAL_GPUS=0,1,...   default uses all 8 from SLURM allocation
#   LL_MOTION_EVAL_DATASETS=<>    default ucf,loveu

set -e

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
    echo "[SLURM][error] cannot locate LongLive repo" >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Data + cache paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"
export WANDB_DIR="$PROJECT_DIR/wandb"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/motion_eval_runs" "$TORCH_HOME"

##############################
# Locate Phase 3 ckpt
##############################
if [ -n "${LL_PHASE3_LOGDIR:-}" ]; then
    LOGDIR="$LL_PHASE3_LOGDIR"
else
    # Auto-discovery: most recently modified motion_dmd_skateboarding_v1_*/
    LOGDIR=$(ls -dt "$PROJECT_DIR"/logs/motion_dmd_skateboarding_v1_*/ 2>/dev/null | head -1 || true)
    if [ -z "$LOGDIR" ]; then
        echo "[SLURM][error] no Phase 3 logdir found under logs/motion_dmd_skateboarding_v1_*/" >&2
        echo "  override with LL_PHASE3_LOGDIR=<path>" >&2
        exit 1
    fi
fi
LOGDIR="${LOGDIR%/}"
echo "[SLURM] Phase 3 logdir: $LOGDIR"

# Pick most-recently-modified checkpoint_model_*/model.pt — save order is
# monotonic with step, so latest mtime = highest step ckpt.
CKPT=$(ls -t "$LOGDIR"/checkpoint_model_*/model.pt 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[SLURM][error] no model.pt under $LOGDIR/checkpoint_model_*/" >&2
    exit 1
fi
echo "[SLURM] eval ckpt: $CKPT"

##############################
# Compose run-id + forward to run_motion_eval.sh
##############################
: "${LL_PHASE3_RUN_ID:=motiondirector_v1}"
RUN_ID="${LL_PHASE3_RUN_ID}_${SLURM_JOB_ID}"
: "${LL_MOTION_EVAL_CONFIG:=configs/motion_eval_inference.yaml}"
: "${LL_MOTION_EVAL_GPUS:=0,1,2,3,4,5,6,7}"
: "${LL_MOTION_EVAL_DATASETS:=ucf,loveu}"

EXTRA_ARGS=()
[ -n "${LL_MOTION_EVAL_LIMIT:-}" ] && EXTRA_ARGS+=(--limit "$LL_MOTION_EVAL_LIMIT")
EXTRA_ARGS+=(--gpus "$LL_MOTION_EVAL_GPUS")
EXTRA_ARGS+=(--datasets "$LL_MOTION_EVAL_DATASETS")

echo "[SLURM] run_id    = $RUN_ID"
echo "[SLURM] config    = $LL_MOTION_EVAL_CONFIG"
echo "[SLURM] gpus      = $LL_MOTION_EVAL_GPUS"
echo "[SLURM] datasets  = $LL_MOTION_EVAL_DATASETS"
[ -n "${LL_MOTION_EVAL_LIMIT:-}" ] && echo "[SLURM] limit     = $LL_MOTION_EVAL_LIMIT"

bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
    "$CKPT" "$LL_MOTION_EVAL_CONFIG" "$RUN_ID" "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished."
