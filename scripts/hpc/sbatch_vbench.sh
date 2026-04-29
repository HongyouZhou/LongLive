#!/bin/bash
#SBATCH --job-name=vbench
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
# 8 workers each load Wan-1.3B + T5-XXL + LoRA → ~18 GB CPU each at boot,
# ~150 GB total before VBench scoring loads detectron2/CLIP/DOVER on top.
# Empirical 200 G triggered cgroup pressure → NFS IO stalled during boot.
# 800 G matches sbatch_train.sh's headroom.
#SBATCH --mem=800G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Same exclusions as training — DGX A100s lack VRAM headroom for the
# detectron2 + CLIP + Wan-1.3B mix even with 8 workers.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Usage:
#   sbatch scripts/hpc/sbatch_vbench.sh <ckpt> [<run_id_prefix>]
#
# <ckpt> can be:
#   - absolute path
#   - relative to $LL_DATA (e.g. "longlive_models/models/lora.pt")
#   - relative to $PROJECT_DIR / repo root (e.g. "logs/.../model.pt")
#
# Examples:
#   # Reproduce paper Table 1: NVlabs released ckpt, all 946 prompts × 5 samples
#   LL_VBENCH_NUM_SAMPLES=5 sbatch scripts/hpc/sbatch_vbench.sh \
#       longlive_models/models/lora.pt paper_baseline
#
#   # Smoke test: 8 prompts on the released ckpt
#   LL_VBENCH_LIMIT=8 sbatch scripts/hpc/sbatch_vbench.sh \
#       longlive_models/models/lora.pt smoke
#
#   # Evaluate one of our trained ckpts
#   sbatch scripts/hpc/sbatch_vbench.sh \
#       logs/longlive_train_long_hpc_<id>/checkpoint_model_3000/model.pt our_run
#
# Env-var overrides:
#   LL_VBENCH_LIMIT=N         pass --limit N to run_vbench.sh (smoke runs)
#   LL_VBENCH_CONFIG=<path>   default configs/vbench_short.yaml
#   LL_VBENCH_GPUS=0,1,...    default uses all 8 from SLURM allocation

set -e

if [ "$#" -lt 1 ]; then
    echo "[SLURM][error] usage: sbatch $0 <ckpt> [<run_id_prefix>]" >&2
    exit 1
fi

CKPT_ARG="$1"
RUN_ID_PREFIX="${2:-vbench}"

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"

##############################
# Activate mamba environment (longlive — run_vbench.sh switches envs internally)
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
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo. Set LL_REPO or sbatch from repo root." >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Data source — same conventions as sbatch_train.sh
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"

# VBench-specific paths
: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"
: "${VBENCH_INFO:=$VBENCH_REPO_DIR/vbench/VBench_full_info.json}"
export VBENCH_REPO_DIR VBENCH_INFO

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/vbench_runs"

if [ ! -f "$VBENCH_INFO" ]; then
    echo "[SLURM][error] VBench_full_info.json missing at $VBENCH_INFO." >&2
    echo "             Run scripts/vbench/setup_vbench_env.sh first." >&2
    exit 1
fi

##############################
# Resolve ckpt path: absolute > $LL_DATA-relative > $PROJECT_DIR-relative
##############################
case "$CKPT_ARG" in
    /*) CKPT="$CKPT_ARG" ;;
     *) if   [ -f "$LL_DATA/$CKPT_ARG"     ]; then CKPT="$LL_DATA/$CKPT_ARG"
        elif [ -f "$PROJECT_DIR/$CKPT_ARG" ]; then CKPT="$PROJECT_DIR/$CKPT_ARG"
        else CKPT="$CKPT_ARG"  # let run_vbench.sh emit the error with full path
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
# Compose run-id and forward to run_vbench.sh
##############################
RUN_ID="${RUN_ID_PREFIX}_${SLURM_JOB_ID}"
: "${LL_VBENCH_CONFIG:=configs/vbench_short.yaml}"
: "${LL_VBENCH_GPUS:=0,1,2,3,4,5,6,7}"

EXTRA_ARGS=()
if [ -n "${LL_VBENCH_LIMIT:-}" ]; then
    EXTRA_ARGS+=(--limit "$LL_VBENCH_LIMIT")
fi
EXTRA_ARGS+=(--gpus "$LL_VBENCH_GPUS")

echo "[SLURM] ckpt          = $CKPT"
echo "[SLURM] run_id        = $RUN_ID"
echo "[SLURM] config        = $LL_VBENCH_CONFIG"
echo "[SLURM] gpus          = $LL_VBENCH_GPUS"
[ -n "${LL_VBENCH_LIMIT:-}" ] && echo "[SLURM] limit         = $LL_VBENCH_LIMIT"

bash "$PROJECT_DIR/scripts/vbench/run_vbench.sh" \
    "$CKPT" "$LL_VBENCH_CONFIG" "$RUN_ID" "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished."
