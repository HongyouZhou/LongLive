#!/bin/bash
#SBATCH --job-name=motiondirector_alpha_sweep
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Alpha sweep on Phase 2 teacher LoRA: regenerate 6 paper Skateboarding
# videos at LoRA inference-time strengths alpha={64,32,16,8} (same trained
# Phase 2 ckpt, no retraining), then compute tracklet amplitude per video.
#
# Goal: localise the amplitude collapse observed in Phase 3 eval. If
# amplitude rises monotonically as alpha drops, inference-time LoRA
# scaling is an effective lever (Phase 2 doesn't need re-training).
#
# Usage (always via submit.sh):
#
#   source scripts/hpc/submit.sh sbatch_motiondirector_alpha_sweep.sh \\
#       motiondirector_runs/skateboarding_v1/teacher_lora_final.pt
#
# <ckpt> resolution: absolute path > $LL_DATA-relative > repo-relative.
#
# Optional env-var overrides:
#   LL_MD_ALPHAS="64 32 16 8"        space-separated alpha values
#   LL_MD_SWEEP_DIR=<path>           explicit output root (default
#                                    $LL_DATA/motiondirector_runs/skateboarding_v1/alpha_sweep)

set -e

if [ "$#" -lt 1 ]; then
    echo "[SLURM][error] usage: sbatch $0 <lora_ckpt>" >&2
    exit 1
fi
CKPT_ARG="$1"

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
echo "[SLURM] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"

##############################
# Env + repo
##############################
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
echo "[SLURM] Working dir: $(pwd)"

##############################
# Paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"

##############################
# Resolve LoRA ckpt
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
    exit 1
fi
echo "[SLURM] lora ckpt: $CKPT"

##############################
# Output dir + alpha list
##############################
: "${LL_MD_ALPHAS:=64 32 16 8}"
# Default output root mirrors Phase 2 runs/skateboarding_v1/, so the sweep
# stays grouped with the LoRA it tests.
LORA_PARENT="$(dirname "$CKPT")"
: "${LL_MD_SWEEP_DIR:=$LORA_PARENT/alpha_sweep}"
mkdir -p "$LL_MD_SWEEP_DIR"
echo "[SLURM] sweep dir: $LL_MD_SWEEP_DIR"
echo "[SLURM] alphas:    $LL_MD_ALPHAS"

##############################
# Distributed env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

##############################
# Generate: loop alphas, fresh torchrun per alpha (clean state isolation)
##############################
for ALPHA in $LL_MD_ALPHAS; do
    ALPHA_OUT="$LL_MD_SWEEP_DIR/alpha_${ALPHA}"
    mkdir -p "$ALPHA_OUT"
    # SLURM_JOB_ID is constant across the loop; use ALPHA in master_port to
    # avoid stale-socket conflicts when reusing the same compute node.
    MASTER_PORT=$((20000 + (SLURM_JOB_ID + ALPHA) % 20000))
    echo ""
    echo "[SLURM] === alpha=$ALPHA → $ALPHA_OUT (master_port=$MASTER_PORT) ==="
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        scripts/local/motiondirector_sanity_inference.py \
        --lora-ckpt "$CKPT" \
        --ckpt-dir "$WAN_MODELS_ROOT/Wan2.1-T2V-14B" \
        --output-dir "$ALPHA_OUT" \
        --rank 64 --alpha "$ALPHA"
done

##############################
# Amplitude analysis (single GPU, rank 0 — CoTracker3 on the 24 videos)
##############################
echo ""
echo "[SLURM] === amplitude analysis ==="
python scripts/local/amplitude_sweep_analysis.py \
    --sweep-dir "$LL_MD_SWEEP_DIR" \
    --output "$LL_MD_SWEEP_DIR/amplitude.csv"

echo "[SLURM] Job finished. videos + amplitude.csv at $LL_MD_SWEEP_DIR"
