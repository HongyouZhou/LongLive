#!/bin/bash
#SBATCH --job-name=motiondirector_sanity
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Wan-14B does not fit on 40 GB GPUs. Charité's single-GPU GRES type pin
# (`--gres=gpu:nvidia_h200:1`) has been observed to bind to a 40 GB A100
# anyway, so go with full-node integer GRES (`gpu:8`) which reliably lands
# on a 80 GB+ pgpu node (s-sc-pgpu08 H100-80GB, etc.) — same trick as
# sbatch_motiondirector_train.sh.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Phase 2 sanity inference: load a teacher LoRA ckpt onto Wan-14B and
# generate 6 Skateboarding videos (paper verbatim prompts) for visual
# inspection. Each torchrun rank handles its share of prompts × seeds via
# the same data-parallel pattern as scripts/local/teacher_boundary.py.
#
# Usage (always via submit.sh):
#
#   source scripts/hpc/submit.sh sbatch_motiondirector_sanity.sh \\
#       motiondirector_runs/skateboarding_v1/teacher_lora_final.pt
#
# <ckpt> resolution: absolute path > $LL_DATA-relative > repo-relative.
#
# Optional env-var overrides:
#   LL_MD_PROMPTS="prompt 1|prompt 2"  pipe-separated prompts (default = paper 6)
#   LL_MD_SEEDS=2                       seeds per prompt (default 1 → 6 jobs total)
#   LL_MD_OUT_DIR=<path>                explicit output dir

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

##############################
# Resolve LoRA ckpt path
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
echo "[SLURM] lora ckpt: $CKPT"

##############################
# Output dir + run-id
##############################
: "${LL_MD_OUT_DIR:=$LL_DATA/motiondirector_runs/sanity_${SLURM_JOB_ID}}"
mkdir -p "$LL_MD_OUT_DIR"
echo "[SLURM] out_dir:   $LL_MD_OUT_DIR"

##############################
# Distributed env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

##############################
# Optional CLI overrides
##############################
EXTRA_ARGS=()
if [ -n "${LL_MD_PROMPTS:-}" ]; then
    IFS='|' read -ra _prompts <<< "$LL_MD_PROMPTS"
    EXTRA_ARGS+=(--prompts "${_prompts[@]}")
fi
if [ -n "${LL_MD_SEEDS:-}" ]; then
    EXTRA_ARGS+=(--seeds "$LL_MD_SEEDS")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

echo "[SLURM] launching torchrun on $GPUS_PER_NODE GPU(s), master=$MASTER_ADDR:$MASTER_PORT"
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    scripts/local/motiondirector_sanity_inference.py \
    --lora-ckpt "$CKPT" \
    --ckpt-dir "$WAN_MODELS_ROOT/Wan2.1-T2V-14B" \
    --output-dir "$LL_MD_OUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo "[SLURM] Job finished. videos at $LL_MD_OUT_DIR"
