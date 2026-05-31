#!/bin/bash
#SBATCH --job-name=per_reference_adaptation
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02],s-sc-pgpu13,s-sc-pgpu14
#
# Per-reference adaptation protocol: resolve units, train each selected unit,
# generate only that unit's eval prompts, score against the same reference
# video, and aggregate results.  One sbatch owns train + eval.
#
# Usage:
#   source scripts/hpc/submit.sh sbatch_per_reference_adaptation.sh
#
# Full EM-RAM:
#   LL_PER_REF_CONFIG=configs/per_reference_em_ram.yaml \
#     source scripts/hpc/submit.sh sbatch_per_reference_adaptation.sh
#
# Run only one method from a matrix config:
#   LL_PER_REF_METHODS=em_ram \
#     source scripts/hpc/submit.sh sbatch_per_reference_adaptation.sh

set -euo pipefail

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
echo "[SLURM] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"

set +u
source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/per_reference/run_protocol.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../scripts/per_reference/run_protocol.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo. Set LL_REPO or submit from repo root." >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

: "${PROJECT_DATA:?PROJECT_DATA not set - add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"
export WANDB_DIR="$PROJECT_DIR/wandb"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/per_reference_adaptation_runs" "$TORCH_HOME"

: "${LL_PER_REF_CONFIG:=configs/per_reference_em_ram_smoke.yaml}"
echo "[SLURM] protocol config: $LL_PER_REF_CONFIG"

ARGS=(--config "$LL_PER_REF_CONFIG")
if [ -n "${LL_PER_REF_METHODS:-}" ]; then
    ARGS+=(--methods "$LL_PER_REF_METHODS")
    echo "[SLURM] method filter: $LL_PER_REF_METHODS"
fi
if [ -n "${LL_PER_REF_LIMIT_UNITS:-}" ]; then
    ARGS+=(--limit-units "$LL_PER_REF_LIMIT_UNITS")
    echo "[SLURM] unit limit: $LL_PER_REF_LIMIT_UNITS"
fi
if [ -n "${LL_PER_REF_DRY_RUN:-}" ]; then
    ARGS+=(--dry-run)
    echo "[SLURM] dry-run enabled"
fi

python scripts/per_reference/run_protocol.py "${ARGS[@]}"

echo "[SLURM] Per-reference protocol finished."
