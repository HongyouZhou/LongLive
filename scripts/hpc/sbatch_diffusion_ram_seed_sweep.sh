#!/bin/bash
#SBATCH --job-name=ram_seed_sweep
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 10 seeds × (~50 min train + ~30 min motion_eval + ~60 min vbench) ≈ 23 h
# on 8 H200.  48 h cap (pgpu max) gives ~25 h margin for I/O / queue stalls.
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# RAM single-category multi-seed sweep — 10 seeds for one UCF Sports category
# in a single SLURM allocation so all 10 runs share node / GPUs / driver
# version.  Purpose: quantify RL-training inter-seed variance (mean ± SE) on
# the cross-reference baseline before reading any single-seed result as a
# method-level claim.
#
# Usage (always via submit.sh):
#   LL_RAM_CATEGORY=skateboarding source scripts/hpc/submit.sh sbatch_diffusion_ram_seed_sweep.sh
#   LL_RAM_CATEGORY=lifting       source scripts/hpc/submit.sh sbatch_diffusion_ram_seed_sweep.sh
#
# Optional: override seed range
#   LL_RAM_SEEDS="0,1,2,3,4"  (default 0..9)
#
# Pin GPU type:
#   sbatch --gres=gpu:nvidia_h200:8 scripts/hpc/sbatch_diffusion_ram_seed_sweep.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
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
# Data + cache paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"

echo "[SLURM] Data root:       $LL_DATA"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/diffusion_ram_runs" \
         "$LL_DATA/motion_eval_runs" "$LL_DATA/vbench_runs" "$TORCH_HOME"

##############################
# VBench info path
##############################
: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"
: "${VBENCH_INFO:=$VBENCH_REPO_DIR/vbench/VBench_full_info.json}"
export VBENCH_REPO_DIR VBENCH_INFO

##############################
# Distributed env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TORCH_NCCL_BLOCKING_WAIT=1

##############################
# Sweep config
##############################
: "${LL_RAM_CATEGORY:=skateboarding}"
: "${LL_RAM_SEEDS:=0,1,2,3,4,5,6,7,8,9}"
IFS=',' read -r -a SEEDS <<< "$LL_RAM_SEEDS"

CFG="longlive/methods/diffusion_ram/configs/${LL_RAM_CATEGORY}_ram.yaml"
if [ ! -f "$CFG" ]; then
    echo "[SLURM][error] config missing: $CFG (LL_RAM_CATEGORY=$LL_RAM_CATEGORY)" >&2
    exit 1
fi

EVAL_CONFIG_MOTION="configs/motion_eval_inference_diffusion_ram.yaml"
EVAL_CONFIG_VBENCH="configs/vbench_short_diffusion_ram.yaml"
EVAL_GPUS="0,1,2,3,4,5,6,7"

echo "[SLURM] Category: $LL_RAM_CATEGORY"
echo "[SLURM] Config:   $CFG"
echo "[SLURM] Seeds:    ${SEEDS[*]}"

# Per-iter status table.
declare -a STATUS_TRAIN STATUS_MOTION STATUS_VBENCH

##############################
# Main loop — one seed per iter, no result reuse.
##############################
ITER=0
for SEED in "${SEEDS[@]}"; do
    ITER=$((ITER + 1))
    echo
    echo "###############################################################"
    echo "[SLURM] === iter $ITER/${#SEEDS[@]}  cat=$LL_RAM_CATEGORY  seed=$SEED  $(date -Iseconds) ==="
    echo "###############################################################"

    SUFFIX="_seed${SEED}"
    MASTER_PORT=$((20000 + (SLURM_JOB_ID + ITER) % 20000))

    set +e

    ##############################
    # 1. Train (--seed $SEED --out-suffix _seed$SEED)
    ##############################
    echo "[SLURM][seed=$SEED] === train ==="
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        -m longlive.methods.diffusion_ram.train \
        --config "$CFG" \
        --seed "$SEED" \
        --out-suffix "$SUFFIX"
    RC_TRAIN=$?
    STATUS_TRAIN[$ITER]=$RC_TRAIN
    echo "[SLURM][seed=$SEED] train exit=$RC_TRAIN"

    # Resolve ckpt path: out_dir from yaml + _seed{N} suffix appended by train.py
    CKPT_DIR=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CFG')
print(OmegaConf.to_container(cfg, resolve=True)['out_dir'])
")
    CKPT_DIR="${CKPT_DIR}${SUFFIX}"
    CKPT="$CKPT_DIR/lora_final.pt"

    if [ "$RC_TRAIN" -ne 0 ] || [ ! -f "$CKPT" ]; then
        echo "[SLURM][seed=$SEED][warn] lora_final.pt missing — skipping eval" >&2
        STATUS_MOTION[$ITER]="skipped"
        STATUS_VBENCH[$ITER]="skipped"
        set -e
        continue
    fi

    RUN_ID="${LL_RAM_CATEGORY}_ram${SUFFIX}_${SLURM_JOB_ID}"

    ##############################
    # 2. motion_eval — full UCF 60 + LOVEU 304
    ##############################
    echo "[SLURM][seed=$SEED] === motion_eval (UCF 60 + LOVEU 304 full) ==="
    bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
        "$CKPT" "$EVAL_CONFIG_MOTION" "$RUN_ID" \
        --gpus "$EVAL_GPUS" \
        --datasets ucf,loveu
    RC_MOTION=$?
    STATUS_MOTION[$ITER]=$RC_MOTION
    echo "[SLURM][seed=$SEED] motion_eval exit=$RC_MOTION"

    ##############################
    # 3. vbench — full 944
    ##############################
    echo "[SLURM][seed=$SEED] === vbench (944 full) ==="
    if [ ! -f "$VBENCH_INFO" ]; then
        echo "[SLURM][seed=$SEED][warn] VBench_full_info.json missing — skipping vbench." >&2
        STATUS_VBENCH[$ITER]="missing_vbench_info"
    else
        bash "$PROJECT_DIR/scripts/vbench/run_vbench.sh" \
            "$CKPT" "$EVAL_CONFIG_VBENCH" "$RUN_ID" \
            --gpus "$EVAL_GPUS"
        RC_VBENCH=$?
        STATUS_VBENCH[$ITER]=$RC_VBENCH
        echo "[SLURM][seed=$SEED] vbench exit=$RC_VBENCH"
    fi

    set -e
done

##############################
# Final summary
##############################
echo
echo "###############################################################"
echo "[SLURM] === Seed sweep summary (category=$LL_RAM_CATEGORY) ==="
echo "###############################################################"
printf "%-3s %-6s %-10s %-12s %-12s\n" "#" "seed" "train" "motion_eval" "vbench"
for i in $(seq 1 ${#SEEDS[@]}); do
    s="${SEEDS[$((i-1))]}"
    printf "%-3s %-6s %-10s %-12s %-12s\n" "$i" "$s" "${STATUS_TRAIN[$i]:-?}" "${STATUS_MOTION[$i]:-?}" "${STATUS_VBENCH[$i]:-?}"
done
echo "[SLURM] Job finished at $(date -Iseconds)."
