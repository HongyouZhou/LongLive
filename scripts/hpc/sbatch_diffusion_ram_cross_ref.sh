#!/bin/bash
#SBATCH --job-name=ram_cross_ref
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 10 UCF Sports categories × (~50 min train + ~30 min motion_eval + ~60 min vbench)
# ≈ 23 h sequential.  48 h cap (pgpu max) — generous margin for queue stalls,
# slow VAE I/O, or any category that drifts long during training.
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# RAM cross-reference sweep: train one LoRA per UCF Sports category, then full
# motion_eval (UCF 60 + LOVEU 304) + vbench (944) per ckpt.  Single SLURM
# allocation so all 10 ckpts share node / code state / cudnn version — the
# whole point of running this as one sbatch instead of 10.
#
# Usage (always via submit.sh):
#   source scripts/hpc/submit.sh sbatch_diffusion_ram_cross_ref.sh
#
# Pin GPU type:
#   sbatch --gres=gpu:nvidia_h200:8 scripts/hpc/sbatch_diffusion_ram_cross_ref.sh

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
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/diffusion_ram_runs" \
         "$LL_DATA/motion_eval_runs" "$LL_DATA/vbench_runs" "$TORCH_HOME"

##############################
# VBench info path (required for vbench)
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
# The 10 UCF Sports categories — full standard set, no exclusions.
##############################
CATEGORIES=(
    "diving"
    "golf_swing"
    "kicking"
    "lifting"
    "riding_horse"
    "run_side"
    "skateboarding"
    "swing_bench"
    "swing_sideangle"
    "walk_front"
)

CONFIG_DIR="longlive/methods/diffusion_ram/configs"
EVAL_CONFIG_MOTION="configs/motion_eval_inference_diffusion_ram.yaml"
EVAL_CONFIG_VBENCH="configs/vbench_short_diffusion_ram.yaml"
EVAL_GPUS="0,1,2,3,4,5,6,7"

# Per-iter status table (printed at end).
declare -a STATUS_TRAIN STATUS_MOTION STATUS_VBENCH

##############################
# Main loop — train + eval per category, no result reuse.
##############################
ITER=0
for cat in "${CATEGORIES[@]}"; do
    ITER=$((ITER + 1))
    echo
    echo "###############################################################"
    echo "[SLURM] === iter $ITER/10  category=$cat  $(date -Iseconds) ==="
    echo "###############################################################"

    CFG="$CONFIG_DIR/${cat}_ram.yaml"
    if [ ! -f "$CFG" ]; then
        echo "[SLURM][error] config missing: $CFG — skipping iter $ITER" >&2
        STATUS_TRAIN[$ITER]="missing_config"
        STATUS_MOTION[$ITER]="skipped"
        STATUS_VBENCH[$ITER]="skipped"
        continue
    fi

    # Re-derive master port per iter to dodge any lingering NCCL state.
    MASTER_PORT=$((20000 + (SLURM_JOB_ID + ITER) % 20000))

    set +e

    ##############################
    # 1. Train
    ##############################
    echo "[SLURM][$cat] === train ==="
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        -m longlive.methods.diffusion_ram.train \
        --config "$CFG"
    RC_TRAIN=$?
    STATUS_TRAIN[$ITER]=$RC_TRAIN
    echo "[SLURM][$cat] train exit=$RC_TRAIN"

    # Resolve ckpt path from the config's out_dir.
    CKPT_DIR=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CFG')
print(OmegaConf.to_container(cfg, resolve=True)['out_dir'])
")
    CKPT="$CKPT_DIR/lora_final.pt"

    if [ "$RC_TRAIN" -ne 0 ] || [ ! -f "$CKPT" ]; then
        echo "[SLURM][$cat][warn] lora_final.pt missing — skipping eval for $cat" >&2
        STATUS_MOTION[$ITER]="skipped"
        STATUS_VBENCH[$ITER]="skipped"
        set -e
        continue
    fi

    RUN_ID="${cat}_ram_${SLURM_JOB_ID}"

    ##############################
    # 2. motion_eval — full UCF 60 + LOVEU 304, no --limit.
    ##############################
    echo "[SLURM][$cat] === motion_eval (UCF 60 + LOVEU 304 full) ==="
    bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
        "$CKPT" "$EVAL_CONFIG_MOTION" "$RUN_ID" \
        --gpus "$EVAL_GPUS" \
        --datasets ucf,loveu
    RC_MOTION=$?
    STATUS_MOTION[$ITER]=$RC_MOTION
    echo "[SLURM][$cat] motion_eval exit=$RC_MOTION"

    ##############################
    # 3. vbench — full 944, no --limit.
    ##############################
    echo "[SLURM][$cat] === vbench (944 full) ==="
    if [ ! -f "$VBENCH_INFO" ]; then
        echo "[SLURM][$cat][warn] VBench_full_info.json missing at $VBENCH_INFO — skipping vbench." >&2
        STATUS_VBENCH[$ITER]="missing_vbench_info"
    else
        bash "$PROJECT_DIR/scripts/vbench/run_vbench.sh" \
            "$CKPT" "$EVAL_CONFIG_VBENCH" "$RUN_ID" \
            --gpus "$EVAL_GPUS"
        RC_VBENCH=$?
        STATUS_VBENCH[$ITER]=$RC_VBENCH
        echo "[SLURM][$cat] vbench exit=$RC_VBENCH"
    fi

    set -e
done

##############################
# Final summary table
##############################
echo
echo "###############################################################"
echo "[SLURM] === Cross-reference sweep summary ==="
echo "###############################################################"
printf "%-3s %-18s %-10s %-12s %-12s\n" "#" "category" "train" "motion_eval" "vbench"
for i in $(seq 1 ${#CATEGORIES[@]}); do
    cat="${CATEGORIES[$((i-1))]}"
    printf "%-3s %-18s %-10s %-12s %-12s\n" "$i" "$cat" "${STATUS_TRAIN[$i]:-?}" "${STATUS_MOTION[$i]:-?}" "${STATUS_VBENCH[$i]:-?}"
done
echo "[SLURM] Job finished at $(date -Iseconds)."
