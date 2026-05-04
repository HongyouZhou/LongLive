#!/bin/bash
#SBATCH --job-name=longlive
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 14B teacher load needs ~30 GB CPU per rank × 8 ranks = ~240 GB peak.
# 200G triggers cgroup memory pressure → kernel evicts page cache → NFS
# reads stall to ~0 MB/s. Give plenty of headroom.
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
# Avoid the 40GB DGX A100s — Wan2.1-T2V-14B teacher OOMs there even with FSDP.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# To pin a specific GPU type, override --gres on the sbatch command line:
#   sbatch --gres=gpu:nvidia_h200:8        scripts/hpc/sbatch_train.sh   # 141 GB Hopper, fastest
#   sbatch --gres=gpu:nvidia_h100_80gb:8   scripts/hpc/sbatch_train.sh   # only on s-sc-pgpu08
#   sbatch --gres=gpu:nvidia_a100-sxm4:8   scripts/hpc/sbatch_train.sh   # 80GB A100 (dgx excluded above)
# Confirm the exact GRES strings on your cluster:
#   scontrol show node s-sc-pgpu11 | grep -i gres

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
# Print cgroup memory limit (helps diagnose memory-pressure-induced IO stalls)
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
# Working directory (= the repo; data lives inside it)
##############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/local/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../scripts/local/train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM] Error: cannot locate LongLive repo. Set LL_REPO or sbatch from repo root."
    exit 1
fi

cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Config (paths already patched by fetch_data.sh)
##############################
: "${LL_CONFIG:=configs/longlive_train_long.yaml}"
HPC_CONFIG="${LL_CONFIG%.yaml}_hpc.yaml"
[ -f "$HPC_CONFIG" ] && LL_CONFIG="$HPC_CONFIG"
echo "[SLURM] Config: $LL_CONFIG"

##############################
# Per-run logdir: <config>_<jobid> so concurrent / sequential runs of
# different configs cannot overwrite each other's checkpoints.
#
# Resume modes (in priority order):
#   LL_RESUME_LOGDIR=<path>   Explicit: reuse this directory.
#   LL_AUTO_RESUME=1          Auto-find: pick the most recently modified
#                             logs/<config>_*/ that contains a saved
#                             checkpoint. Falls back to fresh start
#                             when no prior logdir qualifies.
# Default (neither set): fresh start with a new logdir.
##############################
CONFIG_BASENAME="$(basename "${LL_CONFIG%.yaml}")"
EXTRA_TRAIN_ARGS=""

if [ -z "${LL_RESUME_LOGDIR:-}" ] && [ -n "${LL_AUTO_RESUME:-}" ]; then
    # mtime-sorted scan; first directory containing a model.pt wins.
    while IFS= read -r d; do
        if ls "$d"checkpoint_model_*/model.pt >/dev/null 2>&1; then
            LL_RESUME_LOGDIR="${d%/}"
            echo "[SLURM] LL_AUTO_RESUME: picked $LL_RESUME_LOGDIR"
            break
        fi
    done < <(ls -dt "$PROJECT_DIR/logs/${CONFIG_BASENAME}_"*/ 2>/dev/null)

    if [ -z "${LL_RESUME_LOGDIR:-}" ]; then
        echo "[SLURM] LL_AUTO_RESUME: no prior logdir with a checkpoint, starting fresh"
    fi
fi

if [ -n "${LL_RESUME_LOGDIR:-}" ]; then
    case "$LL_RESUME_LOGDIR" in
        /*) RUN_LOGDIR="$LL_RESUME_LOGDIR" ;;
         *) RUN_LOGDIR="$PROJECT_DIR/$LL_RESUME_LOGDIR" ;;
    esac
    if [ ! -d "$RUN_LOGDIR" ]; then
        echo "[SLURM][error] LL_RESUME_LOGDIR=$LL_RESUME_LOGDIR not found"
        exit 1
    fi
    EXTRA_TRAIN_ARGS="--auto-resume"
    echo "[SLURM] RESUMING from $RUN_LOGDIR"
else
    RUN_NAME="${CONFIG_BASENAME}_$(date +%y%m%d_%H%M)"
    RUN_LOGDIR="$PROJECT_DIR/logs/$RUN_NAME"
    mkdir -p "$RUN_LOGDIR"
fi
echo "[SLURM] Run logdir: $RUN_LOGDIR"

##############################
# Distributed + NCCL env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
NNODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5_0

##############################
# Data source — explicit, no symlinks
##############################
# LongLive claims the "wm" sub-namespace under the shared $PROJECT_DATA.
: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"     # longlive/utils/wan_wrapper.py reads this
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
# WANDB_API_KEY + HF_TOKEN come from ~/.bashrc.
export WANDB_DIR="$PROJECT_DIR/wandb"

echo "[SLURM] Data root:       $LL_DATA"
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"
echo "[SLURM] HF_HOME:          $HF_HOME"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/wandb" "$LL_DATA/hf_cache"

##############################
# Start training
##############################
echo "[SLURM] Launching torchrun on $NNODES node(s), $GPUS_PER_NODE GPU(s) each"
echo "[SLURM] Rendezvous: $MASTER_ADDR:$MASTER_PORT"

if [ "$NNODES" -gt 1 ]; then
    # Multi-node: srun fans out one launcher per node.
    srun --kill-on-bad-exit=1 bash -c "
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --node_rank=\$SLURM_NODEID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_id=$SLURM_JOB_ID \
            scripts/local/train.py \
            --config_path $LL_CONFIG \
            --logdir $RUN_LOGDIR \
            --wandb-save-dir $PROJECT_DIR/wandb \
            $EXTRA_TRAIN_ARGS
    "
else
    # Single-node: simpler launch, matches your run_lab.sh / run_slurm_aue.sh style.
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        scripts/local/train.py \
        --config_path "$LL_CONFIG" \
        --logdir "$RUN_LOGDIR" \
        --wandb-save-dir "$PROJECT_DIR/wandb" \
        $EXTRA_TRAIN_ARGS
fi

echo "[SLURM] Job finished."
