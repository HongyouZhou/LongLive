#!/bin/bash
#SBATCH --job-name=longlive
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"

##############################
# Activate mamba environment
##############################
source ~/.bashrc
# Prefix env on project FS (created by setup_mamba_env.sh)
: "${PROJECT_HOME:?PROJECT_HOME not set}"
: "${LL_ENV_PREFIX:=$PROJECT_HOME/envs/longlive}"
mamba activate "$LL_ENV_PREFIX"

##############################
# Working directory
##############################
: "${LL_WORK:=$PROJECT_HOME/longlive}"

if [ -d "$LL_WORK/repo" ]; then
    PROJECT_DIR="$LL_WORK/repo"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    echo "[SLURM] Error: cannot locate LongLive repo (LL_WORK=$LL_WORK)"
    exit 1
fi

cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Config (paths already patched by fetch_data.sh)
##############################
: "${LL_CONFIG:=configs/longlive_finetune_motion_cross.yaml}"
HPC_CONFIG="${LL_CONFIG%.yaml}_hpc.yaml"
[ -f "$HPC_CONFIG" ] && LL_CONFIG="$HPC_CONFIG"
echo "[SLURM] Config: $LL_CONFIG"

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

# WANDB_API_KEY + HF_TOKEN come from ~/.bashrc.
export WANDB_DIR="$LL_WORK/wandb"
export HF_HOME="$LL_WORK/hf_cache"
export TRANSFORMERS_CACHE="$LL_WORK/hf_cache"

mkdir -p "$LL_WORK/logs" "$LL_WORK/wandb"

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
            train.py \
            --config_path $LL_CONFIG \
            --logdir $LL_WORK/logs \
            --wandb-save-dir $LL_WORK/wandb
    "
else
    # Single-node: simpler launch, matches your run_lab.sh / run_slurm_aue.sh style.
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        train.py \
        --config_path "$LL_CONFIG" \
        --logdir "$LL_WORK/logs" \
        --wandb-save-dir "$LL_WORK/wandb"
fi

echo "[SLURM] Job finished."
