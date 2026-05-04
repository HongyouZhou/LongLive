#!/bin/bash
#SBATCH --job-name=longlive_motion
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
# Motion-DMD HPC training. Mirrors scripts/hpc/sbatch_train.sh:
#   - same SLURM resources (8×H100, 64 CPU, 900G, 48h, dgx excluded)
#   - same env (mamba, $LL_DATA, $WAN_MODELS_ROOT, NCCL, etc.)
#   - same auto-resume + per-run logdir naming
# Differs only in:
#   - default LL_CONFIG → configs/longlive_train_motion.yaml
#   - one-time V_ref latent precache before torchrun (skipped if cache exists)
#
# Submit:
#   sbatch scripts/hpc/sbatch_motion_dmd.sh
#
# To resume the latest motion run:
#   LL_AUTO_RESUME=1 sbatch scripts/hpc/sbatch_motion_dmd.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
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
    echo "[SLURM] Error: cannot locate LongLive repo. Set LL_REPO or sbatch from repo root."
    exit 1
fi

cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Config
##############################
: "${LL_CONFIG:=configs/longlive_train_motion.yaml}"
HPC_CONFIG="${LL_CONFIG%.yaml}_hpc.yaml"
[ -f "$HPC_CONFIG" ] && LL_CONFIG="$HPC_CONFIG"
echo "[SLURM] Config: $LL_CONFIG"

##############################
# Per-run logdir + resume logic (identical to sbatch_train.sh)
##############################
CONFIG_BASENAME="$(basename "${LL_CONFIG%.yaml}")"
EXTRA_TRAIN_ARGS=""

if [ -z "${LL_RESUME_LOGDIR:-}" ] && [ -n "${LL_AUTO_RESUME:-}" ]; then
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

##############################
# Data source
##############################
: "${LL_DATA:=$PROJECT_DATA/wm}"
# Export so torchrun children can resolve `$LL_DATA` in yaml paths via
# `os.path.expandvars` (e.g. generator_ckpt, motion.refs_path).
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"

echo "[SLURM] Data root:       $LL_DATA"
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"
echo "[SLURM] HF_HOME:          $HF_HOME"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/wandb" "$LL_DATA/hf_cache"

##############################
# Motion-DMD: V_ref latent precache (one-time, ~30s on a single H100)
##############################
# `motion.refs_path` in the config defaults to $LL_DATA/motion_dmd/walking_v1.latents.pt;
# build it on the fly with rank-0 if missing. The 7 idle ranks just spin briefly,
# but this is a single one-time hit per dataset (~30s) — cheaper than a separate
# sbatch round-trip. Subsequent runs reuse the NFS-resident cache.
MOTION_REFS_JSONL="${LL_MOTION_REFS:-prompts/walking_refs_v1.jsonl}"
MOTION_CACHE_PATH="${LL_MOTION_CACHE:-$LL_DATA/motion_dmd/walking_v1.latents.pt}"

if [ ! -f "$MOTION_CACHE_PATH" ]; then
    echo "[SLURM] motion cache not found at $MOTION_CACHE_PATH"
    echo "[SLURM] running precache_motion_refs.py on rank-0 (others idle ~30s)"
    mkdir -p "$(dirname "$MOTION_CACHE_PATH")"
    # Use a single GPU for VAE encode; CUDA_VISIBLE_DEVICES=0 isolates rank-0.
    CUDA_VISIBLE_DEVICES=0 python scripts/motion_dmd/precache_motion_refs.py \
        --refs_jsonl "$MOTION_REFS_JSONL" \
        --output "$MOTION_CACHE_PATH" \
        --refs_root "$LL_DATA/motion_refs" \
        --device cuda
    echo "[SLURM] motion cache built: $MOTION_CACHE_PATH"
else
    echo "[SLURM] motion cache already present: $MOTION_CACHE_PATH"
fi
# The trainer reads `motion.refs_path` from the yaml; that value already uses
# $LL_DATA so it resolves to the same path the precache step writes to.

##############################
# Start training
##############################
echo "[SLURM] Launching torchrun on $NNODES node(s), $GPUS_PER_NODE GPU(s) each"
echo "[SLURM] Rendezvous: $MASTER_ADDR:$MASTER_PORT"

if [ "$NNODES" -gt 1 ]; then
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
