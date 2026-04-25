#!/bin/bash
# Launch LongLive training on the lab machine (2x RTX PRO 6000 Blackwell) from arp.
#
# Architecture:
#   - Code lives on arp at /home/hongyou/dev/LongLive/ (this repo)
#   - Lab sees the same code via sshfs at ~/mnt/arp/dev/LongLive/
#   - Conda env lives on arp at ~/miniforge3/envs/longlive-blackwell/
#   - Lab accesses the env via sshfs at ~/mnt/arp/miniforge3/envs/longlive-blackwell/
#   - Training logs are written back to arp via sshfs (low write rate so OK)
#
# Usage:
#   ./scripts/run_lab.sh [CONFIG_FILE]       # CONFIG_FILE defaults to configs/longlive_train_long.yaml
#   ./scripts/run_lab.sh --gpus 0,1          # (NOT IMPLEMENTED YET — set CUDA_VISIBLE_DEVICES below)
#
# Kill:
#   ssh lab pkill -9 -f configs/longlive_train

set -euo pipefail

CONFIG="${1:-configs/longlive_train_long.yaml}"
LOGDIR="${LOGDIR:-logs}"           # per-run checkpoint/vis dir (override for isolated experiments)
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/lab_train_${STAMP}.log"

mkdir -p "$(dirname "$LOGFILE")"

# Hybrid workdir on lab:
#   ~/longlive_work/wan_models/       → LOCAL (rsync'd from arp, avoids sshfs EPERM on concurrent open)
#   ~/longlive_work/logs/             → LOCAL (fast writes)
#   ~/longlive_work/wandb/            → LOCAL (fast writes)
#   ~/longlive_work/*                 → symlinks to ~/mnt/arp/dev/LongLive/* (code + ckpts + prompts)
REMOTE_WORKDIR='/home/hongyou/longlive_work'
# NOTE: must activate via full path — conda on lab does not know env name
#       "longlive-blackwell" since the envs dir is not in its local search path.
REMOTE_CONDA='source ~/mnt/arp/miniforge3/etc/profile.d/conda.sh && conda activate ~/mnt/arp/miniforge3/envs/longlive-blackwell'

# Blackwell notes:
#   NCCL_P2P_DISABLE=0 — RTX PRO 6000 Max-Q does have PCIe P2P (unlike A40 ACS lockups)
#   Leave NCCL_IB_DISABLE=1 since lab has no InfiniBand.
#   PYTORCH_CUDA_ALLOC_CONF — expandable_segments helps but Blackwell already supports native.
# PYTHONNOUSERSITE=1 is important — arp's ~/.local/ is NOT visible on lab, but
# if lab happens to have its own ~/.local/ with stale packages it would shadow
# the env packages. Force env-only imports.
# LL_LOW_CPU_MEM=1: rank 0 loads weights, broadcast via FSDP to other ranks.
# Required on lab — the box only has 125 GB RAM and 2× full 14B in CPU is
# already half of that. HPC nodes leave this unset (NVlabs standard path).
REMOTE_ENV='PYTHONNOUSERSITE=1 NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_LOW_CPU_MEM=1'

REMOTE_CMD="ulimit -n 65536 && \
mkdir -p /home/hongyou/dev && \
[ -e /home/hongyou/dev/data ] || ln -sfn /home/hongyou/mnt/arp/dev/data /home/hongyou/dev/data && \
cd ${REMOTE_WORKDIR} && \
${REMOTE_CONDA} && \
CUDA_VISIBLE_DEVICES=${GPUS} ${REMOTE_ENV} \
python -m torch.distributed.run --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
  train.py \
  --config_path ${CONFIG} \
  --logdir ${LOGDIR} \
  --wandb-save-dir wandb"

echo "[run_lab] Launching on lab with GPUs=${GPUS} nproc=${NPROC}"
echo "[run_lab] Config: ${CONFIG}"
echo "[run_lab] Log:    ${LOGFILE}"
echo

# -tt forces pseudo-tty so pkill etc. propagate on Ctrl+C.
ssh -tt lab "${REMOTE_CMD}" 2>&1 | tee "${LOGFILE}"
