#!/bin/bash
#SBATCH --job-name=teacher_boundary
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 14B teacher: 8 ranks x ~30 GB CPU = ~240 GB peak. Same headroom as
# sbatch_train.sh; under-provisioning triggers cgroup pressure → NFS stalls.
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# 40GB DGX A100s lack VRAM headroom for 14B + T5-XXL even with t5_cpu.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Wan2.1-T2V-14B teacher boundary test: pure 50-step UniPC, no LoRA, no DMD.
# Sweeps prompts/teacher_boundary_v1.jsonl x N seeds across 8 GPUs.
#
# Usage:
#   sbatch scripts/hpc/sbatch_teacher_boundary.sh
#   LL_BOUNDARY_SEEDS=2 sbatch scripts/hpc/sbatch_teacher_boundary.sh   # smoke
#   LL_BOUNDARY_PROMPTS=prompts/foo.jsonl sbatch scripts/hpc/sbatch_teacher_boundary.sh
#
# To pin GPU type, override --gres on the sbatch CLI:
#   sbatch --gres=gpu:nvidia_h200:8       scripts/hpc/sbatch_teacher_boundary.sh
#   sbatch --gres=gpu:nvidia_h100_80gb:8  scripts/hpc/sbatch_teacher_boundary.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"

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
# Data + ckpt paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"

CKPT_DIR="$WAN_MODELS_ROOT/Wan2.1-T2V-14B"
if [ ! -f "$CKPT_DIR/Wan2.1_VAE.pth" ]; then
    echo "[SLURM][error] Wan 14B ckpt not found at $CKPT_DIR" >&2
    echo "  run scripts/hpc/fetch_data.sh on a login node first" >&2
    exit 1
fi

##############################
# Run config
##############################
: "${LL_BOUNDARY_PROMPTS:=prompts/teacher_boundary_v1.jsonl}"
: "${LL_BOUNDARY_SEEDS:=4}"
: "${LL_BOUNDARY_FRAMES:=81}"
: "${LL_BOUNDARY_WIDTH:=832}"
: "${LL_BOUNDARY_HEIGHT:=480}"
: "${LL_BOUNDARY_STEPS:=50}"
: "${LL_BOUNDARY_GUIDE_SCALE:=5.0}"

RUN_NAME="teacher_boundary_$(date +%y%m%d_%H%M)_${SLURM_JOB_ID}"
OUT_DIR="$LL_DATA/teacher_boundary/$RUN_NAME"
mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"

echo "[SLURM] Run name:   $RUN_NAME"
echo "[SLURM] Prompts:    $LL_BOUNDARY_PROMPTS"
echo "[SLURM] Output:     $OUT_DIR"
echo "[SLURM] Ckpt:       $CKPT_DIR"
echo "[SLURM] Seeds:      $LL_BOUNDARY_SEEDS"
echo "[SLURM] Size:       ${LL_BOUNDARY_WIDTH}x${LL_BOUNDARY_HEIGHT}  Frames: $LL_BOUNDARY_FRAMES  Steps: $LL_BOUNDARY_STEPS"

##############################
# Distributed setup
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TORCH_NCCL_BLOCKING_WAIT=1

##############################
# Launch
##############################
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    scripts/local/teacher_boundary.py \
    --prompts "$LL_BOUNDARY_PROMPTS" \
    --ckpt-dir "$CKPT_DIR" \
    --output-dir "$OUT_DIR" \
    --seeds "$LL_BOUNDARY_SEEDS" \
    --width "$LL_BOUNDARY_WIDTH" \
    --height "$LL_BOUNDARY_HEIGHT" \
    --frames "$LL_BOUNDARY_FRAMES" \
    --steps "$LL_BOUNDARY_STEPS" \
    --guide-scale "$LL_BOUNDARY_GUIDE_SCALE"

# Merge per-rank manifests (rank 0 already finished, this runs after torchrun returns)
python - <<EOF
import json
from pathlib import Path
out = Path("$OUT_DIR")
records = []
for p in sorted(out.glob("manifest_rank*.jsonl")):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
records.sort(key=lambda r: (r["group"], r["idx"], r["seed"]))
merged = out / "manifest.jsonl"
with open(merged, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[merge] wrote {merged} ({len(records)} records)")
EOF

echo "[SLURM] DONE. Run name = $RUN_NAME"
echo "[SLURM] Output       = $OUT_DIR"
echo "[SLURM] Pull to arp:  bash scripts/local/pull_hpc_results.sh $RUN_NAME"
