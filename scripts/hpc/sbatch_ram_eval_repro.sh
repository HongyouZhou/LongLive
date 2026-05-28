#!/bin/bash
#SBATCH --job-name=ram_eval_repro
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# N × motion_eval (UCF 60 + LOVEU 304) on ONE fixed ckpt, no retrain.
# ~30 min/eval → N=2 ≈ 1h.  4h cap is generous.
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Eval-side reproducibility probe for the 27% diagnosis.  Runs motion_eval
# N times against the SAME LoRA ckpt — any spread in motion_fidelity is
# pure eval-side nondeterminism (generation 4-step DMD forward kernel
# autotuning + CoTracker GPU atomics), isolated from training-side variance.
#
#   eval spread ≈ 0      → eval pipeline deterministic → 27% is TRAINING-side
#   eval spread sizable  → eval-side autotuning/CoTracker contributes
#
# Usage:
#   source scripts/hpc/submit.sh sbatch_ram_eval_repro.sh
#   LL_REPRO_N=4 source scripts/hpc/submit.sh sbatch_ram_eval_repro.sh
#   LL_REPRO_CKPT=<abs path> source scripts/hpc/submit.sh sbatch_ram_eval_repro.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPU info: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/local/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

: "${PROJECT_DATA:?PROJECT_DATA not set}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"
mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/motion_eval_runs" "$TORCH_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

# Fixed ckpt — cross-ref skateboarding lora_final.pt by default.
: "${LL_REPRO_CKPT:=$LL_DATA/diffusion_ram_runs/skateboarding_ram/lora_final.pt}"
: "${LL_REPRO_N:=2}"
: "${LL_REPRO_CONFIG:=configs/motion_eval_inference_diffusion_ram.yaml}"
EVAL_GPUS="0,1,2,3,4,5,6,7"

if [ ! -f "$LL_REPRO_CKPT" ]; then
    echo "[SLURM][error] ckpt missing: $LL_REPRO_CKPT" >&2
    exit 1
fi

echo "[SLURM] ckpt:    $LL_REPRO_CKPT"
echo "[SLURM] ckpt md5: $(md5sum "$LL_REPRO_CKPT" | awk '{print $1}')"
echo "[SLURM] N evals:  $LL_REPRO_N"
echo "[SLURM] config:   $LL_REPRO_CONFIG"

declare -a STATUS
for i in $(seq 1 "$LL_REPRO_N"); do
    echo
    echo "############### eval repeat $i/$LL_REPRO_N  $(date -Iseconds) ###############"
    RUN_ID="skate_evalrepro_rep${i}_${SLURM_JOB_ID}"
    set +e
    bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
        "$LL_REPRO_CKPT" "$LL_REPRO_CONFIG" "$RUN_ID" \
        --gpus "$EVAL_GPUS" \
        --datasets ucf,loveu
    STATUS[$i]=$?
    set -e
    echo "[SLURM] eval repeat $i exit=${STATUS[$i]}  run_id=$RUN_ID"
done

echo
echo "############### eval-repro summary ###############"
printf "%-6s %-10s\n" "rep" "exit"
for i in $(seq 1 "$LL_REPRO_N"); do
    printf "%-6s %-10s\n" "$i" "${STATUS[$i]:-?}"
done
echo "[SLURM] run_ids: skate_evalrepro_rep{1..$LL_REPRO_N}_${SLURM_JOB_ID}"
echo "[SLURM] Job finished at $(date -Iseconds)."
