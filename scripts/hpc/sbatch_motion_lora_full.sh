#!/bin/bash
#SBATCH --job-name=motion-lora-full
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# Each rank loads Wan2.1-T2V-1.3B (~5GB bf16) + UMT5-XXL T5 (~22GB) + VAE
# (~1GB) + variant cache (~10GB for 6 refs × 30 augments). 8 ranks × ~40GB =
# ~320GB peak before headroom. Match sbatch_train.sh's 900G to stay clear of
# cgroup memory pressure that stalls NFS.
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Same exclusions as training — older A100 40GB DGX nodes lack VRAM headroom.
#SBATCH --exclude=s-sc-dgx[01-02]
#
# 8-GPU DDP motion-LoRA training (full-scale). Mirrors sbatch_train.sh's
# resource pattern. Use sbatch_motion_lora.sh for 1-GPU smoke iteration.
#
# Usage:
#   Multi-ref (recommended):
#     sbatch scripts/hpc/sbatch_motion_lora_full.sh \
#         scripts/motion_lora/walking_refs_v1.jsonl \
#         walking_class_full
#
#   Single-ref (works too, but 8-GPU is overkill for 1 ref):
#     sbatch scripts/hpc/sbatch_motion_lora_full.sh \
#         "celebv_X.mp4" "a person walking ..." walking_v1
#
# Env-var overrides:
#   LL_MOTION_CONFIG=...   default configs/motion_lora_full.yaml
#   LL_ANCHOR_PROMPTS=...  default scripts/motion_lora/anchor_prompts_default.txt
#   LL_INF_PROMPTS=...     newline-separated inference prompts
#   LL_INF_GPU_PARALLEL=1  (default) parallelize Stage 2 across all 8 GPUs;
#                          set to 0 to run inference sequentially on rank 0

set -e

REF_ARG="${1:-}"
case "$REF_ARG" in
    *.jsonl)
        if [ "$#" -lt 1 ]; then
            echo "[SLURM][error] usage: sbatch $0 <refs.jsonl> [<run_id>]" >&2
            exit 1
        fi
        REF_MODE="multi"
        REF_JSONL_ARG="$1"
        RUN_ID="${2:-motion_lora_full}_${SLURM_JOB_ID}"
        ;;
    *)
        if [ "$#" -lt 2 ]; then
            echo "[SLURM][error] usage: sbatch $0 <reference.mp4> <caption> [<run_id>]" >&2
            echo "[SLURM][error]    or: sbatch $0 <refs.jsonl> [<run_id>]" >&2
            exit 1
        fi
        REF_MODE="single"
        REF_VIDEO="$1"
        REF_CAPTION="$2"
        RUN_ID="${3:-motion_lora_full}_${SLURM_JOB_ID}"
        ;;
esac

echo "[SLURM] Job ID:     $SLURM_JOB_ID"
echo "[SLURM] Node:       $(hostname)"
echo "[SLURM] GPUs:       ${SLURM_GPUS_ON_NODE:-8}"
echo "[SLURM] Mode:       $REF_MODE-ref (DDP across $SLURM_GPUS_ON_NODE GPUs)"
echo "[SLURM] Run ID:     $RUN_ID"
if [ -r /sys/fs/cgroup/memory.max ]; then
    echo "[SLURM] cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
fi

##############################
# Activate mamba env
##############################
source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

##############################
# Working dir + paths
##############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo" >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

: "${PROJECT_DATA:?PROJECT_DATA not set — add to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"

: "${LL_MOTION_CONFIG:=configs/motion_lora_full.yaml}"
: "${LL_ANCHOR_PROMPTS:=scripts/motion_lora/anchor_prompts_default.txt}"

OUT_DIR="$LL_DATA/motion_lora_runs/$RUN_ID"
mkdir -p "$OUT_DIR/inference"
echo "[SLURM] Output dir:  $OUT_DIR"
echo "[SLURM] Config:      $LL_MOTION_CONFIG"

##############################
# Distributed env (mirrors sbatch_train.sh)
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
# Resolve reference path(s)
##############################
_resolve_path() {
    local arg="$1"
    case "$arg" in
        /*) echo "$arg"; return ;;
    esac
    for cand in "$LL_DATA/motion_refs/$arg" "$LL_DATA/$arg" "$PROJECT_DIR/$arg"; do
        if [ -e "$cand" ]; then echo "$cand"; return; fi
    done
    echo "$arg"
}

if [ "$REF_MODE" = "multi" ]; then
    REF_JSONL_PATH="$(_resolve_path "$REF_JSONL_ARG")"
    if [ ! -f "$REF_JSONL_PATH" ]; then
        echo "[SLURM][error] references jsonl not found: $REF_JSONL_ARG" >&2
        exit 1
    fi
    REF_VIDEO_PATH=$(python -c "
import json
with open('$REF_JSONL_PATH') as f:
    row = json.loads(f.readline())
print(row.get('path') or row.get('video', ''))
")
    if [ -z "$REF_VIDEO_PATH" ] || [ ! -f "$REF_VIDEO_PATH" ]; then
        REF_VIDEO_PATH="$(_resolve_path "$REF_VIDEO_PATH")"
    fi
else
    REF_VIDEO_PATH="$(_resolve_path "$REF_VIDEO")"
    if [ ! -f "$REF_VIDEO_PATH" ]; then
        echo "[SLURM][error] reference video not found: $REF_VIDEO" >&2
        exit 1
    fi
fi
echo "[SLURM] Stage 3 eval reference: $REF_VIDEO_PATH"

##############################
# Stage 1: train motion LoRA via torchrun DDP
##############################
echo "[SLURM] === Stage 1: train motion LoRA on $GPUS_PER_NODE GPUs (DDP) ==="
echo "[SLURM] Rendezvous: $MASTER_ADDR:$MASTER_PORT"

if [ "$REF_MODE" = "multi" ]; then
    REF_ARGS=(--references_jsonl "$REF_JSONL_PATH")
else
    REF_ARGS=(--reference_video "$REF_VIDEO_PATH" --reference_caption "$REF_CAPTION")
fi

torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    scripts/motion_lora/train.py \
    --config "$LL_MOTION_CONFIG" \
    "${REF_ARGS[@]}" \
    --anchor_prompts "$LL_ANCHOR_PROMPTS" \
    --output_dir "$OUT_DIR" \
    --seed 0

if [ ! -f "$OUT_DIR/motion_lora.pt" ]; then
    echo "[SLURM][error] motion_lora.pt not produced" >&2
    exit 1
fi

##############################
# Stage 2: inference (optionally parallelized across GPUs)
##############################
echo "[SLURM] === Stage 2: inference ==="

DEFAULT_PROMPTS=(
    "a different person walking through a busy city street"
    "an elderly man walking slowly in a hospital corridor"
    "a woman in a red dress walking on a beach at sunset"
    "a child walking with a dog through autumn leaves"
    "a hiker walking on a mountain trail in the morning fog"
)
if [ -n "${LL_INF_PROMPTS:-}" ]; then
    mapfile -t INF_PROMPTS < <(printf '%s\n' "$LL_INF_PROMPTS" | tr '\n' '\0' | xargs -0 -n1)
else
    INF_PROMPTS=("${DEFAULT_PROMPTS[@]}")
fi

PROMPTS_JSONL="$OUT_DIR/inference/prompts.jsonl"
> "$PROMPTS_JSONL"

# Build a list of (prompt_idx, mode, gpu) jobs for parallel scheduling
: "${LL_INF_GPU_PARALLEL:=1}"
N_PROMPTS=${#INF_PROMPTS[@]}
N_JOBS=$((N_PROMPTS * 2))   # baseline + motion_lora per prompt

run_one_inference() {
    local i=$1
    local mode=$2
    local gpu=$3
    local prompt="${INF_PROMPTS[$i]}"
    local out_mp4
    if [ "$mode" = "baseline" ]; then
        out_mp4="$OUT_DIR/inference/baseline_${i}.mp4"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/motion_lora/inference.py \
            --prompt "$prompt" \
            --output_mp4 "$out_mp4" \
            --seed 0
    else
        out_mp4="$OUT_DIR/inference/motion_lora_${i}.mp4"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/motion_lora/inference.py \
            --prompt "$prompt" \
            --motion_lora "$OUT_DIR/motion_lora.pt" \
            --output_mp4 "$out_mp4" \
            --seed 0
    fi
}

# Generate prompts.jsonl (single shot — same data regardless of parallel mode)
for i in "${!INF_PROMPTS[@]}"; do
    PROMPT="${INF_PROMPTS[$i]}"
    python -c "
import json
with open('$PROMPTS_JSONL', 'a') as f:
    for v in ['baseline_${i}.mp4', 'motion_lora_${i}.mp4']:
        f.write(json.dumps({'video': v, 'prompt': '''$PROMPT'''}, ensure_ascii=False) + '\n')
"
done

if [ "$LL_INF_GPU_PARALLEL" = "1" ]; then
    echo "[SLURM] Stage 2: parallelizing $N_JOBS inference jobs across $GPUS_PER_NODE GPUs"
    job_idx=0
    for i in "${!INF_PROMPTS[@]}"; do
        for mode in baseline motion_lora; do
            gpu=$((job_idx % GPUS_PER_NODE))
            run_one_inference "$i" "$mode" "$gpu" &
            # Throttle so we don't have more concurrent jobs than GPUs
            if [ $(( (job_idx + 1) % GPUS_PER_NODE )) -eq 0 ]; then
                wait
            fi
            job_idx=$((job_idx + 1))
        done
    done
    wait
else
    echo "[SLURM] Stage 2: sequential inference on GPU 0"
    for i in "${!INF_PROMPTS[@]}"; do
        run_one_inference "$i" baseline 0
        run_one_inference "$i" motion_lora 0
    done
fi

##############################
# Stage 3: pose-similarity + CLIP-text eval
##############################
echo "[SLURM] === Stage 3: pose + CLIP eval ==="
SCORES_JSONL="$OUT_DIR/inference/scores.jsonl"
python scripts/motion_lora/eval_pose.py \
    --reference "$REF_VIDEO_PATH" \
    --generated_dir "$OUT_DIR/inference" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --output "$SCORES_JSONL" || echo "[SLURM][warn] eval_pose failed — check deps (mediapipe, transformers)"

if [ -f "$SCORES_JSONL" ]; then
    echo "[SLURM] === Score summary (lower pose_dist = better; higher clip_text = better) ==="
    python -c "
import json
rows = [json.loads(l) for l in open('$SCORES_JSONL')]
for r in rows:
    print(f\"  {r['generated']:30s} pose_dist={r['pose_dist']:.4f}  clip_text={r['clip_text']:.4f}\")
b = [r for r in rows if r['generated'].startswith('baseline_')]
l = [r for r in rows if r['generated'].startswith('motion_lora_')]
import statistics as st
def avg(xs, k):
    vals = [x[k] for x in xs if x[k] == x[k]]
    return st.mean(vals) if vals else float('nan')
print()
print(f'  baseline    avg pose_dist={avg(b,\"pose_dist\"):.4f}  clip_text={avg(b,\"clip_text\"):.4f}  (n={len(b)})')
print(f'  motion_lora avg pose_dist={avg(l,\"pose_dist\"):.4f}  clip_text={avg(l,\"clip_text\"):.4f}  (n={len(l)})')
print()
print('  motion_lora win condition: pose_dist (vs baseline) DECREASES, clip_text not far below baseline')
"
fi

echo "[SLURM] DONE → $OUT_DIR"
