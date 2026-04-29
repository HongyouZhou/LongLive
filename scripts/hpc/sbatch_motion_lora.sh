#!/bin/bash
#SBATCH --job-name=motion-lora
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Single GPU is enough — we train per-reference LoRA on a single video.
# Multi-reference parallelism = launch N independent jobs, each on 1 GPU.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# Train motion LoRA on one or more reference videos, then run inference on a
# few new prompts and score baseline-vs-LoRA on pose-similarity + CLIP-text.
#
# Two calling conventions:
#
#   Single ref (instance-level):
#     sbatch scripts/hpc/sbatch_motion_lora.sh \
#         "$LL_DATA/motion_refs/celebv_X.mp4" \
#         "a man walking down a street" \
#         walking_v1
#
#   Multi-ref (class-level — recommended after single-ref smoke validates):
#     # 1. produce refs.jsonl via pick_reference.py
#     python scripts/motion_lora/pick_reference.py --keyword walking ... \
#         --output $LL_DATA/motion_refs/walking_refs.jsonl
#     # 2. train on the JSONL
#     sbatch scripts/hpc/sbatch_motion_lora.sh \
#         "$LL_DATA/motion_refs/walking_refs.jsonl" \
#         walking_class_v1
#
# (The script auto-detects mode by extension: *.jsonl → multi-ref, else single.)
#
# Environment overrides:
#   LL_MOTION_CONFIG=...   default configs/motion_lora.yaml
#   LL_ANCHOR_PROMPTS=...  default scripts/motion_lora/anchor_prompts_default.txt
#   LL_INF_PROMPTS=...     newline-separated inference prompts, default 5 walking inline

set -e

REF_ARG="${1:-}"
case "$REF_ARG" in
    *.jsonl)
        # Multi-ref mode: $1 = JSONL, $2 = run_id
        if [ "$#" -lt 1 ]; then
            echo "[SLURM][error] usage: sbatch $0 <refs.jsonl> [<run_id>]" >&2
            exit 1
        fi
        REF_MODE="multi"
        REF_JSONL_ARG="$1"
        RUN_ID="${2:-motion_lora}_${SLURM_JOB_ID}"
        ;;
    *)
        # Single-ref mode: $1 = mp4, $2 = caption, $3 = run_id
        if [ "$#" -lt 2 ]; then
            echo "[SLURM][error] usage: sbatch $0 <reference_video.mp4> <caption> [<run_id>]" >&2
            echo "[SLURM][error]    or: sbatch $0 <refs.jsonl> [<run_id>]" >&2
            exit 1
        fi
        REF_MODE="single"
        REF_VIDEO="$1"
        REF_CAPTION="$2"
        RUN_ID="${3:-motion_lora}_${SLURM_JOB_ID}"
        ;;
esac

echo "[SLURM] Job ID:     $SLURM_JOB_ID"
echo "[SLURM] Node:       $(hostname)"
echo "[SLURM] Mode:       $REF_MODE-ref"
if [ "$REF_MODE" = "multi" ]; then
    echo "[SLURM] References: $REF_JSONL_ARG"
else
    echo "[SLURM] Reference:  $REF_VIDEO"
    echo "[SLURM] Caption:    $REF_CAPTION"
fi
echo "[SLURM] Run ID:     $RUN_ID"

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

: "${LL_MOTION_CONFIG:=configs/motion_lora.yaml}"
: "${LL_ANCHOR_PROMPTS:=scripts/motion_lora/anchor_prompts_default.txt}"

OUT_DIR="$LL_DATA/motion_lora_runs/$RUN_ID"
mkdir -p "$OUT_DIR/inference"
echo "[SLURM] Output dir:  $OUT_DIR"

##############################
# Resolve reference path (single mp4 OR multi-ref JSONL).
#   Tries: absolute > $LL_DATA/motion_refs > $LL_DATA > $PROJECT_DIR.
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
    REF_VIDEO_PATH=""  # used by Stage 3 eval; pick first ref from JSONL
    REF_VIDEO_PATH=$(python -c "
import json
with open('$REF_JSONL_PATH') as f:
    row = json.loads(f.readline())
print(row.get('path') or row.get('video', ''))
")
    if [ -z "$REF_VIDEO_PATH" ] || [ ! -f "$REF_VIDEO_PATH" ]; then
        REF_VIDEO_PATH="$(_resolve_path "$REF_VIDEO_PATH")"
    fi
    echo "[SLURM] Stage 3 eval reference (1st in JSONL): $REF_VIDEO_PATH"
else
    REF_VIDEO_PATH="$(_resolve_path "$REF_VIDEO")"
    if [ ! -f "$REF_VIDEO_PATH" ]; then
        echo "[SLURM][error] reference video not found: $REF_VIDEO" >&2
        echo "  tried: $REF_VIDEO_PATH" >&2
        exit 1
    fi
fi

##############################
# Stage 1: train motion LoRA
##############################
echo "[SLURM] === Stage 1: train motion LoRA ==="
if [ "$REF_MODE" = "multi" ]; then
    python scripts/motion_lora/train.py \
        --config "$LL_MOTION_CONFIG" \
        --references_jsonl "$REF_JSONL_PATH" \
        --anchor_prompts "$LL_ANCHOR_PROMPTS" \
        --output_dir "$OUT_DIR" \
        --seed 0
else
    python scripts/motion_lora/train.py \
        --config "$LL_MOTION_CONFIG" \
        --reference_video "$REF_VIDEO_PATH" \
        --reference_caption "$REF_CAPTION" \
        --anchor_prompts "$LL_ANCHOR_PROMPTS" \
        --output_dir "$OUT_DIR" \
        --seed 0
fi

if [ ! -f "$OUT_DIR/motion_lora.pt" ]; then
    echo "[SLURM][error] motion_lora.pt not produced" >&2
    exit 1
fi

##############################
# Stage 2: inference — baseline (no LoRA) + with motion LoRA, on K prompts
##############################
echo "[SLURM] === Stage 2: inference ==="

# Default 5 walking-domain prompts that should test motion transfer
# (different appearance / scene, same dynamics class).
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

# Reset prompts.jsonl (consumed by Stage 3 eval for CLIP-text alignment)
PROMPTS_JSONL="$OUT_DIR/inference/prompts.jsonl"
> "$PROMPTS_JSONL"

for i in "${!INF_PROMPTS[@]}"; do
    PROMPT="${INF_PROMPTS[$i]}"
    echo "[SLURM] inf $i/$(( ${#INF_PROMPTS[@]} - 1 )): $PROMPT"

    BASE_MP4="baseline_${i}.mp4"
    LORA_MP4="motion_lora_${i}.mp4"

    # Baseline (no LoRA) — run once per prompt
    python scripts/motion_lora/inference.py \
        --prompt "$PROMPT" \
        --output_mp4 "$OUT_DIR/inference/$BASE_MP4" \
        --seed 0

    # With motion LoRA
    python scripts/motion_lora/inference.py \
        --prompt "$PROMPT" \
        --motion_lora "$OUT_DIR/motion_lora.pt" \
        --output_mp4 "$OUT_DIR/inference/$LORA_MP4" \
        --seed 0

    # Record prompts → mp4 mapping for Stage 3 eval (CLIP-text alignment)
    python -c "
import json
with open('$PROMPTS_JSONL', 'a') as f:
    for v in ['$BASE_MP4', '$LORA_MP4']:
        f.write(json.dumps({'video': v, 'prompt': '''$PROMPT'''}, ensure_ascii=False) + '\n')
"
done

##############################
# Stage 3: pose-similarity + CLIP-text eval (vs the training reference)
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
# group baseline vs motion_lora
b = [r for r in rows if r['generated'].startswith('baseline_')]
l = [r for r in rows if r['generated'].startswith('motion_lora_')]
import statistics as st
def avg(xs, k):
    vals = [x[k] for x in xs if x[k] == x[k]]  # NaN check
    return st.mean(vals) if vals else float('nan')
print()
print(f'  baseline    avg pose_dist={avg(b,\"pose_dist\"):.4f}  clip_text={avg(b,\"clip_text\"):.4f}  (n={len(b)})')
print(f'  motion_lora avg pose_dist={avg(l,\"pose_dist\"):.4f}  clip_text={avg(l,\"clip_text\"):.4f}  (n={len(l)})')
print()
print('  motion_lora win condition: pose_dist (vs baseline) DECREASES, clip_text not far below baseline')
"
fi

echo "[SLURM] DONE → $OUT_DIR"
