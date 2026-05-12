#!/bin/bash
# End-to-end motion-customization eval (MotionDirector-aligned, LOVEU + UCF).
#
# Usage:
#   bash scripts/motion_eval/run_motion_eval.sh <ckpt.pt> <config.yaml> <run-id> [--limit N] [--gpus 0,1,...] [--datasets ucf,loveu]
#
# Stages (all in longlive env — no env switching, unlike VBench):
#   1. build prompt manifest      (build_manifest.py reads UCF YAML + LOVEU CSV)
#   2. dispatch generation        (eval_dispatch.py across GPUs)
#   3. post-hoc scoring           (run_eval.py — 3 CLIP metrics + Yatim motion fidelity)
#
# Each stage is idempotent: re-running skips work already done.
#
# Required input data layout (run scripts/prepare_motion_eval.py first):
#   $LL_DATA/ucf_sports/{videos,manifest.csv}
#   $LL_DATA/loveu_tgve/{videos,prompts.csv}

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <ckpt.pt> <config.yaml> <run-id> [--limit N] [--gpus 0,1,...] [--datasets ucf,loveu]"
    exit 1
fi

CKPT="$1"
CONFIG="$2"
RUN_ID="$3"
shift 3

LIMIT=""
GPUS="${LL_MOTION_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
DATASETS="${LL_MOTION_EVAL_DATASETS:-ucf,loveu}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit)    LIMIT="--limit $2"; shift 2 ;;
        --gpus)     GPUS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
: "${LL_ENV_NAME:=longlive}"

# Keep model caches under $LL_DATA so HPC ~ doesn't fill up. PickScore alone
# is ~3 GB; CoTracker3 ckpt + CLIP-L/14 + CLIP-H/14 stack pushes 5+ GB.
# Mirrors sbatch_motion_eval.sh / fetch_data.sh conventions.
export HF_HOME="${HF_HOME:-$LL_DATA/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$LL_DATA/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$LL_DATA/hf_cache/torch_hub}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Activate longlive env so `python` resolves correctly across HPC / lab / arp.
eval "$(mamba shell hook --shell bash)"
mamba activate "$LL_ENV_NAME"

if [ ! -f "$CKPT" ]; then
    echo "[motion_eval][error] ckpt not found: $CKPT" >&2
    echo "                     check that LL_DATA is exported and the path is correct" >&2
    exit 1
fi

# Sanity-check that data prep has been done.
for ds in $(echo "$DATASETS" | tr ',' ' '); do
    case "$ds" in
        ucf)
            if [ ! -f "$LL_DATA/ucf_sports/manifest.csv" ]; then
                echo "[motion_eval][error] $LL_DATA/ucf_sports/manifest.csv missing." >&2
                echo "                     Run: python scripts/prepare_motion_eval.py --datasets ucf" >&2
                exit 1
            fi ;;
        loveu)
            if [ ! -f "$LL_DATA/loveu_tgve/prompts.csv" ]; then
                echo "[motion_eval][error] $LL_DATA/loveu_tgve/prompts.csv missing." >&2
                echo "                     Run: python scripts/prepare_motion_eval.py --datasets loveu" >&2
                exit 1
            fi ;;
    esac
done

RUN_DIR="$LL_DATA/motion_eval_runs/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "[motion_eval] run_id    = $RUN_ID"
echo "[motion_eval] ckpt      = $CKPT"
echo "[motion_eval] config    = $CONFIG"
echo "[motion_eval] datasets  = $DATASETS"
echo "[motion_eval] run_dir   = $RUN_DIR"
echo "[motion_eval] gpus      = $GPUS"
[ -n "$LIMIT" ] && echo "[motion_eval] $LIMIT"

LONGLIVE_PY="$(which python)"

# -------- phase 1: build prompt manifest --------
PROMPTS_JSONL="$RUN_DIR/prompts_manifest.jsonl"
if [ ! -f "$PROMPTS_JSONL" ]; then
    echo "[motion_eval] phase 1: building prompt manifest"
    python "$REPO_ROOT/scripts/motion_eval/build_manifest.py" \
        --data_root "$LL_DATA" \
        --datasets "$DATASETS" \
        --output "$PROMPTS_JSONL"
else
    echo "[motion_eval] phase 1: manifest already exists, skipping"
fi

# -------- phase 2: dispatch generation across GPUs --------
echo "[motion_eval] phase 2: dispatching generation on GPUs $GPUS"
python "$REPO_ROOT/scripts/motion_eval/eval_dispatch.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --manifest "$PROMPTS_JSONL" \
    --output_dir "$RUN_DIR" \
    --gpu_ids "$GPUS" \
    --python_bin "$LONGLIVE_PY" \
    $LIMIT

# -------- phase 3: post-hoc scoring --------
echo "[motion_eval] phase 3: scoring"
SCORES_CSV="$RUN_DIR/scores.csv"
python "$REPO_ROOT/scripts/motion_eval/run_eval.py" \
    --prompts_manifest "$PROMPTS_JSONL" \
    --gen_dir "$RUN_DIR" \
    --ref_root "$LL_DATA" \
    --output "$SCORES_CSV"

echo "[motion_eval] DONE  ->  $SCORES_CSV"
