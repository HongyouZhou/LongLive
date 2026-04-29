#!/bin/bash
# End-to-end VBench short eval pipeline (Table 1).
#
# Usage:
#   bash scripts/vbench/run_vbench.sh <ckpt.pt> <config.yaml> <run-id> [--limit N]
#
# Stages:
#   1. (longlive env) build prompt manifest + dispatch generation across GPUs
#   2. (vbench env)   run VBench scorer on generated mp4s
#   3. (any env)      aggregate per-dim scores → summary.json + 1-line stdout
#
# Each stage is idempotent: re-running skips work already done. Crash during
# generation → re-run picks up where it stopped (existing mp4s skipped).
# Crash during scoring → re-run repeats vbench scoring; vbench 自己也按 dim
# 分文件,已完成的 dim 不会重算(取决于 vbench 版本)。

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <ckpt.pt> <config.yaml> <run-id> [--limit N] [--gpus 0,1,...]"
    exit 1
fi

CKPT="$1"
CONFIG="$2"
RUN_ID="$3"
shift 3

LIMIT=""
GPUS="${VBENCH_GPUS:-0,1,2,3,4,5,6,7}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="--limit $2"; shift 2 ;;
        --gpus)  GPUS="$2"; shift 2 ;;
        *)       echo "unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROJECT_DATA:?PROJECT_DATA not set (export it via ~/.bashrc to your data root, e.g. ~/dev/data)}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"
: "${VBENCH_INFO:=$VBENCH_REPO_DIR/vbench/VBench_full_info.json}"
: "${LL_ENV_NAME:=longlive}"
: "${VBENCH_ENV:=vbench}"

# Activate longlive env so `python` resolves to the right interpreter
# (mamba install path varies across HPC vs lab vs arp; don't hardcode).
eval "$(mamba shell hook --shell bash)"
mamba activate "$LL_ENV_NAME"

# Fail-fast on missing ckpt — common pitfall when shell doesn't export $LL_DATA
# and the caller passes "$LL_DATA/..." which expands to "/...".
if [ ! -f "$CKPT" ]; then
    echo "[vbench][error] ckpt not found: $CKPT" >&2
    echo "                check that LL_DATA / paths are set in your shell" >&2
    exit 1
fi

RUN_DIR="$LL_DATA/vbench_runs/$RUN_ID"
PROMPT_DIR="$RUN_DIR/prompts"
mkdir -p "$RUN_DIR" "$PROMPT_DIR"

echo "[vbench] run_id      = $RUN_ID"
echo "[vbench] ckpt        = $CKPT"
echo "[vbench] config      = $CONFIG"
echo "[vbench] run_dir     = $RUN_DIR"
echo "[vbench] gpus        = $GPUS"
[ -n "$LIMIT" ] && echo "[vbench] $LIMIT"

LONGLIVE_PY="$(which python)"

# -------- build prompt manifest --------
if [ ! -f "$PROMPT_DIR/manifest.jsonl" ]; then
    echo "[vbench] phase 1a: building VBench prompt manifest"
    python "$REPO_ROOT/scripts/vbench/build_vbench_prompts.py" \
        --vbench_info "$VBENCH_INFO" \
        --output_dir "$PROMPT_DIR" \
        ${LIMIT}
else
    echo "[vbench] phase 1a: manifest already exists, skipping"
fi

# -------- dispatch generation across GPUs --------
: "${LL_VBENCH_NUM_SAMPLES:=1}"

# Optional: feed LLM-augmented prompts to the generator (paper protocol for
# Wan-based models). VBench ships them at
# prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt.
# If both files exist, the dispatcher will line-pair raw↔aug and feed aug
# while keeping raw filenames for VBench's standard-mode globbing.
: "${VBENCH_AUG_PROMPTS:=$VBENCH_REPO_DIR/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt}"
: "${VBENCH_RAW_PROMPTS:=$VBENCH_REPO_DIR/prompts/all_dimension.txt}"
AUG_ARGS=""
if [ -f "$VBENCH_AUG_PROMPTS" ] && [ -f "$VBENCH_RAW_PROMPTS" ]; then
    echo "[vbench] using LLM-augmented prompts from $VBENCH_AUG_PROMPTS"
    AUG_ARGS="--augmented_prompts $VBENCH_AUG_PROMPTS --raw_prompts $VBENCH_RAW_PROMPTS"
else
    echo "[vbench] WARN: augmented prompts not found; falling back to raw VBench prompts"
    echo "         expected: $VBENCH_AUG_PROMPTS"
    echo "         expected: $VBENCH_RAW_PROMPTS"
fi

echo "[vbench] phase 1b: dispatching generation on GPUs $GPUS  (num_samples=$LL_VBENCH_NUM_SAMPLES)"
python "$REPO_ROOT/scripts/vbench/eval_dispatch.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --manifest "$PROMPT_DIR/manifest.jsonl" \
    --output_dir "$RUN_DIR" \
    --gpu_ids "$GPUS" \
    --num_samples "$LL_VBENCH_NUM_SAMPLES" \
    --python_bin "$LONGLIVE_PY" \
    $AUG_ARGS

# -------- score generated mp4s with vbench --------
echo "[vbench] phase 2: scoring with VBench"
mkdir -p "$RUN_DIR/vbench_out"
mamba deactivate
mamba activate "$VBENCH_ENV"
# LL_VBENCH_DIMS lets the caller restrict the dim list — useful when one
# of VBench's auto-downloaded model ckpts is unreachable through your
# proxy (e.g. grit_b_densecap_objectdet.pth needs Azure-private auth on
# Charite proxy → skip multiple_objects).
: "${LL_VBENCH_DIMS:=all}"
python "$REPO_ROOT/scripts/vbench/run_vbench_eval.py" \
    --videos_dir "$RUN_DIR/videos" \
    --vbench_info "$VBENCH_INFO" \
    --output_dir "$RUN_DIR/vbench_out" \
    --dimensions "$LL_VBENCH_DIMS"

# -------- aggregate to Total/Quality/Semantic (back in longlive env for omegaconf) --------
echo "[vbench] phase 3: aggregating scores"
mamba deactivate
mamba activate "$LL_ENV_NAME"
python "$REPO_ROOT/scripts/vbench/aggregate_scores.py" "$RUN_DIR"

echo "[vbench] DONE  →  $RUN_DIR/summary.json"
