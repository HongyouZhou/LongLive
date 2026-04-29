#!/bin/bash
#SBATCH --job-name=motion_precache
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
#
# One-time motion-DMD V_ref latent cache build. Faster than running it inline
# inside sbatch_motion_dmd.sh because it only requests 1 GPU.
#
# Submit:
#   sbatch scripts/hpc/sbatch_precache_motion.sh
# Or override the refs jsonl / output path:
#   LL_MOTION_REFS=prompts/dancing_refs.jsonl \
#   LL_MOTION_CACHE=$LL_DATA/motion_dmd/dancing_v1.latents.pt \
#     sbatch scripts/hpc/sbatch_precache_motion.sh
#
# After this completes, sbatch_motion_dmd.sh will auto-detect the cache and
# skip its inline precache step.

set -e

source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -f "$SCRIPT_DIR/../../train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[precache] cannot locate repo"; exit 1
fi
cd "$PROJECT_DIR"

: "${LL_DATA:=$PROJECT_DATA/wm}"
export WAN_MODELS_ROOT="$LL_DATA/wan_models"

MOTION_REFS_JSONL="${LL_MOTION_REFS:-prompts/walking_refs_v1.jsonl}"
MOTION_CACHE_PATH="${LL_MOTION_CACHE:-$LL_DATA/motion_dmd/walking_v1.latents.pt}"

echo "[precache] refs_jsonl  = $MOTION_REFS_JSONL"
echo "[precache] cache path  = $MOTION_CACHE_PATH"
echo "[precache] LL_DATA     = $LL_DATA"

mkdir -p "$(dirname "$MOTION_CACHE_PATH")"
python scripts/motion_dmd/precache_motion_refs.py \
    --refs_jsonl "$MOTION_REFS_JSONL" \
    --output "$MOTION_CACHE_PATH" \
    --refs_root "$LL_DATA/motion_refs" \
    --device cuda

echo "[precache] done. cache = $MOTION_CACHE_PATH"
ls -la "$MOTION_CACHE_PATH"
