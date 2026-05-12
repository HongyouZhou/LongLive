#!/bin/bash
# Stage UCF Sports Action + LOVEU-TGVE 2023 eval data INTO $LL_DATA on HPC.
#
# Default = direct download (curl + gdown), since HPC ↔ lab has no direct
# SSH channel (confirmed 2026-05-12) and neither dataset has an HF Hub
# mirror. Run on a *login* node — compute nodes lack outbound network.
#
# Usage:
#   bash scripts/hpc/fetch_motion_eval.sh
#
# Env-var overrides:
#   LL_REMOTE_HOST=<user@host>        opt-in rsync from a reachable peer
#                                     (not auto-set; default = direct DL)
#   LL_MOTION_EVAL_DATASETS=ucf,loveu subset (default both)
#   LL_DATA=<path>                    override data root
#   LL_ENV_NAME=longlive              mamba env name

set -euo pipefail

: "${LL_ENV_NAME:=longlive}"
: "${LL_REPO:=$PWD}"
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
: "${LL_MOTION_EVAL_DATASETS:=ucf,loveu}"

# LL_REMOTE_HOST is opt-in only. We don't auto-default to "hongyou@lab"
# because HPC ↔ lab has no direct SSH path. If the user sets it manually,
# we respect that (escape hatch for a side-channel they've configured).

if [ ! -f "$LL_REPO/scripts/prepare_motion_eval.py" ]; then
    echo "[motion_eval][error] LL_REPO=$LL_REPO doesn't look like the LongLive repo" >&2
    echo "                     (missing scripts/prepare_motion_eval.py)" >&2
    echo "                     cd into the repo before running, or export LL_REPO=/path/to/LongLive" >&2
    exit 1
fi

cd "$LL_REPO"

echo "[motion_eval] LL_REPO          = $LL_REPO"
echo "[motion_eval] LL_DATA          = $LL_DATA"
echo "[motion_eval] datasets         = $LL_MOTION_EVAL_DATASETS"
if [ -n "${LL_REMOTE_HOST:-}" ]; then
    echo "[motion_eval] LL_REMOTE_HOST   = $LL_REMOTE_HOST  (rsync mode)"
else
    echo "[motion_eval] mode             = direct download (curl + gdown)"
fi

mkdir -p "$LL_DATA"

# Activate env. `mamba shell hook` is bash-only and isn't loaded in non-interactive
# subshells; eval it explicitly. Skip if env is already active.
if [ "${CONDA_DEFAULT_ENV:-}" != "$LL_ENV_NAME" ]; then
    if command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash 2>/dev/null || conda shell.bash hook)"
    else
        eval "$(conda shell.bash hook)"
    fi
    mamba activate "$LL_ENV_NAME" 2>/dev/null || conda activate "$LL_ENV_NAME"
fi

# Defensive: prepare_motion_eval.py needs gdown for direct LOVEU download,
# and rsync (always). In rsync mode, gdown is unused.
if [ -z "${LL_REMOTE_HOST:-}" ]; then
    python -c "import gdown" 2>/dev/null || {
        echo "[motion_eval][error] gdown not installed in $LL_ENV_NAME — needed for LOVEU direct download." >&2
        echo "                     Run: bash scripts/motion_eval/setup_motion_eval_env.sh" >&2
        exit 1
    }
fi
command -v rsync >/dev/null 2>&1 || {
    echo "[motion_eval][error] rsync not found" >&2
    exit 1
}

# Idempotent — prepare_motion_eval.py skips datasets whose output already exists.
EXTRA_ARGS=()
if [ -n "${LL_REMOTE_HOST:-}" ]; then
    EXTRA_ARGS+=(--remote_host "$LL_REMOTE_HOST")
fi

python scripts/prepare_motion_eval.py \
    --data_root "$LL_DATA" \
    --datasets "$LL_MOTION_EVAL_DATASETS" \
    "${EXTRA_ARGS[@]}"

echo
echo "[motion_eval] DONE. Data staged under:"
for ds in $(echo "$LL_MOTION_EVAL_DATASETS" | tr ',' ' '); do
    case "$ds" in
        ucf)
            echo "  $LL_DATA/ucf_sports/videos/    $(ls -1 "$LL_DATA/ucf_sports/videos/" 2>/dev/null | wc -l) categories"
            echo "  $LL_DATA/ucf_sports/manifest.csv"
            ;;
        loveu)
            echo "  $LL_DATA/loveu_tgve/videos/    $(ls -1 "$LL_DATA/loveu_tgve/videos/" 2>/dev/null | wc -l) videos"
            echo "  $LL_DATA/loveu_tgve/prompts.csv"
            ;;
    esac
done
echo
echo "Next:"
echo "  sbatch scripts/hpc/sbatch_motion_eval.sh longlive_models/models/lora.pt baseline_v1"
