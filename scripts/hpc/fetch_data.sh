#!/bin/bash
# Stage model weights INTO $LL_DATA (matches arp layout: weights live next to
# the code under .gitignore).
#
# Default sources (no extra env needed):
#   - Wan2.1-T2V-{14B,1.3B}      → HuggingFace Hub  (Wan-AI/Wan2.1-T2V-*)
#   - LongLive-1.3B base ckpt    → HuggingFace Hub  (Efficient-Large-Model/LongLive-1.3B)
#
# Override LL_REMOTE_HOST=hongyou@lab to rsync from a peer machine instead of
# pulling from HF.
#
# Run on a *login* node (compute nodes typically have no outbound network).
# Run from the cloned repo root.

set -euo pipefail

: "${LL_ENV_NAME:=longlive}"
: "${LL_REPO:=$PWD}"
: "${LL_CONFIG:=configs/longlive_train_long.yaml}"
# Data lives under $PROJECT_DATA (shared data root, common across projects).
# This script claims the "wm" sub-namespace under it. Override LL_DATA to
# put it elsewhere.
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"

if [ ! -f "$LL_REPO/$LL_CONFIG" ]; then
  echo "[data][error] LL_REPO=$LL_REPO doesn't look like the LongLive repo (missing $LL_CONFIG)." >&2
  echo "             cd into the repo before running, or export LL_REPO=/path/to/LongLive." >&2
  exit 1
fi

cd "$LL_REPO"
echo "[data] LL_REPO   = $LL_REPO"
echo "[data] LL_DATA   = $LL_DATA"
echo "[data] LL_CONFIG = $LL_CONFIG"
[ -n "${LL_REMOTE_HOST:-}" ] && echo "[data] LL_REMOTE_HOST = $LL_REMOTE_HOST (rsync mode)"

# All real data lives flat under $LL_DATA (single namespace, no subdirs):
#   wan_models/        Wan2.1-T2V-{14B,1.3B}/
#   longlive_models/   models/longlive_base.pt
#   hf_cache/          HF download cache
# - longlive/utils/wan_wrapper.py reads WAN_MODELS_ROOT (set by sbatch_train.sh)
# - generator_ckpt is rewritten to an absolute path in the rendered *_hpc.yaml below
mkdir -p "$LL_DATA"/{wan_models,longlive_models/models,hf_cache}

# logs/wandb stay local to the repo (small writes, per-job artifacts).
mkdir -p logs wandb

# Activate env (needs `hf` + omegaconf). mamba shell hook isn't loaded in
# non-interactive subshells, so eval it explicitly. Skip if env is already
# active (e.g. user ran `mamba activate longlive` in parent shell).
if [ "${CONDA_DEFAULT_ENV:-}" != "$LL_ENV_NAME" ]; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate "$LL_ENV_NAME"
fi

# HF downloads cached in $LL_DATA/hf_cache (shared across runs / branches).
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
: "${HF_TOKEN:?HF_TOKEN not set — your .bashrc should export it}"

# -------- helpers --------
remote_mode() { [ -n "${LL_REMOTE_HOST:-}" ]; }

sync_remote() {  # rsync remote_path local_path
  local src="$LL_REMOTE_HOST:$1"
  local dst="$2"
  echo "[data] (remote) rsync $src -> $dst"
  rsync -aP --inplace "$src/" "$dst/"
}

# -------- 1. LongLive-1.3B base ckpt + prompt corpora (~5.6 GB) --------
# Pull base weights, LoRA weights, and prompt text files.
if [ ! -f "$LL_DATA/longlive_models/models/longlive_base.pt" ] \
   || [ ! -f "$LL_DATA/longlive_models/models/lora.pt" ] \
   || [ ! -f "$LL_DATA/longlive_models/prompts/vidprom_filtered_extended.txt" ]; then
  if remote_mode; then
    sync_remote "/home/hongyou/dev/LongLive/longlive_models" "$LL_DATA/longlive_models"
  else
    echo "[data] downloading LongLive-1.3B from HF Hub ..."
    # `hf download` treats trailing positional args after `--include` as
    # filenames (URL-encodes glob '*'), so split into one call per filter.
    hf download Efficient-Large-Model/LongLive-1.3B \
        --local-dir "$LL_DATA/longlive_models" \
        --include "models/longlive_base.pt"
    hf download Efficient-Large-Model/LongLive-1.3B \
        --local-dir "$LL_DATA/longlive_models" \
        --include "models/lora.pt"
    hf download Efficient-Large-Model/LongLive-1.3B \
        --local-dir "$LL_DATA/longlive_models" \
        --include "prompts/*"
  fi
else
  echo "[data] base + lora + prompts already present, skipping."
fi

# Repo-relative symlinks `prompts/*.txt -> ../longlive_models/prompts/*.txt`
# need a `longlive_models` entry inside the repo. Stage one as a symlink
# to $LL_DATA so the upstream config's `data_path: prompts/...` resolves
# without copying ~270 MB into the repo tree.
if [ ! -e "$LL_REPO/longlive_models" ]; then
  ln -s "$LL_DATA/longlive_models" "$LL_REPO/longlive_models"
  echo "[data] symlinked $LL_REPO/longlive_models -> $LL_DATA/longlive_models"
fi

# -------- 2. Wan2.1 weights (~82 GB total) --------
if [ ! -f "$LL_DATA/wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth" ]; then
  if remote_mode; then
    sync_remote "/home/hongyou/dev/LongLive/wan_models" "$LL_DATA/wan_models"
  else
    echo "[data] downloading Wan2.1-T2V-14B from HF Hub (~65 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-14B \
        --local-dir "$LL_DATA/wan_models/Wan2.1-T2V-14B"
  fi
else
  echo "[data] Wan2.1-T2V-14B already present, skipping."
fi

if [ ! -f "$LL_DATA/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" ]; then
  if remote_mode; then
    : # already covered by the wan_models rsync above
  else
    echo "[data] downloading Wan2.1-T2V-1.3B from HF Hub (~17 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir "$LL_DATA/wan_models/Wan2.1-T2V-1.3B"
  fi
else
  echo "[data] Wan2.1-T2V-1.3B already present, skipping."
fi

# -------- 3. Render an HPC-pathed config --------
# Rewrite repo-relative generator_ckpt to absolute path under $LL_DATA so
# the training process doesn't need a symlink shim.
SRC_YAML="$LL_REPO/$LL_CONFIG"
DST_YAML="${SRC_YAML%.yaml}_hpc.yaml"

python - <<EOF
import os
from omegaconf import OmegaConf

src  = "$SRC_YAML"
dst  = "$DST_YAML"
data = "$LL_DATA"
cfg  = OmegaConf.load(src)

def remap_ckpt(p):
    if not p or os.path.isabs(p):
        return p
    if "longlive_init" in p or "longlive_base" in p:
        return f"{data}/longlive_models/models/longlive_base.pt"
    return os.path.join(data, p)  # generic fallback

if "generator_ckpt" in cfg:
    cfg.generator_ckpt = remap_ckpt(cfg.generator_ckpt)

OmegaConf.save(cfg, dst)
print(f"[data] wrote {dst}")
EOF

echo
echo "[data] DONE. All data flat under \$LL_DATA, no symlinks in repo:"
echo "  $LL_DATA/wan_models/                  — Wan2.1-T2V-{14B,1.3B}"
echo "  $LL_DATA/longlive_models/models/      — longlive_base.pt"
echo "  $LL_DATA/hf_cache/                    — HF download cache"
echo "  $LL_REPO/${LL_CONFIG%.yaml}_hpc.yaml  — config with absolute paths to data"
echo
echo "Training picks up data via:"
echo "  * sbatch_train.sh exports  WAN_MODELS_ROOT=\$LL_DATA/wan_models"
echo "  * sbatch_train.sh exports  HF_HOME=\$LL_DATA/hf_cache"
echo "  * *_hpc.yaml has absolute  generator_ckpt"
echo
echo "Next: sbatch scripts/hpc/sbatch_train.sh"
