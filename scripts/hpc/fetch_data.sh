#!/bin/bash
# Stage model weights + motion data into $LL_WORK (default $PROJECT_HOME/longlive).
# Code/repo lives wherever you cloned it — pass it in via LL_REPO or run this
# script from inside the repo (PWD is used as fallback).
#
# Run on a *login* node (needs outbound network for HF Hub or ssh to arp).
#
# Two source modes for weights/data:
#   (a) rsync from arp           — fastest if HPC can ssh to your arp box.
#       Set LL_REMOTE_HOST=hongyou@arp.example before running.
#   (b) HuggingFace Hub fallback — used automatically when LL_REMOTE_HOST
#       unset OR ssh to it fails. Pulls Wan-AI/Wan2.1-T2V-{14B,1.3B} via $HF_TOKEN
#       (already exported by your .bashrc). NOTE: longlive_base.pt and the motion
#       refs are NOT on a public mirror — those *require* mode (a).
#
# Run after setup_mamba_env.sh (uses huggingface-cli from the env).
#
# Usage (from inside the cloned repo):
#   LL_REMOTE_HOST=hongyou@arp bash scripts/hpc/fetch_data.sh

set -euo pipefail

: "${PROJECT_HOME:?PROJECT_HOME not set — source ~/.bashrc first}"
: "${LL_WORK:=$PROJECT_HOME/longlive}"
: "${LL_ENV_PREFIX:=$PROJECT_HOME/envs/longlive}"
# LL_REPO: path to the cloned LongLive repo. Defaults to PWD when run from repo root.
: "${LL_REPO:=$PWD}"
: "${LL_REMOTE_REPO:=/home/hongyou/dev/LongLive}"
: "${LL_REMOTE_DATA:=/home/hongyou/dev/data/wm}"
: "${LL_CONFIG:=configs/longlive_finetune_motion_cross.yaml}"

if [ ! -f "$LL_REPO/$LL_CONFIG" ]; then
  echo "[data][error] LL_REPO=$LL_REPO doesn't look like the LongLive repo (missing $LL_CONFIG)." >&2
  echo "             Either cd into the repo before running, or export LL_REPO=/path/to/LongLive." >&2
  exit 1
fi

echo "[data] LL_WORK        = $LL_WORK"
echo "[data] LL_REPO        = $LL_REPO"
echo "[data] LL_REMOTE_HOST = ${LL_REMOTE_HOST:-<unset → HF fallback>}"
echo "[data] LL_CONFIG      = $LL_CONFIG"

mkdir -p "$LL_WORK"/{logs,wandb,checkpoints,wan_models,longlive_models/models,data/wm,hf_cache}

# Need huggingface-cli + omegaconf from the env we built.
source /opt/miniforge/etc/profile.d/conda.sh
conda activate "$LL_ENV_PREFIX"

# Cache HF downloads on project FS, not $HOME.
export HF_HOME="$LL_WORK/hf_cache"
export TRANSFORMERS_CACHE="$LL_WORK/hf_cache"

# -------- helpers --------
have_remote() {
  [ -n "${LL_REMOTE_HOST:-}" ] && \
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$LL_REMOTE_HOST" true 2>/dev/null
}

sync_remote() {  # rsync remote_path local_path
  local src="$LL_REMOTE_HOST:$1"
  local dst="$2"
  echo "[data] rsync $src -> $dst"
  rsync -aP --inplace "$src/" "$dst/"
}

# -------- 1. longlive_base.pt (5.3 GB; NOT on public hub) --------
if [ ! -f "$LL_WORK/longlive_models/models/longlive_base.pt" ]; then
  if have_remote; then
    sync_remote "$LL_REMOTE_REPO/longlive_models/models" "$LL_WORK/longlive_models/models"
  else
    echo "[data][error] longlive_base.pt missing and arp unreachable." >&2
    echo "             scp it manually:  scp arp:$LL_REMOTE_REPO/longlive_models/models/longlive_base.pt \\" >&2
    echo "                               $LL_WORK/longlive_models/models/" >&2
    exit 1
  fi
fi
ln -sfn "$LL_WORK/longlive_models/models/longlive_base.pt" \
        "$LL_WORK/checkpoints/longlive_init.pt"

# -------- 2. Wan2.1 weights (~82 GB total) --------
if [ ! -f "$LL_WORK/wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth" ]; then
  if have_remote; then
    sync_remote "$LL_REMOTE_REPO/wan_models" "$LL_WORK/wan_models"
  else
    echo "[data] arp unreachable, pulling Wan2.1 from HF Hub (~82 GB, can take 1-3 h) ..."
    : "${HF_TOKEN:?HF_TOKEN must be exported (your .bashrc does this)}"
    huggingface-cli download Wan-AI/Wan2.1-T2V-14B  --local-dir "$LL_WORK/wan_models/Wan2.1-T2V-14B"
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$LL_WORK/wan_models/Wan2.1-T2V-1.3B"
  fi
else
  echo "[data] Wan2.1-T2V-14B already present, skipping."
fi

# -------- 3. Motion data (~1.2 GB; NOT on public hub) --------
if [ ! -d "$LL_WORK/data/wm/motion_refs" ] || [ -z "$(ls -A "$LL_WORK/data/wm/motion_refs" 2>/dev/null)" ]; then
  if have_remote; then
    sync_remote "$LL_REMOTE_DATA" "$LL_WORK/data/wm"
  else
    echo "[data][error] motion data only lives on arp; set LL_REMOTE_HOST." >&2
    exit 1
  fi
else
  echo "[data] motion data already present, skipping."
fi

# -------- 4. Render an HPC-pathed config (written into the repo) --------
# The yamls hard-code arp paths (motion_pair_jsonl, motion_ref_root etc.).
# We render configs/<name>_hpc.yaml with paths rewritten to $LL_WORK,
# so the original yaml stays untouched and the sbatch script picks up the override.
# Tip: add `configs/*_hpc.yaml` to .gitignore to keep `git status` clean.
SRC_YAML="$LL_REPO/$LL_CONFIG"
DST_YAML="${SRC_YAML%.yaml}_hpc.yaml"

python - <<EOF
import os
from omegaconf import OmegaConf

src  = "$SRC_YAML"
dst  = "$DST_YAML"
work = "$LL_WORK"
cfg  = OmegaConf.load(src)

def remap_jsonl(p):
    return f"{work}/data/wm/prompts/" + os.path.basename(p)

if "motion_pair_jsonl" in cfg:
    cfg.motion_pair_jsonl     = remap_jsonl(cfg.motion_pair_jsonl)
if "val_motion_pair_jsonl" in cfg:
    cfg.val_motion_pair_jsonl = remap_jsonl(cfg.val_motion_pair_jsonl)
if "motion_ref_root" in cfg:
    cfg.motion_ref_root       = f"{work}/data/wm/motion_refs"

# generator_ckpt is repo-relative ("checkpoints/longlive_init.pt").
# Rewrite it to absolute under \$LL_WORK so it works regardless of CWD.
if "generator_ckpt" in cfg and not os.path.isabs(cfg.generator_ckpt):
    cfg.generator_ckpt = f"{work}/" + cfg.generator_ckpt

OmegaConf.save(cfg, dst)
print(f"[data] wrote {dst}")
EOF

echo
echo "[data] DONE. Layout:"
echo "  $LL_REPO                             — code (this clone)"
echo "  $LL_WORK/wan_models/                 — Wan2.1-T2V-{14B,1.3B}"
echo "  $LL_WORK/longlive_models/models/     — longlive_base.pt"
echo "  $LL_WORK/checkpoints/longlive_init.pt → longlive_base.pt symlink"
echo "  $LL_WORK/data/wm/                    — motion refs + prompts"
echo "  $LL_REPO/${LL_CONFIG%.yaml}_hpc.yaml — patched config"
echo
echo "Next: sbatch scripts/hpc/sbatch_train.sh   (from $LL_REPO)"
