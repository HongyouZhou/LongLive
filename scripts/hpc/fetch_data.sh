#!/bin/bash
# Stage model weights + motion data INTO the repo (matches arp layout: weights
# live next to the code under .gitignore).
#
# Default sources (no extra env needed):
#   - Wan2.1-T2V-{14B,1.3B}      → HuggingFace Hub  (Wan-AI/Wan2.1-T2V-*)
#   - LongLive-1.3B base ckpt    → HuggingFace Hub  (Efficient-Large-Model/LongLive-1.3B)
#   - Motion data (motion_refs + jsonl prompts)
#                                → MUST already exist at $LL_REPO/data/wm
#                                  (no public mirror; you have to upload it)
#
# Motion data sources you can use (pick one, do this BEFORE running the script):
#
#   (a) scp from a machine that has it:
#       rsync -aP /home/hongyou/dev/data/wm/  hozh10@<HPC>:$PWD_ON_HPC/data/wm/
#
#   (b) symlink a copy you already have on HPC:
#       ln -sfn /sc-projects/sc-proj-.../path/to/wm  $LL_REPO/data/wm
#
#   (c) opt-in to rsync from arp via this script (legacy, only if reachable):
#       LL_REMOTE_HOST=hongyou@arp.example bash scripts/hpc/fetch_data.sh
#
# Run on a *login* node (compute nodes typically have no outbound network).
# Run from the cloned repo root.

set -euo pipefail

: "${LL_ENV_NAME:=longlive}"
: "${LL_REPO:=$PWD}"
: "${LL_CONFIG:=configs/longlive_finetune_motion_cross.yaml}"

if [ ! -f "$LL_REPO/$LL_CONFIG" ]; then
  echo "[data][error] LL_REPO=$LL_REPO doesn't look like the LongLive repo (missing $LL_CONFIG)." >&2
  echo "             cd into the repo before running, or export LL_REPO=/path/to/LongLive." >&2
  exit 1
fi

cd "$LL_REPO"
echo "[data] LL_REPO   = $LL_REPO"
echo "[data] LL_CONFIG = $LL_CONFIG"
[ -n "${LL_REMOTE_HOST:-}" ] && echo "[data] LL_REMOTE_HOST = $LL_REMOTE_HOST (arp rsync mode)"

mkdir -p logs wandb hf_cache \
         wan_models longlive_models/models checkpoints \
         data/wm

# Activate env (needs `hf` + omegaconf). mamba shell hook isn't loaded in
# non-interactive subshells, so eval it explicitly. Skip if env is already
# active (e.g. user ran `mamba activate longlive` in parent shell).
if [ "${CONDA_DEFAULT_ENV:-}" != "$LL_ENV_NAME" ]; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate "$LL_ENV_NAME"
fi

# Cache HF downloads inside the repo's hf_cache/ (in .gitignore).
export HF_HOME="$LL_REPO/hf_cache"
export TRANSFORMERS_CACHE="$LL_REPO/hf_cache"
: "${HF_TOKEN:?HF_TOKEN not set — your .bashrc should export it}"

# -------- helpers --------
arp_mode() { [ -n "${LL_REMOTE_HOST:-}" ]; }

sync_arp() {  # rsync remote_path local_path
  local src="$LL_REMOTE_HOST:$1"
  local dst="$2"
  echo "[data] (arp) rsync $src -> $dst"
  rsync -aP --inplace "$src/" "$dst/"
}

# -------- 1. LongLive-1.3B base ckpt (5.3 GB) --------
# HF repo Efficient-Large-Model/LongLive-1.3B contains longlive_base.pt
# (and assets/prompts that ship in the repo already, harmless to redownload).
if [ ! -f "longlive_models/models/longlive_base.pt" ]; then
  if arp_mode; then
    sync_arp "/home/hongyou/dev/LongLive/longlive_models/models" "longlive_models/models"
  else
    echo "[data] downloading LongLive-1.3B from HF Hub ..."
    hf download Efficient-Large-Model/LongLive-1.3B \
        --local-dir longlive_models \
        --include "models/longlive_base.pt"
  fi
else
  echo "[data] longlive_base.pt already present, skipping."
fi
ln -sfn "../longlive_models/models/longlive_base.pt" "checkpoints/longlive_init.pt"

# -------- 2. Wan2.1 weights (~82 GB total) --------
if [ ! -f "wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth" ]; then
  if arp_mode; then
    sync_arp "/home/hongyou/dev/LongLive/wan_models" "wan_models"
  else
    echo "[data] downloading Wan2.1-T2V-14B from HF Hub (~65 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-14B \
        --local-dir wan_models/Wan2.1-T2V-14B
  fi
else
  echo "[data] Wan2.1-T2V-14B already present, skipping."
fi

if [ ! -f "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" ]; then
  if arp_mode; then
    : # already covered by the wan_models rsync above
  else
    echo "[data] downloading Wan2.1-T2V-1.3B from HF Hub (~17 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir wan_models/Wan2.1-T2V-1.3B
  fi
else
  echo "[data] Wan2.1-T2V-1.3B already present, skipping."
fi

# -------- 3. Motion data (~1.2 GB; private, not on any public hub) --------
# Sub-layout expected:
#   data/wm/motion_refs/*.mp4            (CelebV reference clips)
#   data/wm/prompts/motion_pairs_*.jsonl (train/val pair prompts)
need_motion=0
[ ! -d "data/wm/motion_refs" ] && need_motion=1
[ -d "data/wm/motion_refs" ] && [ -z "$(ls -A data/wm/motion_refs 2>/dev/null)" ] && need_motion=1
[ ! -d "data/wm/prompts"    ] && need_motion=1

if [ "$need_motion" -eq 1 ]; then
  if arp_mode; then
    sync_arp "/home/hongyou/dev/data/wm" "data/wm"
  else
    cat >&2 <<EOF
[data][error] Motion data missing at $LL_REPO/data/wm.
              Expected:
                data/wm/motion_refs/*.mp4
                data/wm/prompts/motion_pairs_{train,val}.jsonl
                data/wm/prompts/motion_pairs_cross_{train,val}.jsonl

              No public mirror exists — upload it yourself:
                # from a machine that has the data:
                rsync -aP /home/hongyou/dev/data/wm/  \\
                      hozh10@<HPC>:$LL_REPO/data/wm/

              Or, if you already have a copy on HPC, symlink it:
                ln -sfn /path/to/existing/wm  $LL_REPO/data/wm

              Or, opt into arp rsync (only if reachable):
                LL_REMOTE_HOST=hongyou@arp bash scripts/hpc/fetch_data.sh
EOF
    exit 1
  fi
else
  echo "[data] motion data present, skipping."
fi

# -------- 4. Render an HPC-pathed config --------
# yamls hard-code arp paths for motion_pair_jsonl / motion_ref_root.
# Rewrite to $LL_REPO/data/wm/. generator_ckpt is repo-relative — works as-is.
SRC_YAML="$LL_REPO/$LL_CONFIG"
DST_YAML="${SRC_YAML%.yaml}_hpc.yaml"

python - <<EOF
import os
from omegaconf import OmegaConf

src  = "$SRC_YAML"
dst  = "$DST_YAML"
repo = "$LL_REPO"
cfg  = OmegaConf.load(src)

def remap_jsonl(p):
    return f"{repo}/data/wm/prompts/" + os.path.basename(p)

if "motion_pair_jsonl" in cfg:
    cfg.motion_pair_jsonl     = remap_jsonl(cfg.motion_pair_jsonl)
if "val_motion_pair_jsonl" in cfg:
    cfg.val_motion_pair_jsonl = remap_jsonl(cfg.val_motion_pair_jsonl)
if "motion_ref_root" in cfg:
    cfg.motion_ref_root       = f"{repo}/data/wm/motion_refs"

OmegaConf.save(cfg, dst)
print(f"[data] wrote {dst}")
EOF

echo
echo "[data] DONE. Layout:"
echo "  $LL_REPO/wan_models/                  — Wan2.1-T2V-{14B,1.3B}"
echo "  $LL_REPO/longlive_models/models/      — longlive_base.pt"
echo "  $LL_REPO/checkpoints/longlive_init.pt → ../longlive_models/models/longlive_base.pt"
echo "  $LL_REPO/data/wm/                     — motion refs + prompts (your upload)"
echo "  $LL_REPO/${LL_CONFIG%.yaml}_hpc.yaml  — patched config"
echo
echo "Next: sbatch scripts/hpc/sbatch_train.sh"
