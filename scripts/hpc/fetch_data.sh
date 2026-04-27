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
# Motion data is built from the OpenVid-1M HF dataset by scripts/prepare_openvid.py:
#   - downloads `nkp37/OpenVid-1M` part 0 zip (~25 GB) + caption CSV
#   - filters on motion_score / aesthetic / duration, extracts 1000 clips
#   - emits self-pair + cross-pair JSONLs
# Override via:
#   LL_OPENVID_PART=0          which OpenVid zip part to use
#   LL_OPENVID_NUM_KEEP=1000   how many clips to keep
#   LL_REMOTE_HOST=hongyou@lab opt-in to rsync from lab (data source-of-truth
#                              since 2026-04-26: ~360 GB motion_refs +
#                              headmotion-mined JSONLs; arp's wm/ is now
#                              just the original 1000-clip backup).
#
# Run on a *login* node (compute nodes typically have no outbound network).
# Run from the cloned repo root.

set -euo pipefail

: "${LL_ENV_NAME:=longlive}"
: "${LL_REPO:=$PWD}"
: "${LL_CONFIG:=configs/longlive_unidad_motion.yaml}"
: "${LL_OPENVID_PART:=0}"
: "${LL_OPENVID_NUM_KEEP:=1000}"
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
#   motion_refs/       *.mp4 from OpenVid
#   meta/              data/train/OpenVid-1M.csv
#   prompts/           motion_pairs_{,cross_}{train,val}.jsonl
#   hf_cache/          HF download cache
# - utils/wan_wrapper.py reads WAN_MODELS_ROOT (set by sbatch_train.sh)
# - generator_ckpt / motion_* paths get rewritten to absolute paths in the
#   rendered *_hpc.yaml below
mkdir -p "$LL_DATA"/{wan_models,longlive_models/models,motion_refs,prompts,meta,hf_cache}

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
arp_mode()    { remote_mode; }   # backward-compat alias

sync_remote() {  # rsync remote_path local_path
  local src="$LL_REMOTE_HOST:$1"
  local dst="$2"
  echo "[data] (remote) rsync $src -> $dst"
  rsync -aP --inplace "$src/" "$dst/"
}
sync_arp() { sync_remote "$@"; }   # backward-compat alias

# -------- 1. LongLive-1.3B base ckpt (5.3 GB) --------
if [ ! -f "$LL_DATA/longlive_models/models/longlive_base.pt" ]; then
  if arp_mode; then
    sync_arp "/home/hongyou/dev/LongLive/longlive_models/models" "$LL_DATA/longlive_models/models"
  else
    echo "[data] downloading LongLive-1.3B from HF Hub ..."
    hf download Efficient-Large-Model/LongLive-1.3B \
        --local-dir "$LL_DATA/longlive_models" \
        --include "models/longlive_base.pt"
  fi
else
  echo "[data] longlive_base.pt already present, skipping."
fi

# -------- 2. Wan2.1 weights (~82 GB total) --------
if [ ! -f "$LL_DATA/wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth" ]; then
  if arp_mode; then
    sync_arp "/home/hongyou/dev/LongLive/wan_models" "$LL_DATA/wan_models"
  else
    echo "[data] downloading Wan2.1-T2V-14B from HF Hub (~65 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-14B \
        --local-dir "$LL_DATA/wan_models/Wan2.1-T2V-14B"
  fi
else
  echo "[data] Wan2.1-T2V-14B already present, skipping."
fi

if [ ! -f "$LL_DATA/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" ]; then
  if arp_mode; then
    : # already covered by the wan_models rsync above
  else
    echo "[data] downloading Wan2.1-T2V-1.3B from HF Hub (~17 GB) ..."
    hf download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir "$LL_DATA/wan_models/Wan2.1-T2V-1.3B"
  fi
else
  echo "[data] Wan2.1-T2V-1.3B already present, skipping."
fi

# -------- 3. Motion data (lab is source-of-truth since 2026-04-26) --------
# Sub-layout expected on the source host:
#   /home/hongyou/dev/data/wm/motion_refs/*.mp4            (~360 GB,
#                                                           ~118k OpenVid
#                                                           clips post
#                                                           head-motion sweep)
#   /home/hongyou/dev/data/wm/prompts/motion_pairs_*.jsonl  (train/val pairs,
#                                                           includes
#                                                           motion_pairs_cross_headmotion_*.jsonl
#                                                           emitted by
#                                                           scripts/build_headmotion_jsonl.py)
#   /home/hongyou/dev/data/wm/meta/data/train/OpenVid-1M.csv (~400 MB)
#
# We rsync unconditionally in remote_mode -- rsync is incremental, so this
# only transfers deltas (new mp4s + updated JSONLs). The previous logic
# guarded on motion_pairs_train.jsonl existence, which silently skipped the
# new headmotion JSONLs whenever any old motion pair file was present.
if remote_mode; then
  sync_remote "/home/hongyou/dev/data/wm/motion_refs" "$LL_DATA/motion_refs"
  sync_remote "/home/hongyou/dev/data/wm/meta"        "$LL_DATA/meta"
  sync_remote "/home/hongyou/dev/data/wm/prompts"     "$LL_DATA/prompts"
elif [ -z "$(ls -A "$LL_DATA/motion_refs" 2>/dev/null)" ] \
     || [ ! -f "$LL_DATA/prompts/motion_pairs_train.jsonl" ]; then
  # Fallback (no LL_REMOTE_HOST): build a tiny self-contained dataset from
  # OpenVid-1M via prepare_openvid.py. This produces only the legacy 1000
  # self-pair clips; head-motion-aware mining is NOT done here (see
  # scripts/mine_openvid_headmotion.py + build_headmotion_jsonl.py, which
  # require the full 6.7 TB sweep and produce the JSONLs synced above).
  if [ ! -f "$LL_DATA/meta/data/train/OpenVid-1M.csv" ]; then
    echo "[data] downloading OpenVid-1M caption CSV ..."
    hf download nkp37/OpenVid-1M --repo-type dataset \
        --include "data/train/OpenVid-1M.csv" \
        --local-dir "$LL_DATA/meta"
  fi
  echo "[data] preparing motion data from OpenVid-1M part=$LL_OPENVID_PART, num_keep=$LL_OPENVID_NUM_KEEP"
  echo "[data] (downloads ~25 GB zip; takes 10-30 min depending on network)"
  python scripts/prepare_openvid.py \
      --part "$LL_OPENVID_PART" \
      --num_keep "$LL_OPENVID_NUM_KEEP" \
      --data_root "$LL_DATA"
  python scripts/prepare_openvid.py \
      --part "$LL_OPENVID_PART" \
      --num_keep "$LL_OPENVID_NUM_KEEP" \
      --data_root "$LL_DATA" \
      --cross_pair --output_suffix _cross \
      --skip_download --skip_extract
else
  echo "[data] motion data present (no LL_REMOTE_HOST set, skipping HF rebuild)."
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
data = "$LL_DATA"
cfg  = OmegaConf.load(src)

def remap_jsonl(p):
    return f"{data}/prompts/" + os.path.basename(p)

if "motion_pair_jsonl" in cfg:
    cfg.motion_pair_jsonl     = remap_jsonl(cfg.motion_pair_jsonl)
if "val_motion_pair_jsonl" in cfg:
    cfg.val_motion_pair_jsonl = remap_jsonl(cfg.val_motion_pair_jsonl)
if "motion_ref_root" in cfg:
    cfg.motion_ref_root       = f"{data}/motion_refs"

# generator_ckpt + score_init_from were repo-relative ("checkpoints/longlive_init.pt").
# Rewrite to absolute path under \$LL_DATA so no symlink shim is needed.
def remap_ckpt(p):
    if not p or os.path.isabs(p):
        return p
    if "longlive_init" in p or "longlive_base" in p:
        return f"{data}/longlive_models/models/longlive_base.pt"
    return os.path.join(data, p)  # generic fallback

if "generator_ckpt" in cfg:
    cfg.generator_ckpt = remap_ckpt(cfg.generator_ckpt)
if "unidad" in cfg and "dual_domain_dmd" in cfg.unidad and "score_init_from" in cfg.unidad.dual_domain_dmd:
    cfg.unidad.dual_domain_dmd.score_init_from = remap_ckpt(cfg.unidad.dual_domain_dmd.score_init_from)

OmegaConf.save(cfg, dst)
print(f"[data] wrote {dst}")
EOF

echo
echo "[data] DONE. All data flat under \$LL_DATA, no symlinks in repo:"
echo "  $LL_DATA/wan_models/                  — Wan2.1-T2V-{14B,1.3B}"
echo "  $LL_DATA/longlive_models/models/      — longlive_base.pt"
echo "  $LL_DATA/motion_refs/                 — OpenVid motion clips"
echo "  $LL_DATA/meta/data/train/             — OpenVid-1M.csv"
echo "  $LL_DATA/prompts/                     — motion_pairs_*.jsonl"
echo "  $LL_DATA/hf_cache/                    — HF download cache"
echo "  $LL_REPO/${LL_CONFIG%.yaml}_hpc.yaml  — config with absolute paths to data"
echo
echo "Training picks up data via:"
echo "  * sbatch_train.sh exports  WAN_MODELS_ROOT=\$LL_DATA/wan_models"
echo "  * sbatch_train.sh exports  HF_HOME=\$LL_DATA/hf_cache"
echo "  * *_hpc.yaml has absolute  generator_ckpt / motion_ref_root / motion_pair_jsonl"
echo
echo "Next: sbatch scripts/hpc/sbatch_train.sh"
