#!/bin/bash
# Upload motion_refs/ + head-motion manifests to a private HuggingFace dataset.
#
# Why: HPC currently re-downloads the entire 6.7 TB OpenVid-1M and extracts
# only the ~360 GB worth of head-motion-mined clips. Mirroring just the
# 360 GB on a private HF dataset cuts HPC's network 18x.
#
# Run on lab (where motion_refs/ + master_all.json live).
#
# Required env / args:
#   LL_HF_REPO    HF dataset id (e.g. hongyou/longlive-headmotion).
#                 Will be created as a PRIVATE dataset if it doesn't exist.
#   HF_TOKEN      authenticated to push (typically already in ~/.bashrc).
#
# Usage:
#   LL_HF_REPO=hongyou/longlive-headmotion bash scripts/upload_motion_refs_to_hf.sh
#
# Time: 360 GB / typical lab uplink. ~50 min @ 1 Gbps, ~3 h @ 200 Mbps.
# Resumable: hf upload-large-folder commits in chunks, re-running
# continues from where it left off.

set -euo pipefail

: "${LL_HF_REPO:?LL_HF_REPO not set, e.g. hongyou/longlive-headmotion}"
# HF_TOKEN can come from env OR from a prior `hf auth login` (stored at
# ~/.cache/huggingface/token). Don't fail hard if env-only is missing.
: "${MOTION_REFS_DIR:=/home/hongyou/dev/data/wm/motion_refs}"
: "${MASTER_JSON:=/home/hongyou/longlive_work/logs/master_all.json}"
: "${PROMPTS_DIR:=/home/hongyou/dev/data/wm/prompts}"
: "${STAGING_DIR:=/home/hongyou/longlive_work/hf_upload_staging}"

# Resolve a Python interpreter that has huggingface_hub installed. Prefer
# an explicit override; fall back to the lab-local env we rsync'd, then
# the sshfs-mounted arp env. We invoke python -m huggingface_hub instead
# of the `hf` wrapper because the latter's shebang points to arp paths
# which fail on lab after rsync.
: "${LL_PY:=}"
if [ -z "$LL_PY" ]; then
    if [ -x "/home/hongyou/longlive_envs/longlive-blackwell/bin/python" ]; then
        LL_PY="/home/hongyou/longlive_envs/longlive-blackwell/bin/python"
    elif [ -x "/home/hongyou/mnt/arp/miniforge3/envs/longlive-blackwell/bin/python" ]; then
        LL_PY="/home/hongyou/mnt/arp/miniforge3/envs/longlive-blackwell/bin/python"
    else
        echo "[upload][error] cannot locate Python; set LL_PY=/path/to/python" >&2
        exit 1
    fi
fi
hf_cli() { "$LL_PY" -m huggingface_hub.commands.huggingface_cli "$@"; }
echo "[upload] LL_PY          = $LL_PY"

echo "[upload] LL_HF_REPO     = $LL_HF_REPO"
echo "[upload] motion_refs    = $MOTION_REFS_DIR"
echo "[upload] master         = $MASTER_JSON"
echo "[upload] prompts        = $PROMPTS_DIR"
n_mp4=$(ls -1 "$MOTION_REFS_DIR" | grep -c "\.mp4$" || echo 0)
sz=$(du -sh "$MOTION_REFS_DIR" | awk '{print $1}')
echo "[upload] mp4 count      = $n_mp4  ($sz)"

# 1. Create the dataset repo if missing (idempotent).
echo "[upload] ensuring repo $LL_HF_REPO exists ..."
hf_cli repo create "$LL_HF_REPO" --repo-type dataset --private -y 2>&1 | tail -3 || true

# 2. Stage manifests next to motion_refs in a single folder so the upload
# is structured as motion_refs/*.mp4 + manifests/*.json* at repo root.
mkdir -p "$STAGING_DIR/motion_refs" "$STAGING_DIR/manifests"
echo "[upload] staging manifests under $STAGING_DIR ..."
cp -u "$MASTER_JSON" "$STAGING_DIR/manifests/master_all.json"
cp -u "$PROMPTS_DIR"/motion_pairs_cross_headmotion_train.jsonl "$STAGING_DIR/manifests/" 2>/dev/null || true
cp -u "$PROMPTS_DIR"/motion_pairs_cross_headmotion_val.jsonl   "$STAGING_DIR/manifests/" 2>/dev/null || true
cp -u "$PROMPTS_DIR"/clip_to_part_cross_headmotion.json        "$STAGING_DIR/manifests/" 2>/dev/null || true
# Hard-link the mp4s into staging so we don't blow up disk; rsync also OK.
echo "[upload] hard-linking mp4s into staging (no extra disk used) ..."
ln -f "$MOTION_REFS_DIR"/*.mp4 "$STAGING_DIR/motion_refs/" 2>/dev/null || true

# 3. Upload via upload-large-folder (chunked, parallel, resumable).
echo "[upload] running hf upload-large-folder (resumable; safe to re-run) ..."
hf_cli upload-large-folder "$LL_HF_REPO" "$STAGING_DIR" \
    --repo-type=dataset \
    --num-workers=8

echo "[upload] DONE. Browse: https://huggingface.co/datasets/$LL_HF_REPO"
echo "[upload] HPC fetch:    LL_HF_DATASET=$LL_HF_REPO bash scripts/hpc/fetch_data.sh"
