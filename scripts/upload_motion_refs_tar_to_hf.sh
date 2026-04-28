#!/bin/bash
# Tar-shard the 118k motion_refs/*.mp4 and push to a private HF dataset.
#
# Why tar shards: HF git repos enforce a hard 10000-files-per-directory
# limit; the previous "upload each mp4 as its own LFS file" hit it at
# 9993/118562 and rejected commits server-side. Bundling clips into
# ~12 tar shards (each 10k mp4) sidesteps the limit and is dramatically
# faster on the HF backend (12 LFS commits vs 118k).
#
# Output layout on the dataset repo:
#   shards/000.tar ... shards/011.tar    (each ~30 GB, 9881 mp4 inside,
#                                         flat: clip_name.mp4 at tar root)
#   manifests/master_all.json
#   manifests/motion_pairs_cross_headmotion_train.jsonl
#   manifests/motion_pairs_cross_headmotion_val.jsonl
#   manifests/clip_to_part_cross_headmotion.json
#   manifests/shard_index.json   (clip_name -> shard_idx, for HPC fetcher)
#
# Run on lab.
#
# Required env:
#   LL_HF_REPO    e.g. HongyouZhou/longlive-headmotion (private dataset).
#   HF_TOKEN      auth; falls back to ~/.cache/huggingface/token.
#
# Time: ~30 min tar build (local nvme, no compression) + upload time
# (12 × 30 GB on lab uplink ≈ 1-3 h depending on bandwidth).

set -euo pipefail

: "${LL_HF_REPO:?LL_HF_REPO not set, e.g. HongyouZhou/longlive-headmotion}"
TOKEN_SRC=""
if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN_SRC="env"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    TOKEN_SRC="~/.cache/huggingface/token"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[upload-tar][error] no HF_TOKEN found." >&2
    exit 1
fi
echo "[upload-tar] token source = $TOKEN_SRC  (len=${#HF_TOKEN})"

: "${MOTION_REFS_DIR:=/home/hongyou/dev/data/wm/motion_refs}"
: "${MASTER_JSON:=/home/hongyou/longlive_work/logs/master_all.json}"
: "${PROMPTS_DIR:=/home/hongyou/dev/data/wm/prompts}"
: "${STAGING_DIR:=/home/hongyou/longlive_work/hf_upload_staging_tar}"
: "${SHARDS_PER_REPO:=12}"      # 118562 / 12 = 9881 per shard, < 10k limit

# Resolve Python that has huggingface_hub.
: "${LL_PY:=}"
if [ -z "$LL_PY" ]; then
    if [ -x "/home/hongyou/longlive_envs/longlive-blackwell/bin/python" ]; then
        LL_PY="/home/hongyou/longlive_envs/longlive-blackwell/bin/python"
    elif [ -x "/home/hongyou/mnt/arp/miniforge3/envs/longlive-blackwell/bin/python" ]; then
        LL_PY="/home/hongyou/mnt/arp/miniforge3/envs/longlive-blackwell/bin/python"
    else
        echo "[upload-tar][error] cannot locate Python; set LL_PY=/path/to/python" >&2
        exit 1
    fi
fi
hf_cli() { "$LL_PY" -m huggingface_hub.commands.huggingface_cli "$@"; }
echo "[upload-tar] LL_PY        = $LL_PY"
echo "[upload-tar] LL_HF_REPO   = $LL_HF_REPO"
echo "[upload-tar] motion_refs  = $MOTION_REFS_DIR"
echo "[upload-tar] staging      = $STAGING_DIR"

# --- 1. Inventory mp4s, sorted (deterministic shard composition).
mkdir -p "$STAGING_DIR/shards" "$STAGING_DIR/manifests"
INVENTORY="$STAGING_DIR/.inventory.txt"
echo "[upload-tar] inventorying $MOTION_REFS_DIR ..."
find "$MOTION_REFS_DIR" -maxdepth 1 -name "*.mp4" -printf "%f\n" | LC_ALL=C sort > "$INVENTORY"
n_total=$(wc -l < "$INVENTORY")
echo "[upload-tar]   $n_total mp4 inventoried"
if [ "$n_total" -lt 1 ]; then
    echo "[upload-tar][error] no mp4 found" >&2
    exit 1
fi

# --- 2. Split into SHARDS_PER_REPO equal-sized chunks. Use awk for stable
# round-robin assignment so each shard is also size-balanced (since file
# names alone don't predict size).
# Strategy: assign clip i to shard (i % SHARDS_PER_REPO). With sorted
# input + uniform shuffling-by-modulo this gives ~9881 mp4 per shard.
LISTS_DIR="$STAGING_DIR/.shard_lists"
mkdir -p "$LISTS_DIR"
rm -f "$LISTS_DIR"/*.txt
echo "[upload-tar] splitting into $SHARDS_PER_REPO shards ..."
awk -v n="$SHARDS_PER_REPO" -v dir="$LISTS_DIR" '{
    f = sprintf("%s/%03d.txt", dir, NR % n)
    print >> f
}' "$INVENTORY"
for i in $(seq 0 $((SHARDS_PER_REPO - 1))); do
    f=$(printf "%s/%03d.txt" "$LISTS_DIR" "$i")
    cnt=$(wc -l < "$f")
    echo "[upload-tar]   shard $(printf %03d $i): $cnt files"
done

# --- 3. Emit manifests/shard_index.json (clip_name -> shard_idx).
echo "[upload-tar] writing shard_index.json ..."
"$LL_PY" - <<PY
import json, os
SH = "$LISTS_DIR"
N = $SHARDS_PER_REPO
idx = {}
for i in range(N):
    p = os.path.join(SH, f"{i:03d}.txt")
    with open(p) as f:
        for line in f:
            name = line.strip()
            if name:
                idx[name] = i
out = "$STAGING_DIR/manifests/shard_index.json"
with open(out, "w") as f:
    json.dump(idx, f, separators=(",", ":"))
print(f"[upload-tar]   {len(idx)} entries -> {out}")
PY

# --- 4. Build tar shards. Use --files-from + -C for ARG_MAX safety; no
# compression (mp4 is already H.264, gzip won't help and costs CPU).
echo "[upload-tar] building tar shards (no compression) ..."
for i in $(seq 0 $((SHARDS_PER_REPO - 1))); do
    list=$(printf "%s/%03d.txt" "$LISTS_DIR" "$i")
    out=$(printf "%s/shards/%03d.tar" "$STAGING_DIR" "$i")
    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt $((1024 * 1024 * 1024)) ]; then
        echo "[upload-tar]   shard $(printf %03d $i) exists ($(du -h "$out" | awk '{print $1}')), skip"
        continue
    fi
    t0=$(date +%s)
    # --verbatim-files-from: don't treat lines starting with "-" as
    # options. OpenVid clip names frequently start with "-" (the
    # YouTube-id alphabet includes "-"), so this is mandatory.
    tar -cf "$out" -C "$MOTION_REFS_DIR" --verbatim-files-from --files-from="$list"
    sz=$(du -h "$out" | awk '{print $1}')
    dt=$(( $(date +%s) - t0 ))
    echo "[upload-tar]   shard $(printf %03d $i) -> $sz in ${dt}s"
done

# --- 5. Stage manifests.
echo "[upload-tar] staging manifests ..."
cp -u "$MASTER_JSON" "$STAGING_DIR/manifests/master_all.json" 2>/dev/null || true
cp -u "$PROMPTS_DIR"/motion_pairs_cross_headmotion_train.jsonl "$STAGING_DIR/manifests/" 2>/dev/null || true
cp -u "$PROMPTS_DIR"/motion_pairs_cross_headmotion_val.jsonl   "$STAGING_DIR/manifests/" 2>/dev/null || true
cp -u "$PROMPTS_DIR"/clip_to_part_cross_headmotion.json        "$STAGING_DIR/manifests/" 2>/dev/null || true
ls -lh "$STAGING_DIR/manifests/"

# --- 6. Create repo (idempotent) + upload.
echo "[upload-tar] ensuring repo $LL_HF_REPO exists ..."
hf_cli repo create "$LL_HF_REPO" --repo-type dataset --private --token "$HF_TOKEN" -y 2>&1 | tail -3 || true

echo "[upload-tar] uploading via upload-large-folder (watchdog wrapper) ..."

# Watchdog: huggingface_hub's requests-based uploader has no socket-level
# timeout, so AWS LB silently dropping idle TCP connections leaves workers
# stuck in poll() forever (CLOSE-WAIT to 18.64.x.x, tx=0). We wrap the
# upload in a loop that monitors tx and force-restarts on stall. The
# upload's own metadata cache makes restart cheap (already-committed
# files are skipped, and S3 multipart sessions resume mid-stream).
: "${WATCHDOG_STALL_SECS:=180}"   # tx==0 for this long -> kill
: "${WATCHDOG_MAX_ATTEMPTS:=30}"
: "${WATCHDOG_IFACE:=}"           # auto-detect if empty
: "${EXPECTED_TAR_COUNT:=$SHARDS_PER_REPO}"

# Auto-detect the active uplink interface (one with the most tx bytes).
if [ -z "$WATCHDOG_IFACE" ]; then
    WATCHDOG_IFACE=$(awk 'NR>2 {gsub(":","",$1); print $10, $1}' /proc/net/dev \
                     | sort -rn | awk '$2 != "lo" {print $2; exit}')
fi
echo "[watchdog] iface = $WATCHDOG_IFACE  stall_secs = $WATCHDOG_STALL_SECS  max = $WATCHDOG_MAX_ATTEMPTS"

count_committed_tars() {
    "$LL_PY" - <<PY 2>/dev/null
from huggingface_hub import HfApi
import os
token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
api = HfApi(token=token)
try:
    files = api.list_repo_files("$LL_HF_REPO", repo_type="dataset")
    print(sum(1 for f in files if f.endswith(".tar")))
except Exception:
    print(0)
PY
}

read_tx() {
    # /proc/net/dev columns: iface: rx_bytes ... (col 2), ..., tx_bytes (col 10)
    awk -v ifc="$WATCHDOG_IFACE" '$1 == ifc":" {print $10}' /proc/net/dev
}

attempt=0
while [ $attempt -lt $WATCHDOG_MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    n_done=$(count_committed_tars)
    echo "[watchdog] attempt $attempt/$WATCHDOG_MAX_ATTEMPTS  committed_tars=$n_done/$EXPECTED_TAR_COUNT"
    if [ "$n_done" -ge "$EXPECTED_TAR_COUNT" ]; then
        echo "[watchdog] all $EXPECTED_TAR_COUNT tar shards committed -- DONE"
        break
    fi

    hf_cli upload-large-folder "$LL_HF_REPO" "$STAGING_DIR" \
        --repo-type=dataset \
        --num-workers=8 \
        --token "$HF_TOKEN" &
    upid=$!
    echo "[watchdog]   spawned upload pid=$upid"

    last_tx=$(read_tx); [ -z "$last_tx" ] && last_tx=0
    stall=0
    sleep 60   # warmup before tx checks (hash phase has no tx)
    while kill -0 "$upid" 2>/dev/null; do
        cur_tx=$(read_tx); [ -z "$cur_tx" ] && cur_tx=0
        if [ "$cur_tx" = "$last_tx" ]; then
            stall=$((stall + 30))
        else
            stall=0
        fi
        if [ "$stall" -ge "$WATCHDOG_STALL_SECS" ]; then
            cw=$(ss -tn state close-wait 2>/dev/null | wc -l)
            echo "[watchdog]   STALL detected (tx=0 for ${stall}s, close_wait=$cw); SIGTERM pid $upid"
            kill -TERM "$upid" 2>/dev/null || true
            sleep 5
            if kill -0 "$upid" 2>/dev/null; then
                echo "[watchdog]   still alive, SIGKILL"
                kill -KILL "$upid" 2>/dev/null || true
            fi
            break
        fi
        last_tx="$cur_tx"
        sleep 30
    done
    wait "$upid" 2>/dev/null
    rc=$?
    echo "[watchdog]   upload exited rc=$rc"
    if [ "$rc" -eq 0 ]; then
        # Re-check: rc=0 from hf_cli usually means done, but verify.
        n_done=$(count_committed_tars)
        if [ "$n_done" -ge "$EXPECTED_TAR_COUNT" ]; then
            echo "[watchdog] confirmed done (rc=0, $n_done/$EXPECTED_TAR_COUNT)"
            break
        fi
    fi
    echo "[watchdog]   cooldown 15s before retry"
    sleep 15
done

if [ "$attempt" -ge "$WATCHDOG_MAX_ATTEMPTS" ]; then
    echo "[watchdog][error] exhausted $WATCHDOG_MAX_ATTEMPTS attempts" >&2
    exit 1
fi

echo "[upload-tar] DONE. Browse: https://huggingface.co/datasets/$LL_HF_REPO"
echo "[upload-tar] HPC fetch:    LL_HF_DATASET=$LL_HF_REPO bash scripts/hpc/fetch_data.sh"
