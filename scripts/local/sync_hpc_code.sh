#!/usr/bin/env bash
# Sync this local LongLive worktree to the Charite HPC project directory.
#
# This is a CODE sync only. It intentionally excludes data, checkpoints,
# model weights, logs, wandb runs, and media. Stage data with
# scripts/hpc/fetch_data.sh and pull results with scripts/local/pull_hpc_results.sh.
#
# Default mode is dry-run. Use --apply to actually update the remote tree.
#
# Required on the local machine:
#   - active VPN namespace (default: vpnns)
#   - sshpass
#   - HPC_PASS exported
#
# Usage:
#   export HPC_PASS=...
#   bash scripts/local/sync_hpc_code.sh
#   bash scripts/local/sync_hpc_code.sh --apply
#
# Overrides:
#   LL_SYNC_REMOTE_USER    default: hozh10
#   LL_SYNC_REMOTE_HOST    default: s-sc-frontend1.charite.de
#   LL_SYNC_REMOTE_REPO    default: /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive
#   LL_VPN_NS              default: vpnns
#   LL_SYNC_INCLUDE_GIT=1  also sync .git/ (default excludes it)

set -euo pipefail

APPLY=0
DELETE=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --apply)
            APPLY=1
            ;;
        --no-delete)
            DELETE=0
            ;;
        -h|--help)
            sed -n '1,38p' "$0"
            exit 0
            ;;
        *)
            echo "[sync][error] unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

: "${LL_SYNC_REMOTE_USER:=hozh10}"
: "${LL_SYNC_REMOTE_HOST:=s-sc-frontend1.charite.de}"
: "${LL_SYNC_REMOTE_REPO:=/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive}"
: "${LL_VPN_NS:=vpnns}"
: "${LL_SYNC_INCLUDE_GIT:=0}"

if [[ "$LL_SYNC_REMOTE_REPO" =~ [[:space:]] ]]; then
    echo "[sync][error] LL_SYNC_REMOTE_REPO must not contain whitespace" >&2
    exit 2
fi

if [ -z "${HPC_PASS:-}" ]; then
    echo "[sync][error] HPC_PASS env var not set" >&2
    echo "  export HPC_PASS=<your-hpc-password>" >&2
    exit 1
fi

for bin in rsync sshpass sudo ip; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "[sync][error] required command not found: $bin" >&2
        exit 1
    fi
done

if ! ip netns list 2>/dev/null | grep -q "^$LL_VPN_NS"; then
    echo "[sync][error] network namespace '$LL_VPN_NS' not found" >&2
    echo "  run this on the local machine that has the Charite VPN namespace" >&2
    exit 1
fi

SOURCE_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$SOURCE_ROOT"

EXCLUDE_FILE="$SOURCE_ROOT/.hpc-sync-exclude"
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "[sync][error] missing exclude file: $EXCLUDE_FILE" >&2
    exit 1
fi

REMOTE="${LL_SYNC_REMOTE_USER}@${LL_SYNC_REMOTE_HOST}"
REMOTE_DEST="${REMOTE}:${LL_SYNC_REMOTE_REPO}/"
SSH_OPTS=(-o StrictHostKeyChecking=no)
NETNS=(sudo ip netns exec "$LL_VPN_NS")
SSHPASS=(sshpass -p "$HPC_PASS")

RSYNC_ARGS=(
    -azP
    --itemize-changes
    --filter=":- .gitignore"
    --exclude-from="$EXCLUDE_FILE"
    --no-perms
    --no-owner
    --no-group
    --chmod=ugo=rwX
    -e "ssh ${SSH_OPTS[*]}"
)

if [ "$DELETE" -eq 1 ]; then
    RSYNC_ARGS+=(--delete --delete-delay)
fi
if [ "$APPLY" -eq 0 ]; then
    RSYNC_ARGS+=(--dry-run)
fi
if [ "$LL_SYNC_INCLUDE_GIT" != "1" ]; then
    RSYNC_ARGS+=(--exclude=".git/")
fi

echo "[sync] source: $SOURCE_ROOT/"
echo "[sync] remote: $REMOTE_DEST"
echo "[sync] mode:   $([ "$APPLY" -eq 1 ] && echo apply || echo dry-run)"
echo "[sync] delete: $([ "$DELETE" -eq 1 ] && echo enabled || echo disabled)"

if [ "$APPLY" -eq 1 ]; then
    "${NETNS[@]}" "${SSHPASS[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE" \
        "mkdir -p '$LL_SYNC_REMOTE_REPO/.sync'"
fi

"${NETNS[@]}" "${SSHPASS[@]}" rsync "${RSYNC_ARGS[@]}" "$SOURCE_ROOT/" "$REMOTE_DEST"

if [ "$APPLY" -eq 0 ]; then
    echo "[sync] dry-run only. Re-run with --apply to update HPC."
    exit 0
fi

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT
{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_host=$(hostname)"
    echo "source_root=$SOURCE_ROOT"
    echo "remote_repo=$LL_SYNC_REMOTE_REPO"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo detached)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_dirty=$([ -n "$(git status --short)" ] && echo yes || echo no)"
        echo
        echo "[git_status_short]"
        git status --short
    else
        echo "git_branch=none"
        echo "git_commit=none"
        echo "git_dirty=unknown"
    fi
} > "$MANIFEST"

"${NETNS[@]}" "${SSHPASS[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cat > '$LL_SYNC_REMOTE_REPO/.sync/last_code_sync.txt'" < "$MANIFEST"

echo "[sync] wrote remote manifest: $LL_SYNC_REMOTE_REPO/.sync/last_code_sync.txt"
echo "[sync] done"
