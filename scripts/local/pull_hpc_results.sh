#!/usr/bin/env bash
# Pull a teacher_boundary (or other) result directory from HPC sc-projects
# down to the LOCAL machine that has the VPN namespace + VPN session set up.
#
# IMPORTANT: This must run on the machine that has the `vpnns` network
# namespace and an active VPN session (typically the user's laptop).
# arp does NOT have VPN access to Charite HPC — running this on arp will fail.
#
# Required env:
#   HPC_PASS                VPN / HPC SSH password
#
# Usage:
#   export HPC_PASS=...
#   bash scripts/local/pull_hpc_results.sh <run_name> [<local_dest_dir>]
#
# Examples:
#   # Default destination (~/Downloads/longlive_results/<run_name>/):
#   bash scripts/local/pull_hpc_results.sh teacher_boundary_260510_1430_12345
#
#   # Custom destination:
#   bash scripts/local/pull_hpc_results.sh teacher_boundary_260510_1430_12345 ~/data/
#
#   # Pull arbitrary subpath under HPC's $PROJECT_DATA/wm:
#   LL_PULL_SUBPATH=logs/longlive_train_long_hpc_<id> \
#       bash scripts/local/pull_hpc_results.sh
#
# Env overrides:
#   LL_PULL_SUBPATH         path under HPC's $PROJECT_DATA/wm
#                           (default: teacher_boundary/<run_name>)
#   LL_PULL_REMOTE_USER     HPC user                (default: hozh10)
#   LL_PULL_REMOTE_HOST     HPC front node          (default: s-sc-frontend1.charite.de)
#   LL_PULL_REMOTE_DATA     HPC data root           (default: /sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm)
#   LL_VPN_NS               network namespace name  (default: vpnns)

set -euo pipefail

: "${LL_PULL_REMOTE_USER:=hozh10}"
: "${LL_PULL_REMOTE_HOST:=s-sc-frontend1.charite.de}"
: "${LL_PULL_REMOTE_DATA:=/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm}"
: "${LL_VPN_NS:=vpnns}"

if [ -n "${LL_PULL_SUBPATH:-}" ]; then
    SUBPATH="$LL_PULL_SUBPATH"
    DEFAULT_DEST="$HOME/Downloads/longlive_results/$(basename "$SUBPATH")/"
elif [ "$#" -ge 1 ]; then
    SUBPATH="teacher_boundary/$1"
    DEFAULT_DEST="$HOME/Downloads/longlive_results/$1/"
else
    echo "[pull][error] usage: $0 <run_name> [<local_dest_dir>]" >&2
    echo "  or: LL_PULL_SUBPATH=<path-under-HPC-PROJECT_DATA/wm> $0 [<local_dest_dir>]" >&2
    exit 1
fi

LOCAL_DEST="${2:-$DEFAULT_DEST}"
[[ "$LOCAL_DEST" != */ ]] && LOCAL_DEST="$LOCAL_DEST/"
mkdir -p "$LOCAL_DEST"

REMOTE_PATH="$LL_PULL_REMOTE_DATA/$SUBPATH"

if [ -z "${HPC_PASS:-}" ]; then
    echo "[pull][error] HPC_PASS env var not set" >&2
    echo "  export HPC_PASS=<your-hpc-password>" >&2
    exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
    echo "[pull][error] sshpass not installed (sudo apt install sshpass)" >&2
    exit 1
fi

if ! ip netns list 2>/dev/null | grep -q "^$LL_VPN_NS"; then
    echo "[pull][error] network namespace '$LL_VPN_NS' not found." >&2
    echo "  this script must run on the LOCAL machine that has the VPN namespace" >&2
    echo "  active (i.e. the laptop that runs charitefront), NOT on arp" >&2
    exit 1
fi

echo "[pull] remote: $LL_PULL_REMOTE_USER@$LL_PULL_REMOTE_HOST:$REMOTE_PATH/"
echo "[pull] local:  $LOCAL_DEST"
echo "[pull] vpn ns: $LL_VPN_NS"

sudo ip netns exec "$LL_VPN_NS" \
    sshpass -p "$HPC_PASS" \
    rsync -avzP \
        --no-perms --no-owner --no-group --chmod=ugo=rwX \
        -e 'ssh -o StrictHostKeyChecking=no' \
        "${LL_PULL_REMOTE_USER}@${LL_PULL_REMOTE_HOST}:${REMOTE_PATH}/" \
        "${LOCAL_DEST}"

echo "[pull] done -> $LOCAL_DEST"
