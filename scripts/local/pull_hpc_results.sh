#!/usr/bin/env bash
# Pull a teacher_boundary (or other) result directory from HPC sc-projects
# down to arp's local ~/dev/data/wm/.
#
# Uses the same VPN-namespace + sshpass pattern documented in CLAUDE.md
# under "Three-machine layout":
#   sudo ip netns exec vpnns sshpass -p "$(cat ~/ovpn/.vpn_fixed_pass)" \
#       rsync ... hozh10@s-sc-frontend1.charite.de:<remote> <local>
#
# Usage:
#   bash scripts/local/pull_hpc_results.sh <run_name>
#   bash scripts/local/pull_hpc_results.sh teacher_boundary_260510_1430_12345
#
# Or pull a specific subpath under $LL_DATA:
#   LL_PULL_SUBPATH=logs/longlive_train_long_hpc_<id> \
#       bash scripts/local/pull_hpc_results.sh
#
# Env overrides:
#   LL_PULL_SUBPATH         path under $LL_DATA on HPC (default: teacher_boundary/<run>)
#   LL_PULL_LOCAL_ROOT      local destination root  (default: ~/dev/data/wm)
#   LL_PULL_REMOTE_USER     HPC user                (default: hozh10)
#   LL_PULL_REMOTE_HOST     HPC front node          (default: s-sc-frontend1.charite.de)
#   LL_PULL_REMOTE_DATA     HPC data root           (default: /sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm)
#   LL_VPN_NS               network namespace name  (default: vpnns)
#   LL_VPN_PASS_FILE        path to VPN password    (default: ~/ovpn/.vpn_fixed_pass)

set -euo pipefail

: "${LL_PULL_REMOTE_USER:=hozh10}"
: "${LL_PULL_REMOTE_HOST:=s-sc-frontend1.charite.de}"
: "${LL_PULL_REMOTE_DATA:=/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm}"
: "${LL_PULL_LOCAL_ROOT:=$HOME/dev/data/wm}"
: "${LL_VPN_NS:=vpnns}"
: "${LL_VPN_PASS_FILE:=$HOME/ovpn/.vpn_fixed_pass}"

if [ -n "${LL_PULL_SUBPATH:-}" ]; then
    SUBPATH="$LL_PULL_SUBPATH"
elif [ "$#" -ge 1 ]; then
    SUBPATH="teacher_boundary/$1"
else
    echo "[pull][error] usage: $0 <run_name>   (or set LL_PULL_SUBPATH=...)" >&2
    exit 1
fi

REMOTE_PATH="$LL_PULL_REMOTE_DATA/$SUBPATH"
LOCAL_PATH="$LL_PULL_LOCAL_ROOT/$SUBPATH"

if [ ! -r "$LL_VPN_PASS_FILE" ]; then
    echo "[pull][error] VPN password file not readable: $LL_VPN_PASS_FILE" >&2
    echo "  set LL_VPN_PASS_FILE or run from a shell with VPN already up + ssh key auth" >&2
    exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
    echo "[pull][error] sshpass not installed (apt: sudo apt install sshpass)" >&2
    exit 1
fi

if ! ip netns list 2>/dev/null | grep -q "^$LL_VPN_NS"; then
    echo "[pull][error] network namespace '$LL_VPN_NS' not found." >&2
    echo "  bring up the VPN namespace first (see ~/ovpn / charitefront alias)" >&2
    exit 1
fi

mkdir -p "$LOCAL_PATH"

echo "[pull] remote: $LL_PULL_REMOTE_USER@$LL_PULL_REMOTE_HOST:$REMOTE_PATH"
echo "[pull] local:  $LOCAL_PATH"
echo "[pull] vpn ns: $LL_VPN_NS"

sudo ip netns exec "$LL_VPN_NS" \
    sshpass -p "$(cat "$LL_VPN_PASS_FILE")" \
    rsync -avzP --inplace \
        -e 'ssh -o StrictHostKeyChecking=no' \
        "${LL_PULL_REMOTE_USER}@${LL_PULL_REMOTE_HOST}:${REMOTE_PATH}/" \
        "${LOCAL_PATH}/"

echo "[pull] done -> $LOCAL_PATH"
