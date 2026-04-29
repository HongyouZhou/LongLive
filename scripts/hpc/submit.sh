# Wrap `sbatch ...` so the returned job id lands in $JID for follow-up.
# MUST be sourced (otherwise $JID stays in the subshell):
#
#   source scripts/hpc/submit.sh sbatch_vbench.sh longlive_models/models/lora.pt smoke
#   srun --jobid=$JID --overlap nvidia-smi
#   tail -f logs/vbench-$JID.out
#   scancel $JID
#
# Pass through env-var overrides exactly as you would to sbatch:
#
#   LL_VBENCH_LIMIT=8 \
#     source scripts/hpc/submit.sh sbatch_vbench.sh longlive_models/models/lora.pt smoke
#
# Resolves the script arg against scripts/hpc/ so callers don't have to
# type the prefix every time.

# Soft check: warn if not sourced (still runs, but $JID won't survive).
if [ -n "${BASH_SOURCE-}" ] && [ "${BASH_SOURCE[0]}" = "${0-}" ]; then
    echo "[submit][warn] not sourced — \$JID will not persist in your shell" >&2
    echo "             use: source scripts/hpc/submit.sh <args>" >&2
fi

if [ "$#" -lt 1 ]; then
    echo "[submit] usage: source scripts/hpc/submit.sh <sbatch_script> [args...]" >&2
    return 1 2>/dev/null || exit 1
fi

# Resolve `sbatch_vbench.sh` etc. relative to scripts/hpc/ if it isn't a path.
__ll_submit_script="$1"
case "$__ll_submit_script" in
    /*|*/*) ;;  # absolute or has slash → leave as-is
    *)
        __ll_hpc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        if [ -f "$__ll_hpc_dir/$__ll_submit_script" ]; then
            __ll_submit_script="$__ll_hpc_dir/$__ll_submit_script"
        fi
        ;;
esac
shift

__ll_submit_out="$(sbatch "$__ll_submit_script" "$@")" || {
    echo "[submit][error] sbatch failed" >&2
    unset __ll_submit_script __ll_submit_out __ll_hpc_dir
    return 1 2>/dev/null || exit 1
}
echo "$__ll_submit_out"
JID="$(awk '{print $NF}' <<<"$__ll_submit_out")"
export JID

# Pull the real --job-name from the script header so the log-path hint
# matches what SLURM actually wrote (#SBATCH --output=logs/%x-%j.out
# expands %x to the job-name, not the script filename).
__ll_jobname="$(grep -m1 -E '^#SBATCH[[:space:]]+--job-name=' "$__ll_submit_script" \
                | sed -E 's/.*--job-name=([^[:space:]]+).*/\1/')"
[ -z "$__ll_jobname" ] && __ll_jobname="$(basename "${__ll_submit_script%.sh}" | sed 's/sbatch_//')"

echo "[submit] JID=$JID exported"
echo "[submit] monitor : srun --jobid=\$JID --overlap nvidia-smi"
echo "[submit] log     : tail -f logs/$__ll_jobname-\$JID.out"
echo "[submit] cancel  : scancel \$JID"

unset __ll_submit_script __ll_submit_out __ll_hpc_dir __ll_jobname
