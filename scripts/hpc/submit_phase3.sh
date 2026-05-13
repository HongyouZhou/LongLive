# Submit Phase 3 DMD training + auto Phase 1 motion eval (afterok dependency).
#
# Submits scripts/hpc/sbatch_train.sh with the Phase 3 config, captures the
# resulting JID, then submits scripts/hpc/sbatch_motion_eval_phase3.sh with
# `--dependency=afterok:<train_jid>`. The eval job waits in queue until
# training succeeds, then auto-discovers the latest ckpt and evaluates it
# against Phase 1's UCF + LOVEU prompt set.
#
# MUST be sourced (so $TRAIN_JID and $EVAL_JID stay in your shell):
#
#   source scripts/hpc/submit_phase3.sh
#   tail -f logs/motion_dmd_skateboarding_v1_*-$TRAIN_JID.out
#
# Override config:
#
#   LL_CONFIG=configs/motion_dmd_skateboarding_v2.yaml \
#       source scripts/hpc/submit_phase3.sh
#
# To attach eval to an ALREADY-RUNNING train (skip train submission):
#
#   LL_PHASE3_EXISTING_TRAIN_JID=<jid> source scripts/hpc/submit_phase3.sh
#
# To skip the eval step entirely:
#
#   LL_PHASE3_SKIP_EVAL=1 source scripts/hpc/submit_phase3.sh

# Soft check: warn if not sourced
if [ -n "${BASH_SOURCE-}" ] && [ "${BASH_SOURCE[0]}" = "${0-}" ]; then
    echo "[submit_phase3][warn] not sourced — \$TRAIN_JID / \$EVAL_JID will not persist" >&2
    echo "                    use: source scripts/hpc/submit_phase3.sh" >&2
fi

__sp3_hpc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${LL_CONFIG:=configs/motion_dmd_skateboarding_v1.yaml}"

echo "[submit_phase3] config: $LL_CONFIG"

# Step 1: training (unless caller is attaching to an existing train)
if [ -n "${LL_PHASE3_EXISTING_TRAIN_JID:-}" ]; then
    TRAIN_JID="$LL_PHASE3_EXISTING_TRAIN_JID"
    echo "[submit_phase3] attaching to existing TRAIN_JID=$TRAIN_JID (skip train submit)"
else
    __sp3_train_out="$(LL_CONFIG="$LL_CONFIG" sbatch --parsable "$__sp3_hpc_dir/sbatch_train.sh")" || {
        echo "[submit_phase3][error] train submit failed" >&2
        unset __sp3_hpc_dir __sp3_train_out
        return 1 2>/dev/null || exit 1
    }
    TRAIN_JID="$__sp3_train_out"
    echo "[submit_phase3] TRAIN_JID=$TRAIN_JID"
fi
export TRAIN_JID
# Keep $JID set to TRAIN_JID for compatibility with submit.sh users.
export JID="$TRAIN_JID"

# Step 2: eval, depending on training success
if [ -n "${LL_PHASE3_SKIP_EVAL:-}" ]; then
    echo "[submit_phase3] LL_PHASE3_SKIP_EVAL=1 — eval not submitted"
    unset __sp3_hpc_dir __sp3_train_out
    return 0 2>/dev/null || exit 0
fi

__sp3_eval_out="$(sbatch --parsable --dependency=afterok:$TRAIN_JID "$__sp3_hpc_dir/sbatch_motion_eval_phase3.sh")" || {
    echo "[submit_phase3][error] eval submit failed" >&2
    unset __sp3_hpc_dir __sp3_train_out __sp3_eval_out
    return 1 2>/dev/null || exit 1
}
EVAL_JID="$__sp3_eval_out"
export EVAL_JID
echo "[submit_phase3] EVAL_JID=$EVAL_JID (afterok:$TRAIN_JID)"

echo "[submit_phase3] monitor train: tail -f logs/motion_dmd_skateboarding_v1_*-\$TRAIN_JID.out"
echo "[submit_phase3] monitor eval:  tail -f logs/motiondirector_eval-\$EVAL_JID.out"
echo "[submit_phase3] cancel both:   scancel \$TRAIN_JID \$EVAL_JID"

unset __sp3_hpc_dir __sp3_train_out __sp3_eval_out
