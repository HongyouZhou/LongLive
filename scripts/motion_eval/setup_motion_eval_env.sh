#!/bin/bash
# Add motion-eval dependencies to the existing `longlive` mamba env.
#
# Why an additive script instead of a separate env: the eval driver
# (run_eval.py) loads only CLIP + PickScore + CoTracker3 — none of which
# pin to a torch / cu version that conflicts with the longlive env
# (unlike VBench's detectron2, which forces torch 2.4). Keeping a single
# env avoids the env-switch dance during the 3-phase pipeline.
#
# Run on a login node (HPC) or interactively (lab). Idempotent.

set -euo pipefail

: "${LL_ENV_NAME:=longlive}"

echo "[motion_eval/setup] activating $LL_ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$LL_ENV_NAME"

# Core eval deps the longlive env doesn't ship.
# - decord: fast video decoding (faster than torchvision for batch reads)
# - Pillow: required by transformers' CLIPProcessor
# - gdown: needed by prepare_motion_eval.py on lab for the LOVEU-TGVE GDrive
#   download. Not used on HPC (rsync-from-lab path), installed here anyway
#   so lab/HPC envs stay symmetric.
# - cotracker: Yatim's motion-fidelity metric. Repo install (no PyPI package).
PIP_PKGS=(
    "Pillow"
    "decord"
    "gdown"
)

echo "[motion_eval/setup] pip install: ${PIP_PKGS[*]}"
pip install --upgrade "${PIP_PKGS[@]}"

# CoTracker3 — install from upstream repo. Skip if already importable.
if python -c "import cotracker" 2>/dev/null; then
    echo "[motion_eval/setup] cotracker already installed"
else
    echo "[motion_eval/setup] installing cotracker (facebookresearch/co-tracker)"
    pip install "git+https://github.com/facebookresearch/co-tracker.git@main"
fi

# Smoke import check
echo "[motion_eval/setup] verifying imports..."
python - <<'PY'
import importlib, sys
problems = []
for name in ("PIL", "decord", "cotracker", "transformers", "torch", "yaml", "gdown"):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"  OK  {name:14s} {ver}")
    except Exception as e:
        problems.append(name)
        print(f"  MISSING  {name}: {e}")
if problems:
    sys.exit(f"missing: {problems}")
PY

echo "[motion_eval/setup] done. You can now run:"
echo "    python scripts/prepare_motion_eval.py --datasets ucf,loveu"
echo "    bash   scripts/motion_eval/run_motion_eval.sh <ckpt> configs/motion_eval_inference.yaml <run_id>"
