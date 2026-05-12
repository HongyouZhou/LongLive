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

# ffmpeg / ffprobe — used by prepare_motion_eval.py to remux UCF .avi
# clips to .mp4. HPC nodes typically have no system ffmpeg, so install
# it into the env (conda-forge build, ~50 MB). Idempotent.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[motion_eval/setup] installing ffmpeg from conda-forge"
    mamba install -y -n "$LL_ENV_NAME" -c conda-forge ffmpeg
else
    echo "[motion_eval/setup] ffmpeg already on PATH ($(command -v ffmpeg))"
fi

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
# No --upgrade: respect existing versions. HPC longlive ships pillow 11.3.0
# which is fine; --upgrade would push it to 12.x and surface a spurious
# moviepy<12.0 metadata warning. moviepy 2.1.2 is runtime-compatible with
# pillow 12 anyway but the cleaner state is "don't churn what works".
pip install "${PIP_PKGS[@]}"

# CoTracker3 — install from upstream repo. Skip if already importable.
if python -c "import cotracker" 2>/dev/null; then
    echo "[motion_eval/setup] cotracker already installed"
else
    echo "[motion_eval/setup] installing cotracker (facebookresearch/co-tracker)"
    pip install "git+https://github.com/facebookresearch/co-tracker.git@main"
fi

# Smoke import + ffmpeg check
echo "[motion_eval/setup] verifying imports + ffmpeg..."
python - <<'PY'
import importlib, sys, shutil
problems = []
for name in ("PIL", "decord", "cotracker", "transformers", "torch", "yaml", "gdown"):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"  OK  {name:14s} {ver}")
    except Exception as e:
        problems.append(name)
        print(f"  MISSING  {name}: {e}")
ff = shutil.which("ffmpeg")
if ff:
    print(f"  OK  ffmpeg         {ff}")
else:
    problems.append("ffmpeg")
    print("  MISSING  ffmpeg (not on PATH)")
if problems:
    sys.exit(f"missing: {problems}")
PY

echo "[motion_eval/setup] done. You can now run:"
echo "    python scripts/prepare_motion_eval.py --datasets ucf,loveu"
echo "    bash   scripts/motion_eval/run_motion_eval.sh <ckpt> configs/motion_eval_inference.yaml <run_id>"
