#!/bin/bash
# Build or update the lab Blackwell environment used for optional NVFP4 work.
#
# This is intentionally separate from scripts/hpc/setup_mamba_env.sh. HPC H200
# jobs should keep using the BF16/Hopper environment and do not need
# FourOverSix, TransformerEngine, or local FP4 CUDA extensions.
#
# Usage:
#   bash scripts/lab/setup_blackwell_env.sh
#
# Useful overrides:
#   LL_ENV_PREFIX=/home/hongyou/longlive_envs/longlive-blackwell bash scripts/lab/setup_blackwell_env.sh
#   LL_FOUROVERSIX_SRC=/path/to/fouroversix bash scripts/lab/setup_blackwell_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

: "${MAMBA_BIN:=/home/hongyou/miniforge3/bin/mamba}"
: "${LL_ENV_PREFIX:=/home/hongyou/longlive_envs/longlive-blackwell}"
: "${REQ_FILE:=$REPO_ROOT/requirements.txt}"
: "${CUDA_ARCHS:=120}"

if [ ! -x "$MAMBA_BIN" ]; then
  echo "[blackwell-env][error] mamba not found at $MAMBA_BIN" >&2
  echo "Set MAMBA_BIN=/path/to/mamba and rerun." >&2
  exit 1
fi

echo "[blackwell-env] env prefix = $LL_ENV_PREFIX"
echo "[blackwell-env] req file   = $REQ_FILE"
echo "[blackwell-env] CUDA_ARCHS = $CUDA_ARCHS"

if [ ! -d "$LL_ENV_PREFIX" ]; then
  "$MAMBA_BIN" create -y -p "$LL_ENV_PREFIX" python=3.10
fi

# shellcheck disable=SC1091
source "$(dirname "$MAMBA_BIN")/../etc/profile.d/conda.sh"
conda activate "$LL_ENV_PREFIX"

python -m pip install --upgrade pip

if ! python -c "import torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null; then
  python -m pip install \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
fi

# Keep build tooling aligned with PyTorch cu128 even if the host also has CUDA 13.
"$MAMBA_BIN" install -y -p "$LL_ENV_PREFIX" -c nvidia cuda-toolkit=12.8

tmp_req="$(mktemp)"
grep -vE '^\s*(nvidia-(pyindex|tensorrt)|pycuda|onnx[a-z]*|flask)\b' "$REQ_FILE" > "$tmp_req"
python -m pip install -r "$tmp_req"
rm -f "$tmp_req"

if ! python -c "import flash_attn; assert flash_attn.__version__.startswith('2.8.3')" 2>/dev/null; then
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.8.3 --no-build-isolation
fi

python -m pip install ninja packaging psutil "setuptools>=77.0.3"

export CUDA_ARCHS
if [ -n "${LL_FOUROVERSIX_SRC:-}" ]; then
  python -m pip install --no-build-isolation -e "$LL_FOUROVERSIX_SRC"
else
  python -m pip install fouroversix --no-build-isolation
fi

python -m pip install --no-build-isolation "transformer-engine[pytorch]"

if [ -f "$REPO_ROOT/utils/kernel/setup.py" ]; then
  (
    cd "$REPO_ROOT/utils/kernel"
    python setup.py build_ext --inplace
  )
else
  echo "[blackwell-env] no utils/kernel/setup.py in this branch; skipping FP4 KV extension build."
fi

python "$REPO_ROOT/scripts/local/probe_backends.py" --backend auto

echo "[blackwell-env] done"
