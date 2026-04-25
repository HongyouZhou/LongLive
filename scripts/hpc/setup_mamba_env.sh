#!/bin/bash
# Build the longlive mamba env on HPC. Run on a *login* node (compute nodes
# usually have no outbound network).
#
# Designed against the .bashrc you shared:
#   - mamba shell hook is already loaded
#   - /opt/miniforge is the system base (read-only for us)
#   - $PROJECT_HOME points at the project FS (NOT $HOME) — we put the env there
#     so the 7+ GB env doesn't eat your /home quota
#   - HF_TOKEN / WANDB_API_KEY already exported (we don't touch them)
#
# Idempotent: re-running only redoes the missing pieces.
#
# Usage:
#   bash scripts/hpc/setup_mamba_env.sh
#
# Override the env location:
#   LL_ENV_PREFIX=/some/other/path/envs/longlive bash scripts/hpc/setup_mamba_env.sh

set -euo pipefail

: "${PROJECT_HOME:?PROJECT_HOME not set — source ~/.bashrc first}"
: "${LL_ENV_PREFIX:=$PROJECT_HOME/envs/longlive}"
REQ_FILE="${REQ_FILE:-$(dirname "$0")/../../requirements.txt}"

if ! command -v mamba >/dev/null; then
  echo "[env][error] mamba not on PATH — source ~/.bashrc first." >&2
  exit 1
fi

mkdir -p "$(dirname "$LL_ENV_PREFIX")"

echo "[env] LL_ENV_PREFIX = $LL_ENV_PREFIX"
echo "[env] REQ_FILE      = $REQ_FILE"

# -------- 1. Create env (prefix-style, lives on project FS) --------
if [ ! -f "$LL_ENV_PREFIX/bin/python" ]; then
  echo "[env] creating mamba env at $LL_ENV_PREFIX (python 3.10) ..."
  mamba create -y -p "$LL_ENV_PREFIX" python=3.10
else
  echo "[env] env exists, skipping create."
fi

# Activate via conda (mamba activate is noisy in non-interactive shells).
# shellcheck disable=SC1091
source /opt/miniforge/etc/profile.d/conda.sh
conda activate "$LL_ENV_PREFIX"

# -------- 2. PyTorch (matches arp: 2.8.0 + cu128) --------
if ! python -c "import torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null; then
  echo "[env] installing PyTorch 2.8.0 + cu128 ..."
  pip install --upgrade pip
  pip install \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
else
  echo "[env] torch 2.8.0 already installed."
fi

# -------- 3. Project deps (drop TRT / inference-only stuff) --------
# nvidia-tensorrt / pycuda / onnx* / flask are TRT inference tooling; not needed
# for training, and they often fail to build on HPCs lacking system TRT libs.
echo "[env] installing project deps ..."
TMP_REQ=$(mktemp)
grep -vE '^\s*(nvidia-(pyindex|tensorrt)|pycuda|onnx[a-z]*|flask)\b' "$REQ_FILE" > "$TMP_REQ"
pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"

# -------- 4. flash-attn 2.8.3 --------
# Try the official prebuilt wheel for cu12torch2.8 cxx11abiFALSE first.
# Falls back to source build (~30 min, single-GPU compile).
if ! python -c "import flash_attn; assert flash_attn.__version__.startswith('2.8.3')" 2>/dev/null; then
  echo "[env] installing flash-attn 2.8.3 ..."
  FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  if ! pip install "$FA_WHEEL"; then
    echo "[env] prebuilt wheel failed, building from source (~30 min) ..."
    MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
  fi
else
  echo "[env] flash-attn 2.8.3 already installed."
fi

# -------- 5. Smoke check (CPU-only, no GPU needed) --------
python - <<'EOF'
import torch, importlib
print(f"torch={torch.__version__}  cuda_built={torch.version.cuda}")
for mod in ("flash_attn", "transformers", "diffusers", "peft", "omegaconf",
            "einops", "accelerate", "wandb", "huggingface_hub"):
    importlib.import_module(mod)
    print(f"  OK  {mod}")
print("[env] all imports OK — GPU check happens on a compute node.")
EOF

echo
echo "[env] DONE. To verify GPU on a compute node:"
echo "  srun -p gpu --gpus=1 --pty bash -c \\"
echo "    'source /opt/miniforge/etc/profile.d/conda.sh && conda activate $LL_ENV_PREFIX && python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
