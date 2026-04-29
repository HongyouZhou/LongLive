#!/bin/bash
# One-shot bootstrap for the `vbench` mamba env (idempotent).
#
# Why a separate env: VBench depends on detectron2, which has prebuilt
# wheels only for torch 2.4 / cu121. Our `longlive` env is on torch 2.8 +
# cu128 — incompatible. Keeping `vbench` separate also means we can
# upgrade either side without breakage.
#
# Usage:
#   bash scripts/vbench/setup_vbench_env.sh

set -euo pipefail

: "${VBENCH_ENV:=vbench}"
: "${VBENCH_PY:=3.10}"
# torch 2.4 / cu121 — what detectron2 prebuilt wheels expect.
: "${TORCH_VERSION:=2.4.0}"
: "${TORCH_CUDA:=cu121}"
# Where VBench's GitHub repo gets cloned (we need VBench_full_info.json).
: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"

echo "[vbench-setup] env name        : $VBENCH_ENV"
echo "[vbench-setup] python           : $VBENCH_PY"
echo "[vbench-setup] torch            : $TORCH_VERSION+$TORCH_CUDA"
echo "[vbench-setup] vbench repo dir  : $VBENCH_REPO_DIR"

# Create env if missing.
if ! mamba env list | awk '{print $1}' | grep -qx "$VBENCH_ENV"; then
    echo "[vbench-setup] creating mamba env $VBENCH_ENV"
    mamba create -y -n "$VBENCH_ENV" "python=$VBENCH_PY" pip
else
    echo "[vbench-setup] env $VBENCH_ENV already exists, reusing"
fi

# Activate inside this shell.
eval "$(mamba shell hook --shell bash)"
mamba activate "$VBENCH_ENV"

echo "[vbench-setup] python: $(which python)"

# Install torch first (detectron2 wheel resolution depends on torch already
# being present).
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}" "torchvision" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# detectron2: build from source against the torch we just installed.
# detectron2's setup.py imports torch at config time, but pip's PEP 517
# build isolation hides our env's torch. Use --no-build-isolation and
# pre-install build deps so the build subprocess inherits torch.
pip install setuptools wheel ninja
pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@v0.6"

# vbench itself.
pip install vbench

# Common transitive deps + version pins for the detectron2 v0.6 era.
# detectron2 v0.6 (released 2021) hardcodes APIs that newer libs removed:
#   - PIL.Image.LINEAR     → renamed to BILINEAR in Pillow 10
#   - numpy<2 (vbench 0.1.5 requires; opencv-python-headless 4.13 drags 2.x)
#   - pkg_resources        → removed from setuptools 70+; OpenAI clip uses it
# All four pins are needed; missing any one of them breaks evaluate() at runtime.
pip install \
    "numpy<2" \
    "Pillow<10" \
    "setuptools<70" \
    "scikit-image<0.25" \
    opencv-python-headless decord

# Clone VBench repo for the prompt-info JSON (VBench_full_info.json) and
# any per-dim asset files. Re-clone idempotent.
mkdir -p "$(dirname "$VBENCH_REPO_DIR")"
if [ ! -d "$VBENCH_REPO_DIR/.git" ]; then
    echo "[vbench-setup] cloning VBench repo to $VBENCH_REPO_DIR"
    git clone https://github.com/Vchitect/VBench.git "$VBENCH_REPO_DIR"
else
    echo "[vbench-setup] VBench repo already present, pulling latest"
    git -C "$VBENCH_REPO_DIR" pull --ff-only || true
fi

INFO_JSON="$VBENCH_REPO_DIR/vbench/VBench_full_info.json"
if [ ! -f "$INFO_JSON" ]; then
    echo "[vbench-setup][error] VBench_full_info.json not found at $INFO_JSON"
    echo "                       (VBench repo layout may have changed; locate"
    echo "                        the prompt info json and update VBENCH_INFO"
    echo "                        paths in run_vbench.sh accordingly)"
    exit 1
fi

# -------- Pre-stage VBench model assets that fail proxy auto-download --------
# VBench's evaluate() auto-downloads several model checkpoints from
# Microsoft Azure blob (datarelease.blob.core.windows.net) on first run.
# Charite's outbound proxy returns 409 on that endpoint. Pre-stage the
# files into VBench's cache so its existence-check skips the wget.
VBENCH_CACHE="$HOME/.cache/vbench"

# GRiT model — needed by `multiple_objects` dim. Mirror on HF: trimble/GRiT.
GRIT_TARGET="$VBENCH_CACHE/grit_model/grit_b_densecap_objectdet.pth"
if [ ! -f "$GRIT_TARGET" ]; then
    echo "[vbench-setup] pre-staging GRiT ckpt from HF (trimble/GRiT, ~417 MB)"
    mkdir -p "$(dirname "$GRIT_TARGET")"
    hf download trimble/GRiT \
        --include "models/grit_b_densecap_objectdet.pth" \
        --local-dir "$VBENCH_CACHE/grit_model_tmp"
    mv "$VBENCH_CACHE/grit_model_tmp/models/grit_b_densecap_objectdet.pth" \
       "$GRIT_TARGET"
    rm -rf "$VBENCH_CACHE/grit_model_tmp"
    echo "[vbench-setup] staged $GRIT_TARGET"
else
    echo "[vbench-setup] GRiT ckpt already staged, skip"
fi

# -------- Sanity: import the things VBench needs at evaluate() time --------
echo "[vbench-setup] sanity-check imports"
python - <<'PY'
import sys
errors = []
def chk(name, test):
    try:
        test()
        print(f"  ok  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        errors.append(name)

chk("torch + cuda build", lambda: __import__("torch").cuda.is_available)
chk("torchvision",        lambda: __import__("torchvision"))
chk("detectron2",         lambda: __import__("detectron2"))
chk("vbench",             lambda: __import__("vbench"))
chk("clip (openai)",      lambda: __import__("clip"))
chk("pkg_resources",      lambda: __import__("pkg_resources"))
chk("PIL.Image.LINEAR",   lambda: getattr(__import__("PIL.Image", fromlist=["Image"]), "LINEAR"))
chk("numpy<2",            lambda: (lambda v: v.startswith("1.") or sys.exit("numpy>=2 installed"))(__import__("numpy").__version__))
chk("opencv-python-headless", lambda: __import__("cv2"))
chk("decord",             lambda: __import__("decord"))

if errors:
    print(f"[vbench-setup][error] {len(errors)} sanity check(s) failed: {errors}")
    sys.exit(1)
print("[vbench-setup] sanity OK")
PY

echo "[vbench-setup] DONE"
echo "  env       : $VBENCH_ENV"
echo "  info json : $INFO_JSON"
echo "  vbench cache: $VBENCH_CACHE"
echo
echo "Next: source scripts/hpc/submit.sh sbatch_vbench.sh longlive_models/models/lora.pt smoke"
