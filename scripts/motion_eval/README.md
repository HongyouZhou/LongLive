# scripts/motion_eval — motion-customization eval framework

Method-agnostic eval for video-diffusion checkpoints, aligned to
MotionDirector (arXiv:2310.08465). Complements `scripts/vbench/` —
VBench measures generic quality, this measures motion fidelity to a
reference video plus appearance / temporal / prompt-alignment.

## What it computes

Four metrics per generated video, copying LOVEU-TGVE 2023's reference
implementation for the three CLIP-based scores and Yatim et al.
(CVPR 2024) for motion fidelity:

| Metric             | Source                                 | Model                                          |
|--------------------|----------------------------------------|------------------------------------------------|
| Appearance Diversity (`app_div`)         | LOVEU-TGVE `run_eval.py:clip_score_text` | `openai/clip-vit-large-patch14`            |
| Temporal Consistency (`temp_consist`)    | LOVEU-TGVE `run_eval.py:clip_score_frame` | `openai/clip-vit-large-patch14`            |
| PickScore (`pick_score`)                 | LOVEU-TGVE `run_eval.py:pick_score`       | `yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` processor |
| Motion Fidelity (`motion_fidelity`)      | Yatim et al. CVPR 2024 tracklet-velocity cosine | `facebookresearch/co-tracker` (CoTracker3) |

MotionDirector itself reports motion fidelity via MTurk; we substitute
the Yatim auto metric because it (i) is the field-standard since 2024
(MotionMatcher / MoTrans / MotionInversion all use it), (ii) is designed
for cross-subject comparison ("human dunking" vs "cat dunking"), and
(iii) is fully reproducible.

## Two benchmarks

- **LOVEU-TGVE 2023** (single-video setting): 76 reference videos ×
  4 edit prompts each. **All prompts are released** in
  `LOVEU-TGVE-2023_Dataset.csv` inside the official zip; numerical
  scores can be compared directly to MotionDirector paper Table 2.
- **UCF Sports Action** (multi-video setting): 10 motion categories ×
  6 cross-appearance prompts each = 60 test prompts. **The 72-prompt
  list and 95-clip filter are NOT released by MotionDirector** —
  this is our reconstruction (paper §A.3 prose + Fig 8a / 10 / 9
  verbatim entries for Skateboarding + Pommel Horse, others
  PATTERN-reconstructed). **UCF Sports numbers are NOT a direct
  comparison to paper Table 1**; they're an internal baseline against
  our own MotionDirector reproduction.

### Deviations from paper

1. **10 vs 12 UCF categories.** MotionDirector's paper splits Golf-Swing
   into 3 view subclasses for 12 total, but their public repo ships
   only 4 sport demo folders with a single combined `playing_golf`
   LoRA. We follow the released convention: 10 UCF native classes,
   no Golf-Swing view split. See `prompts/ucf_sports.yaml` for the
   exact reconstruction.
2. **MTurk → auto metric for motion fidelity.** See "What it computes"
   above. The Yatim metric is auto-reproducible; MTurk human eval
   could be added later for paper-quality reports.
3. **UCF cross-appearance prompts.** Only Skateboarding (all 6
   inference prompts + train caption) and a few scattered entries are
   verbatim from paper figures; the remaining 54 prompts are
   subject-swap PATTERN reconstruction with the 6-subject pool
   {monkey, bear, panda, lion, alien, robot}. Marked per-category in
   `prompts/ucf_sports.yaml` as `paper_verbatim: true | partial | false`.

## End-to-end usage

### 0. One-time env setup (on lab or HPC login node)

The longlive env needs three extra deps for motion eval: `Pillow`, `decord`,
and `cotracker`. The setup script adds them additively (no env fork, since
none of them conflict with the longlive torch/cu stack):

```bash
bash scripts/motion_eval/setup_motion_eval_env.sh
```

### 1. Stage data (one-time)

**On lab** (full internet — direct download mode):

```bash
python scripts/prepare_motion_eval.py --datasets ucf,loveu
```

Reads `$LL_DATA` (= `~/dev/data/wm` on lab). Downloads UCF Sports
(`curl`, 1.66 GiB → filter → ~100 MB of mp4s) and LOVEU-TGVE 2023
(`gdown`, ~500 MB). The original zips are removed after layout
unless ``--keep_zip`` is set.

**On HPC** (use the wrapper — auto-detects HPC, defaults to rsync from lab):

```bash
bash scripts/hpc/fetch_motion_eval.sh
```

The wrapper activates the longlive env, sets `LL_DATA = $PROJECT_DATA/wm`,
and rsyncs the already-prepared `ucf_sports/` and `loveu_tgve/` directories
from `hongyou@lab` (the canonical HPC ← lab data pattern). No HF mirror
exists for either dataset, and compute nodes have no outbound network, so
rsync from lab is the only viable HPC path. Login-node only.

Override defaults via env vars (see header of `fetch_motion_eval.sh`):
- `LL_REMOTE_HOST=""` force direct download even on HPC (rarely useful)
- `LL_MOTION_EVAL_DATASETS=loveu` subset

### 2. Run eval end-to-end

**Local (lab, single command, multi-GPU):**

```bash
bash scripts/motion_eval/run_motion_eval.sh \
    "$LL_DATA/longlive_models/models/lora.pt" \
    configs/motion_eval_inference.yaml \
    baseline_v1
```

**HPC (SLURM):**

```bash
sbatch scripts/hpc/sbatch_motion_eval.sh \
    longlive_models/models/lora.pt baseline_v1
```

**Smoke test (8 prompts):**

```bash
LL_MOTION_EVAL_LIMIT=8 sbatch scripts/hpc/sbatch_motion_eval.sh \
    longlive_models/models/lora.pt smoke
```

Outputs land in `$LL_DATA/motion_eval_runs/<run_id>/`:

```
prompts_manifest.jsonl       # joined prompt list
config.snapshot.yaml         # frozen inference config
videos/<prompt_id>.mp4       # generated videos
manifest.json                # prompt_id -> mp4 (eval input contract)
dispatch_log.jsonl           # generation audit log
cache/tracklets/<sha10>.npz  # CoTracker3 tracklets (per-ref cache)
scores.csv                   # per-prompt 4 metrics
```

End-of-run stdout prints per-dataset means of all 4 metrics.

### 3. Inspect results

```bash
python -c "
import csv, statistics, collections
rows = list(csv.DictReader(open('$LL_DATA/motion_eval_runs/baseline_v1/scores.csv')))
by_ds = collections.defaultdict(list)
for r in rows:
    if r['ok'] != 'True': continue
    by_ds[r['dataset']].append(r)
for ds, rs in by_ds.items():
    print(ds, len(rs),
          {k: round(statistics.mean(float(r[k]) for r in rs), 4)
           for k in ('app_div','temp_consist','pick_score','motion_fidelity')})
"
```

## Component layout

```
scripts/
  prepare_motion_eval.py        # data download + filter + manifest
  motion_eval/
    prompts/ucf_sports.yaml     # 10×6 prompt reconstruction (committed)
    build_manifest.py           # prompts × ref clips -> unified JSONL manifest
    eval_dispatch.py            # multi-GPU generation dispatcher (reuses vbench worker)
    metrics/
      clip_metrics.py           # 3 LOVEU-TGVE metrics
      motion_fidelity.py        # Yatim CoTracker3 metric
      video_io.py               # decord mp4 reader
    run_eval.py                 # post-hoc scoring driver (CSV out)
    run_motion_eval.sh          # 3-phase orchestrator
  hpc/
    sbatch_motion_eval.sh       # SLURM wrapper around run_motion_eval.sh
configs/
  motion_eval_inference.yaml    # generation config (mirrors vbench_short.yaml)
```

The generation worker is `scripts/vbench/eval_worker.py` (verbatim — it
already accepts a `prompt / seed / output_mp4` JSON contract). No fork.

## Licenses

- **UCF Sports Action** (Soomro & Zamir 2015): no written license on the
  CRCV page; academic-only by convention. Footage sourced from BBC /
  Getty / ESPN broadcasts. Cite Soomro & Zamir 2015.
- **LOVEU-TGVE 2023** (Wu et al. CVPR 2023 workshop): Creative Commons
  (variant not stated; safe for academic eval). Cite Wu et al. 2023.
- **CoTracker3** (`facebookresearch/co-tracker`): **CC BY-NC 4.0**
  (non-commercial). Acceptable for paper-table reporting; flag for any
  productisation path.
- **PickScore** (`yuvalkirstain/PickScore_v1`): Apache-2.0.
- **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`): MIT.

## Smoke tests per component

```bash
# 1. Data prep on a tiny subset
python scripts/prepare_motion_eval.py --datasets loveu --keep_zip
ls "$LL_DATA/loveu_tgve/videos/" | head; wc -l "$LL_DATA/loveu_tgve/prompts.csv"

# 2. Manifest build
python scripts/motion_eval/build_manifest.py \
    --data_root "$LL_DATA" --datasets loveu --output /tmp/m.jsonl
wc -l /tmp/m.jsonl  # ~304 rows (76 videos × 4 edit prompts)

# 3. CLIP metrics on any mp4
python -m scripts.motion_eval.metrics.clip_metrics \
    --mp4 some_video.mp4 --prompt "a cat dunking"

# 4. Motion fidelity (self-vs-self should be near 1.0)
python -m scripts.motion_eval.metrics.motion_fidelity \
    --gen some_video.mp4 --ref some_video.mp4

# 5. End-to-end with --limit
LL_MOTION_EVAL_LIMIT=4 bash scripts/motion_eval/run_motion_eval.sh \
    "$LL_DATA/longlive_models/models/lora.pt" \
    configs/motion_eval_inference.yaml smoke
```
