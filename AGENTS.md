# AGENTS.md

LongLive is a video-diffusion motion finetune research framework built on
Wan2.1 and DMD distillation. Core framework code lives under
`longlive/{model,pipeline,trainer,utils}`. Method ideas live under
`longlive/methods/<idea>/` and should stay self-contained. `wan/` is vendored
upstream model code and should not be changed casually.

## Working Rules

- Read relevant code, configs, and latest numbered docs before discussing
  architecture, algorithms, or prior decisions.
- Do not speculate about external methods from names or summaries. Read the
  paper/code first; if unavailable, say so.
- Discuss architecture, algorithm, interface, and experiment-scope changes
  before implementing them.
- Confirm resolved experiment configs before launching training or eval:
  model size, data, hyperparameters, checkpoints, and output logdir all need
  explicit user approval.
- Stay on the current question. Do not answer adjacent future questions unless
  asked.
- Explain every code/config change and why it was made.
- After code changes, run:

```bash
python -m pytest tests/ -x -q
```

For doc-only changes, tests are not required unless the user asks.

## Coding Style

- Keep changes minimal and scoped to the requested task.
- Avoid silent fallbacks. Do not hide errors with broad `try/except` or
  defaulted `getattr` unless an existing local pattern requires it.
- Keep method implementations in `longlive/methods/<idea>/`; shared utilities
  belong in `longlive/utils/`.
- Do not mix model definition, training loop, and data preparation in one
  module unless the existing method pattern already does so.
- Prefer capability-based module names. The trainer, scripts, or sbatch files
  should own orchestration.
- Use concise comments only where they clarify non-obvious behavior.

## Environment

Python 3.10 / mamba env `longlive`. Non-interactive shell commands that need
the project env should use:

```bash
source ~/.bashrc && mamba activate longlive && <command>
```

First-time HPC env build:

```bash
bash scripts/hpc/setup_mamba_env.sh
```

Run setup/data-fetch scripts on login nodes, not compute nodes.

## Machines And Paths

Use `$LL_DATA` in configs instead of hard-coded absolute paths.

| Machine | Role | Data root |
|---|---|---|
| arp | Code orchestration only; no heavy compute | `~/dev/data/wm/` |
| lab | Primary `wm` data host and training via sshfs | `~/dev/data/wm/` through arp mount |
| HPC | Charite SLURM training cluster | `$PROJECT_DATA/wm` |

On HPC, project storage is under:

```bash
/sc-projects/sc-proj-cc09-repair/hongyou
```

Do not propose `/dev/shm`, `/tmp`, `$HOME`, or node-local storage for project
data, checkpoints, logs, or caches. Use `$PROJECT_HOME` and below.

Important HPC defaults:

- `PROJECT_HOME=/sc-projects/sc-proj-cc09-repair/hongyou`
- `PROJECT_DEV=$PROJECT_HOME/dev`
- `PROJECT_DATA=$PROJECT_DEV/data`
- `LL_DATA=$PROJECT_DATA/wm`
- `WAN_MODELS_ROOT=$LL_DATA/wan_models`
- `HF_HOME=$LL_DATA/hf_cache`
- `TRANSFORMERS_CACHE=$LL_DATA/hf_cache`

Submit SLURM jobs through the project wrapper:

```bash
source scripts/hpc/submit.sh <sbatch_script> [args...]
```

Do not use bare `sbatch` unless explicitly requested; the wrapper exports
`$JID` and prints the resolved log path.

## Common Commands

From repo root:

```bash
python scripts/local/train.py --config_path configs/<config>.yaml
torchrun --nproc_per_node=8 scripts/local/train.py --config_path configs/<config>.yaml
python scripts/local/inference.py --config_path configs/<config>.yaml
bash scripts/local/train_long.sh
python -m pytest tests/
bash scripts/local/sync_hpc_code.sh          # dry-run code sync to HPC
bash scripts/local/sync_hpc_code.sh --apply  # apply code sync to HPC
bash scripts/hpc/fetch_data.sh
LL_ON_POLICY_CONTEXT_DISTILLATION_SMOKE=1 source scripts/hpc/submit.sh sbatch_on_policy_context_distillation_train.sh
```

For multi-config or multi-dataset work, use SLURM/orchestrators instead of a
single-GPU bash loop.

## Architecture Map

| Path | Role |
|---|---|
| `wan/` | Vendored Wan2.1 model code |
| `wan_models/` | Local Wan checkpoints |
| `longlive/model/` | `BaseModel`, DMD variants, streaming training model |
| `longlive/pipeline/` | Causal inference, self-forcing, streaming rollout |
| `longlive/trainer/` | `ScoreDistillationTrainer`: FSDP, LoRA, checkpoints, wandb |
| `longlive/utils/` | Wan wrapper, scheduler, loss registry, LoRA utilities, distributed/memory helpers |
| `longlive/methods/` | Independent research methods and adapters |
| `configs/` | OmegaConf YAMLs; merge with `default_config.yaml` where relevant |
| `scripts/local/` | Local train/inference entry points |
| `scripts/hpc/` | SLURM templates and data/env helpers |
| `scripts/motion_eval/` | Motion eval generation/scoring |
| `scripts/vbench/` | VBench generation/scoring |
| `docs/` | Numbered design notes; read the highest-numbered relevant doc first |

## Extension Points

Before adding a new idea, identify which layer it touches.

- L1 parameterization: adapter dispatch in `longlive/utils/lora_utils.py`,
  selected by YAML `adapter.type`.
- L2 forward hooks: PyTorch hooks; preserve output shape/device/dtype.
- L3 teacher score: subclass DMD and override `_compute_kl_grad(...)`.
- L4 loss form: add a `DenoisingLoss` in `longlive/utils/loss.py` and register
  it in `NAME_TO_CLASS`.
- L5 trainer phase: subclass `ScoreDistillationTrainer` only when the training
  loop itself must change.

Most methods should need only YAML plus a small `longlive/methods/<idea>/`
module. If a change seems to require edits across `wan/` or core pipeline
modules, discuss it first.

## Eval Reporting

When reporting checkpoint comparisons, use three separate tables in this order:

1. UCF Sports, 60 prompts:
   `app_div`, `temp_consist`, `pick_score`, `motion_fidelity`, `dynamic_score`
2. LOVEU-TGVE, 304 prompts:
   same five motion-eval metrics
3. VBench, 944 prompts:
   `Total`, `Quality`, `Semantic`, `dynamic_degree`

Rules:

- Put BASE as the first row anchor.
- Columns are metrics; rows are methods.
- Mark every metric header with direction, currently all `↑`.
- Bold the best value per column according to direction.
- Do not merge UCF, LOVEU, and VBench into one table.
- Do not add delta-vs-BASE tables unless asked.
- Full eval means no prompt limit: motion eval over UCF+LOVEU and VBench over
  all 944 prompts.
- Treat very high `dynamic_score` / `dynamic_degree` as potentially useful but
  also possibly a motion artifact; do not over-interpret without video review.

## Current Research Context

The active research problem is efficient few-step-to-few-step distillation /
adaptation on top of the released 4-step LongLive product. Prior notes identify
DMD reverse-KL mode-seeking as a structural cause of motion-amplitude collapse.

Important context from the latest docs:

- Vanilla MotionDirector-style pointwise MSE is poorly matched to a 4-step DMD
  student and tends to overfit reference clips.
- DiffusionNFT-style beta interpolation and EMA self-mirror were tested and
  showed collapse or unstable motion in this setting. The executable
  `diffusion_nft` method has been removed; `docs/04.md` / `docs/05.md` are
  historical failure records only.
- The active new direction is `longlive/methods/on_policy_context_distillation`:
  current-student on-policy rollouts plus frozen context-teacher velocity
  matching on visited states.
- DRaFT-style reward-gradient backprop is memory-prohibitive for 21-frame
  causal video DMD with KV caches.
- RAM-v1 is the current working fast-adaptation baseline: rank-32 LoRA on top
  of `longlive_base.pt + lora.pt`, on-policy 4-step rollouts, motion reward,
  frozen velocity anchor, and short inner-loop distillation.

For new research decisions, read `docs/00.md` and the latest numbered docs
before proposing changes.
