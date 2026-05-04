# CLAUDE.md

LongLive — video diffusion motion finetune research framework. Built on Wan2.1 (1.3B / 14B) with DMD distillation. Core code lives under a single `longlive/` umbrella (`longlive/{model,pipeline,trainer,utils}`); `longlive/methods/<idea>/` hosts independent finetune method implementations (OFT, Schrödinger Bridge, etc.) sharing the core Wan loader + DMD pipeline. `wan/` stays at root as vendored upstream.

## Workflow

- **Confirm experiment config before launch.** Never start a training / eval run without explicit user sign-off on the config (model size, data, hyperparams, ckpt source, output logdir). Print the resolved config and wait for approval, even when the user has asked you to "go run X" — the resolved values may differ from what they expected.
- **Read first, discuss second.** When the user opens a discussion that touches code, configs, or prior decisions, read the relevant files (linked docs, referenced source paths, latest `docs/NN.md`) before responding. Do not theorize on what the code probably does.
- **One question at a time.** Stay on the question being discussed. Do not pre-emptively answer adjacent or future questions, do not pivot to a different aspect mid-thread, do not pile multiple proposals into one reply. Resolve the current question, then move to the next.
- **Discuss before changing.** Architecture, algorithm, or interface changes must be proposed and approved before implementation.
- **Clarify scope first.** When user describes an architectural change ("put X in the loop", "merge these stages"), ask about scope before implementing. Do not assume a narrower interpretation.
- **Verify before claiming compliance.** Check every requirement against the codebase before proposing fixes.
- **Stay focused.** Prioritize the current task. Unrelated improvements can be mentioned but not implemented without asking.
- **Explain every change.** State what changed and why, even for small edits.
- **Run tests after changes.** `python -m pytest tests/ -x -q` after any code modification.
- **`longlive/methods/<idea>/` is self-contained.** It only reads `wan/` and `longlive/{model,pipeline,utils}/`. It does not modify them. Shared utilities go into `longlive/utils/`, not duplicated under `longlive/methods/`.

## Coding Style

- **No silent fallbacks.** No `getattr(obj, attr, default)` or `try/except` to swallow errors.
- **ABC allowed for plug-in frameworks** (e.g. `DenoisingLoss`, scheduler). New business modules default to duck typing + convention + docstrings; reach for ABC only when a registry / plug-in interface justifies it.
- **Keep it simple.** No over-engineering. No deprecated code. No redundant comments.
- **Modules named by capability.** Sequencing belongs in the orchestrator (trainer / sbatch / `script/`).
- **Separation of concerns.** Do not mix model definition, training loop, and data prep. When in doubt, ask before combining.

## Setup

Python 3.10 / mamba env `longlive` (per `scripts/hpc/setup_mamba_env.sh`). PyTorch 2.8.0 + cu128, flash-attn 2.8.3. Non-interactive shells:
```bash
source ~/.bashrc && mamba activate longlive && <command>
```
First-time HPC env build:
```bash
bash scripts/hpc/setup_mamba_env.sh    # idempotent; run on a login node
```

## Three-machine layout

| Machine | Role |
|---|---|
| **arp** | Code orchestration only. Push code, drive remote training, no heavy compute. |
| **lab** (Mechatron) | **Primary `wm` data host + training** since 2026-04-26. arp's repo + data mounted via sshfs. |
| **HPC** (Charité sc-projects cluster) | Training cluster. `fetch_data.sh` defaults to HuggingFace Hub for public ckpts; set `LL_REMOTE_HOST=hongyou@lab` only when rsync-ing **private** data (motion refs, custom clips) from lab, or as a fallback when HF is slow/unavailable. Sole writable storage is `/sc-projects/sc-proj-cc09-repair/hongyou` — never propose `/dev/shm`, `/tmp`, `$HOME`, or other "node-local" paths to speed up NFS. |

Cross-machine path handling: configs reference **`$LL_DATA`**, not hard-coded absolute paths, so the same YAML works on all three machines. Concrete roots differ:
- **arp / lab**: `~/dev/data/wm/` (lab sees arp's home via sshfs mount).
- **HPC**: `$PROJECT_DATA/wm` = `/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm` (no symlink to `~/dev` — `$HOME` on HPC is `/home/hozh10`).

Do not invent path variants for disambiguation.

### lab (sshfs-mounted from arp)

```bash
# On arp: mount once
ssh lab "sshfs arp:/home/hongyou ~/mnt/arp -o reconnect"

# On lab (interactive SSH, ideally inside a long-lived tmux session):
ssh -t lab
source ~/mnt/arp/miniforge3/etc/profile.d/conda.sh
mamba activate ~/mnt/arp/miniforge3/envs/longlive
cd ~/mnt/arp/dev/LongLive
ulimit -n 65536
```

All paths under `~/mnt/arp/...` are arp's filesystem over sshfs; treat that mount as the source of truth.

### HPC (Charité sc-projects, SLURM)

**Access is VPN-gated.** From arp, the entry alias drops you on the front node:

```bash
alias charitefront='sudo ip netns exec vpnns sshpass -p "$(cat ~/ovpn/.vpn_fixed_pass)" ssh -o StrictHostKeyChecking=no hozh10@s-sc-frontend1.charite.de'
charitefront            # interactive front-node shell
```

Direct `ssh hpc …` from outside the VPN namespace will not work. Long-running work must live in tmux on the front node, not on arp.

#### Front-node environment (`hozh10@s-sc-frontend1`)

Login user is `hozh10` (so `$HOME=/home/hozh10`), but **everything project-related lives under `/sc-projects/sc-proj-cc09-repair/hongyou`** — the sole writable scale storage. There is **no symlink** between `~/dev` and `$PROJECT_DEV`; the sbatch templates intentionally use absolute `$PROJECT_DATA` paths (see `sbatch_train.sh` "Data source — explicit, no symlinks"). Cross-machine portability comes from `$LL_DATA`, not from path-equality.

Never propose `/dev/shm`, `/tmp`, `$HOME`, or any "node-local" path — only `$PROJECT_HOME` and below are writable at scale.

Pre-set in HPC `~/.bashrc`:

| Var / alias | Value |
|---|---|
| `PROJECT_HOME` | `/sc-projects/sc-proj-cc09-repair/hongyou` |
| `PROJECT_DEV` | `$PROJECT_HOME/dev` |
| `PROJECT_DATA` | `$PROJECT_DEV/data` |
| `dev` / `dat` / `proj` | `cd` into the above |
| `HF_TOKEN`, `WANDB_API_KEY` | exported at login |
| `myq` / `wmyq` | one-shot / `watch` of own SLURM queue |
| `compute` | `srun -p compute -c 16 --mem=32G --time=10:00:00 --pty bash` — CPU debug shell |
| `jid [N|jobid|name]` | pick a job and export `$JID`; bare `jid` lists running jobs |
| `gpus [-L] [jid]` | run `nvidia-smi` on the job's compute node via `srun --overlap` |
| `wgpus [-L] [-n SEC] [jid]` | `watch` wrapper around `gpus` |
| `tat <name>` | `tmux attach -t <name>` |

mamba is initialised by default; `mamba activate longlive` is the project env. Conda `(base)` is also auto-activated — do not assume a clean shell.

#### LongLive sbatch conventions

Submit via templates under `scripts/hpc/`:

| Env var | Purpose | Default |
|---|---|---|
| `LL_ENV_NAME` | mamba env name | `longlive` |
| `LL_REPO` | repo path (else SLURM submit dir) | auto-detect |
| `LL_DATA` | data root | `$PROJECT_DATA/wm` (= `$PROJECT_HOME/dev/data/wm`) |
| `WAN_MODELS_ROOT` | Wan checkpoints | `$LL_DATA/wan_models` |
| `HF_HOME`, `TRANSFORMERS_CACHE` | HF cache | `$LL_DATA/hf_cache` |
| `LL_AUTO_RESUME=1` | auto-pick latest logdir with a checkpoint | unset |
| `LL_RESUME_LOGDIR` | explicit resume target | unset |
| `LL_LOW_CPU_MEM=1` | meta-init + FSDP broadcast (rank0 holds full model only) | unset |
| `LL_REMOTE_HOST` | opt-in: switch `fetch_data.sh` from HF Hub to rsync from a peer (e.g. `hongyou@lab`) — only needed for private data or HF fallback | unset (HF mode) |

Wan2.1-T2V-14B teacher needs ≥80GB GPU + ≥240GB CPU peak (8 ranks × 30GB). Default sbatch requests `--mem=900G` and excludes 40GB DGX A100s. Pin GPU type via `--gres=gpu:nvidia_h200:8` etc. on the sbatch CLI.

Submit / monitor examples:
```bash
# core training
sbatch scripts/hpc/sbatch_train.sh                                          # default config configs/longlive_train_long.yaml
LL_AUTO_RESUME=1 sbatch scripts/hpc/sbatch_motion_dmd.sh                    # motion-DMD; auto-resume latest matching logdir

# motion-DMD ref-latent cache (sbatch_motion_dmd.sh auto-runs this inline if missing,
# but the dedicated job is faster — only requests 2 GPUs)
sbatch scripts/hpc/sbatch_precache_motion.sh
LL_MOTION_REFS=prompts/dancing_refs.jsonl \
  LL_MOTION_CACHE='$LL_DATA/motion_dmd/dancing_v1.latents.pt' \
  sbatch scripts/hpc/sbatch_precache_motion.sh

# eval (positional <ckpt> required; resolves abs / $LL_DATA-relative / repo-relative)
sbatch scripts/hpc/sbatch_vbench.sh longlive_models/models/lora.pt paper_baseline
LL_VBENCH_LIMIT=8 sbatch scripts/hpc/sbatch_vbench.sh longlive_models/models/lora.pt smoke

# capture JID into the current shell for follow-up (gpus / scancel / tail):
source scripts/hpc/submit.sh sbatch_motion_dmd.sh
echo $JID

# monitoring
myq                       # current queue
jid && gpus               # latest job's GPU usage on its compute node
wgpus -n 5                # live GPU watch every 5 s
```

`fetch_data.sh` and `setup_mamba_env.sh` must run on a **login node** — compute nodes lack outbound network for HF Hub / pip wheels.

## Commands

```bash
python train.py --config configs/<config>.yaml                            # local train (single node)
torchrun --nproc_per_node=8 train.py --config configs/<config>.yaml       # multi-GPU
python inference.py --config configs/<config>.yaml                        # inference
python -m pytest tests/                                                   # tests
bash scripts/hpc/fetch_data.sh                                            # cross-machine data sync
python scripts/motion_dmd/precache_motion_refs.py --refs_root ...         # VAE-encode refs offline
```

**Always use parallel orchestrators / SLURM array for multi-config or multi-dataset work.** Do NOT loop single-GPU jobs in bash.

## Architecture

| Directory | Role |
|---|---|
| `wan/`, `wan_models/` | Wan2.1 base model code + checkpoints (vendored upstream; minor fork patches in `wan/modules/causal_model*.py`) |
| `longlive/model/` | `BaseModel` (real_score / fake_score / generator triple), `DMD`, `MotionAttnInjector` hook system, `streaming_training` |
| `longlive/pipeline/` | `causal_inference`, `streaming_training`, `self_forcing_training` — inference & rollout |
| `longlive/trainer/` | `ScoreDistillationTrainer` — FSDP, LoRA attach, checkpoint/resume, WandB |
| `longlive/utils/` | `loss.py` (DenoisingLoss plug-in), `wan_wrapper.py`, `scheduler.py`, `lora_utils.py`, `dataset.py`, `distributed.py`, `memory.py` |
| `longlive/methods/` | per-method implementations (OFT, bridge, …); each is self-contained, only **reads** core modules, does not modify them |
| `scripts/hpc/` | sbatch templates (`sbatch_train.sh`, `sbatch_motion_dmd.sh`, `sbatch_vbench.sh`, `sbatch_precache_motion.sh`) + `submit.sh` (source-able, captures `$JID`) + `fetch_data.sh` (HF Hub / rsync data staging, login-node only) + `setup_mamba_env.sh` (env build, login-node only) |
| `scripts/motion_dmd/` | offline motion-ref VAE precache |
| `scripts/vbench/` | distributed VBench eval |
| `configs/` | OmegaConf YAMLs; `default_config.yaml` is the base, others inherit + override |
| `docs/` | numbered design / discussion docs — read the **latest** (highest-numbered) for current context; older may be outdated |

Data layout: `$LL_DATA/` (motion datasets, `wan_models/`, `hf_cache/`, `motion_refs/`, `motion_dmd/`). Resolves to `~/dev/data/wm/` on arp / lab and `$PROJECT_DATA/wm/` on HPC. Logs under `logs/<run>/`, wandb under `wandb/`.

## Extension points (where new ideas plug in)

New ideas live under `longlive/methods/<idea>/` and inject through one of six perturbation layers (L0 data → L5 trainer loop). Before writing code, identify which layer(s) the idea touches and use the existing seam — see **[docs/01.md](docs/01.md)** for the full layer model, registration convention, and `longlive/methods/<idea>/` skeleton.

Quick map of seams:

- **L1 parameterization** (LoRA / OFT / DoRA): adapter dispatch in `longlive/utils/lora_utils.py`, YAML `adapter.type`.
- **L2 forward hook** (attn / FFN runtime override): `MotionAttnInjector`-style hook in `longlive/model/motion_hooks.py`; FSDP-safe.
- **L3 teacher score** (DMD gradient term): subclass `DMD`, override `_compute_kl_grad(...)`. Signature stable.
- **L4 loss form**: subclass `DenoisingLoss` in `longlive/utils/loss.py`, register in `NAME_TO_CLASS`, select via `denoising_loss_type`.
- **L5 trainer phase**: subclass `ScoreDistillationTrainer` (rare; LoRA attach point ≈ line 350).

Most variants need **only YAML + a small `longlive/methods/<idea>/` module**, no edits to `wan/` or `longlive/{model,pipeline,utils}/`. If an idea seems to need a new layer, that's a redesign signal — discuss before adding.
