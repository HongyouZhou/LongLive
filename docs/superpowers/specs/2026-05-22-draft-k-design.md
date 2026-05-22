# DRaFT-K — Reward-Gradient Backprop for 4-step DMD Video Fast Adaptation

## Context

Online RL post-training family on our 4-step DMD-distilled Wan2.1-T2V-1.3B base has
been explored along two mechanisms so far:

| Method | Status | Outcome |
|---|---|---|
| DiffusionNFT(arxiv:2509.16117) | 4 runs (NFT / H1 / H3 / H3') | All failed: amplitude overshoot, direction wrong, or caption collapse (VBench Semantic 12-13 vs BASE 76) |
| RAM(arxiv:2605.10759) | RAM-v1 (JID 8978352) | **PASS** — VBench Total 83.07 vs BASE 83.09; UCF mf 0.314 (BASE 0.348 × 0.9 = 0.31 hit); no caption collapse |

RAM works but uses **scalar reward** with no gradient signal — motion_fidelity is
treated as black-box. Our `motion_fidelity` (CoTracker3 tracklet cosine vs reference
clip) is **end-to-end differentiable** (VAE decode + CoTracker3 transformer +
tracklet cosine — all PyTorch nn.Module composition). RAM paper explicitly markets
"no reward gradients" as a feature; DRaFT (Clark et al., arxiv:2309.17400) takes
the opposite stance and **directly backpropagates the reward gradient** through
the sampler. The two are mechanistically orthogonal.

This spec proposes a `longlive/methods/diffusion_draft/` method that ports DRaFT-K
(K = number of last DMD steps with gradient on) to our setting. Goal: get a new
RL-style fast-adaptation ckpt to enter the baseline ledger as `DRaFT-v1` alongside
`RAM-v1`, with the headline question:

> Does the differentiable reward signal beat RAM's scalar reward on motion_fidelity,
> or do they sit at the same plateau bounded by 4-step DMD's distillation ceiling?

The intended outcome is a frozen ckpt + 3-table eval row, NOT a generalized solver.

## Mechanism (Section 1)

**Core loss (per outer epoch, gradient accumulated)**:

```
L_reward = − reward_coef · motion_fidelity( CoTracker3(VAE.decode(x_0)), ref_tracklets )
L_kl     = β_KL · ‖ v_default(x_t, t_random) − sg(v_anchor(x_t, t_random)) ‖²
L_total  = L_reward + L_kl
```

Where:

- `x_0 = pipeline.rollout(noise, cond; θ)` — 4-step DMD rollout. Last `k_grad_steps`
  steps have grad ON, prior `(4 − k_grad_steps)` steps under `torch.no_grad()`.
- `v_default = generator(x_t, cond, t)` with `set_adapter("default")` — trainable.
- `v_anchor = generator(x_t, cond, t)` with `set_adapter("anchor")` no_grad — equals
  the no-LoRA base (zero-init B-projection).
- `t_random` cycles through `t_anchors = [1000, 750, 500, 250]`.
- `sg` = stop-grad.

**Design decisions (signed off Section 1)**:

| Decision | Choice | Rationale |
|---|---|---|
| `k_grad_steps` | **2** (last 2 of 4 DMD steps) | balance reward signal strength vs memory; K=1 too weak, K=4 too memory-heavy |
| Reward branch sample count | **`k_rollouts_per_outer = 2`** | grad-accumulate across 2 rollouts for variance reduction; **no group_normalize** (DRaFT is not contrastive) |
| Explicit KL anchor | **required**, `β_KL = 1e-3` | DRaFT has no implicit anchor (vs RAM's target form); NFT-H3 collapse proved this is necessary; β_KL stronger than RAM's 0 / NFT's 1e-4 |
| Reward polarity | `loss = -reward_coef · mf` (mf higher better) | motion_fidelity ∈ [-1, 1] |
| `reward_coef` | **1.0** at start, sweep later | conservative start; RAM paper uses 100/1000 for [0,1] binary rewards — different reward distribution |
| `v_anchor` in reward path | **No** — only in KL side branch | reward backprop goes through rollout → decode → CoTracker → mf, doesn't touch v_anchor |

## Code architecture (Section 2)

**New method directory** (sibling of `diffusion_nft/` and `diffusion_ram/`):

```
longlive/methods/diffusion_draft/
├── __init__.py
├── train.py                              ~500 LOC
├── losses.py                             ~60 LOC (kl_anchor_loss only)
└── configs/
    ├── skateboarding_draft.yaml
    └── skateboarding_draft_smoke.yaml
```

`diffusion_draft/train.py` is structurally a fork of `diffusion_ram/train.py` with
the outer/inner loop replaced (see Section 3). `losses.py` only exposes
`kl_anchor_loss`; the reward loss is inline in `train.py` (it's a 1-line
`F.mse_loss`-free expression `-reward_coef * mf`).

**Shared infra additions in `longlive/utils/`** (method-agnostic; benefits future
methods):

| File | New API | Purpose |
|---|---|---|
| `longlive/utils/motion_reward.py` | `CoTrackerWrapper.get_tracklets_from_tensor(video: torch.Tensor, requires_grad: bool) -> (tracks, vis)` | Bypass mp4 round-trip; direct tensor → tracklets. `requires_grad=True` removes inner `torch.no_grad()`. |
| `longlive/utils/motion_reward.py` | `motion_fidelity_pair_grad(...)` — torch tensor variant of `motion_fidelity_pair` | Same algorithm but on torch tensors, with gradient. |
| `longlive/utils/motion_reward.py` | `MotionFidelityRewardGrad` class wrapping the above | Parallel to existing `MotionFidelityReward`; same `__init__` API. |
| `longlive/utils/rl_rollout.py` | `RolloutEngine.rollout_with_grad(noise, k_grad_steps) -> (video_tensor, latent_x0)` | 4-step DMD rollout with last `k_grad_steps` grad ON. VAE decode also grad ON. |

**Infra grad enabling**:

1. **`CausalInferencePipeline.inference` grad gating**:
   The existing `@torch.no_grad()` decorator on the inference loop must be
   replaced with per-step `enable_grad` context. Pipeline code lives in
   `longlive/pipeline/causal_inference.py`. The change is **method-agnostic**
   (an inference capability extension, not a method-specific hack); per CLAUDE.md
   `methods/<idea>/ only reads pipeline/`, this is the one location where the
   pipeline interface needs widening. Add a `k_grad_steps: int = 0` parameter
   that, when > 0, runs the last K steps under `enable_grad()`. Default 0
   preserves all existing NFT/RAM/inference behavior — full backward compat.

2. **VAE decode**:
   `vae.decode_to_pixel(latent)` currently runs under no_grad inside the
   inference pipeline. The grad path uses the same call but inside the
   `enable_grad()` window when `step_idx >= total_steps - k_grad_steps`.

3. **CoTracker3 backprop**:
   `facebookresearch/co-tracker` `cotracker3_offline` is a standard `nn.Module`
   (transformer; no BN). Enabling grad means simply not wrapping in
   `torch.no_grad()`. Set `cotracker.eval()` still (dropout off) but allow
   `requires_grad` to propagate.

**Memory budget** (K_grad=2, K_rollouts_per_outer=2, 8 H200 FSDP, est.):

| Component | Per-rank GB |
|---|---:|
| Wan-1.3B FSDP shard | 5 |
| Wan-1.3B activations × K_grad=2 × K_rollouts=2 | 12 |
| VAE decode activations × 2 rollouts | 4 |
| CoTracker3 (200M params) activations × 2 rollouts | 3 |
| Inner KL branch activations | 2 |
| Optimizer state (AdamW × LoRA) | 0.5 |
| KV cache + misc | 5 |
| **Total** | **~30** (H200 140 GB cap) ✓ |

Fallback knobs if OOM: K_grad: 2→1, K_rollouts_per_outer: 2→1.

## Training loop + yaml schema (Section 3)

**Outer epoch pseudocode**:

```python
for outer in range(outer_epochs):
    optimizer.zero_grad()
    sum_mf, sum_loss_kl = 0.0, 0.0

    # ─── Reward branch (DRaFT-K backprop, expensive) ───
    generator.model.set_adapter("default")
    last_latent_x0 = None
    for k in range(k_rollouts_per_outer):
        noise = sample_noise(latent_shape, seed=outer*1009 + 31*rank + k)
        video, latent_x0 = rollout_engine.rollout_with_grad(
            noise, k_grad_steps=cfg.k_grad_steps
        )
        tracklets_gen = cotracker.get_tracklets_from_tensor(
            video, requires_grad=True
        )
        mf = motion_fidelity_pair_grad(
            gen_tracks=tracklets_gen.tracks,
            gen_visibility=tracklets_gen.vis,
            ref_tracks=ref_tracks_cached,
            ref_visibility=ref_vis_cached,
        )
        loss_reward = -reward_coef * mf / k_rollouts_per_outer
        loss_reward.backward()
        sum_mf += float(mf.detach())
        last_latent_x0 = latent_x0.detach()

    # ─── KL anchor branch (cheap, supervised side, anti-drift) ───
    if beta_kl > 0:
        for kl_step in range(n_kl_steps_per_outer):
            anchor_t = anchors[kl_step % len(anchors)]
            noise = torch.randn_like(last_latent_x0)
            x_t = sched.add_noise(last_latent_x0, noise, anchor_t_tensor)

            generator.model.set_adapter("default")
            v_default, _ = generator(x_t, train_cond, ts)

            generator.model.set_adapter("anchor")
            with torch.no_grad():
                v_anchor, _ = generator(x_t, train_cond, ts)

            generator.model.set_adapter("default")  # restore before backward

            loss_kl = beta_kl * F.mse_loss(v_default, v_anchor.detach()) / n_kl_steps_per_outer
            loss_kl.backward()
            sum_loss_kl += float(loss_kl.detach())

    optimizer.step()
    # log + ckpt
```

Key invariants:

- One optimizer step per outer (grad accumulation across reward + KL branches).
- KL branch reuses **last reward-branch rollout's `latent_x0.detach()`** as the
  x_0 source — no extra rollout cost.
- No `group_normalize`, no `dist.all_gather` (no contrastive / group concept).
- `set_adapter` restore-to-default before `.backward()` is preserved from NFT/RAM
  (gradient checkpointing × PEFT × FSDP safety from run 8939766 lesson).

**yaml schema (`skateboarding_draft.yaml`)**:

| Field | Value | Notes |
|---|---|---|
| `outer_epochs` | 20 | Same budget as NFT/RAM |
| `k_grad_steps` | 2 | last K of 4 DMD steps with grad ON |
| `k_rollouts_per_outer` | 2 | reward-branch grad-on rollouts |
| `reward_coef` | 1.0 | conservative start; sweep later |
| `n_kl_steps_per_outer` | 4 | cheap KL anchor steps |
| `beta_kl` | 1.0e-3 | stronger than RAM (which has implicit anchor) |
| `t_anchors` | `[1000, 750, 500, 250]` | unchanged |
| `adapter rank/alpha` | 32 / 64 | unchanged |
| `lr` | 3.0e-4 | unchanged |
| `denoising_step_list` | `[1000, 750, 500, 250]` | unchanged |
| Dropped vs NFT/RAM | `beta`, `positive_only`, `ema_*`, `adv_clip_max`, `inner_steps`, `k_rollouts`, `k_noisings_per_endpoint`, `g_endpoints_per_outer`, `rollout_adapter` | not applicable |

Smoke yaml (`skateboarding_draft_smoke.yaml`):
`outer_epochs=2`, `k_rollouts_per_outer=1`, `n_kl_steps_per_outer=2`,
`k_grad_steps=1`. ~5-8 min on 1 GPU.

**Wall clock estimate**:
Reward rollout × 2 with grad: ~2 min · KL 4 cheap forwards: ~30 s · Each outer:
~2.5 min · 20 outer training: ~50 min · + motion_eval ~1h + vbench ~1h · **Total
~3 h**.

## Verification (Section 4)

### Smoke pass criteria (1 GPU, outer=2, K_grad=1, K_rollouts=1, n_kl=2)

1. SLURM job exits 0:0
2. `default param count == anchor param count` (both 600)
3. Reward grad backprop functional: `loss_reward < 0`, finite, no NaN in outer 0/1
4. CoTracker3 grad path works: log shows `mf.requires_grad=True`; `mf.backward()` doesn't error
5. VAE decode grad path works: no `torch.no_grad()` blocking decode
6. K_grad=1 correctness: hook verifies t={1000,750,500} forward in no_grad, t=250 has grad
7. `v_default ≈ v_anchor` at outer 0 within 5% (PEFT anchor B=0 confirms)
8. 1 GPU memory < 80 GB (H200 has 140 GB)
9. Wall clock < 10 min

### Full pass criteria (8 H200, outer=20)

1. SLURM COMPLETED 0:0
2. Reward loss curve monotone ↓ (or initial ↑ then ↓, but not divergent)
3. KL loss < 1.0 throughout (anti-drift functional)
4. UCF motion_fidelity ≥ 0.31 AND ≥ RAM-v1 (0.314)
5. LOVEU motion_fidelity ≥ 0.36 AND ≥ RAM-v1 (0.357)
6. VBench Total ≥ 80 (looser than RAM's actual 83.07 — reward grad may be more aggressive)
7. VBench Semantic ≥ 70 (NFT-H3 collapse floor avoidance)
8. Wall clock < 4 h (SLURM cap)

### Verdict matrix

| Outcome | Conclusion |
|---|---|
| All criteria pass + mf > RAM-v1 | **DRaFT > RAM** — reward gradient beats scalar reward on this base |
| All criteria pass but mf ≤ RAM-v1 | **DRaFT ≈ RAM** — gradient signal didn't help; 4-step DMD ceiling not gradient-bound |
| VBench Semantic drops below 70 | **Caption collapse re-emerged** — β_KL=1e-3 still insufficient; sweep to 1e-2 |
| OOM / not converging | **DRaFT-K infeasible at this K_grad/K_rollouts** — fall back to K_grad=1, K_rollouts=1 |

### Risk-aware fallback knobs (yaml-exposed)

- OOM: `k_grad_steps: 2→1`, `k_rollouts_per_outer: 2→1`
- Caption collapse: `beta_kl: 1e-3 → 1e-2` (NFT-H3' lineage)
- mf flat: `reward_coef: 1.0 → 10` (paper precedent)

### Sanity post-implementation

Before launching DRaFT smoke, sanity-check that the `pipeline.inference`
grad-gating addition didn't break NFT/RAM by running a 1-GPU NFT smoke:

```bash
LL_NFT_SMOKE=1 LL_NFT_EVAL=0 source scripts/hpc/submit.sh sbatch_diffusion_nft_train.sh
```

Must finish < 15 min, exit 0. Confirms backward compat of pipeline change.

## File-by-file change summary

| Path | Action |
|---|---|
| `longlive/pipeline/causal_inference.py` | EDIT — add `k_grad_steps` parameter to `inference()` |
| `longlive/utils/motion_reward.py` | EDIT — add `CoTrackerWrapper.get_tracklets_from_tensor`, `motion_fidelity_pair_grad`, `MotionFidelityRewardGrad` |
| `longlive/utils/rl_rollout.py` | EDIT — add `RolloutEngine.rollout_with_grad` method |
| `longlive/methods/diffusion_draft/__init__.py` | NEW |
| `longlive/methods/diffusion_draft/losses.py` | NEW (`kl_anchor_loss`) |
| `longlive/methods/diffusion_draft/train.py` | NEW (~500 LOC) |
| `longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml` | NEW |
| `longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml` | NEW |
| `configs/motion_eval_inference_diffusion_draft.yaml` | NEW (copy of RAM counterpart, "RAM"→"DRaFT" header) |
| `configs/vbench_short_diffusion_draft.yaml` | NEW (copy of RAM counterpart) |
| `scripts/hpc/sbatch_diffusion_draft_train.sh` | NEW (copy of RAM sbatch with `LL_RAM_*` → `LL_DRAFT_*`) |

## Integration with existing baseline ledger

Result enters `/tmp/baseline_pull/aggregate.py`'s `CKPTS` list as:

```python
("DRaFT-v1", "skateboarding_draft_<JID>", "skateboarding_draft_<JID>"),
```

Aggregator emits the standard 3-table comparison (UCF / LOVEU / VBench) per
`[[score-table-format]]` memory (↑ column headers, per-column bold maxima).
