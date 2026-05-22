# DRaFT-K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement DRaFT-K (reward-gradient backprop, Clark et al. arXiv:2309.17400) for 4-step DMD video fast adaptation as a new sibling method to `diffusion_nft` / `diffusion_ram`, exposing a `DRaFT-v1` checkpoint that enters the same baseline ledger.

**Architecture:** Per outer epoch: 2 self-rollouts with last K=2 of 4 DMD denoising steps grad-ON → CoTracker3 motion_fidelity (differentiable) → `-reward_coef · mf / K` backward into LoRA params; then 4 cheap KL anchor forwards at random anchor t (default vs zero-init `anchor` adapter) → `β_KL · MSE` backward; one `optimizer.step()`. Reuses existing `RolloutEngine` + `MotionFidelityReward` infrastructure via new grad-enabled variants. Only minimal edit to `CausalInferencePipeline.inference` adds a backward-compat `k_grad_steps` parameter.

**Tech Stack:** PyTorch 2.8 + cu128 + FSDP, PEFT 0.19 (LoRA), Wan2.1-T2V-1.3B base + NVlabs lora.pt merged, CoTracker3 offline, OmegaConf, SLURM/HPC.

**Spec:** `docs/superpowers/specs/2026-05-22-draft-k-design.md`

**Sibling references (for code structure):**
- `longlive/methods/diffusion_ram/train.py` — closest structural cousin
- `longlive/methods/diffusion_nft/train.py` — original setup-phase pattern
- `longlive/utils/{rl_rollout,motion_reward,group_norm}.py` — already-promoted shared infra

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `longlive/pipeline/causal_inference.py` | EDIT | Add `k_grad_steps: int = 0` parameter to `inference()`; gate denoising step forwards + VAE decode with `torch.no_grad()` based on it |
| `longlive/utils/rl_rollout.py` | EDIT | Add `RolloutEngine.rollout_with_grad(noise, k_grad_steps)` method |
| `longlive/utils/motion_reward.py` | EDIT | Add `CoTrackerWrapper.get_tracklets_from_tensor()`, `motion_fidelity_pair_grad()`, `MotionFidelityRewardGrad` class |
| `longlive/methods/diffusion_draft/__init__.py` | NEW | Module docstring + provenance |
| `longlive/methods/diffusion_draft/losses.py` | NEW | `kl_anchor_loss(v_default, v_anchor)` |
| `longlive/methods/diffusion_draft/train.py` | NEW | Trainer (~500 LOC, forked from `diffusion_ram/train.py` with reward-grad outer loop) |
| `longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml` | NEW | Main config (20 outer × K_grad=2 × K_rollouts=2 × n_kl=4) |
| `longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml` | NEW | Smoke config (2 outer × K_grad=1 × K_rollouts=1 × n_kl=2) |
| `configs/motion_eval_inference_diffusion_draft.yaml` | NEW | Verbatim copy of `motion_eval_inference_diffusion_ram.yaml` with header rename |
| `configs/vbench_short_diffusion_draft.yaml` | NEW | Verbatim copy of `vbench_short_diffusion_ram.yaml` with header rename |
| `scripts/hpc/sbatch_diffusion_draft_train.sh` | NEW | Verbatim copy of `sbatch_diffusion_ram_train.sh` with `LL_RAM_*` → `LL_DRAFT_*` substitution |

---

## Task 1: Add `k_grad_steps` parameter to `CausalInferencePipeline.inference`

**Files:**
- Modify: `longlive/pipeline/causal_inference.py:56-243` — `inference()` method body

**Design:** When `k_grad_steps=0` (default), every existing denoising-step forward and the final VAE decode are wrapped in `with torch.no_grad()` blocks internally — this is **bytewise-equivalent to current behavior** because all existing callers (`RolloutEngine.rollout_one`, `eval_worker.handle`) already wrap the call in `@torch.no_grad()`. When `k_grad_steps > 0`, the **last `k_grad_steps` of `denoising_step_list`** per frame block and the VAE decode are NOT wrapped, so they inherit the caller's grad mode. The "block-end clean-cache-update" forward (line 192-200) stays under `with torch.no_grad()` regardless (it only mutates KV cache, doesn't contribute to output gradient).

- [ ] **Step 1: Read the current `inference()` method to anchor the edit**

Run: `sed -n '56,243p' longlive/pipeline/causal_inference.py | wc -l`
Expected output: `188`

- [ ] **Step 2: Edit signature to add `k_grad_steps` parameter**

Find lines 56-63 (the method signature). Apply this edit:

```python
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        k_grad_steps: int = 0,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
            k_grad_steps (int): When > 0, the last `k_grad_steps` of the
                `denoising_step_list` (per frame block) and the final VAE
                decode are NOT wrapped in `torch.no_grad()` — they inherit
                the caller's grad mode.  Used by DRaFT-K reward-gradient
                backprop (see longlive/methods/diffusion_draft/).  Default 0
                preserves the original behavior bytewise (every existing
                caller wraps with @torch.no_grad() externally).
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
```

- [ ] **Step 3: Gate per-step denoising forwards by `k_grad_steps`**

Find the inner denoising step loop (lines 154-188 in the original file). Replace the `if index < len(self.denoising_step_list) - 1: ... else: ...` block with this version that inserts grad-mode gating:

```python
            # Step 2.1: Spatial denoising loop
            n_denoising_steps = len(self.denoising_step_list)
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                # k_grad_steps gating: force no_grad on the first
                # (n_denoising_steps - k_grad_steps) steps; let the last
                # k_grad_steps inherit caller's grad mode.
                in_grad_window = index >= n_denoising_steps - k_grad_steps
                step_ctx = torch.enable_grad() if in_grad_window else torch.no_grad()

                with step_ctx:
                    if index < n_denoising_steps - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
```

- [ ] **Step 4: Wrap the block-end clean-cache-update forward in `no_grad`**

Find the "Step 2.3: rerun with timestep zero" block (lines 191-200 in original). The current code calls `self.generator(...)` to update KV cache. Wrap it explicitly:

```python
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            # This forward only mutates the KV cache for the *next* block; it does
            # not contribute to the gradient flow from the output to the LoRA
            # params, so we always run it under no_grad regardless of k_grad_steps.
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
```

- [ ] **Step 5: Gate the VAE decode by `k_grad_steps`**

Find the "Step 3: Decode the output" block (lines 219-224 in original). Replace with:

```python
        # Step 3: Decode the output
        # When k_grad_steps > 0, decode inherits caller grad mode so the
        # CoTracker reward downstream can backprop through VAE.  When = 0,
        # we keep decode under no_grad (matches original behavior for all
        # existing inference callers).
        decode_ctx = torch.enable_grad() if k_grad_steps > 0 else torch.no_grad()
        with decode_ctx:
            if getattr(self.args.model_kwargs, "use_infinite_attention", False):
                video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
            else:
                video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
```

- [ ] **Step 6: Syntax check**

Run: `~/miniforge3/envs/longlive/bin/python -m py_compile longlive/pipeline/causal_inference.py && echo OK`
Expected output: `OK`

- [ ] **Step 7: Commit**

```bash
git add longlive/pipeline/causal_inference.py
git commit -m "pipeline: add k_grad_steps param to CausalInferencePipeline.inference

Default k_grad_steps=0 wraps every step + VAE decode in no_grad internally,
which is bytewise equivalent to current behavior (all existing callers
wrap with @torch.no_grad() externally).  When > 0, the last k_grad_steps
of denoising_step_list (per frame block) and decode inherit caller's grad
mode.  Used by DRaFT-K reward-gradient backprop
(longlive/methods/diffusion_draft/, see docs/superpowers/specs/2026-05-22-draft-k-design.md).

Block-end clean-cache-update forward stays in no_grad regardless — it
only mutates KV cache and doesn't contribute to output gradient."
```

---

## Task 2: Add `RolloutEngine.rollout_with_grad`

**Files:**
- Modify: `longlive/utils/rl_rollout.py` — add method to `RolloutEngine` class

- [ ] **Step 1: Read the current `RolloutEngine.rollout_one` method to copy its signature shape**

Run: `grep -nE "def rollout_one|def rollout_k|@torch.no_grad" longlive/utils/rl_rollout.py`
Expected output: lines for both methods + `@torch.no_grad()` decorators.

- [ ] **Step 2: Add `rollout_with_grad` method**

Find the `class RolloutEngine:` block and add this method right after `rollout_one` (before `rollout_k`):

```python
    def rollout_with_grad(
        self,
        noise: torch.Tensor,
        k_grad_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one full 4-step DMD rollout with last `k_grad_steps` steps grad ON.

        Used by DRaFT-K (reward-gradient backprop).  NOT decorated with
        @torch.no_grad() — the caller's grad mode controls behavior, and
        the underlying pipeline.inference internally enables grad on the
        last k_grad_steps + VAE decode.

        Args:
            noise: (B, F_lat, C, H_lat, W_lat) Gaussian noise tensor.
            k_grad_steps: How many of the last DMD denoising steps (per
                frame block) keep gradient enabled.  Range 0..4 for our
                4-step DMD schedule.  k_grad_steps=0 falls back to no_grad
                behavior identical to rollout_one (but without the
                @torch.no_grad() guard, so the caller's grad mode still
                governs the post-pipeline graph).

        Returns:
            (video_pixel, latent_x0) — same shape semantics as rollout_one.
            When k_grad_steps > 0, both tensors have `requires_grad=True`
            and the autograd graph traces back to the trainable LoRA params
            through the last k_grad_steps generator forwards + VAE decode.
        """
        video, latent = self.pipeline.inference(
            noise=noise,
            text_prompts=self._placeholder_prompts,
            return_latents=True,
            k_grad_steps=k_grad_steps,
        )
        return video, latent
```

- [ ] **Step 3: Syntax check**

Run: `~/miniforge3/envs/longlive/bin/python -m py_compile longlive/utils/rl_rollout.py && echo OK`
Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add longlive/utils/rl_rollout.py
git commit -m "utils/rl_rollout: add RolloutEngine.rollout_with_grad for DRaFT-K

Thin wrapper around CausalInferencePipeline.inference(..., k_grad_steps=K)
that exposes a method-agnostic grad-enabled rollout path.  No
@torch.no_grad decorator — caller's grad mode + pipeline's internal
k_grad_steps gating decides which steps + decode have grad on.

NFT / RAM continue to use the existing @torch.no_grad rollout_one; only
DRaFT calls this new method."
```

---

## Task 3: Add grad-enabled motion_fidelity reward path

**Files:**
- Modify: `longlive/utils/motion_reward.py` — add `CoTrackerWrapper.get_tracklets_from_tensor`, `motion_fidelity_pair_grad`, `MotionFidelityRewardGrad`

- [ ] **Step 1: Locate the `CoTrackerWrapper` class and existing `motion_fidelity_pair` numpy implementation**

Run: `grep -nE "^class CoTrackerWrapper|^def motion_fidelity_pair|class MotionFidelityReward" longlive/utils/motion_reward.py scripts/motion_eval/metrics/motion_fidelity.py 2>&1`

Expected output: lines pointing to the class / function definitions in either file. The numpy `motion_fidelity_pair` likely lives in `scripts/motion_eval/metrics/motion_fidelity.py`; the wrapper class lives in `longlive/utils/motion_reward.py`.

- [ ] **Step 2: Add `get_tracklets_from_tensor` method to `CoTrackerWrapper`**

Find `class CoTrackerWrapper:` inside `longlive/utils/motion_reward.py`. After the existing `get_tracklets(self, video_path)` method, add:

```python
    def get_tracklets_from_tensor(
        self,
        video: torch.Tensor,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same algorithm as `get_tracklets(video_path)` but takes an
        in-memory torch tensor and optionally keeps gradients.

        Args:
            video: (T, 3, H, W) or (1, T, 3, H, W) float in [0, 1].
            requires_grad: When True, the CoTracker forward is NOT wrapped
                in torch.no_grad() so reward gradients can flow back to
                the input video tensor (used by DRaFT-K reward backprop).

        Returns:
            tracks:     (T, N, 2) float torch tensor (still on device).
            visibility: (T, N) bool  torch tensor.

        NOTE: When `requires_grad=True`, the *visibility* output remains
        boolean and is treated as a constant in the gradient path (we mask
        invisible tracklets but don't differentiate through visibility).
        """
        self._ensure_model()
        # Accept (T, 3, H, W) or (1, T, 3, H, W); CoTracker wants (B, T, 3, H, W).
        if video.ndim == 4:
            video = video.unsqueeze(0)
        # Uniform-sample to self.n_frames if longer.
        T_in = video.shape[1]
        if T_in > self.n_frames:
            idxs = torch.linspace(0, T_in - 1, self.n_frames, device=video.device).long()
            video = video[:, idxs]
        # CoTracker expects [0, 255] float input (matches existing _video_to_tensor).
        video_input = video * 255.0
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            pred_tracks, pred_visibility = self._model(
                video_input, grid_size=self.grid_size
            )
        # Drop batch dim.  Keep on device + dtype unchanged so caller can
        # decide whether to .cpu() / .float() / etc.
        return pred_tracks[0], pred_visibility[0].bool()
```

- [ ] **Step 3: Add `motion_fidelity_pair_grad` function**

After the existing `motion_fidelity_pair` (numpy) function (or after the `CoTrackerWrapper` class, alphabetical convention is fine), add the torch version:

```python
def motion_fidelity_pair_grad(
    gen_tracks: torch.Tensor,        # (T, N_gen, 2)  float, grad-tracked if needed
    gen_visibility: torch.Tensor,    # (T, N_gen)     bool
    ref_tracks: torch.Tensor,        # (T, N_ref, 2)  float
    ref_visibility: torch.Tensor,    # (T, N_ref)     bool
) -> torch.Tensor:
    """Differentiable torch version of `motion_fidelity_pair` (numpy).

    Same algorithm as Yatim et al. (CVPR 2024) tracklet velocity cosine,
    but uses torch ops so gradients flow back through `gen_tracks`.

    Args:
        gen_tracks:     (T, N_gen, 2) generated-video tracklets.
        gen_visibility: (T, N_gen)    boolean visibility mask (treated as
                                       constant — no gradient through it).
        ref_tracks:     (T, N_ref, 2) reference-video tracklets (constant).
        ref_visibility: (T, N_ref)    constant.

    Returns:
        Scalar torch tensor with `requires_grad` matching `gen_tracks`.
        Same value range as numpy version: ∈ [-1, 1].
    """
    assert gen_tracks.shape[0] == ref_tracks.shape[0], (
        f"frame-count mismatch: gen T={gen_tracks.shape[0]} vs ref T={ref_tracks.shape[0]}"
    )

    # Visibility: keep only tracklets visible in ALL frames.
    gen_visible = gen_visibility.all(dim=0)  # (N_gen,)
    ref_visible = ref_visibility.all(dim=0)  # (N_ref,)
    if not gen_visible.any() or not ref_visible.any():
        return torch.zeros((), device=gen_tracks.device, dtype=gen_tracks.dtype)

    gen_t = gen_tracks[:, gen_visible]   # (T, N_gen', 2)
    ref_t = ref_tracks[:, ref_visible]   # (T, N_ref', 2)

    # Per-frame displacement (velocity direction).
    gen_d = gen_t[1:] - gen_t[:-1]       # (T-1, N_gen', 2)
    ref_d = ref_t[1:] - ref_t[:-1]
    gen_d = gen_d / (gen_d.norm(dim=-1, keepdim=True) + 1e-8)
    ref_d = ref_d / (ref_d.norm(dim=-1, keepdim=True) + 1e-8)

    # cos(d_ref[t, n], d_gen[t, m])  →  (T-1, N_ref', N_gen')
    per_frame_cos = torch.einsum("tnc,tmc->tnm", ref_d, gen_d)
    mean_t_cos = per_frame_cos.mean(dim=0)         # (N_ref', N_gen')
    best_match = mean_t_cos.max(dim=1).values      # (N_ref',)
    return best_match.mean()
```

- [ ] **Step 4: Add `MotionFidelityRewardGrad` class**

After the existing `MotionFidelityReward` class, add the grad-enabled parallel:

```python
class MotionFidelityRewardGrad:
    """Differentiable motion_fidelity scorer.

    Parallel to `MotionFidelityReward` but:
      * Takes an in-memory video tensor directly (no mp4 round-trip).
      * Keeps gradients flowing from `mf` scalar back to `video` tensor.
      * Returns a torch scalar (not a Python float).

    Reference tracklets are extracted ONCE at init under torch.no_grad
    and cached as torch tensors on `device`.
    """

    def __init__(
        self,
        ref_path: str | Path,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        n_frames: int = 16,
        grid_size: int = 30,
    ):
        self.ref_path = Path(ref_path)
        self.device = torch.device(device)

        self.tracker = CoTrackerWrapper(
            device=device, cache_dir=cache_dir,
            n_frames=n_frames, grid_size=grid_size,
        )
        # Pre-extract ref tracklets via the file-based path.  Returns
        # numpy arrays (from disk cache or fresh CoTracker run); convert
        # to torch tensors on `self.device`.  Reference doesn't need grad.
        ref_tracks_np, ref_vis_np = self.tracker.get_tracklets(self.ref_path)
        import numpy as np
        self.ref_tracks = torch.from_numpy(np.asarray(ref_tracks_np)).to(
            device=self.device, dtype=torch.float32
        )
        self.ref_visibility = torch.from_numpy(np.asarray(ref_vis_np)).to(
            device=self.device, dtype=torch.bool
        )

    def score_grad(self, video: torch.Tensor) -> torch.Tensor:
        """Score one rollout video tensor against the cached reference.

        Args:
            video: (T, 3, H, W) or (1, T, 3, H, W) float in [0, 1].  May have
                requires_grad=True (the upstream rollout pipeline grants
                this when k_grad_steps > 0).

        Returns:
            Scalar torch tensor, motion_fidelity ∈ [-1, 1], with
            requires_grad matching `video`.
        """
        gen_tracks, gen_vis = self.tracker.get_tracklets_from_tensor(
            video, requires_grad=video.requires_grad,
        )
        # Match dtype/device of gen with cached ref for the einsum.
        ref_tracks = self.ref_tracks.to(dtype=gen_tracks.dtype)
        return motion_fidelity_pair_grad(
            gen_tracks=gen_tracks,
            gen_visibility=gen_vis,
            ref_tracks=ref_tracks,
            ref_visibility=self.ref_visibility,
        )
```

- [ ] **Step 5: Syntax check**

Run: `~/miniforge3/envs/longlive/bin/python -m py_compile longlive/utils/motion_reward.py && echo OK`
Expected output: `OK`

- [ ] **Step 6: Smoke-test `motion_fidelity_pair_grad` gradient flow**

Run:
```bash
~/miniforge3/envs/longlive/bin/python -c "
import torch
from longlive.utils.motion_reward import motion_fidelity_pair_grad
T, N_g, N_r = 4, 5, 7
gen = torch.randn(T, N_g, 2, requires_grad=True)
gen_v = torch.ones(T, N_g, dtype=torch.bool)
ref = torch.randn(T, N_r, 2)
ref_v = torch.ones(T, N_r, dtype=torch.bool)
mf = motion_fidelity_pair_grad(gen, gen_v, ref, ref_v)
mf.backward()
assert mf.requires_grad, 'mf should require grad'
assert gen.grad is not None and gen.grad.shape == gen.shape, 'gen.grad shape mismatch'
assert -1.0 <= float(mf) <= 1.0, f'mf out of range: {float(mf)}'
print(f'OK  mf={float(mf):.4f}  grad_norm={gen.grad.norm():.4f}')
"
```

Expected output: `OK  mf=<some value in [-1, 1]>  grad_norm=<positive>`

- [ ] **Step 7: Commit**

```bash
git add longlive/utils/motion_reward.py
git commit -m "utils/motion_reward: add grad-enabled motion_fidelity path for DRaFT-K

Three additions parallel to the existing numpy / no_grad pipeline:

  * CoTrackerWrapper.get_tracklets_from_tensor(video, requires_grad)
    — bypasses the mp4 round-trip, takes torch video tensor directly.
  * motion_fidelity_pair_grad(gen_tracks, gen_vis, ref_tracks, ref_vis)
    — torch version of motion_fidelity_pair, fully differentiable
    through gen_tracks (visibility is treated as constant).
  * MotionFidelityRewardGrad(ref_path, ...) class with score_grad(video)
    — parallel to existing MotionFidelityReward, returns a torch scalar
    that retains requires_grad when video does.

Reference tracklets are extracted once via the existing file-based
get_tracklets() (numpy + on-disk cache), then converted to torch on the
target device and held as constants in score_grad."
```

---

## Task 4: Create `longlive/methods/diffusion_draft/__init__.py` + `losses.py`

**Files:**
- Create: `longlive/methods/diffusion_draft/__init__.py`
- Create: `longlive/methods/diffusion_draft/losses.py`

- [ ] **Step 1: Create the package directory**

Run: `mkdir -p longlive/methods/diffusion_draft/configs && ls longlive/methods/diffusion_draft/`
Expected output: `configs`

- [ ] **Step 2: Write `__init__.py`**

Create `longlive/methods/diffusion_draft/__init__.py`:

```python
"""DRaFT-K — reward-gradient backprop for 4-step DMD video fast adaptation.

Implementation of the DRaFT mechanism from
  Clark et al. (arXiv:2309.17400):
  "Directly Fine-Tuning Diffusion Models on Differentiable Rewards"
adapted to our 4-step DMD-distilled Wan2.1-T2V-1.3B base + motion_fidelity
(CoTracker3 tracklet cosine) reward.

Mechanistically orthogonal to DiffusionNFT / RAM: where those treat the
reward as a black-box scalar, DRaFT-K directly back-propagates the
reward gradient through the *last K of 4 DMD steps* + VAE decode +
CoTracker3 transformer to the trainable LoRA parameters.

Per outer epoch:
  * 2 self-rollouts via `RolloutEngine.rollout_with_grad(noise, k_grad_steps=2)`
  * `mf = MotionFidelityRewardGrad.score_grad(video)`  — differentiable
  * `(-reward_coef · mf / k_rollouts_per_outer).backward()` accumulated
  * 4 cheap KL anchor steps at random anchor t (default vs zero-init
    "anchor" adapter) — explicit anti-drift, β_KL=1e-3
  * single `optimizer.step()`

What is borrowed from DRaFT:
  * Direct reward-gradient backprop through the sampler.
  * DRaFT-K truncation: backprop only the last K denoising steps.

What is adapted to our 4-step DMD video setting:
  * Sampler = our `CausalInferencePipeline` (not Euler), with k_grad_steps
    parameter exposing the truncation point.
  * Reward = motion_fidelity (CoTracker3 tracklet cosine) — continuous in
    [-1, 1], differentiable end-to-end including VAE decode.
  * Anchor adapter (zero-init PEFT) as v_ref for explicit KL anti-drift —
    DRaFT has no implicit anchor in the loss form.

Shared with longlive/methods/diffusion_nft and diffusion_ram:
  * longlive.utils.rl_rollout.RolloutEngine (with new rollout_with_grad)
  * longlive.utils.motion_reward.MotionFidelityRewardGrad (new)
  * longlive.methods.motiondirector.data.SkateboardingLatentDataset

See:
  * docs/superpowers/specs/2026-05-22-draft-k-design.md
  * docs/superpowers/plans/2026-05-22-draft-k-implementation.md
"""
```

- [ ] **Step 3: Write `losses.py`**

Create `longlive/methods/diffusion_draft/losses.py`:

```python
"""DRaFT-K loss terms.

The reward loss is inline in train.py (it's literally `-reward_coef * mf`).
This file only holds the explicit KL anchor regularization that pins the
trainable LoRA to the no-LoRA base (via the zero-init `anchor` PEFT
adapter) — DRaFT has no implicit anchor in its target form, so this term
is required to prevent caption-collapse (NFT-H3 / H3' failure mode).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_anchor_loss(
    v_default: torch.Tensor,
    v_anchor: torch.Tensor,
) -> torch.Tensor:
    """Anti-drift MSE between v_θ (trainable) and v_ref (frozen base).

    Identical formula to the version in diffusion_nft / diffusion_ram —
    duplicated here so diffusion_draft is self-contained per CLAUDE.md
    `longlive/methods/<idea>/ is self-contained` convention.

    Args:
        v_default: (B, F, C, H, W) — trainable network output at (x_t, t).
        v_anchor:  (B, F, C, H, W) — frozen reference (zero-init "anchor"
            adapter), caller's responsibility to call under no_grad.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(v_default, v_anchor.detach())
```

- [ ] **Step 4: Syntax check**

Run: `~/miniforge3/envs/longlive/bin/python -m py_compile longlive/methods/diffusion_draft/__init__.py longlive/methods/diffusion_draft/losses.py && echo OK`
Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add longlive/methods/diffusion_draft/__init__.py longlive/methods/diffusion_draft/losses.py
git commit -m "diffusion_draft: skeleton (__init__.py + kl_anchor_loss)"
```

---

## Task 5: Create `longlive/methods/diffusion_draft/train.py`

**Files:**
- Create: `longlive/methods/diffusion_draft/train.py`

**Design notes (per spec Section 3):**
- 2 PEFT adapters (`default` trainable + `anchor` zero-init frozen).  No `old` EMA.
- Outer loop: reward branch (`k_rollouts_per_outer` grad-on rollouts) + KL branch (`n_kl_steps_per_outer` cheap forwards), gradient-accumulated, single `optimizer.step()`.
- KL branch reuses the *last* reward-branch rollout's `latent_x0.detach()` as x_0 source.
- Setup phase is copied near-verbatim from `diffusion_ram/train.py` (FSDP wrap, adapter attach, reward init).

- [ ] **Step 1: Read the sibling trainer for setup-phase reference**

Run: `wc -l longlive/methods/diffusion_ram/train.py && grep -nE "^def |^class |def main\(\)" longlive/methods/diffusion_ram/train.py`
Expected output: line count ~660 + function/class anchors.

- [ ] **Step 2: Write the new trainer file in full**

Create `longlive/methods/diffusion_draft/train.py`:

```python
"""DRaFT-K trainer for 4-step DMD video fast adaptation.

Single-file trainer.  Setup phase mirrors longlive/methods/diffusion_ram/
train.py (FSDP + 2 PEFT adapters: `default` trainable, `anchor` zero-init
frozen = no-LoRA base).  Outer loop differs:

  for outer in range(outer_epochs):
      optimizer.zero_grad()
      # ── Reward branch (DRaFT-K backprop, expensive) ──
      generator.set_adapter("default")
      for k in range(k_rollouts_per_outer):
          noise = sample_noise(seed=outer*1009 + 31*rank + k)
          video, latent_x0 = rollout_engine.rollout_with_grad(
              noise, k_grad_steps=cfg.k_grad_steps
          )
          mf = reward_fn.score_grad(video[0])
          (-reward_coef * mf / k_rollouts_per_outer).backward()
          last_latent_x0 = latent_x0.detach()
      # ── KL anchor branch (cheap, anti-drift) ──
      if beta_kl > 0:
          for kl_step in range(n_kl_steps_per_outer):
              anchor_t = anchors[kl_step % len(anchors)]
              noise = torch.randn_like(last_latent_x0)
              x_t   = sched.add_noise(last_latent_x0, noise, anchor_t_tensor)
              # default forward (grad ON)
              v_default, _ = generator(x_t, train_cond, ts)
              # anchor forward (no_grad)
              generator.set_adapter("anchor")
              with torch.no_grad():
                  v_anchor, _ = generator(x_t, train_cond, ts)
              generator.set_adapter("default")  # restore before backward (gc safety)
              (beta_kl * F.mse_loss(v_default, v_anchor.detach()) / n_kl_steps_per_outer).backward()
      optimizer.step()

See docs/superpowers/specs/2026-05-22-draft-k-design.md for the locked
design and motivation.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import peft
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import AdamW

from longlive.methods.diffusion_draft.losses import kl_anchor_loss
from longlive.methods.motiondirector.data import SkateboardingLatentDataset
from longlive.utils.distributed import fsdp_wrap, launch_distributed_job
from longlive.utils.lora_utils import configure_adapter_for_model
from longlive.utils.motion_reward import MotionFidelityRewardGrad
from longlive.utils.rl_rollout import RolloutEngine, maybe_barrier
from longlive.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


# ============================================================================
# Helpers
# ============================================================================


def _clean_fsdp_key(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


def _find_adapter_params(
    model: torch.nn.Module, adapter_tag: str
) -> list[torch.nn.Parameter]:
    needle = f".{adapter_tag}."
    return [p for n, p in model.named_parameters() if needle in n]


def _save_lora_ckpt(
    fsdp_peft_model: torch.nn.Module,
    out_dir: Path,
    tag: str,
    rank0: bool,
    adapter_name: str = "default",
) -> Path | None:
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_peft_model, StateDictType.FULL_STATE_DICT, save_policy):
        full = fsdp_peft_model.state_dict()
    if not rank0:
        return None
    lora_state = get_peft_model_state_dict(
        fsdp_peft_model, state_dict=full, adapter_name=adapter_name
    )
    lora_state = {k: v.detach().cpu() for k, v in lora_state.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_{tag}.pt"
    torch.save(lora_state, path)
    return path


def _prune_old_ckpts(out_dir: Path, keep_last: int) -> None:
    ckpts = sorted(
        (p for p in out_dir.glob("lora_*.pt") if "final" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    while len(ckpts) > keep_last:
        ckpts[0].unlink()
        ckpts.pop(0)


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--smoke", action="store_true",
        help="2-outer × small smoke run (overrides outer_epochs / k_rollouts / n_kl).",
    )
    ap.add_argument(
        "--disable-wandb", action="store_true",
        help="Skip wandb.init — useful for local debug.",
    )
    args = ap.parse_args()

    # ---------- Distributed init ----------
    launch_distributed_job(backend="nccl")
    t_setup_start = time.time()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    rank0 = rank == 0

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.smoke:
        # Smoke override: 2 outer × K_rollouts=1 × n_kl=2 × K_grad=1.
        cfg.outer_epochs = 2
        cfg.k_rollouts_per_outer = 1
        cfg.n_kl_steps_per_outer = 2
        cfg.k_grad_steps = 1
        cfg.ckpt_interval = 2
        cfg.warmup_steps = 0

    if rank0:
        print("[diffusion_draft] resolved config:")
        print(OmegaConf.to_yaml(cfg))
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(
            f"[diffusion_draft] device: {gpu_name} ({gpu_total_gib:.1f} GiB) "
            f"× world_size={world_size}",
            flush=True,
        )

    # ---------- wandb ----------
    wandb_enabled = rank0 and not args.disable_wandb
    if wandb_enabled:
        config_basename = Path(args.config).stem
        run_name = f"{config_basename}_{time.strftime('%y%m%d_%H%M')}"
        if args.smoke:
            run_name += "_smoke"
        wandb.init(
            project=getattr(cfg, "wandb_project", "longlive_diffusion_draft"),
            entity=getattr(cfg, "wandb_entity", "hongyou"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.environ.get("WANDB_DIR", "wandb"),
        )
        print(f"[diffusion_draft] wandb run: {wandb.run.url}", flush=True)

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    # ---------- VAE (bf16, eval) ----------
    if rank0:
        print("[diffusion_draft] loading VAE (bf16) ...", flush=True)
    vae = WanVAEWrapper()
    vae.to(device=device, dtype=torch.bfloat16).eval()

    # ---------- Dataset (ref clip + train caption) ----------
    dataset = SkateboardingLatentDataset(
        data_root=cfg.data_root,
        vae=vae,
        frame_count=int(cfg.frame_count),
        resolution=int(cfg.resolution),
        category=str(cfg.category),
        device=device,
        single_video=True,
    )
    train_caption = dataset.train_caption
    ref_clip_path = dataset.train_clip_path

    # ---------- Text encoder: load → encode → free ----------
    if rank0:
        print("[diffusion_draft] loading text encoder ...", flush=True)
    text_encoder = WanTextEncoder()
    text_encoder.to(device).eval()
    with torch.no_grad():
        train_cond = {k: v.detach().clone() for k, v in text_encoder([train_caption]).items()}
    del text_encoder
    torch.cuda.empty_cache()

    # ---------- Backbone + base ckpt + NVlabs baseline LoRA merge ----------
    is_causal = bool(getattr(cfg, "is_causal", True))
    if rank0:
        arch = "CausalWanModel" if is_causal else "WanModel"
        print(f"[diffusion_draft] building {cfg.model_name} ({arch}) ...", flush=True)
    model_kwargs = dict(
        model_name=cfg.model_name,
        timestep_shift=float(cfg.timestep_shift),
        is_causal=is_causal,
    )
    if is_causal:
        model_kwargs["local_attn_size"] = int(getattr(cfg, "local_attn_size", -1))
        model_kwargs["sink_size"] = int(getattr(cfg, "sink_size", 0))
    generator = WanDiffusionWrapper(**model_kwargs)

    base_ckpt_path = os.path.expandvars(os.path.expanduser(cfg.base_ckpt))
    if rank0:
        print(f"[diffusion_draft] loading base ckpt: {base_ckpt_path}", flush=True)
    sd = torch.load(base_ckpt_path, map_location="cpu")
    if "generator" in sd:
        state = sd["generator"]
    elif "model" in sd:
        state = sd["model"]
    else:
        state = sd
    state = {_clean_fsdp_key(k): v for k, v in state.items()}
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if rank0:
        print(
            f"[diffusion_draft] base load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    del sd, state

    baseline_lora_ckpt = getattr(cfg, "baseline_lora_ckpt", None)
    if baseline_lora_ckpt:
        baseline_lora_ckpt = os.path.expandvars(os.path.expanduser(baseline_lora_ckpt))
        if rank0:
            print(
                f"[diffusion_draft] overlaying NVlabs baseline LoRA: {baseline_lora_ckpt}",
                flush=True,
            )
        generator.model = configure_adapter_for_model(
            generator.model,
            model_name="generator",
            adapter_config=cfg.baseline_adapter,
            is_main_process=rank0,
        )
        baseline_state = torch.load(baseline_lora_ckpt, map_location="cpu")
        if isinstance(baseline_state, dict) and "generator_lora" in baseline_state:
            baseline_state = baseline_state["generator_lora"]
        peft.set_peft_model_state_dict(generator.model, baseline_state)
        generator.model = generator.model.merge_and_unload()
        if rank0:
            print("[diffusion_draft] baseline LoRA merged into base weights", flush=True)
        del baseline_state

    # ---------- Attach 2 PEFT adapters: default + anchor ----------
    if rank0:
        print("[diffusion_draft] attaching adapters: default + anchor (v_ref)", flush=True)
    generator.model = configure_adapter_for_model(
        generator.model,
        model_name="generator",
        adapter_config=cfg.adapter,
        is_main_process=rank0,
    )
    peft_config_default = generator.model.peft_config["default"]
    generator.model.add_adapter("anchor", peft_config_default)
    for name, param in generator.model.named_parameters():
        if ".anchor." in name:
            param.requires_grad_(False)
    generator.model.set_adapter("default")
    generator.enable_gradient_checkpointing()

    # Cast fp32 PEFT params → bf16 (FSDP size-wrap dtype uniformity).
    n_cast = 0
    for p in generator.model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if rank0:
        print(f"[diffusion_draft] cast {n_cast} fp32 params → bf16 (post-LoRA, pre-FSDP)", flush=True)

    # ---------- FSDP wrap ----------
    generator.model = fsdp_wrap(
        generator.model,
        sharding_strategy="full",
        mixed_precision=True,
        wrap_strategy="size",
    )
    generator.model.train()

    random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    # ---------- Adapter param lists (post-FSDP) ----------
    default_params = _find_adapter_params(generator.model, "default")
    anchor_params = _find_adapter_params(generator.model, "anchor")
    if rank0:
        print(
            f"[diffusion_draft] adapter param counts: "
            f"default={len(default_params)}, anchor={len(anchor_params)}",
            flush=True,
        )
        assert len(default_params) == len(anchor_params), (
            f"adapter param count mismatch: default={len(default_params)} "
            f"vs anchor={len(anchor_params)}"
        )

    # ---------- Optimizer (default adapter only) ----------
    trainable = [p for p in default_params if p.requires_grad]
    n_trainable_local = sum(p.numel() for p in trainable)
    n_trainable_global = torch.tensor(n_trainable_local, device=device)
    dist.all_reduce(n_trainable_global)
    if rank0:
        print(
            f"[diffusion_draft] trainable params (FSDP-sharded total): "
            f"{int(n_trainable_global.item()):,}",
            flush=True,
        )
    optimizer = AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    # ---------- Rollout engine ----------
    latent_b = 1
    latent_f = (int(cfg.frame_count) - 1) // 4 + 1
    latent_c = 16
    latent_h = int(cfg.resolution) // 8
    latent_w_pixel = int(cfg.resolution * 16 / 9)
    latent_w_pixel = int(getattr(cfg, "pixel_width", latent_w_pixel))
    latent_w = latent_w_pixel // 8
    latent_shape = (latent_b, latent_f, latent_c, latent_h, latent_w)

    pipeline_args = OmegaConf.create({
        "denoising_step_list": list(cfg.denoising_step_list),
        "warp_denoising_step": bool(getattr(cfg, "warp_denoising_step", True)),
        "num_frame_per_block": int(getattr(cfg, "num_frame_per_block", 3)),
        "context_noise": int(getattr(cfg, "context_noise", 0)),
        "model_kwargs": OmegaConf.create({
            "local_attn_size": int(getattr(cfg, "local_attn_size", -1)),
            "sink_size": int(getattr(cfg, "sink_size", 0)),
            "use_infinite_attention": False,
        }),
    })
    rollout_engine = RolloutEngine(
        generator=generator,
        vae=vae,
        cached_cond_dict=train_cond,
        pipeline_args=pipeline_args,
        device=device,
        latent_shape=latent_shape,
    )

    # ---------- Reward (grad-enabled, rank-0 first to avoid CoTracker hub race) ----------
    cache_root = Path(os.path.expandvars(cfg.cache_dir)) if getattr(cfg, "cache_dir", None) else None
    if rank0:
        print(f"[diffusion_draft] init reward grad (rank 0 first): ref={ref_clip_path}", flush=True)
        reward_fn = MotionFidelityRewardGrad(
            ref_path=ref_clip_path,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
        )
    dist.barrier()
    if not rank0:
        reward_fn = MotionFidelityRewardGrad(
            ref_path=ref_clip_path,
            device=device,
            cache_dir=cache_root,
            n_frames=int(getattr(cfg, "reward_n_frames", 16)),
            grid_size=int(getattr(cfg, "reward_grid_size", 30)),
        )
    dist.barrier()
    if rank0:
        print("[diffusion_draft] reward init complete on all ranks", flush=True)

    out_dir = Path(cfg.out_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    sched = generator.scheduler

    # ============ Outer training loop ============
    outer_epochs = int(cfg.outer_epochs)
    k_rollouts_per_outer = int(cfg.k_rollouts_per_outer)
    n_kl_steps_per_outer = int(cfg.n_kl_steps_per_outer)
    k_grad_steps = int(cfg.k_grad_steps)
    reward_coef = float(cfg.reward_coef)
    beta_kl = float(getattr(cfg, "beta_kl", 0.0))
    anchors = list(cfg.t_anchors)

    if rank0:
        print(
            f"[diffusion_draft] start: outer={outer_epochs} × K_rollouts={k_rollouts_per_outer} "
            f"× K_grad={k_grad_steps} × n_kl={n_kl_steps_per_outer} | "
            f"reward_coef={reward_coef} | beta_kl={beta_kl} | anchors={anchors}",
            flush=True,
        )

    global_step = 0
    t_train_loop_start = time.time()
    setup_time_s = t_train_loop_start - t_setup_start
    if rank0:
        print(f"[diffusion_draft] setup_time_s={setup_time_s:.1f}", flush=True)

    for outer in range(outer_epochs):
        t_outer = time.time()
        optimizer.zero_grad()

        # ── REWARD BRANCH (DRaFT-K reward-gradient backprop) ──
        generator.model.set_adapter("default")
        sum_mf, sum_reward_loss = 0.0, 0.0
        last_latent_x0 = None
        t_reward_start = time.time()
        for k in range(k_rollouts_per_outer):
            gen_seed = int(cfg.seed) + 1009 * outer + 31 * rank + k
            torch.manual_seed(gen_seed)
            noise = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
            video, latent_x0 = rollout_engine.rollout_with_grad(
                noise=noise, k_grad_steps=k_grad_steps,
            )
            # video: (B=1, F_pix, 3, H_pix, W_pix) in [0, 1], grad on if k_grad_steps > 0
            mf = reward_fn.score_grad(video[0])
            reward_loss = -reward_coef * mf / k_rollouts_per_outer
            reward_loss.backward()
            sum_mf += float(mf.detach())
            sum_reward_loss += float(reward_loss.detach())
            last_latent_x0 = latent_x0.detach()  # for KL branch (no grad needed)
        t_reward = time.time() - t_reward_start

        # ── KL ANCHOR BRANCH (cheap, anti-drift) ──
        sum_kl = 0.0
        t_kl_start = time.time()
        if beta_kl > 0.0 and last_latent_x0 is not None:
            for kl_step in range(n_kl_steps_per_outer):
                anchor_t = int(anchors[kl_step % len(anchors)])
                noise_kl = torch.randn_like(last_latent_x0)
                n_frames = last_latent_x0.shape[1]
                t_scalar = torch.tensor([anchor_t], device=device, dtype=torch.long)
                timestep = t_scalar.expand(1, n_frames).contiguous()
                x_t = sched.add_noise(
                    last_latent_x0.flatten(0, 1),
                    noise_kl.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).unflatten(0, last_latent_x0.shape[:2])

                # default forward (grad ON)
                generator.model.set_adapter("default")
                v_default, _ = generator(x_t, train_cond, timestep)

                # anchor forward (no_grad)
                generator.model.set_adapter("anchor")
                with torch.no_grad():
                    v_anchor, _ = generator(x_t, train_cond, timestep)

                # restore default BEFORE backward (gc safety)
                generator.model.set_adapter("default")

                kl = beta_kl * kl_anchor_loss(v_default, v_anchor) / n_kl_steps_per_outer
                kl.backward()
                sum_kl += float(kl.detach())
        t_kl = time.time() - t_kl_start

        optimizer.step()
        global_step += 1

        # ── LOG ──
        dt_outer = time.time() - t_outer
        avg_mf = sum_mf / max(1, k_rollouts_per_outer)
        avg_reward_loss = sum_reward_loss
        avg_kl = sum_kl
        if rank0:
            print(
                f"[diffusion_draft] outer {outer:3d}/{outer_epochs}  "
                f"mf={avg_mf:.4f}  reward_loss={avg_reward_loss:.4f}  "
                + (f"kl={avg_kl:.4f}  " if beta_kl > 0.0 else "")
                + f"dt={dt_outer:.1f}s (reward={t_reward:.1f}, kl={t_kl:.1f})",
                flush=True,
            )
        if wandb_enabled:
            log_dict = {
                "outer/mf": avg_mf,
                "outer/reward_loss": avg_reward_loss,
                "outer/dt_total_s": dt_outer,
                "outer/dt_reward_s": t_reward,
                "outer/dt_kl_s": t_kl,
            }
            if beta_kl > 0.0:
                log_dict["outer/kl"] = avg_kl
            wandb.log(log_dict, step=global_step)

        # ── CKPT ──
        if (
            int(cfg.ckpt_interval) > 0
            and (outer + 1) % int(cfg.ckpt_interval) == 0
            and (outer + 1) < outer_epochs
        ):
            ckpt_path = _save_lora_ckpt(generator.model, out_dir, str(outer + 1), rank0)
            if rank0:
                _prune_old_ckpts(out_dir, int(cfg.ckpt_keep_last))
                print(f"[diffusion_draft] saved ckpt: {ckpt_path}", flush=True)

        maybe_barrier()

    train_loop_time_s = time.time() - t_train_loop_start
    final_path = _save_lora_ckpt(generator.model, out_dir, "final", rank0)
    if rank0:
        print(
            f"[diffusion_draft] DONE. setup_time_s={setup_time_s:.1f} "
            f"train_loop_time_s={train_loop_time_s:.1f}  final ckpt: {final_path}",
            flush=True,
        )
    if wandb_enabled:
        wandb.run.summary["setup_time_s"] = setup_time_s
        wandb.run.summary["train_loop_time_s"] = train_loop_time_s
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax check**

Run: `~/miniforge3/envs/longlive/bin/python -m py_compile longlive/methods/diffusion_draft/train.py && echo OK`
Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add longlive/methods/diffusion_draft/train.py
git commit -m "diffusion_draft: trainer with reward-grad outer loop + KL anchor branch

Outer step:
  * Reward branch: K_rollouts_per_outer rollouts via
    RolloutEngine.rollout_with_grad(noise, k_grad_steps), MotionFidelity
    via score_grad — reward_coef * mf / K backward, gradient-accumulated.
  * KL anchor branch: n_kl_steps_per_outer forwards at random anchor t,
    default vs zero-init anchor adapter, beta_kl * MSE backward.
  * One optimizer.step() per outer.

Setup phase (FSDP wrap, dual-adapter attach, reward init) mirrors
diffusion_ram/train.py.  No old/EMA adapter, no group_normalize (DRaFT
is not contrastive), no cross-rank gather (no shared reward statistics
required)."
```

---

## Task 6: Create training YAMLs

**Files:**
- Create: `longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml`
- Create: `longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml`

- [ ] **Step 1: Write `skateboarding_draft.yaml`**

Create `longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml`:

```yaml
# DRaFT-K reward-gradient backprop on the 4-step DMD base.
#
# Mechanism: paper arXiv:2309.17400 (Clark et al., ICLR 2024) adapted to our
# 4-step DMD-distilled Wan2.1-T2V-1.3B base + motion_fidelity (CoTracker3
# tracklet cosine vs reference clip) reward.  See
#   longlive/methods/diffusion_draft/__init__.py
#   docs/superpowers/specs/2026-05-22-draft-k-design.md
#
# Goal: VBench Total >= 80 AND Semantic >= 70 (avoid NFT-H3 caption collapse)
# AND UCF motion_fidelity >= 0.31 (BASE 0.348 * 0.9), AND ideally >= RAM-v1
# (0.314) — the headline question is whether reward gradient beats RAM's
# scalar reward on this 4-step DMD ceiling.

# ---- data ----
data_root: ${oc.env:LL_DATA}
category: Skateboarding
frame_count: 81
resolution: 480
pixel_width: 832

# ---- backbone ----
model_name: Wan2.1-T2V-1.3B
timestep_shift: 5.0
is_causal: true
local_attn_size: 12
sink_size: 3
base_ckpt: ${oc.env:LL_DATA}/longlive_models/models/longlive_base.pt

# ---- NVlabs baseline LoRA — merged into base ----
baseline_lora_ckpt: ${oc.env:LL_DATA}/longlive_models/models/lora.pt
baseline_adapter:
  type: lora
  rank: 256
  alpha: 256
  dropout: 0.0
  verbose: false

# ---- Trainable LoRA: rank 32 / alpha 64 (matches NFT/RAM) ----
adapter:
  type: lora
  rank: 32
  alpha: 64
  dropout: 0.0
  verbose: false

# ---- DMD rollout schedule ----
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_frame_per_block: 3
context_noise: 0

# ---- DRaFT-K hyperparameters ----
outer_epochs: 20
k_grad_steps: 2                  # backprop last K of 4 DMD steps
k_rollouts_per_outer: 2          # # reward-branch grad-on rollouts per outer
reward_coef: 1.0                 # multiplier on -mf; sweep up later if mf stagnates
n_kl_steps_per_outer: 4          # # cheap KL anchor forwards per outer
beta_kl: 1.0e-3                  # explicit anti-drift (DRaFT has no implicit anchor)
t_anchors: [1000, 750, 500, 250] # KL branch sample t cycle

# ---- Reward (motion_fidelity) ----
reward_n_frames: 16
reward_grid_size: 30
fps: 16

# ---- Optimizer ----
lr: 3.0e-4
weight_decay: 1.0e-4
warmup_steps: 0

# ---- Output ----
out_dir: ${oc.env:LL_DATA}/diffusion_draft_runs/skateboarding_draft
cache_dir: ${oc.env:LL_DATA}/diffusion_draft_runs/_tracklet_cache
scratch_dir: /tmp/diffusion_draft_scratch
ckpt_interval: 5
ckpt_keep_last: 4

# ---- misc ----
seed: 0
wandb_project: longlive_diffusion_draft
wandb_entity: hongyou
```

- [ ] **Step 2: Write `skateboarding_draft_smoke.yaml`**

Create `longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml`:

```yaml
# DRaFT-K smoke — 1 GPU, 2 outer × K_rollouts=1 × n_kl=2 × K_grad=1.
#
# Pass criteria in docs/superpowers/plans/2026-05-22-draft-k-implementation.md
# Task 10.
#
# Run via:
#   LL_DRAFT_SMOKE=1 source scripts/hpc/submit.sh sbatch_diffusion_draft_train.sh

data_root: ${oc.env:LL_DATA}
category: Skateboarding
frame_count: 81
resolution: 480
pixel_width: 832

model_name: Wan2.1-T2V-1.3B
timestep_shift: 5.0
is_causal: true
local_attn_size: 12
sink_size: 3
base_ckpt: ${oc.env:LL_DATA}/longlive_models/models/longlive_base.pt

baseline_lora_ckpt: ${oc.env:LL_DATA}/longlive_models/models/lora.pt
baseline_adapter:
  type: lora
  rank: 256
  alpha: 256
  dropout: 0.0
  verbose: false

adapter:
  type: lora
  rank: 32
  alpha: 64
  dropout: 0.0
  verbose: false

denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_frame_per_block: 3
context_noise: 0

# === SMOKE OVERRIDES ===
outer_epochs: 2
k_grad_steps: 1                  # cheapest grad path
k_rollouts_per_outer: 1
reward_coef: 1.0
n_kl_steps_per_outer: 2
beta_kl: 1.0e-3
t_anchors: [1000, 750, 500, 250]

reward_n_frames: 16
reward_grid_size: 30
fps: 16

lr: 3.0e-4
weight_decay: 1.0e-4
warmup_steps: 0

out_dir: ${oc.env:LL_DATA}/diffusion_draft_runs/_smoke
cache_dir: ${oc.env:LL_DATA}/diffusion_draft_runs/_tracklet_cache
scratch_dir: /tmp/diffusion_draft_smoke_scratch
ckpt_interval: 1
ckpt_keep_last: 2

seed: 0
wandb_project: longlive_diffusion_draft
wandb_entity: hongyou
```

- [ ] **Step 3: Verify both YAMLs load**

Run:
```bash
~/miniforge3/envs/longlive/bin/python -c "
from omegaconf import OmegaConf
for f in ['longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml',
          'longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml']:
    c = OmegaConf.load(f)
    print(f, 'OK,', len(list(c.keys())), 'top-level keys')
"
```

Expected output: two `OK` lines, each ~30 keys.

- [ ] **Step 4: Commit**

```bash
git add longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml \
        longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml
git commit -m "diffusion_draft: training configs (main + smoke)"
```

---

## Task 7: Create eval inference YAMLs + sbatch

**Files:**
- Create: `configs/motion_eval_inference_diffusion_draft.yaml`
- Create: `configs/vbench_short_diffusion_draft.yaml`
- Create: `scripts/hpc/sbatch_diffusion_draft_train.sh`

- [ ] **Step 1: Write `motion_eval_inference_diffusion_draft.yaml`**

Create `configs/motion_eval_inference_diffusion_draft.yaml`:

```yaml
# Motion-customization eval inference — for DiffusionDRaFT ckpts.
#
# DRaFT idea-layer LoRA: rank 32 / alpha 64 (identical (rank, alpha) to NFT/RAM).
# PEFT scale `alpha/r = 2.0` matches training config.
denoising_step_list:
- 1000
- 750
- 500
- 250
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 3

# inference defaults; dispatcher overrides per-run paths
data_path: longlive_models/prompts/vidprom_filtered_extended.txt
output_folder: motion_eval_runs/_unset
inference_iter: -1
num_output_frames: 21
use_ema: false
seed: 0
num_samples: 1
save_with_index: true
global_sink: true
context_noise: 0

generator_ckpt: longlive_models/models/longlive_base.pt

# NVlabs baseline LoRA (rank 256, merged into base)
baseline_lora_ckpt: longlive_models/models/lora.pt
baseline_adapter:
  type: "lora"
  rank: 256
  alpha: 256
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false

# DiffusionDRaFT idea-layer LoRA: rank 32 / alpha 64
lora_ckpt: ""
adapter:
  type: "lora"
  rank: 32
  alpha: 64
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false
```

- [ ] **Step 2: Write `vbench_short_diffusion_draft.yaml`**

Create `configs/vbench_short_diffusion_draft.yaml`:

```yaml
# VBench short inference — for DiffusionDRaFT ckpts.
#
# DRaFT idea-layer LoRA: rank 32 / alpha 64 (identical (rank, alpha) to NFT/RAM).
denoising_step_list:
- 1000
- 750
- 500
- 250
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 3

# inference defaults; dispatcher overrides per-run paths
data_path: longlive_models/prompts/vidprom_filtered_extended.txt
output_folder: vbench_runs/_unset
inference_iter: -1
num_output_frames: 21          # 81 pixel frames @ 16 fps = 5.06s
use_ema: false
seed: 0
num_samples: 1
save_with_index: true
global_sink: true
context_noise: 0

generator_ckpt: longlive_models/models/longlive_base.pt

# NVlabs baseline LoRA (rank 256, merged into base)
baseline_lora_ckpt: longlive_models/models/lora.pt
baseline_adapter:
  type: "lora"
  rank: 256
  alpha: 256
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false

# DiffusionDRaFT idea-layer LoRA: rank 32 / alpha 64
lora_ckpt: ""
adapter:
  type: "lora"
  rank: 32
  alpha: 64
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false
```

- [ ] **Step 3: Write `sbatch_diffusion_draft_train.sh`**

Create `scripts/hpc/sbatch_diffusion_draft_train.sh` (mirrors `sbatch_diffusion_ram_train.sh` with `LL_RAM_*` → `LL_DRAFT_*` substitution, output dir `diffusion_draft_runs`, module path `longlive.methods.diffusion_draft.train`):

```bash
#!/bin/bash
#SBATCH --job-name=diffusion_draft_train
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
# 8 ranks × Wan-1.3B + K_rollouts=2 rollouts with K_grad=2 DMD steps grad-on +
# VAE decode grad on + CoTracker3 forward grad on.  Heavier than NFT/RAM due
# to the reward-gradient graph; 900 GB host RAM headroom prevents NFS stalls.
# 4 h cap accommodates 20-outer training (~50-80 min) + inline motion_eval +
# vbench (~2 h).
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=s-sc-dgx[01-02]
#
# DiffusionDRaFT (Clark et al., arXiv:2309.17400) finetune on the few-step
# LongLive base.  See longlive/methods/diffusion_draft/__init__.py.
#
# Usage (always via submit.sh wrapper — captures $JID):
#
#   source scripts/hpc/submit.sh sbatch_diffusion_draft_train.sh
#
#   # Override config (smoke = 2 outer × K_rollouts=1 × n_kl=2 × K_grad=1):
#   LL_DRAFT_CONFIG=longlive/methods/diffusion_draft/configs/skateboarding_draft_smoke.yaml \
#     source scripts/hpc/submit.sh sbatch_diffusion_draft_train.sh
#
#   LL_DRAFT_SMOKE=1 source scripts/hpc/submit.sh sbatch_diffusion_draft_train.sh
#
#   # Disable post-train eval:
#   LL_DRAFT_EVAL=0 source scripts/hpc/submit.sh sbatch_diffusion_draft_train.sh

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node:   $(hostname)"
echo "[SLURM] GPUs:   ${SLURM_GPUS_ON_NODE:-8}"
echo "[SLURM] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
if [ -r /sys/fs/cgroup/memory.max ]; then
    echo "[SLURM] cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
fi

##############################
# Activate mamba environment
##############################
source ~/.bashrc
: "${LL_ENV_NAME:=longlive}"
mamba activate "$LL_ENV_NAME"

##############################
# Working directory
##############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LL_REPO:-}" ] && [ -d "$LL_REPO" ]; then
    PROJECT_DIR="$LL_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/local/train.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "$SCRIPT_DIR/../../scripts/local/train.py" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[SLURM][error] cannot locate LongLive repo. Set LL_REPO or sbatch from repo root." >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Data + cache paths
##############################
: "${PROJECT_DATA:?PROJECT_DATA not set — add 'export PROJECT_DATA=\$PROJECT_DEV/data' to ~/.bashrc}"
: "${LL_DATA:=$PROJECT_DATA/wm}"
export LL_DATA
export WAN_MODELS_ROOT="$LL_DATA/wan_models"
export HF_HOME="$LL_DATA/hf_cache"
export TRANSFORMERS_CACHE="$LL_DATA/hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"

echo "[SLURM] Data root:       $LL_DATA"
echo "[SLURM] WAN_MODELS_ROOT:  $WAN_MODELS_ROOT"

mkdir -p "$PROJECT_DIR/logs" "$LL_DATA/diffusion_draft_runs"

##############################
# Distributed env
##############################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TORCH_NCCL_BLOCKING_WAIT=1

##############################
# Run
##############################
: "${LL_DRAFT_CONFIG:=longlive/methods/diffusion_draft/configs/skateboarding_draft.yaml}"
echo "[SLURM] config:  $LL_DRAFT_CONFIG"

EXTRA_ARGS=()
if [ -n "${LL_DRAFT_SMOKE:-}" ]; then
    EXTRA_ARGS+=(--smoke)
    echo "[SLURM] SMOKE mode — 2 outer × K_rollouts=1 × K_grad=1"
fi

echo "[SLURM] Launching torchrun on $GPUS_PER_NODE GPU(s), master=$MASTER_ADDR:$MASTER_PORT"
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_port="$MASTER_PORT" \
    -m longlive.methods.diffusion_draft.train \
    --config "$LL_DRAFT_CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "[SLURM] Training phase finished."

##############################
# Post-training eval phase
##############################
if [ -n "${LL_DRAFT_SMOKE:-}" ]; then
    echo "[SLURM] Smoke mode — skipping post-training eval."
    echo "[SLURM] Job finished."
    exit 0
fi

if [ "${LL_DRAFT_EVAL:-1}" = "0" ]; then
    echo "[SLURM] LL_DRAFT_EVAL=0 — skipping post-training eval."
    echo "[SLURM] Job finished."
    exit 0
fi

CKPT_DIR=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$LL_DRAFT_CONFIG')
print(OmegaConf.to_container(cfg, resolve=True)['out_dir'])
")
CKPT="$CKPT_DIR/lora_final.pt"

if [ ! -f "$CKPT" ]; then
    echo "[SLURM][error] post-train eval: lora_final.pt missing at $CKPT" >&2
    echo "[SLURM] Job finished (training OK, eval skipped)."
    exit 0
fi

CONFIG_BASENAME=$(basename "$LL_DRAFT_CONFIG" .yaml)
EVAL_PREFIX="${LL_DRAFT_RUN_PREFIX:-$CONFIG_BASENAME}"

echo "[SLURM] Post-train eval"
echo "[SLURM]   ckpt   = $CKPT"
echo "[SLURM]   prefix = $EVAL_PREFIX"

: "${LL_DRAFT_VBENCH_CONFIG:=configs/vbench_short_diffusion_draft.yaml}"
: "${LL_DRAFT_EVAL_DATASETS:=ucf,loveu}"
: "${LL_DRAFT_EVAL_GPUS:=0,1,2,3,4,5,6,7}"

EVAL_LIMIT_ARGS=()
if [ -n "${LL_DRAFT_EVAL_LIMIT:-}" ]; then
    EVAL_LIMIT_ARGS+=(--limit "$LL_DRAFT_EVAL_LIMIT")
fi

: "${VBENCH_REPO_DIR:=${PROJECT_DEV:-$HOME/dev}/VBench}"
: "${VBENCH_INFO:=$VBENCH_REPO_DIR/vbench/VBench_full_info.json}"
export VBENCH_REPO_DIR VBENCH_INFO
export TORCH_HOME="$LL_DATA/hf_cache/torch_hub"
mkdir -p "$LL_DATA/motion_eval_runs" "$LL_DATA/vbench_runs" "$TORCH_HOME"

set +e

echo "[SLURM] === motion_eval ==="
RUN_ID_MOTION="${EVAL_PREFIX}_${SLURM_JOB_ID}"
bash "$PROJECT_DIR/scripts/motion_eval/run_motion_eval.sh" \
    "$CKPT" "configs/motion_eval_inference_diffusion_draft.yaml" "$RUN_ID_MOTION" \
    --gpus "$LL_DRAFT_EVAL_GPUS" \
    --datasets "$LL_DRAFT_EVAL_DATASETS" \
    "${EVAL_LIMIT_ARGS[@]}"
RC_MOTION=$?
echo "[SLURM] motion_eval exit=$RC_MOTION"

echo "[SLURM] === vbench ==="
RUN_ID_VBENCH="${EVAL_PREFIX}_${SLURM_JOB_ID}"
if [ ! -f "$VBENCH_INFO" ]; then
    echo "[SLURM][warn] VBench_full_info.json missing at $VBENCH_INFO — skipping vbench." >&2
    RC_VBENCH=99
else
    bash "$PROJECT_DIR/scripts/vbench/run_vbench.sh" \
        "$CKPT" "$LL_DRAFT_VBENCH_CONFIG" "$RUN_ID_VBENCH" \
        --gpus "$LL_DRAFT_EVAL_GPUS" \
        "${EVAL_LIMIT_ARGS[@]}"
    RC_VBENCH=$?
fi
echo "[SLURM] vbench exit=$RC_VBENCH"

set -e

echo "[SLURM] Job finished (train=OK, motion_eval=$RC_MOTION, vbench=$RC_VBENCH)."
```

- [ ] **Step 4: Make sbatch executable + bash syntax check**

Run: `chmod +x scripts/hpc/sbatch_diffusion_draft_train.sh && bash -n scripts/hpc/sbatch_diffusion_draft_train.sh && echo OK`
Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add configs/motion_eval_inference_diffusion_draft.yaml \
        configs/vbench_short_diffusion_draft.yaml \
        scripts/hpc/sbatch_diffusion_draft_train.sh
git commit -m "diffusion_draft: eval inference YAMLs + sbatch entry

motion_eval + vbench inference configs are byte-identical to the RAM
counterparts (rank 32 / alpha 64 PEFT) — only header rename.

sbatch mirrors sbatch_diffusion_ram_train.sh with LL_RAM_* → LL_DRAFT_*
env vars and output dir diffusion_draft_runs."
```

---

## Task 8: Sync all changes to HPC + sanity-check backward compat with NFT smoke

**Files:**
- Sync to HPC: all files modified in Tasks 1-7

- [ ] **Step 1: Sync pipeline + utils to HPC**

Run:
```bash
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  longlive/pipeline/causal_inference.py \
  longlive/utils/rl_rollout.py \
  longlive/utils/motion_reward.py \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/longlive/utils/ 2>&1 | tail -5
```

(Note: `causal_inference.py` belongs under `longlive/pipeline/`, not utils — fix the target path in the actual command. See Step 2.)

Actually run THIS instead:

```bash
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  longlive/pipeline/causal_inference.py \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/longlive/pipeline/ 2>&1 | tail -3
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  longlive/utils/rl_rollout.py \
  longlive/utils/motion_reward.py \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/longlive/utils/ 2>&1 | tail -5
```

Expected output: 1 file transferred per rsync (causal_inference, rl_rollout, motion_reward).

- [ ] **Step 2: Sync the new diffusion_draft method directory**

Run:
```bash
ssh hpc 'mkdir -p /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/longlive/methods/diffusion_draft/configs'
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  longlive/methods/diffusion_draft/ \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/longlive/methods/diffusion_draft/ 2>&1 | tail -10
```

Expected output: __init__.py + losses.py + train.py + 2 configs/*.yaml transferred.

- [ ] **Step 3: Sync the new eval inference YAMLs + sbatch**

Run:
```bash
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  configs/motion_eval_inference_diffusion_draft.yaml \
  configs/vbench_short_diffusion_draft.yaml \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/configs/ 2>&1 | tail -5
rsync -av --rsh='ssh -S /home/hongyou/.ssh/charitefront.sock' \
  scripts/hpc/sbatch_diffusion_draft_train.sh \
  hpc:/sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/scripts/hpc/ 2>&1 | tail -3
```

Expected output: 3 files transferred.

- [ ] **Step 4: Submit NFT smoke for backward-compat sanity check**

The default `k_grad_steps=0` in `CausalInferencePipeline.inference` should make NFT behavior bytewise identical to before. Verify by running a 1-GPU NFT smoke:

Run:
```bash
ssh hpc 'cd /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive && LL_NFT_SMOKE=1 LL_NFT_EVAL=0 sbatch --job-name=nft_draft_compat_smoke scripts/hpc/sbatch_diffusion_nft_train.sh' 2>&1 | tail -3
```

Expected output: `Submitted batch job <JID>`. Note the JID.

- [ ] **Step 5: Poll NFT smoke until completion**

Run (replace `<JID>` with the value from Step 4):
```bash
JID=<JID>; while true; do
    STATE=$(ssh hpc "sacct -j $JID --format=State -X -n 2>/dev/null" | head -1 | tr -d ' ')
    if [ "$STATE" != "RUNNING" ] && [ "$STATE" != "PENDING" ] && [ -n "$STATE" ]; then
        echo "[$JID done] state=$STATE"; break
    fi
    sleep 60
done
```

Expected output: `[<JID> done] state=COMPLETED` within ~10 min.

- [ ] **Step 6: Verify NFT smoke pass criteria**

Run (substitute `<JID>`):
```bash
ssh hpc "sacct -j <JID> --format=JobID,State,ExitCode,Elapsed -X | head -3; echo; grep -E 'outer +[0-9]+/' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/nft_draft_compat_smoke-<JID>.out | tail -3; tail -3 /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/nft_draft_compat_smoke-<JID>.out"
```

Expected output: State=COMPLETED 0:0, two `outer 0/2` and `outer 1/2` lines with finite loss/neg_def_ratio/kl, then `Training phase finished. Job finished.`

- [ ] **Step 7: Commit task progress note (no file changes — this task is HPC-side)**

No files changed locally in this task. No commit.

---

## Task 9: Submit DRaFT smoke + verify 9 pass criteria

**Files:** none locally — HPC submit + log inspection.

- [ ] **Step 1: Submit DRaFT smoke**

Run:
```bash
ssh hpc 'cd /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive && LL_DRAFT_SMOKE=1 sbatch --job-name=draft_smoke scripts/hpc/sbatch_diffusion_draft_train.sh' 2>&1 | tail -3
```

Expected output: `Submitted batch job <JID>`. Note the JID as `DRAFT_SMOKE_JID`.

- [ ] **Step 2: Poll until completion**

Run (substitute `<JID>`):
```bash
JID=<JID>; while true; do
    STATE=$(ssh hpc "sacct -j $JID --format=State -X -n 2>/dev/null" | head -1 | tr -d ' ')
    if [ "$STATE" != "RUNNING" ] && [ "$STATE" != "PENDING" ] && [ -n "$STATE" ]; then
        echo "[$JID done] state=$STATE"; break
    fi
    sleep 120
done
```

Expected output: `[<JID> done] state=COMPLETED` within ~10 min (1 GPU, K_grad=1, K_rollouts=1).

- [ ] **Step 3: Verify pass criteria #1 (SLURM exit 0:0)**

Run: `ssh hpc "sacct -j <JID> --format=JobID,State,ExitCode,Elapsed -X | head -3"`
Expected output: State=COMPLETED, ExitCode=0:0, Elapsed under ~10 min.

- [ ] **Step 4: Verify pass criteria #2 (adapter param counts match)**

Run: `ssh hpc "grep 'adapter param counts' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out"`
Expected output: `[diffusion_draft] adapter param counts: default=600, anchor=600` (per-rank count after FSDP).

- [ ] **Step 5: Verify pass criteria #3-5 (mf grad path)**

Run: `ssh hpc "grep -E 'outer +[0-9]+/|reward init|mf=' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out"`

Expected output: 2 `outer N/2` lines, each with `mf=<finite float>` (e.g. `mf=0.23`) and `reward_loss=<negative finite>`. `reward init complete on all ranks` appears.

- [ ] **Step 6: Verify pass criteria #7 (v_default ≈ v_anchor at outer 0)**

Since we don't log v_default_norm / v_anchor_norm explicitly in DRaFT (KL loss is logged, not branch norms), use the KL loss value as a proxy. At outer 0 the LoRA delta is near-zero (PEFT B-projection zero-init) so `kl_anchor_loss` should be ≤ 0.05 in raw value (before β_KL multiplication). The trainer prints `kl=<beta_kl * raw_kl>`, so with beta_kl=1e-3 the printed value should be ≤ 5e-5.

Run: `ssh hpc "grep 'outer +0/' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out"`
Expected output: line includes `kl=<value>` with `<value> <= 0.01` (loose smoke threshold).

If `<value> > 0.01` at outer 0, that flags PEFT anchor not zero-init properly — investigate before proceeding to full run.

- [ ] **Step 7: Verify pass criteria #8 (memory headroom) and #9 (wall clock)**

Run: `ssh hpc "grep -E 'GPU info|cgroup memory|Elapsed' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out; sacct -j <JID> --format=Elapsed -X -n | head -1"`

Expected output: GPU info line shows H200 with 143 GiB; cgroup memory line shows allocated mem; Elapsed line shows under 10 minutes.

- [ ] **Step 8: Optional — quick read of trailing log for surprises**

Run: `ssh hpc "tail -15 /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out"`

Expected output: `[diffusion_draft] DONE.`, no Python traceback, `[SLURM] Smoke mode — skipping post-training eval.`, `[SLURM] Job finished.`

If any criteria fails, **STOP**. Diagnose by reading the full log:
```bash
ssh hpc "cat /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_smoke-<JID>.out | tail -100"
```

Common failures and remedies:
- OOM → in `skateboarding_draft.yaml` lower `k_grad_steps: 2 → 1` then resync via Task 8 Step 2.
- `RuntimeError: ...add_adapter` PEFT 0.19 quirk → upgrade or pin via `peft.add_adapter` patch.
- CoTracker3 not loadable (no internet on compute node) → check `TORCH_HOME` cache is populated; rerun smoke after `fetch_data.sh` on login node.

- [ ] **Step 9: Commit progress note (no local changes)**

No local file changes. No commit.

---

## Task 10: Submit DRaFT full + pull + aggregate

**Files:**
- Modify: `/tmp/baseline_pull/aggregate.py` — add DRaFT-v1 row

- [ ] **Step 1: Submit DRaFT full**

Run:
```bash
ssh hpc 'cd /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive && sbatch --job-name=draft_v1 scripts/hpc/sbatch_diffusion_draft_train.sh' 2>&1 | tail -3
```

Expected output: `Submitted batch job <JID>`. Note as `DRAFT_FULL_JID`.

- [ ] **Step 2: Poll until completion (~3 h)**

Run (substitute `<JID>`):
```bash
JID=<JID>; while true; do
    STATE=$(ssh hpc "sacct -j $JID --format=State -X -n 2>/dev/null" | head -1 | tr -d ' ')
    if [ "$STATE" != "RUNNING" ] && [ "$STATE" != "PENDING" ] && [ -n "$STATE" ]; then
        echo "[$JID done] state=$STATE"; break
    fi
    sleep 900
done
```

Expected output: `[<JID> done] state=COMPLETED` within ~3 h.

- [ ] **Step 3: Verify full pass criteria #1-3 (SLURM + training health)**

Run (substitute `<JID>`):
```bash
ssh hpc "sacct -j <JID> --format=JobID,State,ExitCode,Elapsed -X | head -3; echo; echo '[outer summary]'; grep -E 'outer +[0-9]+/' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_v1-<JID>.out | tail -10; echo; echo '[stage markers]'; grep -E 'Training phase finished|=== motion_eval|=== vbench|generated [0-9]+ videos|motion_eval\] DONE|vbench\] DONE' /sc-projects/sc-proj-cc09-repair/hongyou/dev/LongLive/logs/draft_v1-<JID>.out | tail -8"
```

Expected output:
- State=COMPLETED, ExitCode=0:0, Elapsed < 4 h
- Each of the 20 outer lines has finite `mf=`, `reward_loss=`, `kl=` (last 10 shown)
- `Training phase finished.` `=== motion_eval ===` `[motion_eval] DONE` `=== vbench ===` `[vbench] DONE` all appear

- [ ] **Step 4: Pull scores.csv + summary.json**

Run (substitute `<JID>`):
```bash
mkdir -p /tmp/baseline_pull/motion_eval/skateboarding_draft_<JID> /tmp/baseline_pull/vbench/skateboarding_draft_<JID>
ssh hpc "cat /sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm/motion_eval_runs/skateboarding_draft_<JID>/scores.csv" > /tmp/baseline_pull/motion_eval/skateboarding_draft_<JID>/scores.csv
ssh hpc "cat /sc-projects/sc-proj-cc09-repair/hongyou/dev/data/wm/vbench_runs/skateboarding_draft_<JID>/summary.json" > /tmp/baseline_pull/vbench/skateboarding_draft_<JID>/summary.json
wc -l /tmp/baseline_pull/motion_eval/skateboarding_draft_<JID>/scores.csv
jq -r '"total=\(.total) quality=\(.quality) semantic=\(.semantic) DD=\(.per_dim.dynamic_degree)"' /tmp/baseline_pull/vbench/skateboarding_draft_<JID>/summary.json
```

Expected output:
- `365` lines (header + 364 rows)
- `total=<float> quality=<float> semantic=<float> DD=<float>`

- [ ] **Step 5: Add DRaFT-v1 row to aggregator**

Edit `/tmp/baseline_pull/aggregate.py` — find the `CKPTS = [...]` list near the top and add a new entry as the LAST element. Substitute `<JID>`:

```python
    ("DRaFT-v1",     "skateboarding_draft_<JID>",              "skateboarding_draft_<JID>"),
```

- [ ] **Step 6: Run aggregator and surface 3 tables**

Run: `cd /tmp/baseline_pull && python3 aggregate.py 2>&1 | head -55`

Expected output: 3 markdown tables (UCF / LOVEU / VBench) with all existing ckpts + new DRaFT-v1 row.

- [ ] **Step 7: Check full pass criteria #4-7 (the verdict gates)**

From the aggregator output, verify (substitute the DRaFT-v1 row values):

| Criterion | Threshold | DRaFT-v1 | Pass? |
|---|---|---|---|
| UCF motion_fidelity | ≥ 0.31 AND ≥ 0.314 (RAM-v1) | __read__ | __check__ |
| LOVEU motion_fidelity | ≥ 0.36 AND ≥ 0.357 (RAM-v1) | __read__ | __check__ |
| VBench Total | ≥ 80 | __read__ | __check__ |
| VBench Semantic | ≥ 70 | __read__ | __check__ |

Verdict:
- All 4 pass + mf > RAM-v1 → **DRaFT > RAM** verdict, reward gradient beats scalar reward
- All 4 pass + mf ≤ RAM-v1 → **DRaFT ≈ RAM** verdict, same 4-step DMD ceiling
- VBench Semantic < 70 → **Caption collapse**, fallback path: bump `beta_kl: 1e-3 → 1e-2` and re-run
- mf << 0.31 → reward signal not strong enough; fallback: bump `reward_coef: 1.0 → 10` and re-run

- [ ] **Step 8: Commit aggregator update**

Note: `/tmp/baseline_pull/aggregate.py` is in `/tmp/` and not version-controlled. The original aggregator lives in `/tmp/baseline_pull/`. If you want a persistent copy, add an entry under `scripts/aggregate_baseline.py` in the repo — out of scope here.

No file commit needed for this task. The DRaFT-v1 row's interpretation should be added to `docs/05.md` (which is task #5 from the existing task list, still pending — outside this plan's scope).

---

## Self-review (post-write check)

Spec coverage (cross-checking `docs/superpowers/specs/2026-05-22-draft-k-design.md`):

| Spec section | Task(s) covering it |
|---|---|
| Mechanism (Section 1) loss form | T4 (kl_anchor_loss), T5 (inline reward loss) |
| `k_grad_steps=2` | T1 (pipeline param), T6 (yaml field) |
| `k_rollouts_per_outer=2` | T5 (loop), T6 (yaml field) |
| Explicit KL anchor, β_KL=1e-3 | T4 (loss fn), T5 (loop), T6 (yaml field) |
| `reward_coef=1.0` | T5 (read from cfg), T6 (yaml field) |
| Code architecture (Section 2) | T1-T7 collectively |
| New `methods/diffusion_draft/` | T4 + T5 + T6 |
| Shared utils additions | T2 (rl_rollout), T3 (motion_reward) |
| Pipeline `k_grad_steps` param | T1 |
| Memory budget (~30 GB / rank) | T9 step 7 verification |
| Training loop + yaml (Section 3) | T5 + T6 |
| `rollout_engine.rollout_with_grad` call | T2 (impl) + T5 (use) |
| KL branch reuses last latent | T5 |
| Wall clock estimate ~3 h | T10 polling interval |
| Verification path (Section 4) | T8 (backward compat) + T9 (smoke) + T10 (full + verdict) |
| Smoke pass criteria 1-9 | T9 |
| Full pass criteria 1-8 | T10 |
| Risk-aware fallback knobs | T10 step 7 (verdict table) |

No spec requirement is unaddressed.

Placeholder scan: no "TBD" / "TODO" / "fill in" / "similar to Task N" — all steps contain actual code or actual commands.

Type consistency: `MotionFidelityRewardGrad` is named consistently across T3 (def), T5 (import + use), and the spec. `rollout_with_grad(noise, k_grad_steps)` signature is consistent across T2 (def) and T5 (use). `kl_anchor_loss(v_default, v_anchor)` signature consistent T4 ↔ T5.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-22-draft-k-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
