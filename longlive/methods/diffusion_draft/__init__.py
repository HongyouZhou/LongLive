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
