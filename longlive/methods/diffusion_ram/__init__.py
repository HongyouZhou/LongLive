"""Reinforce Adjoint Matching (RAM) finetune on the 4-step DMD base.

Implementation of the RAM mechanism from
  Bergmeister, Jegelka, Nüsken, Domingo-Enrich, Pidstrigach (arXiv:2605.10759):
  "Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and
   Flow-Matching Models"

RAM is an independently derived RL post-training method — NOT a DiffusionNFT
variant.  Its derivation chain (paper §3–4):

  KL-regularized reward maximization
    → tilted clean-endpoint distribution (Eq. 7)
    → controlled SDE + stochastic optimal control (Eq. 8–9)
    → adjoint-matching optimality condition (Eq. 12)
    → REINFORCE identity on the adjoint (Eq. 13–14)
    → closed-form Bayes bridge score (Prop 4.1 / Eq. 15)
    → RAM loss (Eq. 17, v-space)

What we lift verbatim from the paper:
  * Single MSE policy loss with the target-shift form (Eq. 17).
  * Frozen LongLive base velocity `v_anchor` as the anchor (no EMA self-mirror).
  * Reward enters as a multiplicative coefficient on the target shift,
    so r→0 collapses the target back to v_anchor (strict base anchor).
  * Analytical Gaussian noising `X_t = (1−σ)·X_0 + σ·ε` identical to the
    pretraining law — no SDE rollout in the loss path.

What we adapt to our 4-step DMD video setting (not validated by the paper):
  * Endpoint sampler = 4-step DMD `CausalInferencePipeline`, not 20-step Euler.
  * Training t restricted to the 4 DMD anchors {1000, 750, 500, 250} (in
    [0, 1000] scaled to t/1000 ∈ {1.0, 0.75, 0.5, 0.25}).
  * Reward = motion_fidelity (CoTracker3 tracklet cosine vs reference clip),
    continuous in [-1, 1] — paper uses [0, 1] partial-credit binary rewards.
  * Cross-rank `dist.all_gather` group_normalize over K×world_size rewards
    (carried over from the NFT-H1 fix, see docs/04.md addendum).
  * Frozen `v_anchor` is implemented as a zero-init PEFT adapter ("anchor"),
    not a separate copy of the base — sidesteps the FSDP × PEFT ×
    gradient_checkpointing `disable_adapter()` incompatibility.

Shared method-agnostic infrastructure:
  * `longlive.utils.rl_rollout.RolloutEngine` (method-agnostic K-rollout sampler)
  * `longlive.utils.motion_reward.MotionFidelityReward`
  * `longlive.utils.group_norm.group_normalize`
  * `longlive.data.motion_refs.SkateboardingLatentDataset`
    (only `train_caption` + `train_clip_path` used; not `sample()`)

See `/home/hongyou/.claude/plans/jaunty-pondering-wall.md` for the full plan.
"""
