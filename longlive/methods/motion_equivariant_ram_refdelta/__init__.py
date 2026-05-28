"""Motion-Equivariant RAM RefDelta for fast motion distillation.

This method keeps RAM's efficient outer loop:

  1. sample a small group of on-policy few-step rollouts,
  2. score them with tracklet direction plus speed-ratio consistency,
  3. group-normalize the scalar rewards across ranks,
  4. run a short inner loop on noised rollout endpoints.

The departure from vanilla RAM is the update geometry.  Vanilla RAM multiplies
the scalar reward into a full latent velocity residual, so a high-reward rollout
can copy appearance, color, texture, and background details.  Motion-Equivariant
RAM instead applies reward only to a coarse temporal-delta representation of
predicted x0, while anchoring frame-shared/static velocity to the frozen base.

In short:

    reward decides which rollout motion to trust;
    the loss decides that only motion is allowed to move.
"""
