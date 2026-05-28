"""Reward-Tilted f-Divergence Flow-Map Distillation.

RT-fDMD-Flow is a conservative successor to the RAM fast-adaptation baseline.
It keeps the same deployment-friendly shape:

  * one trainable LoRA adapter on top of longlive_base + merged NVlabs LoRA;
  * one frozen zero-init "anchor" adapter representing the base velocity;
  * on-policy K-rollouts scored by motion_fidelity;
  * no full rollout backpropagation and no separate teacher copy.

The objective changes the interpretation of reward.  Instead of treating the
motion reward as a raw RL update, reward defines a tilted target distribution
inside the base model's support:

    p_tilt(x | c) proportional to p_base(x | c) exp(beta R_motion(x)).

The implementation uses a bounded JS-style density-ratio transform by default
and adds a local flow-map consistency term so the student transition from the
same noisy point stays close to the frozen base transition.
"""
