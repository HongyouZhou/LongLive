"""EM-RAM finetune on the 4-step DMD base.

EM-RAM treats each outer epoch as a small empirical EM / mirror-descent step:

  E-step:
    sample K on-policy DMD rollouts, score them with motion_fidelity, and form
    a reward-tilted endpoint distribution q_i proportional to exp(A_i / eta) under a
    KL(q || uniform) trust-region budget.

  M-step:
    distill that tilted distribution back into the few-step student using
    RAM's stable residual target, not a direct self-endpoint flow target.

The important design choice is that reward decides which endpoints receive a
RAM correction; the velocity target geometry remains RAM:

    target = v_ref + alpha_i * ((eps - x0_i) - stopgrad(v_theta))

Low/average reward endpoints get alpha_i near 0 and therefore act as anchor
updates.  This is meant to preserve RAM-v1's non-collapse behavior while using
multiple samples per outer epoch more efficiently than scalar group-normalized
RAM.
"""
