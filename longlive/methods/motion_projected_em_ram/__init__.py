"""Motion-Projected EM-RAM finetune on the 4-step DMD base.

Motion-Projected EM-RAM treats each outer epoch as a small empirical EM /
mirror-descent step:

  E-step:
    sample K on-policy DMD rollouts, score them with a tracklet motion reward,
    and form a reward-tilted endpoint distribution q_i proportional to
    exp(A_i / eta) under a KL(q || uniform) trust-region budget.

  M-step:
    distill that tilted distribution back into the few-step student using only
    a projected motion component of RAM's stable residual target.

The important design choice is that reward decides which endpoints receive a
RAM correction; reward does not control full velocity space:

    target = v_anchor + alpha_i * P_motion((eps - x0_i) - stopgrad(v_theta))

Low/average reward endpoints get alpha_i near 0 and therefore act as anchor
updates.  Static, appearance, and texture-scale velocity components stay
anchored unless the base model already supports them.
"""
