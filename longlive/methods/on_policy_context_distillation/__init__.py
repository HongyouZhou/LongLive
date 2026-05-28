"""On-policy context distillation for few-step DMD video adaptation.

This method follows the core mechanism of On-Policy Context Distillation
(arXiv:2602.12275): sample from the current student policy, then distill a
frozen context teacher on the states the student actually visits.

For the 4-step LongLive DMD base we instantiate the reverse-KL signal as a
velocity-matching loss. With a fixed-variance Gaussian transition view, matching
teacher and student rectified-flow velocities is the practical first-order
surrogate:

    x_0 ~ student current 4-step sampler
    x_t = (1 - sigma_t) x_0 + sigma_t eps
    loss = || v_student(x_t, t, prompt)
              - stopgrad(v_teacher(x_t, t, teacher_context)) ||^2

The important distinction from the removed DiffusionNFT experiment is that
reward weighting and negative-aware beta interpolation are not the method's
center. The center is the on-policy state distribution plus a stronger frozen
teacher that carries context or experience.
"""
