"""MotionDirector LoRA finetune on the few-step LongLive model.

Entry-point: ``longlive.methods.motiondirector.train`` (standalone trainer).
Loss: L_temporal_MSE + L_AD (alpha=sqrt(2), beta=1) in epsilon space via
B1 close-form reverse — see docs/00.md §1 candidate idea 1.
"""
