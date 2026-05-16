"""MotionDirector LoRA finetune on the few-step LongLive model.

Entry-point: ``longlive.methods.motiondirector.train`` (standalone trainer).
Loss: L_temporal_MSE + L_AD (alpha=sqrt(2), beta=1) in epsilon space via
B1 close-form reverse.

Framing — see docs/00.md §1.1 (research anchor, route 1) and docs/01.md
(two-LoRA layering + load flow + train.py step mapping).
"""
