"""MotionDirector teacher-finetune method.

Phase 2 (standalone trainer): ``longlive.methods.motiondirector.train``
produces a teacher LoRA from MotionDirector recipe (paper
L_temporal_MSE + L_AD) on Wan-14B.

Phase 3 (DMD distillation): ``MotionDirectorScoreDistillationTrainer``
subclass attaches the Phase 2 LoRA to ``real_score`` before standard
DMD distillation runs. Select via ``trainer: score_distillation_motiondirector``
in the YAML config.
"""
from longlive.trainer import register_trainer

from longlive.methods.motiondirector.trainer import (
    MotionDirectorScoreDistillationTrainer,
)


register_trainer(
    "score_distillation_motiondirector",
    MotionDirectorScoreDistillationTrainer,
)
