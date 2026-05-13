"""Phase 3 (docs/04.md): DMD distillation with a Phase 2 teacher LoRA
attached to ``real_score``.

Subclasses ``longlive.trainer.distillation.Trainer`` (aka
``ScoreDistillationTrainer``) and overrides the L5 hook
``_attach_real_score_lora`` to attach + load + freeze a teacher LoRA
on ``self.model.real_score`` before FSDP wrap.

Reads from config:
  * ``real_score_adapter``     — adapter cfg dict (type / rank / alpha / ...)
  * ``real_score_lora_ckpt``   — path to Phase 2 ``teacher_lora_*.pt``
"""
from __future__ import annotations

import os

import peft
import torch

from longlive.trainer.distillation import Trainer as ScoreDistillationTrainer
from longlive.utils.lora_utils import configure_adapter_for_model


class MotionDirectorScoreDistillationTrainer(ScoreDistillationTrainer):
    """``ScoreDistillationTrainer`` + Phase 2 teacher LoRA on ``real_score``.

    The only override is the L5 hook (``_attach_real_score_lora``); the rest
    of the trainer (LoRA setup on student/critic, FSDP wraps, training loop,
    checkpoint save, DMD loss math) is inherited unchanged.

    Config schema (additions over base trainer's):
        real_score_adapter:
            type: lora                # or any registered adapter
            rank: 64                  # must match Phase 2 training-time rank
            alpha: 64
            dropout: 0.0
        real_score_lora_ckpt: ${oc.env:LL_DATA}/.../teacher_lora_final.pt
    """

    def _attach_real_score_lora(self):
        adapter_cfg = getattr(self.config, "real_score_adapter", None)
        lora_ckpt = getattr(self.config, "real_score_lora_ckpt", None)

        if adapter_cfg is None and lora_ckpt is None:
            # Subclass is selected but config didn't ask for real_score LoRA —
            # behave like base trainer.
            return
        if adapter_cfg is None or lora_ckpt is None:
            raise ValueError(
                "real_score_adapter and real_score_lora_ckpt must be set "
                f"together. adapter={adapter_cfg!r}, ckpt={lora_ckpt!r}"
            )

        if self.is_main_process:
            print(f"[motiondirector trainer] attaching LoRA to real_score: {adapter_cfg}")
        self.model.real_score.model = configure_adapter_for_model(
            self.model.real_score.model,
            "real_score",
            adapter_cfg,
            self.is_main_process,
        )

        # Match base trainer's post-LoRA fp32 → bf16 cast (distillation.py
        # lines 262-276 explanation: FSDP size-based auto-wrap refuses
        # mixed-dtype groups; PEFT initializes LoRA in fp32).
        if self.config.mixed_precision:
            n_cast = 0
            for p in self.model.real_score.parameters():
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
                    n_cast += 1
            if self.is_main_process:
                print(
                    f"[motiondirector trainer] cast {n_cast} real_score "
                    f"fp32 params to bfloat16 (post-LoRA, pre-FSDP)"
                )

        ckpt_path = os.path.expandvars(os.path.expanduser(lora_ckpt))
        if self.is_main_process:
            print(f"[motiondirector trainer] loading real_score LoRA from {ckpt_path}")
        lora_state = torch.load(ckpt_path, map_location="cpu")
        peft.set_peft_model_state_dict(self.model.real_score.model, lora_state)

        # Re-freeze: PEFT marks LoRA params requires_grad=True by default,
        # but real_score is the frozen DMD teacher — never trained.
        self.model.real_score.requires_grad_(False)
        if self.is_main_process:
            print("[motiondirector trainer] real_score LoRA loaded and frozen")
