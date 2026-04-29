"""Motion-DMD attention hooks for the 14B `real_score` teacher.

Provides `MotionAttnInjector`, a context manager that wraps a teacher's
`register_forward_hook` plumbing into two phases:

  1) ``capture`` — the next teacher forward pass over a V_ref noisy latent
     records the post-o-projection output of `blocks[i].self_attn` for each
     `i` in `block_idxs`.
  2) ``inject``  — the next teacher forward pass over the student's noisy
     latent has its `blocks[i].self_attn` output blended with the cached
     V_ref output: ``out = (1 - alpha) * out_student + alpha * out_v_ref``.

The hook contract is "return same shape, same device, same dtype" so FSDP's
asynchronous all-gather scheduler is undisturbed (see plan risk #4).

Per-block VRAM footprint while in inject mode is fixed by the cache: for the
14B teacher with 21-frame chunks it is `[B, 32760, 5120]` bf16 ≈ 335 MB / block.
With the default 3-block plan that is ≈ 1 GB peak; freed on context exit.

Also defines `MotionConfig`, the dataclass attached to BaseModel that carries
all the knobs the DMD branch consults: which blocks to hook, alpha (hook
strength), beta (score-blend strength) with linear warmup, inject_prob, and
the loaded V_ref latent cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch


class MotionAttnInjector:
    """Two-phase forward-hook attached to selected `WanAttentionBlock.self_attn`.

    Usage::

        injector = MotionAttnInjector(real_score.model, [18, 19, 20], alpha=0.5)
        with injector:
            injector.set_mode("capture")
            real_score(v_ref_noisy, ...)               # fills cache
            injector.set_mode("inject")
            _, pred_real_motion = real_score(student_noisy, ...)
    """

    VALID_MODES = (None, "capture", "inject")

    def __init__(
        self,
        real_score_model: torch.nn.Module,
        block_idxs: Iterable[int],
        alpha: float = 0.5,
    ):
        self.model = real_score_model
        self.block_idxs: List[int] = list(block_idxs)
        self.alpha = float(alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {self.alpha}")

        self.cache: dict[int, torch.Tensor] = {}
        self.mode: Optional[str] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def set_mode(self, mode: Optional[str]) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if mode == "inject" and not self.cache:
            raise RuntimeError("set_mode('inject') called before 'capture' filled the cache")
        self.mode = mode

    def _make_hook(self, idx: int):
        def fn(module, args, output):
            if self.mode == "capture":
                self.cache[idx] = output.detach()
                return output
            if self.mode == "inject":
                ref = self.cache[idx]
                if ref.shape != output.shape:
                    raise RuntimeError(
                        f"motion injector shape mismatch at block {idx}: "
                        f"cache={tuple(ref.shape)} vs current={tuple(output.shape)}"
                    )
                a = self.alpha
                return (1.0 - a) * output + a * ref.to(dtype=output.dtype, device=output.device)
            return output
        return fn

    def __enter__(self) -> "MotionAttnInjector":
        if self._handles:
            raise RuntimeError("MotionAttnInjector is already active")
        for i in self.block_idxs:
            block = self.model.blocks[i]
            h = block.self_attn.register_forward_hook(self._make_hook(i))
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.cache.clear()
        self.mode = None

    def attn_norm_summary(self) -> dict[int, float]:
        """Diagnostic: per-block mean abs of cached V_ref attn (post capture)."""
        return {i: float(self.cache[i].abs().mean().item()) for i in self.cache}


@dataclass
class MotionConfig:
    """Knobs for the motion-DMD branch. Defaults = disabled (no behavior change).

    `enabled=False` → DMD path is unchanged; `compute_distribution_matching_loss`
    skips the dispatch and runs vanilla `_compute_kl_grad`.
    """
    enabled: bool = False
    refs_path: Optional[str] = None         # path to .pt cache from precache_motion_refs.py
    block_idxs: List[int] = field(default_factory=lambda: [18, 19, 20])
    alpha: float = 0.5                       # hook-blend weight inside MotionAttnInjector
    beta_max: float = 0.2                    # score-blend weight steady-state
    beta_warmup_steps: int = 200             # linear warmup 0.05 → beta_max
    beta_warmup_start: float = 0.05
    inject_prob: float = 0.7                 # Bernoulli prob a step is motion-injected
    seed: int = 0                            # RNG seed for ref-pick + Bernoulli (rank 0)

    # Filled at trainer init time (not from yaml):
    v_ref_latents: Optional[torch.Tensor] = None  # [N, F=21, C=16, H=60, W=104] bf16, cpu pinned
    captions: Optional[List[str]] = None          # length N

    def beta_at(self, step: int) -> float:
        if step >= self.beta_warmup_steps:
            return self.beta_max
        if self.beta_warmup_steps <= 0:
            return self.beta_max
        frac = max(0, step) / float(self.beta_warmup_steps)
        return self.beta_warmup_start + frac * (self.beta_max - self.beta_warmup_start)


def attach_motion_config(model_obj, args) -> None:
    """Read `args.motion` (OmegaConf or dict) and attach a `motion_cfg` to `model_obj`.

    Always attaches; if `args.motion` is missing or `enabled=False`, the cfg is
    a no-op default. Trainer is responsible for separately loading the latent
    cache into `model_obj.motion_cfg.v_ref_latents` when enabled.
    """
    motion = getattr(args, "motion", None) or {}
    # OmegaConf DictConfig supports attribute access; dict supports get(); be permissive.
    def _get(k, default):
        if hasattr(motion, k):
            return getattr(motion, k)
        if isinstance(motion, dict):
            return motion.get(k, default)
        return default

    cfg = MotionConfig(
        enabled=bool(_get("enabled", False)),
        refs_path=_get("refs_path", None),
        block_idxs=list(_get("block_idxs", [18, 19, 20])),
        alpha=float(_get("alpha", 0.5)),
        beta_max=float(_get("beta_max", 0.2)),
        beta_warmup_steps=int(_get("beta_warmup_steps", 200)),
        beta_warmup_start=float(_get("beta_warmup_start", 0.05)),
        inject_prob=float(_get("inject_prob", 0.7)),
        seed=int(_get("seed", getattr(args, "seed", 0))),
    )
    model_obj.motion_cfg = cfg
