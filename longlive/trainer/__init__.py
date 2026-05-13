"""Trainer registry.

Built-in trainers register themselves here. Method packages under
``longlive/methods/<idea>/`` register their L5 subclasses via side-effect
import (auto-scan in ``longlive.utils.lora_utils``).

Add a new trainer subclass by:
  1. Defining the subclass in ``longlive/methods/<idea>/trainer.py``.
  2. Calling ``register_trainer("<name>", MyTrainer)`` from that method's
     ``__init__.py``.
  3. Setting ``trainer: <name>`` in the YAML config.

The orchestrator (``scripts/local/train.py``) does
``get_trainer_class(config.trainer)(config)`` — methods are never
referenced by name from core.
"""

_TRAINER_REGISTRY = {}


def register_trainer(name, trainer_cls):
    """Register a trainer class under ``name``."""
    if name in _TRAINER_REGISTRY:
        raise KeyError(f"Trainer '{name}' already registered")
    _TRAINER_REGISTRY[name] = trainer_cls


def get_trainer_class(name):
    """Look up a registered trainer class."""
    cls = _TRAINER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown trainer: '{name}'. Registered: {sorted(_TRAINER_REGISTRY)}"
        )
    return cls


# Define register_trainer / get_trainer_class BEFORE importing the default
# trainer / triggering method autoload — partial imports from method
# packages can then call register_trainer on the partial module.
from .distillation import Trainer as ScoreDistillationTrainer  # noqa: E402

register_trainer("score_distillation", ScoreDistillationTrainer)

# Triggers ``longlive.utils.lora_utils._autoload_methods`` at module bottom,
# which imports every ``longlive/methods/<idea>/__init__.py`` exactly once
# — those side-effect-register any L1 adapters or L5 trainer subclasses.
import longlive.utils.lora_utils  # noqa: E402,F401


__all__ = [
    "ScoreDistillationTrainer",
    "register_trainer",
    "get_trainer_class",
]
