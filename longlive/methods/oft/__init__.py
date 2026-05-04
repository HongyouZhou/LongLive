"""OFT (Orthogonal Fine-Tuning) adapter via PEFT 0.19+ block-Cayley
implementation. Touches L1 (parameterization) only; see docs/01.md.

Per the __init__.py discipline: relative imports + register_* calls only,
no heavy module-level code.
"""
import peft

from longlive.utils.lora_utils import register_adapter

register_adapter("oft", lambda cfg, targets: peft.OFTConfig(
    oft_block_size=cfg.get('oft_block_size', 64),
    use_cayley_neumann=cfg.get('use_cayley_neumann', True),
    num_cayley_neumann_terms=cfg.get('num_cayley_neumann_terms', 5),
    module_dropout=cfg.get('module_dropout', 0.0),
    target_modules=targets,
    bias='none',
))
