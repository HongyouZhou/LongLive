"""Group-normalize rollout rewards into an r in [0, 1] signal.

Originally mirrored DiffusionNFT's `PerPromptStatTracker`
(flow_grpo/stat_tracking.py:23-31) plus the
`r = 0.5 + 0.5 * clip(advantage / adv_clip_max, -1, 1)` mapping at
`train_nft_sd3.py:920-929`. It is kept as a method-agnostic reward utility.

Simplified for our single-prompt setting: every adaptation experiment uses
one caption (the reference clip's `train_caption`), so we group-normalize
across the K rollouts of that one caption rather than maintaining a
multi-prompt history.
"""
from __future__ import annotations

import numpy as np
import torch


def group_normalize(
    rewards: list[float] | np.ndarray | torch.Tensor,
    adv_clip_max: float = 5.0,
    std_floor: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Map K rollout-level rewards to r in [0, 1].

    Steps:
      1. Mean-subtract:   adv = r_raw - mean(r_raw)
      2. Std-normalize:   adv /= (std(r_raw) + std_floor)
      3. Clip + remap:    r   = 0.5 + 0.5 * clip(adv / adv_clip_max, -1, 1)

    Args:
        rewards: shape (K,) - one scalar reward per rollout in the group.
        adv_clip_max: advantages outside this many stds saturate.
        std_floor: prevents div-by-0 when all K rewards happen to be equal
            (degenerate iteration — r will be 0.5 for all, no learning).

    Returns:
        (r, diag) - r is shape (K,) in [0, 1], diag has scalar stats for
        logging (mean / std / zero_std_flag).
    """
    if isinstance(rewards, list):
        arr = np.asarray(rewards, dtype=np.float64)
    elif isinstance(rewards, torch.Tensor):
        arr = rewards.detach().cpu().double().numpy()
    else:
        arr = np.asarray(rewards, dtype=np.float64)

    mean = float(arr.mean())
    std_raw = float(arr.std())
    std = std_raw + std_floor

    advantages = (arr - mean) / std
    advantages = np.clip(advantages, -adv_clip_max, adv_clip_max)
    r = 0.5 + 0.5 * (advantages / adv_clip_max)

    diag = {
        "reward/raw_mean": mean,
        "reward/raw_std": std_raw,
        # If this is high (group has no
        # spread), the iteration produces r ≈ 0.5 for everyone and the loss
        # has no useful gradient.  We log a single bool here; trainer
        # aggregates across outer epochs.
        "reward/group_collapsed": float(std_raw < std_floor),
        "reward/r_min": float(r.min()),
        "reward/r_max": float(r.max()),
    }
    return torch.from_numpy(r).float(), diag
