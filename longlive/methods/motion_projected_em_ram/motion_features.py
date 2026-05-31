"""Structured reference-motion features from CoTracker tracklets.

This module keeps the motion signal before it is compressed into a scalar
reward.  It removes robust global translation, selects moving residual
tracklets, and exposes per-bucket motion summaries for diagnostics and future
training objectives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceMotionDescriptor:
    """Motion descriptor built from visible CoTracker tracklets."""

    tracks_xy: np.ndarray
    visibility: np.ndarray
    velocity_xy: np.ndarray
    global_motion_xy: np.ndarray
    residual_velocity_xy: np.ndarray
    moving_mask: np.ndarray
    bucket_summaries: list[dict[str, float]]
    summary: dict[str, float]


def _safe_unit(v: np.ndarray, eps: float) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _normalize_tracks(
    tracks: np.ndarray,
    *,
    frame_height: int | float,
    frame_width: int | float,
) -> np.ndarray:
    scale = np.asarray([float(frame_width), float(frame_height)], dtype=np.float32)
    return tracks.astype(np.float32) / np.maximum(scale, 1.0)


def _bucket_indices(n_steps: int, bucket_count: int) -> list[np.ndarray]:
    if n_steps <= 0:
        return []
    return [idx for idx in np.array_split(np.arange(n_steps), int(bucket_count)) if idx.size]


def build_reference_motion_descriptor(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    frame_height: int | float,
    frame_width: int | float,
    bucket_count: int = 3,
    moving_percentile: float = 60.0,
    min_moving_tracks: int = 4,
    moving_speed_floor: float = 1e-5,
    eps: float = 1e-8,
) -> ReferenceMotionDescriptor:
    """Build a global-motion-removed descriptor from tracklets.

    Args:
        tracks: CoTracker tracks shaped ``(T, N, 2)`` in pixel xy coordinates.
        visibility: CoTracker visibility shaped ``(T, N)``.
        frame_height/frame_width: dimensions of the video frames used for
            coordinate normalization.
    """
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (T, N, 2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"visibility must match tracks[:2], got {visibility.shape} vs {tracks.shape[:2]}"
        )
    if tracks.shape[0] < 2:
        raise ValueError("at least two frames are required for motion features")

    all_visible = visibility.astype(bool).all(axis=0)
    if not all_visible.any():
        raise ValueError("no tracklets are visible across all frames")

    visible_tracks = tracks[:, all_visible]
    tracks_xy = _normalize_tracks(
        visible_tracks,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    velocity_xy = np.diff(tracks_xy, axis=0)
    global_motion_xy = np.median(velocity_xy, axis=1)
    residual_velocity_xy = velocity_xy - global_motion_xy[:, None, :]

    residual_speed = np.linalg.norm(residual_velocity_xy, axis=-1)
    mean_residual_speed = residual_speed.mean(axis=0)
    if mean_residual_speed.size == 0:
        moving_mask = np.zeros((0,), dtype=bool)
    else:
        threshold = np.percentile(mean_residual_speed, float(moving_percentile))
        moving_mask = mean_residual_speed > max(float(threshold), float(moving_speed_floor))
        if int(moving_mask.sum()) < int(min_moving_tracks):
            topk = min(int(min_moving_tracks), mean_residual_speed.size)
            moving_mask = np.zeros_like(mean_residual_speed, dtype=bool)
            moving_mask[np.argsort(mean_residual_speed)[-topk:]] = True

    bucket_summaries = []
    for bucket_id, idx in enumerate(_bucket_indices(velocity_xy.shape[0], bucket_count)):
        residual_bucket = residual_velocity_xy[idx][:, moving_mask]
        speed_bucket = np.linalg.norm(residual_bucket, axis=-1)
        if residual_bucket.size == 0:
            mean_vec = np.zeros((2,), dtype=np.float32)
            mean_speed = 0.0
        else:
            mean_vec = residual_bucket.reshape(-1, 2).mean(axis=0)
            mean_speed = float(speed_bucket.mean())
        direction = _safe_unit(mean_vec[None, :], eps=eps)[0]
        coherence = float(np.linalg.norm(mean_vec) / (mean_speed + eps))
        bucket_summaries.append({
            "bucket": float(bucket_id),
            "start_step": float(idx[0]),
            "end_step": float(idx[-1]),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "mean_speed": mean_speed,
            "coherence": coherence,
        })

    selected_speed = mean_residual_speed[moving_mask] if moving_mask.size else np.asarray([])
    global_speed = np.linalg.norm(global_motion_xy, axis=-1)
    summary = {
        "visible_tracks": float(visible_tracks.shape[1]),
        "moving_tracks": float(moving_mask.sum()),
        "moving_track_rate": float(moving_mask.mean()) if moving_mask.size else 0.0,
        "residual_speed_mean": float(selected_speed.mean()) if selected_speed.size else 0.0,
        "residual_speed_p90": float(np.percentile(selected_speed, 90)) if selected_speed.size else 0.0,
        "global_speed_mean": float(global_speed.mean()),
        "global_speed_p90": float(np.percentile(global_speed, 90)),
    }

    return ReferenceMotionDescriptor(
        tracks_xy=tracks_xy,
        visibility=visibility[:, all_visible],
        velocity_xy=velocity_xy,
        global_motion_xy=global_motion_xy,
        residual_velocity_xy=residual_velocity_xy,
        moving_mask=moving_mask,
        bucket_summaries=bucket_summaries,
        summary=summary,
    )


def descriptor_pair_metrics(
    gen: ReferenceMotionDescriptor,
    ref: ReferenceMotionDescriptor,
    *,
    speed_lower: float = 0.5,
    speed_upper: float = 2.0,
    speed_penalty_coef: float = 0.25,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compare generated and reference residual-motion descriptors."""
    gen_v = gen.residual_velocity_xy[:, gen.moving_mask]
    ref_v = ref.residual_velocity_xy[:, ref.moving_mask]
    n_steps = min(gen_v.shape[0], ref_v.shape[0])
    if n_steps <= 0 or gen_v.shape[1] == 0 or ref_v.shape[1] == 0:
        return {
            "residual_direction": 0.0,
            "residual_speed_ratio": 0.0,
            "residual_speed_penalty": 0.0,
            "residual_score": 0.0,
        }

    gen_v = gen_v[:n_steps]
    ref_v = ref_v[:n_steps]
    gen_speed = np.linalg.norm(gen_v, axis=-1)
    ref_speed = np.linalg.norm(ref_v, axis=-1)
    gen_dir = gen_v / (gen_speed[..., None] + eps)
    ref_dir = ref_v / (ref_speed[..., None] + eps)

    per_frame_cos = np.einsum("tnc,tmc->tnm", ref_dir, gen_dir)
    mean_t_cos = per_frame_cos.mean(axis=0)
    best_idx = mean_t_cos.argmax(axis=1)
    best_cos = mean_t_cos[np.arange(mean_t_cos.shape[0]), best_idx]
    direction = float(best_cos.mean())

    ref_speed_mean = ref_speed.mean(axis=0)
    gen_speed_mean = gen_speed.mean(axis=0)
    matched_gen_speed = gen_speed_mean[best_idx]
    ratio = matched_gen_speed / (ref_speed_mean + eps)

    lower = float(max(speed_lower, eps))
    upper = float(max(speed_upper, lower + eps))
    log_ratio = np.log(ratio + eps)
    under = np.maximum(np.log(lower) - log_ratio, 0.0)
    over = np.maximum(log_ratio - np.log(upper), 0.0)
    speed_penalty = float((under * under + over * over).mean())
    score = direction - float(speed_penalty_coef) * speed_penalty

    global_n = min(gen.global_motion_xy.shape[0], ref.global_motion_xy.shape[0])
    gen_global = gen.global_motion_xy[:global_n]
    ref_global = ref.global_motion_xy[:global_n]
    global_cos = float((
        _safe_unit(gen_global, eps) * _safe_unit(ref_global, eps)
    ).sum(axis=-1).mean()) if global_n else 0.0

    out = {
        "residual_direction": direction,
        "residual_speed_ratio": float(ratio.mean()),
        "residual_speed_penalty": speed_penalty,
        "residual_score": float(score),
        "global_direction": global_cos,
        "gen_residual_speed_mean": gen.summary["residual_speed_mean"],
        "ref_residual_speed_mean": ref.summary["residual_speed_mean"],
        "gen_global_speed_mean": gen.summary["global_speed_mean"],
        "ref_global_speed_mean": ref.summary["global_speed_mean"],
        "gen_moving_tracks": gen.summary["moving_tracks"],
        "ref_moving_tracks": ref.summary["moving_tracks"],
    }

    for idx, (gen_bucket, ref_bucket) in enumerate(zip(gen.bucket_summaries, ref.bucket_summaries)):
        gen_dir_b = np.asarray([gen_bucket["direction_x"], gen_bucket["direction_y"]])
        ref_dir_b = np.asarray([ref_bucket["direction_x"], ref_bucket["direction_y"]])
        bucket_cos = float((gen_dir_b * ref_dir_b).sum())
        bucket_ratio = float(gen_bucket["mean_speed"] / (ref_bucket["mean_speed"] + eps))
        out[f"bucket{idx}_direction"] = bucket_cos
        out[f"bucket{idx}_speed_ratio"] = bucket_ratio
        out[f"bucket{idx}_gen_speed"] = float(gen_bucket["mean_speed"])
        out[f"bucket{idx}_ref_speed"] = float(ref_bucket["mean_speed"])
    return out


def descriptor_to_npz(path: str, descriptor: ReferenceMotionDescriptor) -> None:
    """Persist descriptor arrays plus summary fields as compressed npz."""
    arrays = {
        "tracks_xy": descriptor.tracks_xy,
        "visibility": descriptor.visibility,
        "velocity_xy": descriptor.velocity_xy,
        "global_motion_xy": descriptor.global_motion_xy,
        "residual_velocity_xy": descriptor.residual_velocity_xy,
        "moving_mask": descriptor.moving_mask,
    }
    for key, value in descriptor.summary.items():
        arrays[f"summary/{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)
