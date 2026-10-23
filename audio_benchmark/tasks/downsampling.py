"""Temporal downsampling utilities for frame-level audio tasks."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def downsample_sequence(
    sequence: Sequence[int | float] | np.ndarray,
    target_len: int,
    mode: str = "mean",
) -> np.ndarray:
    """Downsample a 1D sequence of length T to target_len T'.

    For each region in the original sequence mapped to a downsampled frame:
    - If mode is 'mean': computes the average value in the region.
    - If mode is 'mode': selects the most common label in the region.

    Args:
        sequence: Original sequence of length T.
        target_len: Target sequence length T' (T' <= T or T' > 0).
        mode: Aggregation mode: 'mean' (regression) or 'mode' (classification).

    Returns:
        np.ndarray: Downsampled 1D array of length target_len.
    """
    seq = np.asarray(sequence)
    orig_len = len(seq)

    if orig_len == 0:
        raise ValueError("Cannot downsample an empty sequence.")

    if (
        not isinstance(target_len, int)
        or isinstance(target_len, bool)
        or target_len <= 0
    ):
        raise ValueError(f"target_len must be positive integer, got {target_len}.")

    if orig_len == target_len:
        return seq

    out = []
    for i in range(target_len):
        start = round(i * orig_len / target_len)
        end = round((i + 1) * orig_len / target_len)
        # Ensure region has at least one element and stays in bounds
        start = min(start, orig_len - 1)
        end = max(start + 1, min(orig_len, end))
        region = seq[start:end]

        if mode == "mean":
            out.append(float(np.mean(region)))
        elif mode == "mode":
            counts = Counter(region)
            most_common = counts.most_common(1)[0][0]
            out.append(most_common)
        else:
            raise ValueError(
                f"Unknown downsample mode: '{mode}'. Supported: 'mean', 'mode'."
            )

    return np.asarray(out)


def parse_frame_targets(
    raw_target: Any,
    audio_length: float | None = None,
    key_name: str = "frame_targets",
) -> tuple[np.ndarray, list[Any]]:
    """Parse and validate arbitrary-length sequence or dictionary frame targets.

    Supports:
      1. Mapping/dictionary ({timestamp: target}) where timestamps are integer
         or stringified-integer sample indices in [0, audio_length).
      2. 1D sequence or tensor of arbitrary length K >= 1, centered at
         (i + 0.5) * audio_length / K for i in {0, ..., K-1}.

    Args:
        raw_target: Raw target object from sample metadata.
        audio_length: Total audio duration in samples. Required for validating
            dictionary timestamp bounds when provided.
        key_name: Metadata key identifier used in error messages.

    Returns:
        tuple[np.ndarray, list[Any]]: A tuple of (timestamps, target_values)
            where timestamps has shape (K,) and target_values is a list of length K.

    Raises:
        InvalidDatasetError: If raw_target format, shape, key types, or timestamps are invalid.
    """
    from audio_benchmark.exceptions import InvalidDatasetError

    if isinstance(raw_target, dict):
        if len(raw_target) == 0:
            raise InvalidDatasetError(
                f"Target dictionary in '{key_name}' must not be empty (K >= 1)."
            )
        if audio_length is not None and (
            isinstance(audio_length, bool)
            or not isinstance(audio_length, (int, float))
            or audio_length <= 0
        ):
            raise ValueError(
                f"audio_length must be a positive number, got {audio_length!r}."
            )
        parsed_items: list[tuple[int, Any]] = []
        seen_timestamps: set[int] = set()
        for k, v in raw_target.items():
            if isinstance(k, bool):
                raise InvalidDatasetError(
                    f"Timestamp key in '{key_name}' must be an integer, got bool: {k!r}."
                )
            if isinstance(k, int):
                ts = k
            elif isinstance(k, str):
                cleaned = k.strip()
                try:
                    ts = int(cleaned)
                    if str(ts) != cleaned:
                        raise ValueError
                except (ValueError, TypeError) as e:
                    raise InvalidDatasetError(
                        f"Timestamp key {k!r} in '{key_name}' is not a valid integer sample index."
                    ) from e
            else:
                raise InvalidDatasetError(
                    f"Timestamp key {k!r} in '{key_name}' must be an integer or stringified integer, got {type(k).__name__}."
                )

            if ts < 0:
                raise InvalidDatasetError(
                    f"Timestamp key {ts} in '{key_name}' must be non-negative, got {ts}."
                )
            if audio_length is not None and ts >= audio_length:
                raise InvalidDatasetError(
                    f"Timestamp {ts} in '{key_name}' exceeds audio length {audio_length}."
                )
            if ts in seen_timestamps:
                raise InvalidDatasetError(
                    f"Duplicate timestamp key {ts} found in '{key_name}'."
                )
            seen_timestamps.add(ts)
            parsed_items.append((ts, v))

        sorted_items = sorted(parsed_items, key=lambda item: item[0])
        timestamps = np.array([item[0] for item in sorted_items], dtype=np.float64)
        targets = [item[1] for item in sorted_items]
        return timestamps, targets

    if isinstance(raw_target, (str, bytes, bytearray)):
        raise InvalidDatasetError(
            f"Metadata key '{key_name}' cannot be a string or bytes, got {type(raw_target).__name__}."
        )

    if isinstance(raw_target, (torch.Tensor, np.ndarray)):
        if raw_target.ndim != 1:
            raise InvalidDatasetError(
                f"Target sequence in '{key_name}' must be 1D, got ndim={raw_target.ndim} with shape {tuple(raw_target.shape)}."
            )
        targets = (
            raw_target.detach().cpu().tolist()
            if isinstance(raw_target, torch.Tensor)
            else raw_target.tolist()
        )
    elif isinstance(raw_target, (list, tuple)):
        targets = list(raw_target)
    else:
        raise InvalidDatasetError(
            f"Metadata key '{key_name}' must be a dictionary or 1D sequence, got {type(raw_target).__name__}."
        )

    k_count = len(targets)
    if k_count == 0:
        raise InvalidDatasetError(
            f"Target sequence in '{key_name}' must not be empty (K >= 1)."
        )

    if audio_length is not None and (
        isinstance(audio_length, bool)
        or not isinstance(audio_length, (int, float))
        or audio_length <= 0
    ):
        raise ValueError(
            f"audio_length must be a positive number, got {audio_length!r}."
        )

    eff_audio_len = float(audio_length) if audio_length is not None else float(k_count)
    timestamps = (np.arange(k_count, dtype=np.float64) + 0.5) * (
        eff_audio_len / float(k_count)
    )
    return timestamps, targets


def align_features_to_targets(
    representations: np.ndarray | torch.Tensor,
    target_timestamps: np.ndarray | Sequence[float],
    audio_length: float,
    pooling: str = "mean",
) -> np.ndarray:
    """Align temporal representation features to target timestamps.

    When representation length T' >= target count K (downsampling or 1-to-1),
    representation frames are assigned to their nearest target timestamp and
    pooled using the specified pooling mode. If any target receives zero assigned
    representation frames, it falls back to its single nearest representation
    feature in time.

    When representation length T' < target count K (upsampling), each target is
    assigned the single representation feature closest to it in time.

    Args:
        representations: Feature array of shape (T', C) or torch.Tensor.
        target_timestamps: 1D array or sequence of target timestamps in sample indices.
        audio_length: Total audio duration in sample indices (T > 0).
        pooling: Pooling strategy when T' >= K ('mean', 'max', 'min', 'median').

    Returns:
        np.ndarray: Aligned feature array of shape (K, C).

    Raises:
        ValueError: If representations is not 2D, target_timestamps is empty or out of bounds,
            audio_length is not positive, or pooling method is unsupported.
    """
    valid_poolings = ("mean", "max", "min", "median")
    if pooling not in valid_poolings:
        raise ValueError(
            f"Unknown pooling '{pooling}'. Supported pooling methods: {valid_poolings}"
        )

    if (
        isinstance(audio_length, bool)
        or not isinstance(audio_length, (int, float))
        or audio_length <= 0
    ):
        raise ValueError(f"audio_length must be positive, got {audio_length}.")

    if isinstance(representations, torch.Tensor):
        reps = representations.detach().cpu().numpy()
    else:
        reps = np.asarray(representations)

    if reps.ndim != 2:
        raise ValueError(
            f"representations must have shape (T', C) with ndim=2, got shape {reps.shape}."
        )

    t_prime, feature_dim = reps.shape
    if t_prime == 0:
        raise ValueError("representations cannot have temporal length 0.")

    target_ts = np.asarray(target_timestamps, dtype=np.float64)
    if target_ts.ndim != 1 or len(target_ts) == 0:
        raise ValueError(
            f"target_timestamps must be a non-empty 1D array, got shape {target_ts.shape}."
        )

    if np.any(target_ts < 0) or np.any(target_ts >= audio_length):
        raise ValueError(
            f"All target timestamps must be within [0, audio_length) ([0, {audio_length}))."
        )

    k_count = len(target_ts)

    rep_ts = (np.arange(t_prime, dtype=np.float64) + 0.5) * (
        float(audio_length) / float(t_prime)
    )

    if t_prime >= k_count:
        dist = np.abs(rep_ts[:, None] - target_ts[None, :])
        nearest_targets = np.argmin(dist, axis=1)

        assigned: list[list[int]] = [[] for _ in range(k_count)]
        for j, target_idx in enumerate(nearest_targets):
            assigned[target_idx].append(j)

        aligned = np.empty((k_count, feature_dim), dtype=reps.dtype)
        for i in range(k_count):
            if assigned[i]:
                sub = reps[assigned[i]]
                if pooling == "mean":
                    aligned[i] = np.mean(sub, axis=0)
                elif pooling == "max":
                    aligned[i] = np.max(sub, axis=0)
                elif pooling == "min":
                    aligned[i] = np.min(sub, axis=0)
                elif pooling == "median":
                    aligned[i] = np.median(sub, axis=0)
            else:
                rep_dist = np.abs(target_ts[i] - rep_ts)
                nearest_rep = int(np.argmin(rep_dist))
                aligned[i] = reps[nearest_rep]
        return aligned

    dist = np.abs(target_ts[:, None] - rep_ts[None, :])
    nearest_reps = np.argmin(dist, axis=1)
    return reps[nearest_reps]
