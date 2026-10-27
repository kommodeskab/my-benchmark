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

    if not isinstance(target_len, int) or target_len <= 0:
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

    Supports strictly:
      1. Mapping/dictionary ({timestamp: target}) where timestamps are integer
         sample indices in [0, audio_length).
      2. 1D list of arbitrary length K >= 1, centered at
         (i + 0.5) * audio_length / K for i in {0, ..., K-1}.

    Args:
        raw_target: Raw target object from sample metadata (list or dict[int, Any]).
        audio_length: Total audio duration in samples. Required for validating
            dictionary timestamp bounds when provided.
        key_name: Metadata key identifier used in error messages.

    Returns:
        tuple[np.ndarray, list[Any]]: A tuple of (timestamps, target_values)
            where timestamps has shape (K,) and target_values is a list of length K.

    Raises:
        InvalidDatasetError: If raw_target format, shape, key types, or timestamps are invalid.
        ValueError: If audio_length is not a positive number.
    """
    from audio_benchmark.exceptions import InvalidDatasetError

    if isinstance(raw_target, dict):
        if len(raw_target) == 0:
            raise InvalidDatasetError(
                f"Target dictionary in '{key_name}' must not be empty (K >= 1)."
            )
        if audio_length is not None and (
            not isinstance(audio_length, (int, float)) or audio_length <= 0
        ):
            raise ValueError(
                f"audio_length must be a positive number, got {audio_length!r}."
            )
        parsed_items: list[tuple[int, Any]] = []
        seen_timestamps: set[int] = set()
        for k, v in raw_target.items():
            if not isinstance(k, int):
                raise InvalidDatasetError(
                    f"Timestamp key {k!r} in '{key_name}' must be an integer, got {type(k).__name__}."
                )
            ts = k
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

    elif isinstance(raw_target, list):
        targets = raw_target
        k_count = len(targets)
        if k_count == 0:
            raise InvalidDatasetError(
                f"Target sequence in '{key_name}' must not be empty (K >= 1)."
            )
        if audio_length is not None and (
            not isinstance(audio_length, (int, float)) or audio_length <= 0
        ):
            raise ValueError(
                f"audio_length must be a positive number, got {audio_length!r}."
            )
        eff_audio_len = (
            float(audio_length) if audio_length is not None else float(k_count)
        )
        timestamps = (np.arange(k_count, dtype=np.float64) + 0.5) * (
            eff_audio_len / float(k_count)
        )
        return timestamps, targets

    else:
        raise InvalidDatasetError(
            f"Metadata key '{key_name}' must be a list or dict[int, Any], got {type(raw_target).__name__}."
        )


def align_features_to_targets(
    representations: torch.Tensor,
    target_timestamps: np.ndarray | Sequence[float] | torch.Tensor,
    audio_length: float,
    pooling: str = "mean",
) -> torch.Tensor:
    """Align temporal representation features to target timestamps.

    When representation length T' >= target count K (downsampling or 1-to-1),
    representation frames are assigned to their nearest target timestamp and
    pooled using the specified pooling mode. If any target receives zero assigned
    representation frames, it falls back to its single nearest representation
    feature in time.

    When representation length T' < target count K (upsampling), each target is
    assigned the single representation feature closest to it in time.

    Args:
        representations: 2D feature tensor of shape (T', C).
        target_timestamps: 1D array, sequence, or tensor of target timestamps in sample indices.
        audio_length: Total audio duration in sample indices (T > 0).
        pooling: Pooling strategy when T' >= K ('mean', 'max', 'min', 'median').

    Returns:
        torch.Tensor: Aligned 2D feature tensor of shape (K, C).

    Raises:
        TypeError: If representations is not a torch.Tensor.
        ValueError: If representations is not 2D, target_timestamps is empty or out of bounds,
            audio_length is not positive, or pooling method is unsupported.
    """
    valid_poolings = ("mean", "max", "min", "median")
    if pooling not in valid_poolings:
        raise ValueError(
            f"Unknown pooling '{pooling}'. Supported pooling methods: {valid_poolings}"
        )

    if not isinstance(audio_length, (int, float)) or audio_length <= 0:
        raise ValueError(f"audio_length must be positive, got {audio_length}.")

    if not isinstance(representations, torch.Tensor):
        raise TypeError(
            f"representations must be a torch.Tensor, got {type(representations).__name__}."
        )

    if representations.ndim != 2:
        raise ValueError(
            f"representations must have shape (T', C) with ndim=2, got shape {tuple(representations.shape)}."
        )

    t_prime, feature_dim = representations.shape
    if t_prime == 0:
        raise ValueError("representations cannot have temporal length 0.")

    target_ts = torch.as_tensor(
        target_timestamps, dtype=torch.float32, device=representations.device
    )
    if target_ts.ndim != 1 or len(target_ts) == 0:
        raise ValueError(
            f"target_timestamps must be a non-empty 1D array, got shape {tuple(target_ts.shape)}."
        )

    if (target_ts < 0).any() or (target_ts >= audio_length).any():
        raise ValueError(
            f"All target timestamps must be within [0, audio_length) ([0, {audio_length}))."
        )

    k_count = len(target_ts)
    rep_ts = (
        torch.arange(t_prime, dtype=torch.float32, device=representations.device) + 0.5
    ) * (float(audio_length) / float(t_prime))

    if t_prime >= k_count:
        dist = torch.abs(rep_ts[:, None] - target_ts[None, :])
        nearest_targets = torch.argmin(dist, dim=1)

        assigned: list[list[int]] = [[] for _ in range(k_count)]
        for j, target_idx in enumerate(nearest_targets):
            assigned[target_idx.item()].append(j)

        aligned = torch.empty(
            (k_count, feature_dim),
            dtype=representations.dtype,
            device=representations.device,
        )
        for i in range(k_count):
            if assigned[i]:
                sub = representations[assigned[i]]
                if pooling == "mean":
                    aligned[i] = sub.mean(dim=0)
                elif pooling == "max":
                    aligned[i] = sub.amax(dim=0)
                elif pooling == "min":
                    aligned[i] = sub.amin(dim=0)
                elif pooling == "median":
                    aligned[i] = torch.quantile(sub.float(), 0.5, dim=0).to(
                        dtype=representations.dtype
                    )
            else:
                rep_dist = torch.abs(target_ts[i] - rep_ts)
                nearest_rep = int(torch.argmin(rep_dist).item())
                aligned[i] = representations[nearest_rep]
        return aligned

    dist = torch.abs(target_ts[:, None] - rep_ts[None, :])
    nearest_reps = torch.argmin(dist, dim=1)
    return representations[nearest_reps]
