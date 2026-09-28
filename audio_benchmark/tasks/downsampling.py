"""Temporal downsampling utilities for frame-level audio tasks."""

from collections import Counter
from collections.abc import Sequence

import numpy as np


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

    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}.")

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
