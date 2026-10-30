"""Temporal downsampling utilities for frame-level audio tasks."""

from typing import Any

import torch
import torch.nn.functional as F

from audio_benchmark.exceptions import InvalidDatasetError


def parse_frame_targets(
    raw_target: Any,
    key_name: str = "frame_targets",
) -> list[Any]:
    """Parse and validate flat tensor or sequence frame targets.

    Targets must be either a 1D (flat) torch.Tensor or a 1D list of length K >= 1,
    assumed to be equally spaced in time.

    Args:
        raw_target: Raw target object from sample metadata (flat torch.Tensor or list).
        key_name: Metadata key identifier used in error messages.

    Returns:
        list[Any]: List of target values of length K.

    Raises:
        InvalidDatasetError: If raw_target is not a flat torch.Tensor or list,
            or if it is empty.
    """
    if isinstance(raw_target, torch.Tensor):
        if raw_target.ndim != 1:
            raise InvalidDatasetError(
                f"Target tensor in '{key_name}' must be 1D (flat), got shape {tuple(raw_target.shape)}."
            )
        raw_target = raw_target.tolist()
    elif not isinstance(raw_target, list):
        raise InvalidDatasetError(
            f"Metadata key '{key_name}' must be a flat torch.Tensor or list, got {type(raw_target).__name__}."
        )

    if not raw_target:
        raise InvalidDatasetError(
            f"Target sequence in '{key_name}' must not be empty (K >= 1)."
        )
    return raw_target


def align_features_to_targets(
    representations: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """Align temporal representation features to target sequence length using linear interpolation.

    Changes the sequence length of representations (T') to match the sequence length of
    the targets (T) using torch 1D linear interpolation.

    Args:
        representations: 2D feature tensor of shape (T', C).
        target_len: Desired target sequence length T (positive integer).

    Returns:
        torch.Tensor: Aligned 2D feature tensor of shape (T, C).

    Raises:
        TypeError: If representations is not a torch.Tensor.
        ValueError: If representations is not 2D, representations has temporal length 0,
            or target_len is not a positive integer.
    """
    if not isinstance(representations, torch.Tensor):
        raise TypeError(
            f"representations must be a torch.Tensor, got {type(representations).__name__}."
        )
    if representations.ndim != 2:
        raise ValueError(
            f"representations must have shape (T', C) with ndim=2, got shape {tuple(representations.shape)}."
        )
    if representations.shape[0] == 0:
        raise ValueError("representations cannot have temporal length 0.")
    if not isinstance(target_len, int) or target_len <= 0:
        raise ValueError(f"target_len must be a positive integer, got {target_len!r}.")

    if representations.shape[0] == target_len:
        return representations

    x = representations.T.unsqueeze(0)
    return (
        F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        .squeeze(0)
        .T
    )
