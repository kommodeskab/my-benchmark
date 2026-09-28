"""Helper utilities for dataset wrappers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resample_waveform(
    waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000
) -> torch.Tensor:
    """Resample 1D or 2D audio waveform to target sample rate using linear interpolation.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio waveform tensor of shape `(1, T)` or `(T,)`.
    orig_sr : int
        Original sample rate.
    target_sr : int, default 16000
        Target sample rate.

    Returns
    -------
    torch.Tensor
        Resampled waveform with matching dimension.
    """
    if orig_sr == target_sr:
        return waveform

    is_2d = waveform.ndim == 2
    audio_1d = waveform.squeeze().float()
    orig_len = audio_1d.shape[0]

    target_len = round(orig_len * float(target_sr) / float(orig_sr))
    if target_len <= 0:
        target_len = 1

    resampled = F.interpolate(
        audio_1d.view(1, 1, -1),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).view(-1)

    if is_2d:
        return resampled.unsqueeze(0)
    return resampled
