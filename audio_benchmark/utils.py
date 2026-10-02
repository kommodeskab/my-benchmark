"""General utility functions for the audio representation benchmark suite."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor or array-like object to a NumPy array.

    Args:
        x: Input tensor or array-like data structure.

    Returns:
        np.ndarray: NumPy array representation of the input.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device specification to a concrete torch.device.

    If 'auto' or None, resolves to 'cuda' when available, otherwise 'cpu'.

    Args:
        device: Device string (e.g. 'auto', 'cpu', 'cuda') or torch.device instance.

    Returns:
        torch.device: Resolved PyTorch device object.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Embedding / feature dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass adding positional encoding to input tensor.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Position-encoded tensor of shape (B, T, d_model).
        """
        return x + self.pe[:, : x.size(1), :]
