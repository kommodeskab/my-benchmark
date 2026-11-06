"""General utility functions for the audio representation benchmark suite."""

from __future__ import annotations

import math

import torch


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
        self.register_buffer("pe", self._create_pe(max_len, d_model))

    @staticmethod
    def _create_pe(
        max_len: int, d_model: int, device: torch.device | None = None
    ) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass adding positional encoding to input tensor.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Position-encoded tensor of shape (B, T, d_model).
        """
        if x.size(1) > self.pe.size(1):
            new_len = max(x.size(1), self.pe.size(1) * 2)
            self.pe = self._create_pe(new_len, self.pe.size(-1), x.device)

        return x + self.pe[:, : x.size(1), :]


def train_torch_model(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    show_progress: bool = True,
    desc: str = "Training",
) -> None:
    """Train a PyTorch module using mini-batch gradient descent.

    Args:
        model: PyTorch module to train.
        X: Feature tensor.
        y: Target tensor.
        criterion: Loss function.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        weight_decay: L2 penalty for Adam optimizer.
        device: Device to place tensors on.
        show_progress: Whether to display a tqdm progress bar.
        desc: Description text for the tqdm progress bar.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    dataset = TensorDataset(X, y)
    effective_bs = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    epoch_iter = tqdm(
        range(epochs),
        desc=desc,
        disable=not show_progress,
        leave=False,
    )
    for _ in epoch_iter:
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        epoch_iter.set_postfix({"loss": f"{running_loss / max(1, len(loader)):.4f}"})
