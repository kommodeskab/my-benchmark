"""Dummy audio dataset and representation learning model for demonstration and testing."""

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class DummyAudioDataset(Dataset):
    """Synthetic audio dataset for testing and demonstration.

    Generates synthetic sine waves with distinct frequencies per class,
    simulating realistic audio inputs with ground truth labels in metadata.
    """

    def __init__(
        self,
        num_samples: int = 100,
        audio_length: int = 16000,
        num_classes: int = 5,
        sample_rate: int = 16000,
        name: str = "dummy_audio_dataset",
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.name = name

        rng = np.random.default_rng(seed)
        self.labels = [int(i % num_classes) for i in range(num_samples)]
        rng.shuffle(self.labels)

        self.data: list[dict[str, Any]] = []
        t = np.linspace(
            0,
            audio_length / sample_rate,
            audio_length,
            endpoint=False,
            dtype=np.float32,
        )

        for idx, label in enumerate(self.labels):
            freq = 200.0 * (label + 1)
            # Sine wave + small gaussian noise
            signal = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(
                audio_length
            ).astype(np.float32)
            audio_tensor = torch.from_numpy(signal).unsqueeze(0)  # Shape (1, T)

            # Per-frame targets and labels matching length T
            frame_labels = [
                label if i < audio_length // 2 else (label + 1) % num_classes
                for i in range(audio_length)
            ]
            frame_targets = [
                float(freq) + 10.0 * np.sin(4 * np.pi * t[i])
                for i in range(audio_length)
            ]

            # Variable-length target token sequence (e.g. phonemes or words)
            seq_len = 2 + (idx % 3)
            tokens = [f"ph_{(label + k) % num_classes}" for k in range(seq_len)]

            metadata = {
                "id": f"sample_{idx:04d}",
                "label": label,
                "label_name": f"class_{label}",
                "target": float(freq),
                "frame_labels": frame_labels,
                "frame_targets": frame_targets,
                "tokens": tokens,
                "sample_rate": sample_rate,
            }
            self.data.append(
                {
                    "waveform": audio_tensor,
                    "sample_rate": sample_rate,
                    "metadata": metadata,
                }
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class DummyAudioModel(nn.Module):
    """Dummy representation learning model.

    Maps 1D raw audio of shape (B, 1, T) to sequence representations
    of shape (B, T', C') using 1D convolutional layers.
    """

    def __init__(self, in_channels: int = 1, representation_dim: int = 64):
        super().__init__()
        self.representation_dim = representation_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, representation_dim, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Audio tensor of shape (B, 1, T).

        Returns:
            torch.Tensor: Audio representations of shape (B, T', C').
        """
        # x is (B, 1, T) -> conv_layers -> (B, C', T')
        features = self.conv_layers(x)
        # Transpose to (B, T', C') as required by benchmark specification
        return features.transpose(1, 2)
