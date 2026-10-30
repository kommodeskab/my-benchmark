"""PTDB-TUG dataset implementation for pitch tracking and VAD benchmarks."""

import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, random_split


class PTDBTUGDataset(Dataset):
    """PTDB-TUG audio dataset for pitch tracking and voice activity benchmarks.

    Loads the PTDB-TUG dataset from Hugging Face ("gilkeyio/ptdb-tug") and provides
    deterministic train and validation splits with pre-computed pitch and voicing targets.

    Args:
        split: Dataset split to load ('train' or 'val').
        num_bins: Number of voiced pitch bins for cents-based classification. Defaults to 10.
        data_dir: Optional path or cache directory for dataset storage. If None,
            uses the DATA_PATH environment variable if set.
        seed: Random seed used for deterministic 80/20 train/val split. Defaults to 42.

    Raises:
        ValueError: If split is not 'train' or 'val', or if num_bins is not positive.
    """

    def __init__(
        self,
        split: Literal["train", "val"],
        num_bins: int = 10,
        data_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize PTDBTUGDataset."""
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError(f"num_bins must be a positive integer, got {num_bins!r}")

        super().__init__()
        self.split = split
        self.num_bins = num_bins
        self.seed = seed
        self.data_dir = (
            str(data_dir) if data_dir is not None else os.getenv("DATA_PATH")
        )

        # Approximated f0 range for human speech, used for binning
        self.fmax = 550.0
        self.fmin = 55.0
        max_cents = 1200.0 * math.log2(self.fmax / self.fmin)
        self.bin_width = max_cents / float(num_bins)
        self.stride_ms = 10.0

        full_dataset = load_dataset(
            "gilkeyio/ptdb-tug",
            cache_dir=self.data_dir,
            split="train",
        )

        generator = torch.Generator().manual_seed(self.seed)
        trainset, valset = random_split(full_dataset, [0.8, 0.2], generator=generator)

        self.dataset = trainset if split == "train" else valset

    def __len__(self) -> int:
        """Return the number of samples in the split.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample at index idx.

        Args:
            idx: Sample index.

        Returns:
            dict[str, Any]: Dictionary containing:
                - "waveform": torch.Tensor of shape (1, T).
                - "sample_rate": Audio sample rate as int.
                - "metadata": Metadata dictionary with keys 'f0_values',
                  'voice_activity', 'cents', 'cents_class', and 'audio_length'.
        """
        item = self.dataset[idx]
        waveform = torch.as_tensor(
            item["mic_audio"]["array"], dtype=torch.float32
        ).unsqueeze(0)
        sample_rate = int(item["mic_audio"]["sampling_rate"])
        f0_data = torch.as_tensor(item["f0_data"], dtype=torch.float32)

        voice_activity = f0_data[:, 1]
        f0_values = torch.clamp(f0_data[:, 0], min=0.0, max=self.fmax)

        # Find first and last non-zero voice activity frames (with 10-frame padding)
        non_zero_indices = torch.nonzero(voice_activity, as_tuple=False).flatten()
        if len(non_zero_indices) == 0:
            start_index = 0
            end_index = len(voice_activity) - 1
        else:
            start_index = max(int(non_zero_indices[0].item()) - 10, 0)
            end_index = min(
                int(non_zero_indices[-1].item()) + 10, len(voice_activity) - 1
            )

        stride_samples = int(self.stride_ms / 1000.0 * sample_rate)
        start_sample = start_index * stride_samples
        end_sample = min(waveform.shape[-1], (end_index + 1) * stride_samples)

        waveform = waveform[:, start_sample:end_sample]
        voice_activity = voice_activity[start_index : end_index + 1]
        f0_values = f0_values[start_index : end_index + 1]

        # Compute pitch in cents and discrete pitch bins (0 = unvoiced, 1..num_bins = voiced)
        is_voiced = voice_activity > 0.0
        f0_voiced = torch.clamp(f0_values, min=self.fmin, max=self.fmax)
        cents_voiced = 1200.0 * torch.log2(f0_voiced / self.fmin)

        cents = torch.where(
            is_voiced, cents_voiced, torch.full_like(f0_values, float("-inf"))
        )
        voiced_bins = (
            torch.clamp(
                (cents_voiced / self.bin_width).long(),
                min=0,
                max=self.num_bins - 1,
            )
            + 1
        )
        cents_class = torch.where(is_voiced, voiced_bins, torch.zeros_like(voiced_bins))

        metadata = {
            "f0_values": f0_values,
            "voice_activity": voice_activity,
            "cents": cents,
            "cents_class": cents_class,
            "audio_length": waveform.shape[-1],
        }

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
