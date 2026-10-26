"""CREMA-D dataset implementation for audio benchmark."""

import os
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, random_split


class CREMADDataset(Dataset):
    """CREMA-D audio dataset for emotion recognition benchmarking.

    Loads the CREMA-D dataset from Hugging Face ("AbstractTTS/CREMA-D") and provides
    deterministic train and validation splits.

    Args:
        split: Dataset split to load ('train' or 'val').
        data_dir: Optional path or cache directory for dataset storage. If None,
            uses the DATA_PATH environment variable if set.
        seed: Random seed used for deterministic 80/20 train/val split. Defaults to 42.

    Raises:
        ValueError: If split is not 'train' or 'val'.
    """

    def __init__(
        self,
        split: Literal["train", "val"],
        data_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize CREMADDataset."""
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        super().__init__()
        self.split = split
        self.data_dir = (
            str(data_dir) if data_dir is not None else os.getenv("DATA_PATH")
        )
        self.seed = seed

        full_dataset = load_dataset(
            "AbstractTTS/CREMA-D",
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
                - "metadata": Metadata dictionary with keys 'gender', 'transcription', and 'label'.
        """
        sample = self.dataset[idx]
        waveform = torch.as_tensor(
            sample["audio"]["array"], dtype=torch.float32
        ).unsqueeze(0)
        sample_rate = int(sample["audio"]["sampling_rate"])

        metadata = {
            "gender": sample["gender"],
            "transcription": sample["transcription"],
            "label": sample["major_emotion"],
        }

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
