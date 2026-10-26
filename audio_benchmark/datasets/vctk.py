"""VCTK dataset implementation for audio benchmark."""

import os
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class VCTKDataset(Dataset):
    """VCTK corpus audio dataset for speaker and speech benchmarking.

    Loads the VCTK dataset from Hugging Face ("badayvedat/VCTK") with accent,
    age, gender, and transcription text metadata.

    Args:
        split: Dataset split to load ('train' or 'val').
        data_dir: Optional path or cache directory for dataset storage. If None,
            uses the DATA_PATH environment variable if set.

    Raises:
        ValueError: If split is not 'train' or 'val'.
    """

    def __init__(
        self,
        split: Literal["train", "val"],
        data_dir: str | Path | None = None,
    ) -> None:
        """Initialize VCTKDataset."""
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        super().__init__()
        self.split = split
        self.data_dir = (
            str(data_dir) if data_dir is not None else os.getenv("DATA_PATH")
        )

        hf_split = "validation" if split == "val" else split
        self.dataset = load_dataset(
            "badayvedat/VCTK",
            split=hf_split,
            cache_dir=self.data_dir,
        )

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
                - "metadata": Metadata dictionary containing 'accent', 'age',
                  'gender', and 'text'.
        """
        sample = self.dataset[idx]
        waveform = torch.as_tensor(
            sample["flac"]["array"], dtype=torch.float32
        ).unsqueeze(0)
        sample_rate = int(sample["flac"]["sampling_rate"])

        metadata = {
            "accent": sample["accent"],
            "age": sample["age"],
            "gender": sample["gender"],
            "text": sample["txt"],
        }

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
