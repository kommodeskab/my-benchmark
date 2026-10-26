"""TIMIT dataset implementation for audio benchmark."""

import os
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class TIMITDataset(Dataset):
    """TIMIT acoustic-phonetic continuous speech corpus dataset.

    Loads the TIMIT dataset from Hugging Face ("Andemand11/TIMIT") with speaker,
    dialect, phonetic, and speech segment metadata.

    Args:
        split: Dataset split to load ('train' or 'test').
        data_dir: Optional path or cache directory for dataset storage. If None,
            uses the DATA_PATH environment variable if set.

    Raises:
        ValueError: If split is not 'train' or 'test'.
    """

    def __init__(
        self,
        split: Literal["train", "test"],
        data_dir: str | Path | None = None,
    ) -> None:
        """Initialize TIMITDataset."""
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        super().__init__()
        self.split = split
        self.data_dir = (
            str(data_dir) if data_dir is not None else os.getenv("DATA_PATH")
        )

        self.dataset = load_dataset(
            "Andemand11/TIMIT",
            split=split,
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
                - "metadata": Metadata dictionary containing 'dialect_region', 'speaker_id',
                  'text', 'phonemes', 'speech_start', 'speech_end', and 'speech_duration'.
        """
        sample = self.dataset[idx]
        waveform = torch.as_tensor(
            sample["audio"]["array"], dtype=torch.float32
        ).unsqueeze(0)
        sample_rate = int(sample["audio"]["sampling_rate"])

        text = sample["words"]
        phonemes = sample["phonemes"]
        speech_start = float(sample["words"]["start"][0]) / sample_rate
        speech_end = float(sample["words"]["end"][-1]) / sample_rate
        speech_duration = speech_end - speech_start

        metadata = {
            "dialect_region": sample["dialect_region"],
            "speaker_id": sample["speaker_id"],
            "text": text["text"],
            "phonemes": phonemes["text"],
            "speech_start": speech_start,
            "speech_end": speech_end,
            "speech_duration": speech_duration,
        }

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
