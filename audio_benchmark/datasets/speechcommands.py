"""Speech Commands dataset implementation for audio benchmark."""

import os
from pathlib import Path
from typing import Any, Literal

import torchaudio
from torch.utils.data import Dataset

SPEECHCOMMANDS_LABELS: tuple[str, ...] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)


class SpeechCommandsDataset(Dataset):
    """Speech Commands dataset for spoken command classification benchmarking.

    Uses `torchaudio.datasets.SPEECHCOMMANDS` to load 35 spoken command words.

    Args:
        split: Dataset split to load ('train', 'val', or 'test').
        data_dir: Optional path to data storage directory. If None,
            uses the DATA_PATH environment variable or './data'.
        download: Whether to download dataset files if not already present. Defaults to True.

    Raises:
        ValueError: If split is not 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        data_dir: str | Path | None = None,
        download: bool = True,
    ) -> None:
        """Initialize SpeechCommandsDataset."""
        split_map: dict[str, str] = {
            "train": "training",
            "val": "validation",
            "test": "testing",
        }
        if split not in split_map:
            raise ValueError(
                f"split must be one of {list(split_map.keys())}, got {split!r}"
            )

        super().__init__()
        self.split = split
        resolved_root = (
            str(data_dir)
            if data_dir is not None
            else (os.getenv("DATA_PATH") or "./data")
        )
        if resolved_root == "./data":
            os.makedirs(resolved_root, exist_ok=True)
        self.data_dir = resolved_root
        self.download = download

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=self.data_dir,
            subset=split_map[split],
            download=download,
        )
        self.label_to_index: dict[str, int] = {
            label: index for index, label in enumerate(SPEECHCOMMANDS_LABELS)
        }

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
                - "metadata": Metadata dictionary with keys 'label', 'text', 'speaker_id',
                  and 'utterance_number'.
        """
        waveform, sample_rate, label_string, speaker_id, utterance_number = (
            self.dataset[idx]
        )
        label = self.label_to_index[label_string]

        return {
            "waveform": waveform,
            "sample_rate": int(sample_rate),
            "metadata": {
                "label": label,
                "text": label_string,
                "speaker_id": speaker_id,
                "utterance_number": utterance_number,
            },
        }
