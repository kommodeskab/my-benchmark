"""Dataset utilities and validation for the audio benchmark suite."""

from collections.abc import Sequence
from typing import Any

import torch

from audio_benchmark.exceptions import InvalidDatasetError


def parse_dataset_item(item: Any) -> dict[str, Any]:
    """Parse and validate an individual item returned by an audio dataset.

    All datasets supplied to the benchmark suite must return a dictionary with
    exactly three keys: "waveform", "sample_rate", and "metadata":
      - "waveform": torch.Tensor of shape (1, T) or (T,)
      - "sample_rate": int or float
      - "metadata": dict containing sample metadata and task targets

    Args:
        item: The object returned by dataset[idx].

    Returns:
        dict[str, Any]: Dictionary with validated "waveform" (shape (1, T)),
        "sample_rate" (int), and "metadata" (dict).

    Raises:
        InvalidDatasetError: If item is not a dict, missing required keys,
            or contains invalid waveforms/metadata.
    """
    if not isinstance(item, dict):
        raise InvalidDatasetError(
            f"Dataset __getitem__ must return a dict with keys 'waveform', 'sample_rate', and 'metadata', "
            f"but returned {type(item).__name__}. "
            "Please write a simple PyTorch Dataset wrapper around your dataset to format each sample as:\n\n"
            "    class AudioDatasetWrapper(torch.utils.data.Dataset):\n"
            "        def __init__(self, raw_dataset):\n"
            "            self.raw_dataset = raw_dataset\n"
            "        def __len__(self):\n"
            "            return len(self.raw_dataset)\n"
            "        def __getitem__(self, idx):\n"
            "            item = self.raw_dataset[idx]\n"
            "            return {\n"
            "                'waveform': torch.as_tensor(item['audio'], dtype=torch.float32),\n"
            "                'sample_rate': item.get('sample_rate', 16000),\n"
            "                'metadata': {'label': item['label'], 'target': item.get('target', 0.0)},\n"
            "            }"
        )

    required_keys = ("waveform", "sample_rate", "metadata")
    missing_keys = [k for k in required_keys if k not in item]
    if missing_keys:
        raise InvalidDatasetError(
            f"Dataset item is missing required key(s): {missing_keys}. "
            f"Expected keys: ['waveform', 'sample_rate', 'metadata']."
        )

    waveform = item["waveform"]
    sample_rate = item["sample_rate"]
    metadata = item["metadata"]

    if not isinstance(waveform, torch.Tensor):
        raise InvalidDatasetError(
            f"Dataset item['waveform'] must be a torch.Tensor, but got {type(waveform).__name__}. "
            "Please convert your audio samples to torch.Tensor (e.g. `torch.from_numpy(waveform)`)."
        )

    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise InvalidDatasetError(
            f"Dataset item['sample_rate'] must be a positive integer or float, but got {sample_rate!r}."
        )

    if not isinstance(metadata, dict):
        raise InvalidDatasetError(
            f"Dataset item['metadata'] must be a dict, but got {type(metadata).__name__}. "
            "Please provide metadata as a dictionary."
        )

    # Convert to float tensor if integer
    if not waveform.is_floating_point():
        waveform = waveform.float()

    # Waveform should have shape (1, T) or (T,)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Shape (1, T)
    elif waveform.ndim == 2:
        if waveform.shape[0] != 1:
            raise InvalidDatasetError(
                f"Waveform must have 1 channel (shape (1, T)), but got shape {tuple(waveform.shape)}."
            )
    else:
        raise InvalidDatasetError(
            f"Waveform must be 1D (T,) or 2D (1, T), but got shape {tuple(waveform.shape)}."
        )

    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
        "metadata": metadata,
    }


def validate_dataset(
    dataset: Any,
    required_metadata_keys: Sequence[str] | None = None,
    num_samples_to_check: int = 5,
) -> None:
    """Validate that a user-supplied dataset satisfies benchmark requirements.

    Args:
        dataset: The PyTorch-like dataset object.
        required_metadata_keys: Optional list of keys that must exist in metadata
            (e.g., ['label'] for classification, ['target'] for regression).
        num_samples_to_check: Number of samples at the start of the dataset to check.

    Raises:
        InvalidDatasetError: If validation fails.
    """
    if not hasattr(dataset, "__len__"):
        raise InvalidDatasetError("Dataset must implement `__len__`.")
    if not hasattr(dataset, "__getitem__"):
        raise InvalidDatasetError("Dataset must implement `__getitem__`.")

    length = len(dataset)
    if length == 0:
        raise InvalidDatasetError("Dataset is empty (len == 0).")

    samples_to_check = min(length, max(1, num_samples_to_check))
    for idx in range(samples_to_check):
        try:
            item = dataset[idx]
        except Exception as e:
            raise InvalidDatasetError(
                f"Failed to access sample at index {idx} in dataset: {e}"
            ) from e

        parsed = parse_dataset_item(item)
        metadata = parsed["metadata"]

        if required_metadata_keys:
            missing_keys = [k for k in required_metadata_keys if k not in metadata]
            if missing_keys:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata is missing required key(s): {missing_keys}. "
                    f"Available metadata keys: {list(metadata.keys())}."
                )


def collate_audio_batch(
    batch: Sequence[Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Collate a batch of raw dataset items into a batched audio tensor and metadata list.

    Audio tensors are padded along the time dimension if lengths differ, resulting
    in shape (B, 1, max_T).

    Args:
        batch: List of items from dataset.

    Returns:
        tuple[torch.Tensor, list[dict[str, Any]]]: Batched audio (B, 1, T) and metadata dicts.
    """
    parsed_items = [parse_dataset_item(item) for item in batch]
    audios = [item["waveform"] for item in parsed_items]
    metadatas = [item["metadata"] for item in parsed_items]

    # Check if all audios have the same length
    lengths = [audio.shape[-1] for audio in audios]
    max_len = max(lengths)

    if all(l == max_len for l in lengths):
        batched_audio = torch.stack(audios, dim=0)  # Shape (B, 1, T)
    else:
        # Pad with zeros to max_len
        padded_audios = []
        for a in audios:
            pad_amount = max_len - a.shape[-1]
            if pad_amount > 0:
                padded = torch.nn.functional.pad(a, (0, pad_amount))
            else:
                padded = a
            padded_audios.append(padded)
        batched_audio = torch.stack(padded_audios, dim=0)

    return batched_audio, metadatas


def get_dataset_name(dataset: Any, default: str = "dataset") -> str:
    """Retrieve an identifiable name for a dataset."""
    if (
        hasattr(dataset, "name")
        and isinstance(dataset.name, str)
        and dataset.name.strip()
    ):
        return dataset.name.strip()
    if (
        hasattr(dataset, "dataset_name")
        and isinstance(dataset.dataset_name, str)
        and dataset.dataset_name.strip()
    ):
        return dataset.dataset_name.strip()
    return dataset.__class__.__name__
