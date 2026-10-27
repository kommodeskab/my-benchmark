"""Dataset utilities and validation for the audio benchmark suite."""

from collections.abc import Sequence
from typing import Any

import torch
import torchaudio.functional
from torch.utils.data import Dataset

from audio_benchmark.exceptions import InvalidDatasetError


def parse_dataset_item(item: Any) -> dict[str, Any]:
    """Parse and validate an individual item returned by an audio dataset.

    All datasets supplied to the benchmark suite must return a dictionary with
    exactly three keys: "waveform", "sample_rate", and "metadata":
      - "waveform": torch.Tensor of shape (1, T)
      - "sample_rate": positive int
      - "metadata": dict containing sample metadata and task targets

    Args:
        item: The object returned by dataset[idx].

    Returns:
        dict[str, Any]: Validated dataset item dictionary.

    Raises:
        InvalidDatasetError: If item is not a dict, missing required keys,
            or contains invalid waveforms, sample_rate, or metadata.
    """
    if not isinstance(item, dict):
        raise InvalidDatasetError(
            f"Dataset __getitem__ must return a dict with keys 'waveform', 'sample_rate', and 'metadata', "
            f"but returned {type(item).__name__}."
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
            f"Dataset item['waveform'] must be a torch.Tensor, but got {type(waveform).__name__}."
        )

    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise InvalidDatasetError(
            f"Waveform must have shape (1, T), but got shape {tuple(waveform.shape)}."
        )

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise InvalidDatasetError(
            f"Dataset item['sample_rate'] must be a positive integer, but got {sample_rate!r}."
        )

    if not isinstance(metadata, dict):
        raise InvalidDatasetError(
            f"Dataset item['metadata'] must be a dict, but got {type(metadata).__name__}."
        )

    return item


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
    metadatas = []
    for item in parsed_items:
        meta = dict(item["metadata"])
        if "audio_length" not in meta:
            meta["audio_length"] = item["waveform"].shape[-1]
        metadatas.append(meta)

    lengths = [audio.shape[-1] for audio in audios]
    max_len = max(lengths)

    padded_audios = [
        torch.nn.functional.pad(a, (0, max_len - a.shape[-1]))
        if a.shape[-1] < max_len
        else a
        for a in audios
    ]
    return torch.stack(padded_audios, dim=0), metadatas


class BaseDataset(Dataset):
    """Wrapper dataset that resamples audio, normalizes waveforms, and pads or crops audio and frame metadata.

    Wraps an underlying dataset that returns dictionaries containing 'waveform' of shape (1, T),
    'sample_rate', and 'metadata'. Resamples audio to `sample_rate`, optionally normalizes audio
    samples to zero mean and unit variance, and pads or crops to `length` samples. When resampling
    audio, dictionary sample timestamps are scaled while uniform sequence targets are left
    unchanged. When cropping or padding to `length`, dictionary timestamps >= length are dropped,
    and uniform sequence targets are sliced or padded to match.

    Args:
        dataset: Underlying PyTorch-compatible dataset.
        dataset_name: Explicit identifier name for the dataset.
        sample_rate: Target audio sampling rate in Hz.
        length: Target audio length in samples. If None, audio is resampled without padding or cropping.
        normalize: Whether to normalize audio samples to mean 0 and standard deviation 1. Defaults to True.

    Raises:
        TypeError: If normalize is not a boolean.
        ValueError: If dataset_name is empty, or sample_rate/length are not positive integers.
        InvalidDatasetError: If dataset does not implement __len__ and __getitem__.
    """

    def __init__(
        self,
        dataset: Any,
        dataset_name: str,
        sample_rate: int,
        length: int | None = None,
        normalize: bool = True,
    ) -> None:
        """Initialize the BaseDataset wrapper.

        Args:
            dataset: Underlying PyTorch dataset.
            dataset_name: Explicit identifier name for the dataset.
            sample_rate: Target audio sampling rate in Hz.
            length: Target audio length in samples (None for no padding/cropping).
            normalize: Whether to normalize audio samples to mean 0 and standard deviation 1. Defaults to True.

        Raises:
            TypeError: If normalize is not a boolean.
            ValueError: If dataset_name is not a non-empty string, or sample_rate/length are not positive integers.
            InvalidDatasetError: If dataset does not implement __len__ and __getitem__.
        """
        super().__init__()
        if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise InvalidDatasetError(
                f"Wrapped dataset must implement __len__ and __getitem__, got {type(dataset).__name__}."
            )
        if isinstance(dataset, (str, bytes)):
            raise InvalidDatasetError("Wrapped dataset cannot be a string or bytes.")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string.")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate}."
            )
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError(f"length must be a positive integer, got {length}.")
        if not isinstance(normalize, bool):
            raise TypeError(
                f"normalize must be a boolean, got {type(normalize).__name__}."
            )

        self.dataset = dataset
        self.dataset_name = dataset_name.strip()
        self.name = self.dataset_name
        self.sample_rate = sample_rate
        self.length = length
        self.normalize = normalize

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the processed sample at index idx.

        Args:
            idx: Integer index of the sample.

        Returns:
            dict[str, Any]: Dictionary containing 'waveform', 'sample_rate', and 'metadata'.

        Raises:
            InvalidDatasetError: If item structure or types are invalid.
        """
        raw_item = self.dataset[idx]
        return self.process_item(raw_item)

    def process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Process an individual raw dataset sample.

        Resamples waveform audio to `self.sample_rate`, adjusts frame metadata,
        and pads or crops waveform and frame metadata to `self.length`.

        Args:
            item: Raw dataset sample dictionary containing 'waveform', 'sample_rate', and 'metadata'.

        Returns:
            dict[str, Any]: Processed sample dictionary with keys 'waveform', 'sample_rate', and 'metadata'.

        Raises:
            InvalidDatasetError: If item structure, waveform shape, or metadata keys are invalid.
        """
        parsed = parse_dataset_item(item)
        waveform = parsed["waveform"]
        orig_sr = parsed["sample_rate"]
        metadata = dict(parsed["metadata"])

        for key in ("frame_targets", "frame_labels"):
            if key in metadata:
                val = metadata[key]
                if isinstance(val, dict):
                    for k in val:
                        if not isinstance(k, int):
                            raise InvalidDatasetError(
                                f"Timestamp key {k!r} in '{key}' must be an integer, got {type(k).__name__}."
                            )
                elif isinstance(val, list):
                    pass
                else:
                    raise InvalidDatasetError(
                        f"Metadata key '{key}' must be a list or dict[int, Any], got {type(val).__name__}."
                    )

        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.sample_rate
            )
            ratio = self.sample_rate / orig_sr
            for key in ("frame_targets", "frame_labels"):
                if key in metadata:
                    val = metadata[key]
                    if isinstance(val, dict):
                        metadata[key] = {round(k * ratio): v for k, v in val.items()}

        if self.length is not None:
            cur_len = waveform.shape[-1]
            if cur_len > self.length:
                waveform = waveform[..., : self.length]
                for key in ("frame_targets", "frame_labels"):
                    if key in metadata:
                        val = metadata[key]
                        if isinstance(val, dict):
                            metadata[key] = {
                                k: v for k, v in val.items() if k < self.length
                            }
                        else:
                            keep_len = (
                                self.length
                                if len(val) == cur_len
                                else max(1, round(len(val) * self.length / cur_len))
                            )
                            metadata[key] = val[:keep_len]

        if self.normalize:
            waveform = (waveform - waveform.mean()) / waveform.std()

        if self.length is not None and waveform.shape[-1] < self.length:
            cur_len = waveform.shape[-1]
            pad_amount = self.length - cur_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount), value=0.0)
            pad_val_map = {"frame_targets": 0.0, "frame_labels": 0}
            for key in ("frame_targets", "frame_labels"):
                if key in metadata:
                    val = metadata[key]
                    if isinstance(val, dict):
                        pass
                    else:
                        target_pad = (
                            pad_amount
                            if len(val) == cur_len
                            else round(len(val) * self.length / cur_len) - len(val)
                        )
                        if target_pad > 0:
                            metadata[key] = val + [pad_val_map.get(key, 0)] * target_pad

        metadata["audio_length"] = waveform.shape[-1]
        return {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "metadata": metadata,
        }
