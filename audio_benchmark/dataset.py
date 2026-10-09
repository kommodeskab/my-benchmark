"""Dataset utilities and validation for the audio benchmark suite."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torchaudio.functional
from torch.utils.data import Dataset

from audio_benchmark.exceptions import InvalidDatasetError

ALLOWED_METADATA_KEYS = frozenset(
    {"label", "target", "frame_labels", "frame_targets", "tokens"}
)


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

    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
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
    metadatas = [item["metadata"] for item in parsed_items]

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
    """Wrapper dataset that resamples audio and pads or crops audio and frame metadata.

    Wraps an underlying dataset that returns dictionaries containing 'waveform' of shape (1, T),
    'sample_rate', and 'metadata'. Resamples audio to `sample_rate` and pads or crops to `length`
    samples. Frame-level metadata ('frame_labels' and 'frame_targets') is resampled and padded/cropped
    to match the audio waveform.

    Args:
        dataset: Underlying PyTorch-compatible dataset.
        dataset_name: Explicit identifier name for the dataset.
        sample_rate: Target audio sampling rate in Hz.
        length: Target audio length in samples. If None, audio is resampled without padding or cropping.

    Raises:
        ValueError: If dataset_name is empty, or sample_rate/length are not positive integers.
        InvalidDatasetError: If dataset does not implement __len__ and __getitem__.
    """

    def __init__(
        self,
        dataset: Any,
        dataset_name: str,
        sample_rate: int,
        length: int | None = None,
    ) -> None:
        """Initialize the BaseDataset wrapper.

        Args:
            dataset: Underlying PyTorch dataset.
            dataset_name: Explicit identifier name for the dataset.
            sample_rate: Target audio sampling rate in Hz.
            length: Target audio length in samples (None for no padding/cropping).

        Raises:
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
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate}."
            )
        if length is not None and (
            not isinstance(length, int) or isinstance(length, bool) or length <= 0
        ):
            raise ValueError(f"length must be a positive integer, got {length}.")

        self.dataset = dataset
        self.dataset_name = dataset_name.strip()
        self.name = self.dataset_name
        self.sample_rate = sample_rate
        self.length = length

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

        if not any(k in ALLOWED_METADATA_KEYS for k in metadata):
            raise InvalidDatasetError(
                "Metadata must contain at least one of 'label', 'target', "
                "'frame_labels', 'frame_targets', or 'tokens'."
            )

        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.sample_rate
            )
            new_len = waveform.shape[-1]
            if "frame_targets" in metadata:
                metadata["frame_targets"] = self._resample_sequence(
                    metadata["frame_targets"], new_len, mode="mean"
                )
            if "frame_labels" in metadata:
                metadata["frame_labels"] = self._resample_sequence(
                    metadata["frame_labels"], new_len, mode="mode"
                )

        if self.length is not None:
            cur_len = waveform.shape[-1]
            if cur_len > self.length:
                waveform = waveform[..., : self.length]
                if "frame_targets" in metadata:
                    metadata["frame_targets"] = metadata["frame_targets"][: self.length]
                if "frame_labels" in metadata:
                    metadata["frame_labels"] = metadata["frame_labels"][: self.length]
            elif cur_len < self.length:
                pad_amount = self.length - cur_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount), value=0.0)
                if "frame_targets" in metadata:
                    metadata["frame_targets"] = self._pad_sequence(
                        metadata["frame_targets"], pad_amount, pad_val=0.0
                    )
                if "frame_labels" in metadata:
                    metadata["frame_labels"] = self._pad_sequence(
                        metadata["frame_labels"], pad_amount, pad_val=0
                    )

        return {
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "metadata": metadata,
        }

    def _resample_sequence(self, sequence: Any, target_len: int, mode: str) -> Any:
        """Resample a sequence to target_len using downsample_sequence.

        Args:
            sequence: Sequence to resample.
            target_len: Target length in frames.
            mode: Downsample aggregation mode ('mean' or 'mode').

        Returns:
            Resampled sequence preserving container type.
        """
        if len(sequence) == target_len:
            return sequence

        from audio_benchmark.tasks.downsampling import downsample_sequence

        if isinstance(sequence, torch.Tensor):
            res = downsample_sequence(
                sequence.detach().cpu().numpy(), target_len, mode=mode
            )
            return torch.as_tensor(res, dtype=sequence.dtype, device=sequence.device)
        elif isinstance(sequence, np.ndarray):
            res = downsample_sequence(sequence, target_len, mode=mode)
            return res.astype(sequence.dtype)

        res = downsample_sequence(sequence, target_len, mode=mode)
        if mode == "mode" and sequence:
            elem_type = type(sequence[0])
            return [elem_type(x) for x in res]
        return res.tolist()

    def _pad_sequence(self, sequence: Any, pad_amount: int, pad_val: float) -> Any:
        """Pad a sequence with pad_val.

        Args:
            sequence: Sequence to pad.
            pad_amount: Number of elements to pad at the end.
            pad_val: Constant value to pad with.

        Returns:
            Padded sequence preserving container type.
        """
        if isinstance(sequence, torch.Tensor):
            return torch.nn.functional.pad(sequence, (0, pad_amount), value=pad_val)
        elif isinstance(sequence, np.ndarray):
            return np.pad(
                sequence, (0, pad_amount), mode="constant", constant_values=pad_val
            )
        elif isinstance(sequence, tuple):
            return tuple(list(sequence) + [pad_val] * pad_amount)
        return list(sequence) + [pad_val] * pad_amount
