"""Base dataset wrapper for dynamically annotating audio datasets with metadata."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item


class BaseMetadataWrapper(Dataset):
    """Abstract base wrapper for dynamically adding metadata keys to audio datasets.

    Wraps an underlying audio Dataset conforming to the 3-key dictionary format:
    {"waveform": torch.Tensor, "sample_rate": int, "metadata": dict}.

    Parameters
    ----------
    dataset : Dataset
        The underlying audio dataset.
    key : str
        The metadata key that will be added or updated in `metadata`.
    cache : bool, default True
        If True, caches generated metadata in-memory for each index `idx` after first access.
    overwrite : bool, default False
        If False and `key` already exists in sample `metadata`, reuses the existing value
        without re-generating. If True, always re-generates and overwrites.
    """

    def __init__(
        self,
        dataset: Dataset,
        key: str,
        cache: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.dataset = dataset
        self.key = key
        self.cache = cache
        self.overwrite = overwrite
        self._cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def clear_cache(self) -> None:
        """Clears the in-memory annotation cache."""
        self._cache.clear()

    def generate(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        metadata: dict[str, Any],
    ) -> Any:
        """Generate metadata value for a given audio sample.

        Subclasses must implement this method.

        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio waveform tensor of shape `(1, T)` or `(T,)`.
        sample_rate : int
            Audio sample rate.
        metadata : dict[str, Any]
            Current metadata dictionary.

        Returns
        -------
        Any
            The generated metadata value for `self.key`.
        """
        raise NotImplementedError("Subclasses must implement `generate`.")

    def precompute(self, show_progress: bool = False) -> BaseMetadataWrapper:
        """Convenience method to precompute and cache annotations for all samples.

        Parameters
        ----------
        show_progress : bool, default False
            If True and `tqdm` is installed, displays a progress bar.

        Returns
        -------
        BaseMetadataWrapper
            Self reference for chaining.
        """
        indices = range(len(self))
        if show_progress:
            try:
                from tqdm import tqdm

                indices = tqdm(indices, desc=f"Precomputing {self.__class__.__name__}")
            except ImportError:
                pass

        for idx in indices:
            _ = self[idx]
        return self

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        parsed = parse_dataset_item(item)
        waveform = parsed["waveform"]
        sample_rate = parsed["sample_rate"]
        metadata = dict(parsed["metadata"])

        # If key already exists in sample metadata and overwrite is False, keep existing
        if (
            not self.overwrite
            and self.key in metadata
            and metadata[self.key] is not None
        ):
            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "metadata": metadata,
            }

        # Check in-memory cache
        if self.cache and idx in self._cache:
            metadata[self.key] = self._cache[idx]
            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "metadata": metadata,
            }

        # Generate dynamically
        generated_val = self.generate(waveform, sample_rate, metadata)

        if self.cache:
            self._cache[idx] = generated_val

        metadata[self.key] = generated_val
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
