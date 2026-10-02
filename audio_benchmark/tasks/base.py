"""Base interface for benchmark tasks."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, name: str):
        """Initialize base task.

        Args:
            name: Identifier for the task.
        """
        self.name = name

    @abstractmethod
    def validate_dataset(self, dataset: Dataset) -> None:
        """Validate that the dataset contains necessary metadata and format for this task.

        Args:
            dataset: PyTorch dataset.

        Raises:
            InvalidDatasetError: If dataset does not satisfy task requirements.
        """

    @abstractmethod
    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate representations on this task.

        Args:
            representations: Tensor of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each sample.

        Returns:
            dict[str, Any]: Dictionary containing evaluation metrics.
        """

    @abstractmethod
    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of this task.

        Returns dataset characteristics (such as sample counts, splits, class or target
        statistics) and expected baseline metric performance.

        Args:
            dataset: PyTorch dataset.

        Returns:
            dict[str, Any]: Summary statistics and baseline metrics.
        """

    def _validate_for_examination(
        self,
        dataset: Any,
        required_metadata_keys: Any = None,
    ) -> None:
        """Validate dataset before examination.

        If dataset is a full Dataset (with waveforms), delegates to self.validate_dataset.
        If dataset is a list/tuple of metadata dictionaries, validates the metadata format and keys.

        Args:
            dataset: PyTorch dataset or sequence of metadata dicts.
            required_metadata_keys: Sequence of required metadata keys.
        """
        from audio_benchmark.exceptions import InvalidDatasetError, TaskError

        if not hasattr(dataset, "__len__"):
            raise InvalidDatasetError("Dataset must implement `__len__`.")
        if len(dataset) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        first_item = dataset[0]
        # If it's a raw metadata dict without waveform
        if isinstance(first_item, dict) and "waveform" not in first_item:
            if required_metadata_keys:
                for k in required_metadata_keys:
                    if k not in first_item:
                        raise InvalidDatasetError(
                            f"Sample at index 0 metadata is missing required key(s): ['{k}']. Available metadata keys: {list(first_item.keys())}."
                        )
        else:
            self.validate_dataset(dataset)

    @staticmethod
    def extract_metadata(dataset: Any) -> list[dict[str, Any]]:
        """Extract metadata dictionaries from each sample in the dataset.

        Args:
            dataset: PyTorch dataset or sequence of sample metadata dicts.

        Returns:
            list[dict[str, Any]]: List of metadata dictionaries.
        """
        if isinstance(dataset, (list, tuple)) and (
            len(dataset) == 0
            or (isinstance(dataset[0], dict) and "waveform" not in dataset[0])
        ):
            return list(dataset)

        if hasattr(dataset, "metadata") and isinstance(dataset.metadata, list):
            return dataset.metadata

        from audio_benchmark.dataset import parse_dataset_item

        metadata_list = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            if (
                isinstance(item, dict)
                and "metadata" in item
                and isinstance(item["metadata"], dict)
            ):
                metadata_list.append(item["metadata"])
            else:
                parsed = parse_dataset_item(item)
                metadata_list.append(parsed["metadata"])
        return metadata_list

    @staticmethod
    def apply_pooling(
        representations: torch.Tensor,
        pooling: str | Any,
    ) -> Any:
        """Apply temporal pooling across representations of shape (N, T', C').

        Args:
            representations: Tensor of shape (N, T', C').
            pooling: Pooling mode ('mean', 'max', 'first', 'last') or custom callable.

        Returns:
            np.ndarray: Pooled 2D feature matrix of shape (N, C').
        """
        import numpy as np

        if callable(pooling):
            pooled = pooling(representations)
        elif pooling == "mean":
            pooled = representations.mean(dim=1)
        elif pooling == "max":
            pooled = representations.amax(dim=1)
        elif pooling == "first":
            pooled = representations[:, 0, :]
        elif pooling == "last":
            pooled = representations[:, -1, :]
        else:
            raise ValueError(
                f"Unknown pooling method '{pooling}'. Choose from 'mean', 'max', 'first', 'last' or callable."
            )

        if isinstance(pooled, torch.Tensor):
            return pooled.detach().cpu().numpy()
        return np.asarray(pooled)
