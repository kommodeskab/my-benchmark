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
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate representations on this task using explicit train and test sets.

        Args:
            train_representations: Tensor of shape (N_train, T', C').
            train_metadata: List of metadata dicts corresponding to each train sample.
            test_representations: Tensor of shape (N_test, T', C').
            test_metadata: List of metadata dicts corresponding to each test sample.

        Returns:
            dict[str, Any]: Dictionary containing evaluation metrics.
        """

    @abstractmethod
    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of this task.

        Returns dataset characteristics (such as sample counts, class or target
        statistics) and expected baseline metric performance.

        Args:
            dataset: PyTorch dataset for training.
            testset: Optional PyTorch dataset for testing.

        Returns:
            dict[str, Any]: Summary statistics and baseline metrics.
        """

    @staticmethod
    def extract_metadata(dataset: Dataset) -> list[dict[str, Any]]:
        """Extract metadata dictionaries from each sample in the dataset.

        Args:
            dataset: PyTorch dataset.

        Returns:
            list[dict[str, Any]]: List of metadata dictionaries.
        """
        return [dataset[idx]["metadata"] for idx in range(len(dataset))]

    @staticmethod
    def normalize_metric(value: float, m_min: float, m_max: float) -> float:
        """Normalize a raw metric value to [0, 1] using theoretical worst/best bounds.

        Computes M_hat = (M - M^min) / (M^max - M^min), clamped to [0.0, 1.0],
        where M^min is the worst possible value and M^max is the best possible value.

        Args:
            value: Raw metric value M.
            m_min: Worst possible metric bound M^min.
            m_max: Best possible metric bound M^max.

        Returns:
            float: Normalized metric value in [0.0, 1.0].
        """
        normalized = (float(value) - m_min) / (m_max - m_min)
        return float(min(max(normalized, 0.0), 1.0))

    @staticmethod
    def apply_pooling(
        representations: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        """Apply temporal pooling across representations of shape (N, T', C').

        Args:
            representations: Tensor of shape (N, T', C').
            pooling: Pooling mode ('mean', 'max', 'first', 'last').

        Returns:
            torch.Tensor: Pooled 2D feature matrix of shape (N, C').

        Raises:
            ValueError: If pooling method is unsupported.
        """
        if pooling == "mean":
            return representations.mean(dim=1)
        elif pooling == "max":
            return representations.amax(dim=1)
        elif pooling == "first":
            return representations[:, 0, :]
        elif pooling == "last":
            return representations[:, -1, :]
        else:
            raise ValueError(
                f"Unknown pooling method '{pooling}'. Choose from 'mean', 'max', 'first', 'last'."
            )
