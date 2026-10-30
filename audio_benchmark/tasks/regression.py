"""Global audio regression task."""

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset

from audio_benchmark.dataset import validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.regressors.models import BaseRegressor
from audio_benchmark.tasks.base import BaseTask


class RegressionTask(BaseTask):
    """Regression task: predicting a continuous target for the entire audio clip.

    The continuous target is extracted from the metadata dictionary of each sample.
    The sequence representations (N, T', C') are pooled over time to produce
    clip-level features (N, C'), which are then fitted and evaluated using
    an instantiated BaseRegressor model.
    """

    def __init__(
        self,
        regressor: BaseRegressor,
        name: str = "regression",
        target_key: str = "target",
        pooling: str = "mean",
    ):
        """Initialize the RegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Key in the sample metadata dictionary containing the regression target.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last'.

        Raises:
            TypeError: If regressor is not an instance of BaseRegressor.
        """
        super().__init__(name=name)
        if not isinstance(regressor, BaseRegressor):
            raise TypeError(
                f"regressor must be an instantiated BaseRegressor instance, got {type(regressor).__name__}."
            )

        self.regressor = regressor
        self.target_key = target_key
        self.pooling = pooling
        self.metric = "r2"
        self.m_min = 0.0
        self.m_max = 1.0

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected target_key."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

    def pool_representations(self, representations: torch.Tensor) -> torch.Tensor:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

    def _extract_targets(
        self,
        metadata: list[dict[str, Any]],
        prefix: str = "Sample",
    ) -> list[float]:
        """Extract and convert continuous targets from metadata to floats."""
        targets: list[float] = []
        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"{prefix} at index {idx} metadata does not contain target key '{self.target_key}'."
                )
            try:
                targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"{prefix} at index {idx} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e
        return targets

    def _extract_split_targets(
        self,
        representations: Any,
        metadata: list[dict[str, Any]],
        split_name: str,
    ) -> torch.Tensor:
        """Validate split lengths and return float32 target tensor."""
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} {split_name} representations vs "
                f"{len(metadata)} {split_name} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError(f"No {split_name} representations provided for evaluation.")

        raw_targets = self._extract_targets(
            metadata, prefix=f"{split_name.capitalize()} sample"
        )
        return torch.as_tensor(raw_targets, dtype=torch.float32)

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train downstream regressor on pooled representations and return evaluation metrics.

        Args:
            train_representations: Computed train representations of shape (N_train, T', C').
            train_metadata: List of metadata dicts corresponding to each train sample.
            test_representations: Computed test representations of shape (N_test, T', C').
            test_metadata: List of metadata dicts corresponding to each test sample.

        Returns:
            dict[str, Any]: Dictionary of evaluation results.
        """
        y_train = self._extract_split_targets(
            train_representations, train_metadata, "train"
        )
        y_test = self._extract_split_targets(
            test_representations, test_metadata, "test"
        )

        if getattr(self.regressor, "accepts_sequence", False):
            X_train = train_representations
            X_test = test_representations
            pooling_name = "none (transformer)"
        else:
            X_train = self.pool_representations(train_representations)
            X_test = self.pool_representations(test_representations)
            pooling_name = self.pooling

        start_time = time.perf_counter()
        self.regressor.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = self.regressor.evaluate(X_test, y_test)
        train_metrics = self.regressor.evaluate(X_train, y_train)
        value = float(metrics[self.metric])

        return {
            "task": self.name,
            "regressor": self.regressor.__class__.__name__,
            "pooling": pooling_name,
            "value": value,
            "normalized_value": self.normalize_metric(value, self.m_min, self.m_max),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "training_time": training_time,
        }

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of global regression.

        Validates the datasets, checks target distributions across train and test sets,
        and computes a trivial baseline (mean predictor) expected metric.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be an instance of Dataset, got {type(dataset).__name__}."
            )
        if testset is not None and not isinstance(testset, Dataset):
            raise TypeError(
                f"testset must be an instance of Dataset, got {type(testset).__name__}."
            )
        self.validate_dataset(dataset)
        if testset is not None:
            self.validate_dataset(testset)

        train_metadata = self.extract_metadata(dataset)
        test_metadata = (
            self.extract_metadata(testset) if testset is not None else train_metadata
        )

        y_train = np.array(self._extract_targets(train_metadata), dtype=np.float32)
        y_test = np.array(self._extract_targets(test_metadata), dtype=np.float32)

        train_mean = float(np.mean(y_train))
        y_pred_baseline = np.full_like(y_test, train_mean)

        baseline_mse = float(mean_squared_error(y_test, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(y_test, y_pred_baseline))
        except (ValueError, ZeroDivisionError):
            baseline_r2 = 0.0

        return {
            "task": self.name,
            "total_samples": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_samples": len(train_metadata),
            "test_samples": len(test_metadata) if testset is not None else 0,
            "target_key": self.target_key,
            "target_min": float(np.min(y_train)),
            "target_max": float(np.max(y_train)),
            "target_mean": train_mean,
            "target_std": float(np.std(y_train)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": baseline_r2,
            "baseline_normalized_value": self.normalize_metric(
                baseline_r2, self.m_min, self.m_max
            ),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": float(np.sqrt(baseline_mse)),
            "baseline_mae": float(mean_absolute_error(y_test, y_pred_baseline)),
        }
