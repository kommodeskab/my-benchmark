"""Global audio regression task."""

import time
from collections.abc import Callable
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
        pooling: str | Callable[[torch.Tensor], torch.Tensor] = "mean",
        metric: str = "r2",
    ):
        """Initialize the RegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Key in the sample metadata dictionary containing the regression target.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last',
                or a callable f(representations) -> pooled_tensor.
            metric: Primary metric for the task ('r2', 'mse', 'rmse', 'mae', 'pearson_r').
        """
        super().__init__(name=name)
        valid_metrics = ("r2", "mse", "rmse", "mae", "pearson_r")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        self.metric = metric
        self.target_key = target_key
        if regressor is None:
            raise TypeError(
                "regressor must be an instantiated BaseRegressor instance, got None. "
                "Passing None as a model is not allowed."
            )
        elif isinstance(regressor, str):
            raise TypeError(
                f"Passing strings as models is not supported (got '{regressor}'). "
                "Please pass an instantiated BaseRegressor instance."
            )
        elif not isinstance(regressor, BaseRegressor):
            raise TypeError(
                f"regressor must be an instantiated BaseRegressor instance, got {type(regressor).__name__}."
            )
        self.regressor = regressor
        self.pooling = pooling

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected target_key."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

    def pool_representations(self, representations: torch.Tensor) -> np.ndarray:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

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
        if len(train_representations) != len(train_metadata):
            raise TaskError(
                f"Length mismatch: {len(train_representations)} train representations vs {len(train_metadata)} train metadata entries."
            )
        if len(test_representations) != len(test_metadata):
            raise TaskError(
                f"Length mismatch: {len(test_representations)} test representations vs {len(test_metadata)} test metadata entries."
            )
        if len(train_representations) == 0:
            raise TaskError("No train representations provided for evaluation.")
        if len(test_representations) == 0:
            raise TaskError("No test representations provided for evaluation.")

        raw_train_targets = []
        for i, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_train_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Train sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        raw_test_targets = []
        for i, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_test_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Test sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        y_train = np.array(raw_train_targets, dtype=np.float32)
        y_test = np.array(raw_test_targets, dtype=np.float32)

        regressor_inst = self.regressor
        regressor_name = regressor_inst.__class__.__name__

        is_sequence_probe = getattr(regressor_inst, "accepts_sequence", False)
        if is_sequence_probe:
            X_train = train_representations.detach().cpu().numpy()
            X_test = test_representations.detach().cpu().numpy()
            pooling_name = "none (transformer)"
        else:
            X_train = self.pool_representations(train_representations)
            X_test = self.pool_representations(test_representations)
            pooling_name = self.pooling if isinstance(self.pooling, str) else "custom"

        start_time = time.perf_counter()
        regressor_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = regressor_inst.evaluate(X_test, y_test)
        train_metrics = regressor_inst.evaluate(X_train, y_train)

        results = {
            "task": self.name,
            "regressor": regressor_name,
            "pooling": pooling_name,
            "value": float(metrics[self.metric]),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "training_time": training_time,
        }
        if hasattr(regressor_inst, "best_params") and regressor_inst.best_params:
            results["best_params"] = regressor_inst.best_params
        if (
            hasattr(regressor_inst, "best_score")
            and regressor_inst.best_score is not None
        ):
            results["best_cv_score"] = regressor_inst.best_score

        return results

    def examine_data(self, dataset: Any, testset: Any | None = None) -> dict[str, Any]:
        """Examine the dataset in the context of global regression.

        Validates the datasets, checks target distributions across train and test sets,
        and computes a trivial baseline (mean predictor) expected metric.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.target_key])
        train_metadata = self.extract_metadata(dataset)

        if testset is not None:
            self._validate_for_examination(testset, [self.target_key])
            test_metadata = self.extract_metadata(testset)
        else:
            test_metadata = train_metadata

        if len(train_metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        raw_train_targets = []
        for i, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_train_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        raw_test_targets = []
        for i, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_test_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        y_train = np.array(raw_train_targets, dtype=np.float32)
        y_test = np.array(raw_test_targets, dtype=np.float32)

        train_mean = float(np.mean(y_train))
        y_pred_baseline = np.full_like(y_test, train_mean)

        baseline_mse = float(mean_squared_error(y_test, y_pred_baseline))
        baseline_rmse = float(np.sqrt(baseline_mse))
        baseline_mae = float(mean_absolute_error(y_test, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(y_test, y_pred_baseline))
        except (ValueError, ZeroDivisionError):
            baseline_r2 = 0.0

        if self.metric == "r2":
            baseline_value = baseline_r2
        elif self.metric == "mse":
            baseline_value = baseline_mse
        elif self.metric == "rmse":
            baseline_value = baseline_rmse
        elif self.metric == "mae":
            baseline_value = baseline_mae
        elif self.metric == "pearson_r":
            baseline_value = 0.0
        else:
            baseline_value = baseline_r2

        return {
            "task": self.name,
            "total_samples": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_samples": len(train_metadata),
            "test_samples": len(test_metadata) if testset is not None else 0,
            "target_key": self.target_key,
            "target_min": float(np.min(y_train)),
            "target_max": float(np.max(y_train)),
            "target_mean": float(np.mean(y_train)),
            "target_std": float(np.std(y_train)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }
