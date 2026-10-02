"""Global audio regression task."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
        test_size: float = 0.2,
        random_state: int = 42,
        split_key: str | None = None,
        metric: str = "r2",
    ):
        """Initialize the RegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Key in the sample metadata dictionary containing the regression target.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last',
                or a callable f(representations) -> pooled_tensor.
            test_size: Proportion of samples to use for test split if split_key is not used.
            random_state: Random seed for train/test split.
            split_key: Optional metadata key (e.g. 'split') specifying 'train' or 'test'.
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
        self.test_size = test_size
        self.random_state = random_state
        self.split_key = split_key

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected target_key."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

    def pool_representations(self, representations: torch.Tensor) -> np.ndarray:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train downstream regressor on pooled representations and return evaluation metrics.

        Args:
            representations: Computed representations of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each sample.

        Returns:
            dict[str, Any]: Dictionary of evaluation results.
        """
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} representations vs {len(metadata)} metadata entries."
            )

        if len(representations) == 0:
            raise TaskError("No representations provided for evaluation.")

        # Extract continuous targets
        raw_targets = []
        for i, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        targets = np.array(raw_targets, dtype=np.float32)

        # Regressor instance
        regressor_inst = self.regressor
        regressor_name = regressor_inst.__class__.__name__

        is_sequence_probe = getattr(regressor_inst, "accepts_sequence", False)
        if is_sequence_probe:
            features = representations.detach().cpu().numpy()
            pooling_name = "none (transformer)"
        else:
            features = self.pool_representations(representations)
            pooling_name = self.pooling if isinstance(self.pooling, str) else "custom"

        # Handle train/test split
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_mask = np.array([m[self.split_key] == "train" for m in metadata])
            test_mask = np.array([m[self.split_key] == "test" for m in metadata])

            X_train, y_train = features[train_mask], targets[train_mask]
            X_test, y_test = features[test_mask], targets[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(X_train)} train samples, {len(X_test)} test samples."
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                targets,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        # Fit regressor with timing
        start_time = time.perf_counter()
        regressor_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        # Evaluate on test set
        metrics = regressor_inst.evaluate(X_test, y_test)

        # Evaluate on train set for reference
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

    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of global regression.

        Validates the dataset, checks target distribution and train/test splits,
        and computes a trivial baseline (mean predictor) expected metric.

        Args:
            dataset: Audio dataset to examine.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.target_key])
        metadata = self.extract_metadata(dataset)

        if len(metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        raw_targets = []
        for i, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain target key '{self.target_key}'."
                )
            try:
                raw_targets.append(float(meta[self.target_key]))
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Sample at index {i} target '{meta[self.target_key]}' cannot be converted to float: {e}"
                ) from e

        targets = np.array(raw_targets, dtype=np.float32)
        indices = np.arange(len(targets))

        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_mask = np.array([m[self.split_key] == "train" for m in metadata])
            test_mask = np.array([m[self.split_key] == "test" for m in metadata])
            train_indices = indices[train_mask]
            test_indices = indices[test_mask]
            if len(train_indices) == 0 or len(test_indices) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_indices)} train samples, {len(test_indices)} test samples."
                )
        else:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        y_train = targets[train_indices]
        y_test = targets[test_indices]
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
            "total_samples": len(dataset),
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
            "target_key": self.target_key,
            "target_min": float(np.min(targets)),
            "target_max": float(np.max(targets)),
            "target_mean": float(np.mean(targets)),
            "target_std": float(np.std(targets)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }
