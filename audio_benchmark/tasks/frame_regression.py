"""Frame-level audio regression task."""

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.regressors.models import BaseRegressor
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import downsample_sequence


class FrameRegressionTask(BaseTask):
    """Frame-level audio regression task.

    Each frame of the input audio has a continuous target in sample metadata.
    When representation models downsample audio along the temporal dimension
    from T to T', this task downsamples the frame targets to T' by computing
    the mean of values in each downsampled region.
    """

    def __init__(
        self,
        regressor: BaseRegressor,
        name: str = "frame_regression",
        target_key: str = "frame_targets",
        metric: str = "r2",
    ):
        """Initialize the FrameRegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Metadata key containing the per-frame target sequence (length T).
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

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains per-frame targets matching audio length T."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        parsed = parse_dataset_item(dataset[0])
        audio_length = parsed["waveform"].shape[-1]
        targets = parsed["metadata"][self.target_key]
        if not hasattr(targets, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' must be a sequence of length {audio_length}, "
                f"got {type(targets).__name__}."
            )
        if len(targets) != audio_length:
            raise InvalidDatasetError(
                f"Length mismatch in '{self.target_key}': audio length is {audio_length} frames, "
                f"but metadata '{self.target_key}' has {len(targets)} elements."
            )

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Downsample frame targets to T' and evaluate downstream regressor.

        Args:
            train_representations: Tensor of shape (N_train, T', C').
            train_metadata: List of metadata dicts corresponding to each train clip.
            test_representations: Tensor of shape (N_test, T', C').
            test_metadata: List of metadata dicts corresponding to each test clip.

        Returns:
            dict[str, Any]: Frame-level regression metrics.
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

        _, target_len_train, feature_dim = train_representations.shape
        _, target_len_test, _ = test_representations.shape

        downsampled_train: list[np.ndarray] = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            downsampled = downsample_sequence(
                meta[self.target_key], target_len=target_len_train, mode="mean"
            )
            downsampled_train.append(downsampled)

        downsampled_test: list[np.ndarray] = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            downsampled = downsample_sequence(
                meta[self.target_key], target_len=target_len_test, mode="mean"
            )
            downsampled_test.append(downsampled)

        targets_train = np.array(downsampled_train, dtype=np.float32)
        targets_test = np.array(downsampled_test, dtype=np.float32)

        X_train = train_representations.detach().cpu().numpy().reshape(-1, feature_dim)
        y_train = targets_train.reshape(-1)

        X_test = test_representations.detach().cpu().numpy().reshape(-1, feature_dim)
        y_test = targets_test.reshape(-1)

        regressor_inst = self.regressor
        regressor_name = regressor_inst.__class__.__name__

        start_time = time.perf_counter()
        regressor_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = regressor_inst.evaluate(X_test, y_test)
        train_metrics = regressor_inst.evaluate(X_train, y_train)

        results = {
            "task": self.name,
            "regressor": regressor_name,
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
        """Examine the dataset in the context of frame regression.

        Validates the datasets, computes frame target distributions across train and test sets,
        and computes a trivial baseline (frame target mean predictor) expected metric.

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

        train_targets = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            for t in meta[self.target_key]:
                try:
                    train_targets.append(float(t))
                except (ValueError, TypeError) as e:
                    raise TaskError(
                        f"Sample at index {idx} frame target cannot be converted to float: {e}"
                    ) from e

        test_targets = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            for t in meta[self.target_key]:
                try:
                    test_targets.append(float(t))
                except (ValueError, TypeError) as e:
                    raise TaskError(
                        f"Sample at index {idx} frame target cannot be converted to float: {e}"
                    ) from e

        train_targets_np = np.array(train_targets, dtype=np.float32)
        test_targets_np = np.array(test_targets, dtype=np.float32)
        all_targets = (
            np.concatenate([train_targets_np, test_targets_np])
            if testset is not None
            else train_targets_np
        )

        frame_len_sample = (
            len(train_metadata[0][self.target_key]) if train_metadata else 0
        )
        train_mean = float(np.mean(train_targets_np))
        y_pred_baseline = np.full_like(test_targets_np, train_mean)

        baseline_mse = float(mean_squared_error(test_targets_np, y_pred_baseline))
        baseline_rmse = float(np.sqrt(baseline_mse))
        baseline_mae = float(mean_absolute_error(test_targets_np, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(test_targets_np, y_pred_baseline))
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
            "total_clips": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "frame_length": int(frame_len_sample),
            "total_frames": len(train_targets)
            + (len(test_targets) if testset is not None else 0),
            "train_frames": len(train_targets),
            "test_frames": len(test_targets) if testset is not None else 0,
            "target_key": self.target_key,
            "target_min": float(np.min(all_targets)),
            "target_max": float(np.max(all_targets)),
            "target_mean": float(np.mean(all_targets)),
            "target_std": float(np.std(all_targets)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }
